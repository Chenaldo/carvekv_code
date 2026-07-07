# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV2
class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        ##一口气把公式 (38) 的内容部分（nope）和公式 (39) 准备做 RoPE 的位置部分（rope）都生成出来了
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            ##将hidden_size映射到q_lora_rank维度
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            ##用来对q_a_proj的输出进行归一化
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            ##将q_lora_rank维度映射到num_heads * q_head_dim维度，向上解压，输出维度为 num_heads * q_head_dim
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        ##从输入h_t(hidden_size) 中一次性投影出两根向量的拼接体
        ##第一部分维度是 kv_lora_rank，公式 (41) 中大名鼎鼎的 Latent
        ##第二部分维度是 qk_rope_head_dim，它是公式 (43) 中用于位置编码的k_t^R
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)

        ##(self.q_head_dim - self.qk_rope_head_dim) 其实就等于 qk_nope_head_dim（即 Key 的内容维度k_t^C）
        ##再加上 v_head_dim（即 Value 的内容维度v_t^C）。这就完美对应了公式 (42) 和 (45) 的解压过程
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        # ----- 预计算 kv_b_proj 的权重分割（weight absorption 用）-----
        # 在 __init__ 里做一次，避免每次 forward 都重新 view。
        # 注意：这是 view（零拷贝），不新增参数，不影响梯度。
        # W_UK [H, qk_nope_head_dim, kv_lora_rank] : latent → k_nope
        # W_UV [H, v_head_dim,       kv_lora_rank] : latent → v
        # （实际拆分在 forward 里执行，因为 kv_b_proj 的权重在 __init__ 结束后才确定）

        # ----- Latent Eviction Config (MLA KV Cache Compression Research) -----
        # Budget is decided per-sequence via the elbow (knee) of the sorted
        # importance-score curve, ensuring at least `coverage_floor` of total
        # importance mass is retained.  Replaces the old fixed threshold.
        self.latent_eviction = True              # Set True to enable eviction during prefill
        self.latent_eviction_window = 4          # Neighbour radius for redundancy (fallback path only)
        # ---- Adaptive budget: knee detection + coverage floor ----------------
        # keep_k = max( knee_k, cov_k ) where:
        #   knee_k = argmax_k[ C(k) - k/n ]  (elbow of cumulative-mass curve)
        #   cov_k  = min k s.t. C(k) >= coverage_floor
        self.latent_eviction_coverage_floor = 0.85  # min fraction of importance mass to retain
        self.latent_eviction_min_keep       = 0.30  # hard lower bound on mid-token keep ratio
        self.latent_eviction_max_keep       = 0.98  # hard upper bound (always evict >= 2%)
        # Eviction is intentionally restricted to the ONE-SHOT prefill pass. During
        # autoregressive decode the cache is APPEND-ONLY: no scoring, no top-k, no
        # gather/slice. Per-token tensor slicing in decode would force repeated
        # non-contiguous memory copies and tank tokens/s, so it is disabled here.
        self.latent_eviction_prefill_only = True

        # ---- Sink / Recency protection (applied on top of any scoring method) ----
        # Attention sinks: empirically, the first few tokens accumulate disproportionately
        # large attention across layers ("sink" phenomenon). Their L2 norm is often
        # unremarkable, so statistical scoring systematically under-values them.
        # We protect them unconditionally.
        # Recent tokens: the last few tokens in the chunk have only been attended to by
        # themselves (column sum is tiny by construction), so query-aware scoring also
        # under-values them. Protect them unconditionally as well.
        self.latent_eviction_sink_tokens     = 4    # Always keep first N tokens
        self.latent_eviction_recent_tokens   = 16   # Always keep last N tokens
        # Decision layer: eviction indices are computed at this layer and broadcast
        # to all layers. Layers 0..D-1 are retroactively trimmed on the spot.
        # Empirically Config-B (layer-0 shared) degrades heavily; features stabilise
        # around layer 6 in DeepSeek-V2-Lite (oracle gap ≤ 0.5 PPL from L6 onward).
        self.latent_eviction_decision_layer  = 6    # Layer that computes the decision
        # Long-sequence guard: full [n_mid, n_mid] scoring matrix is infeasible at
        # 32k+ prefill (32k × 16 heads × fp32 ≈ 68 GB → OOM).
        # Subsample K query rows instead; column-sum becomes an unbiased estimator.
        # Memory: O(K × n) vs O(n²).  32k @ K=512 ≈ 1 GB;  128k @ K=512 ≈ 4 GB.
        # Lower K to 128–256 if GPU memory is tight; higher K for less rank noise.
        self.latent_eviction_score_queries   = 512  # Max query rows used for scoring
        # -----------------------------------------------------------------------


##根据模型的配置，初始化对应的旋转位置编码（Rotary Position Embedding, 简称 RoPE）模块
##注意力机制（Attention）本身是无法感知词语先后顺序的，必须通过“位置编码”来告诉模型每个词的位置
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    @staticmethod
    def _adaptive_budget(
        info_mid: torch.Tensor,
        coverage_floor: float,
        min_keep: float,
        max_keep: float,
    ) -> int:
        """
        Adaptive keep-k via elbow (knee) detection on the cumulative-mass curve.

        Algorithm
        ---------
        1. Shift info_mid to >= 0 and normalise to a probability mass distribution.
        2. Sort descending; form the concave cumulative-mass curve C(k), k = 1..n.
        3. Knee = argmax_k [ C(k) - k/n ].
           Geometry: the point on the curve furthest above the diagonal (0,0)->(1,1).
           · Concentrated distributions -> knee is early   (aggressive eviction)
           · Flat distributions         -> knee is central (moderate eviction)
        4. Coverage floor: smallest k_cov s.t. C(k_cov) >= coverage_floor.
        5. keep_k = max(knee_k, k_cov) -- adapts to content, never below the floor.
        6. Clamp to [min_keep, max_keep] x n_mid.

        Why max(knee, coverage_floor)?
          For near-uniform importance, cumsum is almost linear and the knee lands
          around n/2, which would evict too much.  The coverage floor of 0.90
          correctly prevents this and keeps enough tokens.
        """
        n      = info_mid.shape[1]
        device = info_mid.device

        # 1. Shift to non-negative, normalise to mass distribution
        scores = info_mid.float()
        scores = scores - scores.min(dim=-1, keepdim=True).values   # [B, n] >= 0
        total  = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        mass   = scores / total                                      # [B, n], row-sums to 1

        # 2. Sort descending, cumulative sum
        sorted_mass = mass.sort(dim=-1, descending=True).values      # [B, n]
        cumsum      = sorted_mass.cumsum(dim=-1)                     # [B, n], 0 -> 1

        # 3. Knee: argmax_k( C(k) - k/n ),  k in {1..n}
        x      = torch.arange(1, n + 1, device=device, dtype=torch.float32) / n  # [n]
        dist   = cumsum - x.unsqueeze(0)                             # [B, n]
        knee_k = int(dist.argmax(dim=-1).max().item()) + 1           # 1-indexed, batch-max

        # 4. Coverage floor: min k s.t. C(k) >= coverage_floor
        cov_k  = int(
            (cumsum >= coverage_floor).long().argmax(dim=-1).max().item()
        ) + 1

        # 5. Take the larger (guarantee coverage floor)
        keep_k = max(knee_k, cov_k)

        # 6. Hard ratio bounds
        keep_k = max(keep_k, max(1, int(n * min_keep)))
        keep_k = min(keep_k, max(1, int(n * max_keep)))

        return keep_k


def compute_latent_info_score(
    compressed_kv: torch.Tensor,
    window: int = 4,
    w_norm: float = 0.4,
    w_entropy: float = 0.3,
    w_variance: float = 0.3,
    w_redundancy: float = 0.5,
) -> torch.Tensor:
    """
    Compute a composite *information score* for each latent KV vector (c_t^{KV}).

    Instead of a single metric, this blends several complementary signals so that
    a token is considered low-information if it is weak across the board OR is
    largely a duplicate of its neighbours. Tokens whose score falls below a
    threshold can then be evicted before the cache write ("read, forget the
    filler, remember the key parts").

    Signals (each min-max normalized to [0, 1] across the sequence):
        - L2 norm            : magnitude of the latent  (larger = richer)
        - negative entropy   : peakedness of softmax(c) (sharper = more specific)
        - per-dim variance   : spread across dimensions  (flatter = more redundant)
    Penalty:
        - neighbour redundancy: max cosine similarity to tokens within `window`
                                (high similarity = duplicate = less worth keeping)

    Composite:
        info = w_norm*norm + w_entropy*neg_entropy + w_variance*variance
               - w_redundancy*redundancy
    The positive weights are intended to sum to 1.0, so the informative part lies
    in [0, 1] and the redundancy penalty pulls duplicates down toward (and below) 0.

    Args:
        compressed_kv: [B, S, kv_lora_rank]  latent vectors after kv_a_proj_with_mqa
        window:        neighbour radius for redundancy detection
        w_norm/w_entropy/w_variance: weights of the informativeness signals
        w_redundancy:  weight of the neighbour-redundancy penalty

    Returns:
        info: [B, S]  composite information score, higher = more worth keeping
    """
    x = compressed_kv.float()
    bsz, seq_len, _ = x.shape

    # --- Signal 1: L2 norm (magnitude / richness) ---
    norm = x.norm(dim=-1)                                      # [B, S]

    # --- Signal 2: negative normalized entropy (distribution peakedness) ---
    prob = F.softmax(x, dim=-1)
    entropy = -(prob * (prob + 1e-10).log()).sum(dim=-1)      # [B, S]
    neg_entropy = -entropy                                     # peaked = more info

    # --- Signal 3: per-dimension variance (spread / non-flatness) ---
    variance = x.var(dim=-1)                                   # [B, S]

    # --- Penalty: neighbour redundancy via local cosine similarity ---
    # Compare each token only with up to `window` neighbours on each side using
    # cheap shifted dot-products (O(S * window * D)), avoiding a full S x S matrix.
    x_unit = F.normalize(x, dim=-1)                            # [B, S, D]
    redundancy = torch.zeros(bsz, seq_len, device=x.device, dtype=x.dtype)
    for offset in range(1, max(1, window) + 1):
        if offset >= seq_len:
            break
        # cosine similarity between token t and token (t - offset)
        sim = (x_unit[:, offset:, :] * x_unit[:, :-offset, :]).sum(dim=-1)  # [B, S-offset]
        # redundancy is symmetric: both tokens in the pair are "duplicates"
        redundancy[:, offset:] = torch.maximum(redundancy[:, offset:], sim)
        redundancy[:, :-offset] = torch.maximum(redundancy[:, :-offset], sim)

    # --- Composite information score ---
    info = (
        w_norm * _robust_normalize(norm)
        + w_entropy * _robust_normalize(neg_entropy)
        + w_variance * _robust_normalize(variance)
        - w_redundancy * _robust_normalize(redundancy)
    )
    return info                                               # [B, S]



    def _compute_keep_indices(
        self,
        compressed_kv: torch.Tensor,
        q_abs: Optional[torch.Tensor] = None,
        W_UV: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[核心优化点 1：Value-Aware Query-Aware 评分 + Sink/Recency 保护]

        在 latent_eviction_decision_layer 调用一次，供所有层共享驱逐决策。

        评分公式（q_abs 和 W_UV 均可用时）：
          Importance[j] = mean_H( Σ_i prob_{i,j} × ‖W_UV @ c_j‖₂ )

        Budget 决策：肘部法（Knee Detection）+ 覆盖率下界
          · 对 info_mid 归一化累积质量曲线 C(k)，求肘部：
              knee_k = argmax_k[ C(k) − k/n ]   （曲线距对角线最远点）
          · cov_k  = min k s.t. C(k) >= latent_eviction_coverage_floor
          · keep_k = max(knee_k, cov_k)  —— 自适应，不低于覆盖率下界

        退化策略（按可用性自动选择）：
          · 仅 q_abs          → 纯列求和（无 Value 权重）
          · 仅 W_UV / 均无    → 统计评分（compute_latent_info_score），可乘 Value 范数

        无条件保护：
          · 前 latent_eviction_sink_tokens   个 token（attention sink）
          · 后 latent_eviction_recent_tokens 个 token（recency 偏差）

        Args:
            compressed_kv : [B, S, kv_lora_rank]          归一化 latent
            q_abs         : [B, H, S, kv_lora_rank] | None  absorbed query
            W_UV          : [H, v_head_dim, kv_lora_rank] | None  Value 投影矩阵

        Returns:
            keep_indices  : [B, keep_k]  按时序升序
        """
        bsz, seq_len, R = compressed_kv.shape
        device = compressed_kv.device

        # ── Sink / Recency 保护边界 ────────────────────────────────────────
        n_sink   = min(self.latent_eviction_sink_tokens,   seq_len)
        n_recent = min(self.latent_eviction_recent_tokens, max(0, seq_len - n_sink))
        n_mid    = seq_len - n_sink - n_recent

        if n_mid <= 0:
            keep_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1).clone()
            logger.debug(
                f"Latent eviction [layer {self.layer_idx}]: "
                f"chunk={seq_len}, no eviction needed (mid={n_mid})"
            )
            return keep_idx

        mid_latent = compressed_kv[:, n_sink : seq_len - n_recent, :]  # [B, n_mid, R]

        # ── Value 范数（预先计算，与评分方法正交）────────────────────────────
        val_norm = None
        if W_UV is not None:
            val_proj = torch.einsum("bnr,hdr->bhnd", mid_latent, W_UV).float()
            val_norm = val_proj.norm(dim=-1).mean(dim=1)              # [B, n_mid]
            v_mean   = val_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            val_norm = val_norm / v_mean

        # ── 主评分（先算分再决定 budget，顺序与旧版相反）──────────────────
        if q_abs is not None:
            # ── Query-Aware：因果 attention 列求和 × Value 范数（可选）───────
            q_mid = q_abs[:, :, n_sink : seq_len - n_recent, :]       # [B, H, n_mid, R]
            K = min(self.latent_eviction_score_queries, n_mid)

            if K < n_mid:
                sample_pos = torch.randperm(n_mid, device=device)[:K].sort().values
                q_scored   = q_mid[:, :, sample_pos, :]                # [B, H, K, R]
                causal_mask = (
                    torch.arange(n_mid, device=device).unsqueeze(0)
                    > sample_pos.unsqueeze(1)
                )  # [K, n_mid] bool
            else:
                q_scored    = q_mid
                causal_mask = ~torch.tril(
                    torch.ones(n_mid, n_mid, device=device, dtype=torch.bool)
                )

            attn_logits = torch.matmul(
                q_scored, mid_latent.unsqueeze(1).transpose(-1, -2)
            ) * self.softmax_scale
            attn_logits = attn_logits.masked_fill(causal_mask[None, None], float("-inf"))
            attn_probs  = torch.softmax(attn_logits, dim=-1, dtype=torch.float32)
            col_sum     = attn_probs.sum(dim=2)                        # [B, H, n_mid]

            if val_norm is not None:
                info_mid = (col_sum * val_norm.unsqueeze(1)).mean(dim=1)  # [B, n_mid]
            else:
                info_mid = col_sum.mean(dim=1)
        else:
            # ── 统计 Fallback（q_abs 不可用时，仅此路径才调用统计评分）──────
            info_stat = compute_latent_info_score(
                compressed_kv, window=self.latent_eviction_window)
            info_mid = info_stat[:, n_sink : seq_len - n_recent].float()
            if val_norm is not None:
                info_mid = info_mid * val_norm

        # ── 自适应 Budget：肘部法 + 覆盖率下界 ────────────────────────────
        n_mid_keep = self._adaptive_budget(
            info_mid,
            coverage_floor = self.latent_eviction_coverage_floor,
            min_keep       = self.latent_eviction_min_keep,
            max_keep       = self.latent_eviction_max_keep,
        )

        if n_mid_keep >= n_mid:
            keep_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1).clone()
            logger.debug(
                f"Latent eviction [layer {self.layer_idx}]: "
                f"chunk={seq_len}, no eviction needed "
                f"(n_mid_keep={n_mid_keep} >= n_mid={n_mid})"
            )
            return keep_idx

        # ── Top-K 选中间段，拼合 Sink + 中间 + Recent ─────────────────────
        mid_order    = info_mid.argsort(dim=-1, descending=True)[:, :n_mid_keep]
        mid_keep_idx = (mid_order + n_sink).sort(dim=-1).values

        sink_idx   = torch.arange(n_sink,              device=device).unsqueeze(0).expand(bsz, -1)
        recent_idx = torch.arange(seq_len - n_recent, seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        keep_idx   = torch.cat([sink_idx, mid_keep_idx, recent_idx], dim=1).sort(dim=-1).values

        evicted = seq_len - keep_idx.shape[1]
        mode = ("value+query-aware" if (q_abs is not None and W_UV is not None) else
                "query-aware"       if q_abs is not None else
                "value+statistical" if W_UV  is not None else
                "statistical")
        logger.debug(
            f"Latent eviction [layer {self.layer_idx}] ({mode}): "
            f"chunk={seq_len}, sink={n_sink}, recent={n_recent}, "
            f"mid_kept={n_mid_keep}/{n_mid}, evicted={evicted} ({evicted/seq_len*100:.1f}%)"
        )
        return keep_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Weight-Absorption MLA Forward — stores compressed latent in KV cache.

        Cache layout (per layer):
            key_cache   [B, 1, S, kv_lora_rank]     ← kv_a_layernorm(c_t^{KV})
            value_cache [B, 1, S, qk_rope_head_dim]  ← RoPE-encoded k_t^R

        Attention via weight absorption (no K/V expansion at inference time):
            content_score = q_absorbed @ cached_latent^T
            where q_absorbed = einsum(q_nope, W_UK)    推导：
              q_nope^T k_nope = q_nope^T (W_UK c) = (W_UK^T q_nope)^T c

            rope_score = q_pe @ cached_kpe^T

            output = einsum(attn @ cached_latent, W_UV)

        Memory: 576 floats/token vs 4096 (expanded K/V) — ~7× smaller per layer.
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead."
            )
        bsz, q_len, _ = hidden_states.size()

        # ── Query projection ───────────────────────────────────────────────
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # q_nope : [B, H, q_len, qk_nope_head_dim]
        # q_pe   : [B, H, q_len, qk_rope_head_dim]

        # ── KV compression → latent + k_pe ────────────────────────────────
        raw = self.kv_a_proj_with_mqa(hidden_states)
        c_kv, k_pe_raw = torch.split(raw, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv_normed = self.kv_a_layernorm(c_kv)                       # [B, q_len, kv_lora_rank]
        k_pe = k_pe_raw.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        # k_pe : [B, 1, q_len, qk_rope_head_dim]

        # ── kv_seq_len & RoPE ──────────────────────────────────────────────
        if self.layer_idx is None and past_key_value is not None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using "
                f"{self.__class__.__name__} for auto-regressive decoding with k/v caching, "
                "please make sure to initialize the attention class with a layer index."
            )

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        # 驱逐后物理 cache 短于真实序列长度；position_ids 仍携带真实绝对位置，
        # 须把 rotary 表扩展到真实最大位置，否则 cos[position_ids] 越界。
        rotary_seq_len = kv_seq_len
        if self.latent_eviction and position_ids is not None:
            rotary_seq_len = max(rotary_seq_len, int(position_ids.max()) + 1)

        cos, sin = self.rotary_emb(k_pe, seq_len=rotary_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        # k_pe : [B, 1, q_len, qk_rope_head_dim]  (RoPE applied)

        # ── 权重吸收：提前拆分 kv_b_proj（q_abs 供 eviction scoring 使用）──────
        # 必须在 cache update 前计算，否则 _compute_keep_indices 拿不到 q_abs。
        # view 是零拷贝操作，不增加参数，不影响梯度。
        W_kv = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        W_UK = W_kv[:, : self.qk_nope_head_dim, :]  # [H, qk_nope_head_dim, kv_lora_rank]
        W_UV = W_kv[:, self.qk_nope_head_dim :, :]  # [H, v_head_dim,       kv_lora_rank]

        # q_abs = q_nope @ W_UK  → [B, H, q_len, kv_lora_rank]
        # 推导：q_nope^T k_nope = q_nope^T W_UK c = (W_UK^T q_nope)^T c
        q_abs = torch.einsum("bhqd,hdr->bhqr", q_nope, W_UK)

        # ── Update cache: 存 latent（key_cache）和 k_pe（value_cache）───────
        # key_cache   ← c_kv_normed  [B, 1, S, kv_lora_rank=512]
        # value_cache ← k_pe         [B, 1, S, qk_rope_head_dim=64]
        # 两者合计 576 维/token，比展开 K/V 的 4096 维节省约 7×。
        c_kv_normed_4d = c_kv_normed.unsqueeze(1)  # [B, 1, q_len, kv_lora_rank]

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}

            # 必须在 update 前检查：update 后本层 cache 必然非空，检查失效
            _is_first_prefill = (
                q_len > 1
                and self.latent_eviction
                and past_key_value.get_seq_length(self.layer_idx) == 0
            )

            # update() 拼接 past + current，返回完整序列的 latent 和 k_pe
            # cached_latent : [B, 1, kv_seq_len, kv_lora_rank]
            # cached_kpe    : [B, 1, kv_seq_len, qk_rope_head_dim]
            cached_latent, cached_kpe = past_key_value.update(
                c_kv_normed_4d, k_pe, self.layer_idx, cache_kwargs
            )

            if _is_first_prefill:
                # [核心优化点 3：决策层（Layer D）驱逐 + 溯源修剪]
                # · Layer < D  → 稠密写入，等待决策层
                # · Layer == D → 计算 keep_indices（Value-Aware Query-Aware），
                #                同时回头修剪已写入的 Layer 0..D-1
                # · Layer > D  → 复用 shared_keep_indices
                # 改用 Layer 6 而非 Layer 0：实验表明 Layer 0 特征未稳定，
                # Config-B PPL 退化严重；Layer 6 起与 Oracle 差距 ≤ 0.5 PPL。
                D = self.latent_eviction_decision_layer

                if self.layer_idx == D:
                    # 传入 q_abs 和 W_UV 启用 Value-Aware Query-Aware 评分
                    keep_indices = self._compute_keep_indices(
                        c_kv_normed, q_abs=q_abs, W_UV=W_UV
                    )
                    past_key_value.shared_keep_indices = keep_indices

                    # [核心优化点 3b：溯源修剪 Layer 0..D-1]
                    # Layer D 算出索引后立即回头裁剪前面已写入的 cache。
                    # 各层 key_cache / value_cache 此时存的是完整 prefill 长度，
                    # 用同一份 keep_indices gather 即可对齐到相同物理长度。
                    _ret_idx = keep_indices.unsqueeze(1).unsqueeze(-1)  # [B,1,keep_k,1]
                    for _prev in range(D):
                        if len(past_key_value.key_cache) > _prev:
                            _pl = past_key_value.key_cache[_prev]
                            _pk = past_key_value.value_cache[_prev]
                            past_key_value.key_cache[_prev]   = _pl.gather(
                                2, _ret_idx.expand(-1, 1, -1, _pl.shape[-1])
                            )
                            past_key_value.value_cache[_prev] = _pk.gather(
                                2, _ret_idx.expand(-1, 1, -1, _pk.shape[-1])
                            )

                elif self.layer_idx > D:
                    keep_indices = getattr(past_key_value, "shared_keep_indices", None)
                else:
                    # layer_idx < D：稠密，不驱逐
                    keep_indices = None

                if keep_indices is not None:
                    # [核心优化点 4：双轨同步切片（latent + k_pe 两条轨道）]
                    idx = keep_indices.unsqueeze(1).unsqueeze(-1)  # [B, 1, keep_k, 1]
                    trimmed_latent = cached_latent.gather(
                        2, idx.expand(-1, 1, -1, cached_latent.shape[-1])
                    )  # [B, 1, keep_k, kv_lora_rank]
                    trimmed_kpe = cached_kpe.gather(
                        2, idx.expand(-1, 1, -1, cached_kpe.shape[-1])
                    )  # [B, 1, keep_k, qk_rope_head_dim]

                    # 强制覆写当前层 cache（驱逐在此发生）
                    # 局部变量 cached_latent / cached_kpe 仍指向覆写前的完整张量，
                    # 当前 forward 的 attention 依然对全量 q_len token 计算；
                    # 裁剪后的 cache 仅供后续 decode 步骤使用。
                    past_key_value.key_cache[self.layer_idx]   = trimmed_latent
                    past_key_value.value_cache[self.layer_idx] = trimmed_kpe

                    # [核心优化点 5：修复 DynamicCache 内置状态]
                    if self.layer_idx == self.config.num_hidden_layers - 1:
                        past_key_value._seen_tokens = trimmed_latent.shape[2]
        else:
            # 无 cache：只用当前 q_len 个 token
            cached_latent = c_kv_normed_4d  # [B, 1, q_len, kv_lora_rank]
            cached_kpe    = k_pe             # [B, 1, q_len, qk_rope_head_dim]

        # ── 注意力分数 ─────────────────────────────────────────────────────
        # W_UK / W_UV / q_abs 已在 cache update 前计算完毕（见上方）。
        # cached_latent : [B, 1, kv_seq_len, kv_lora_rank]  ← 广播到 H 个头
        # cached_kpe    : [B, 1, kv_seq_len, qk_rope_head_dim]
        content_score = torch.matmul(q_abs, cached_latent.transpose(-1, -2))
        rope_score    = torch.matmul(q_pe,  cached_kpe.transpose(-1, -2))
        attn_weights  = (content_score + rope_score) * self.softmax_scale
        # [B, H, q_len, kv_seq_len]

        kv_seq_len = cached_latent.shape[2]  # 以实际 cached_latent 为准

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                f"but is {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_nope.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # ── 输出：通过吸收后的 V 权重聚合 ─────────────────────────────────
        # output = attn @ v = attn @ (W_UV c) = (attn @ c) @ W_UV^T
        # 先对 latent 做加权求和，再用 W_UV 投影一次，不展开所有 S 个 v 向量。
        # attn_weights  : [B, H, q_len, kv_seq_len]
        # cached_latent : [B, 1, kv_seq_len, kv_lora_rank]  ← 广播到 H 个头
        weighted_latent = torch.matmul(attn_weights, cached_latent)
        # [B, H, q_len, kv_lora_rank]
        attn_output = torch.einsum("bhqr,hdr->bhqd", weighted_latent, W_UV)
        # [B, H, q_len, v_head_dim]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
