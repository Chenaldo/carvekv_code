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
        # Enable latent-based token eviction during prefill to compress the KV cache.
        # Each token's c_t^{KV} latent is scored by compute_latent_info_score (a blend
        # of L2 norm, negative entropy, per-dim variance, minus neighbour redundancy).
        # Tokens whose composite info score falls BELOW `threshold` are dropped before
        # the cache write. The number evicted is content-adaptive, NOT a fixed ratio.
        self.latent_eviction = True              # Set True to enable eviction during prefill
        self.latent_eviction_threshold = 0.3     # Drop tokens with info score below this
        self.latent_eviction_window = 4          # Neighbour radius for redundancy detection
        # Eviction is intentionally restricted to the ONE-SHOT prefill pass. During
        # autoregressive decode the cache is APPEND-ONLY: no scoring, no top-k, no
        # gather/slice. Per-token tensor slicing in decode would force repeated
        # non-contiguous memory copies and tank tokens/s, so it is disabled here.
        self.latent_eviction_prefill_only = True
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

    def _compute_keep_indices(self, compressed_kv: torch.Tensor) -> torch.Tensor:
        """[核心优化点 1：先验特征计算]

        仅在 Layer 0 调用一次。对当前 prefill chunk 的每个 token 的
        latent c_t^{KV} 用 compute_latent_info_score 评分，通过自适应阈值
        选出信息量足够的 token 位置索引，供所有层共享使用。

        Args:
            compressed_kv: [B, S, kv_lora_rank]  — 当前 chunk 的 latent 向量

        Returns:
            keep_indices: [B, keep_k]  — 按时序升序排列的保留位置
        """
        bsz, seq_len, _ = compressed_kv.shape

        # 复合信息量评分 [B, S]（沿用原有 compute_latent_info_score）
        info = compute_latent_info_score(
            compressed_kv, window=self.latent_eviction_window
        )

        # 自适应阈值过滤（非固定比例）
        keep_mask   = info >= self.latent_eviction_threshold       # [B, S] bool
        keep_counts = keep_mask.sum(dim=-1).clamp(min=1)           # [B]，统计每个 Sequence 中及格的 Token 数量
        keep_k      = int(keep_counts.max().item())                #找出当前 Batch 中，保留 Token 数量最多的那个 Sequence 的长度

        # 按信息量降序取 top-keep_k；超额槽位循环复用本行最优 token，
        # 而非引入低质 token（batch>1 时保证各行独立，bsz==1 时退化为精确 top-k）
        order    = info.argsort(dim=-1, descending=True)           # [B, S]
        slot     = torch.arange(keep_k, device=info.device)
        slot     = slot.unsqueeze(0) % keep_counts.unsqueeze(1)    # [B, keep_k]
        keep_idx = order.gather(1, slot)
        keep_idx = keep_idx.sort(dim=-1).values                    # 恢复时序顺序

        evicted = seq_len - keep_k
        logger.debug(
            f"Latent eviction [layer {self.layer_idx}]: "
            f"chunk_size={seq_len}, kept={keep_k}, evicted={evicted} "
            f"({evicted / seq_len * 100:.1f}%) | threshold={self.latent_eviction_threshold}"
        )
        return keep_idx  # [B, keep_k]

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
                # [核心优化点 3：跨层对齐]
                # Layer 0 计算驱逐索引并挂载到 Cache 对象上；其余层复用，
                # 保证所有层丢弃完全相同位置的 Token。
                if self.layer_idx == 0:
                    keep_indices = self._compute_keep_indices(c_kv_normed)
                    past_key_value.shared_keep_indices = keep_indices
                else:
                    keep_indices = getattr(past_key_value, "shared_keep_indices", None)

                if keep_indices is not None:
                    # [核心优化点 4：双轨同步切片（latent + k_pe 两条轨道）]
                    idx = keep_indices.unsqueeze(1).unsqueeze(-1)  # [B, 1, keep_k, 1]
                    trimmed_latent = cached_latent.gather(
                        2, idx.expand(-1, 1, -1, cached_latent.shape[-1])
                    )  # [B, 1, keep_k, kv_lora_rank]
                    trimmed_kpe = cached_kpe.gather(
                        2, idx.expand(-1, 1, -1, cached_kpe.shape[-1])
                    )  # [B, 1, keep_k, qk_rope_head_dim]

                    # 强制覆写存储的 cache（驱逐在此发生）
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

        # ── 权重吸收：拆分 kv_b_proj ──────────────────────────────────────
        # kv_b_proj.weight : [H*(qk_nope_head_dim + v_head_dim), kv_lora_rank]
        #   W_UK [H, qk_nope_head_dim, kv_lora_rank]  latent → k_nope
        #   W_UV [H, v_head_dim,       kv_lora_rank]  latent → v
        W_kv = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        W_UK = W_kv[:, : self.qk_nope_head_dim, :]  # [H, qk_nope_head_dim, kv_lora_rank]
        W_UV = W_kv[:, self.qk_nope_head_dim :, :]  # [H, v_head_dim,       kv_lora_rank]

        # ── 吸收后的 Query ─────────────────────────────────────────────────
        # q_nope^T k_nope = q_nope^T W_UK c = (W_UK^T q_nope)^T c
        # 令 q_abs = q_nope @ W_UK，直接与 latent 做内积，不再展开 K。
        # q_nope : [B, H, q_len, qk_nope_head_dim]
        # W_UK   : [H, qk_nope_head_dim, kv_lora_rank]
        # q_abs  : [B, H, q_len, kv_lora_rank]
        q_abs = torch.einsum("bhqd,hdr->bhqr", q_nope, W_UK)

        # ── 注意力分数 ─────────────────────────────────────────────────────
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
