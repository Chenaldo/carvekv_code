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

        # ----- Latent Eviction Config (MLA KV Cache Compression Research) -----
        # Enable latent-based token eviction during prefill to compress the KV cache.
        # Each token's c_t^{KV} latent is scored by compute_latent_info_score (a blend
        # of L2 norm, negative entropy, per-dim variance, minus neighbour redundancy).
        # Tokens whose composite info score falls BELOW `threshold` are dropped before
        # the cache write. The number evicted is content-adaptive, NOT a fixed ratio.
        self.latent_eviction = False             # Set True to enable eviction during prefill
        self.latent_eviction_threshold = 0.3     # Drop tokens with info score below this
        self.latent_eviction_window = 4          # Neighbour radius for redundancy detection
        # Eviction is intentionally restricted to the ONE-SHOT prefill pass. During
        # autoregressive decode the cache is APPEND-ONLY: no scoring, no top-k, no
        # gather/slice. Per-token tensor slicing in decode would force repeated
        # non-contiguous memory copies and tank tokens/s, so it is disabled here.
        self.latent_eviction_prefill_only = True
        # -----------------------------------------------------------------------

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
        keep_counts = keep_mask.sum(dim=-1).clamp(min=1)           # [B]
        keep_k      = int(keep_counts.max().item())

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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            ##将hidden_size h_t 映射到q_lora_rank维度，见thinking 1 2 式
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        ##输入C_t^q,一次性算完，分割出q_t^c和q_t^R
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        ###输入h_t,一次性算完，分割出k_t^R和Latent(C_t^kv),见thinking 3 式
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        ###将Latent(C_t^kv)输入一次性算完，解压分割成k_t^c和v_t^c,见thinking 4 式
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # With latent eviction the physical cache (hence kv_seq_len) is SHORTER than
        # the true sequence length, yet position_ids still carry TRUE absolute
        # positions. The rotary tables must be long enough to be indexed by those
        # true positions, otherwise cos[position_ids] goes out of bounds.
        rotary_seq_len = kv_seq_len
        if self.latent_eviction and position_ids is not None:
            rotary_seq_len = max(rotary_seq_len, int(position_ids.max()) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        ##施加 RoPE 位置编码
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)


        ##拼接完整 Q 和 K
        ##query_states = [q_nope | q_pe]                            # 公式(40)
        ##key_states   = [k_nope | k_pe]                              公式(44)
        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models

            # 在 update 之前判断是否属于首次 prefill（本层 cache 尚为空）
            # 必须在 update 前检查：update 之后本层 cache 必然非空，检查会失效
            _is_first_prefill = (
                q_len > 1
                and self.latent_eviction
                and not (
                    len(past_key_value.key_cache) > self.layer_idx
                    and past_key_value.key_cache[self.layer_idx].numel() > 0
                )
            )

            # [核心优化点 2：区分 Prefill 与 Decode]
            # 先无条件将完整 token 写入 Cache。Decode 阶段（q_len==1）或
            # chunked-prefill 续写阶段到此即结束，不触发任何评分/切片，
            # 彻底杜绝 per-token 非连续内存拷贝导致的 tokens/s 下降。
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            if _is_first_prefill:
                # [核心优化点 3：跨层对齐 (Cross-layer Alignment)]
                # Layer 0 计算驱逐索引并挂载到 Cache 对象上；其余层直接复用，
                # 保证所有层丢弃的是完全相同位置的 Token，DynamicCache
                # 各层长度始终一致，避免后续 decode 因层间长度不一而崩溃。
                if self.layer_idx == 0:
                    keep_indices = self._compute_keep_indices(compressed_kv)
                    past_key_value.shared_keep_indices = keep_indices
                else:
                    keep_indices = getattr(past_key_value, "shared_keep_indices", None)

                if keep_indices is not None:
                    # [核心优化点 4：双轨同步切片]
                    # 用同一份 keep_indices 同时对 key_cache 和 value_cache 做 gather，
                    # 确保 K/V 位置完全对齐，避免错位导致注意力计算错误。
                    idx = keep_indices.unsqueeze(1).unsqueeze(-1)  # [B, 1, keep_k, 1]
                    trimmed_key = key_states.gather(
                        2, idx.expand(-1, self.num_heads, -1, key_states.shape[-1])
                    )   # [B, H, keep_k, q_head_dim]
                    trimmed_val = value_states.gather(
                        2, idx.expand(-1, self.num_heads, -1, value_states.shape[-1])
                    )   # [B, H, keep_k, v_head_dim]

                    # 强制覆写底层 Cache（驱逐在此发生）
                    past_key_value.key_cache[self.layer_idx]   = trimmed_key
                    past_key_value.value_cache[self.layer_idx] = trimmed_val

                    # [核心优化点 5：修复 DynamicCache 内置状态]
                    # update() 在 layer 0 时已将 _seen_tokens 加上了 q_len，
                    # 在最后一层统一修正为实际保留数量，防止多层分别写引发竞争。
                    if self.layer_idx == self.config.num_hidden_layers - 1:
                        past_key_value._seen_tokens = trimmed_key.shape[2]

                    # key_states / value_states 保持完整 [B, H, q_len, D]，
                    # 当前 prefill 对全量 token 做注意力，保证 intra-chunk 注意力精度；
                    # 已剪枝的 Cache 仅作用于后续 Decode（更短 KV 序列 → 更低显存）。

        ##计算注意力权重attn_weights = Q @ K^T * softmax_scale
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        ##加权聚合 V
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        ##输出投影
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value