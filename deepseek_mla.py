import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepseekV2Attention(nn.Module):
    # ... [初始化代码保持原样] ...

    def _compute_keep_indices(self, kv_states):
        """
        [核心优化点 1：先验特征计算]
        仅在 Layer 0 计算一次，利用 L2、方差等指标计算需要保留的 Token 索引。
        """
        bsz, _, seq_len, dim = kv_states.shape
        # 这里放置你的 _robust_normalize 和特征评分逻辑
        # info_score = w1*L2 + w2*Var + ...
        
        # 假设通过你的自适应阈值计算出要保留的 token 数量 keep_k
        # 这里用简单的 top-k 举例
        keep_k = int(seq_len * 0.7) # 比如保留 70%
        
        # 获取 top-k 的索引 (必须 sort 保证相对位置不变)
        # 伪代码： _, keep_indices = torch.topk(info_score, keep_k, dim=-1)
        # keep_indices, _ = torch.sort(keep_indices, dim=-1)
        
        # 这里为了演示返回一个 dummy 索引 (实际替换为你的逻辑)
        keep_indices = torch.arange(seq_len - keep_k, seq_len, device=kv_states.device).unsqueeze(0).expand(bsz, -1)
        return keep_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        # ... [其他参数]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()

        # ... [原生代码：计算 query_states, kv_states (Latent), k_pe (RoPE)] ...

        if past_key_value is not None:
            # 1. 正常把当前的 token 更新进 Cache (原生逻辑)
            # 注意：DeepSeek V2 的 cache 通常把 kv_states 存 key_cache，k_pe 存 value_cache
            kv_states, k_pe = past_key_value.update(kv_states, k_pe, self.layer_idx, {"cache_position": cache_position})

            # =====================================================================
            # [核心优化点 2：区分 Prefill 与 Decode]
            # 仅在输入长度 > 1 时（Prefill 阶段）触发驱逐，Decode 阶段（q_len == 1）直接跳过，防止碎片化
            # =====================================================================
            if q_len > 1 and getattr(self.config, "use_latent_eviction", True):
                
                # =================================================================
                # [核心优化点 3：跨层对齐 (Cross-layer Alignment)]
                # 必须保证所有层的 Cache 丢弃的是相同位置的 Token，否则 DynamicCache 长度不一会崩溃
                # =================================================================
                if self.layer_idx == 0:
                    # 只有第 0 层计算驱逐索引
                    keep_indices = self._compute_keep_indices(kv_states)
                    # 将计算结果挂载到 past_key_value 对象上，跨层共享
                    past_key_value.shared_keep_indices = keep_indices
                else:
                    # 其他层直接复用第 0 层的驱逐索引
                    keep_indices = getattr(past_key_value, "shared_keep_indices", None)

                if keep_indices is not None:
                    # =============================================================
                    # [核心优化点 4：Latent 与 RoPE 双轨同步切片]
                    # 必须使用同一个 keep_indices 同时切片 kv_states 和 k_pe
                    # =============================================================
                    
                    # 扩展索引维度以匹配 kv_states [bsz, 1, seq_len, kv_lora_rank]
                    idx_kv = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, kv_states.shape[-1])
                    kv_states = torch.gather(kv_states, dim=2, index=idx_kv)

                    # 扩展索引维度以匹配 k_pe [bsz, num_heads, seq_len, head_dim]
                    idx_pe = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, k_pe.shape[1], -1, k_pe.shape[-1])
                    k_pe = torch.gather(k_pe, dim=2, index=idx_pe)

                    # 强行覆写底层的 Cache
                    past_key_value.key_cache[self.layer_idx] = kv_states
                    past_key_value.value_cache[self.layer_idx] = k_pe

                    # =============================================================
                    # [核心优化点 5：修复 DynamicCache 内置状态]
                    # 仅在最后一层更新 _seen_tokens，防止覆盖冲突
                    # =============================================================
                    if self.layer_idx == self.config.num_hidden_layers - 1:
                        past_key_value._seen_tokens = kv_states.shape[2]

        # ... [后续的原生 Attention 矩阵乘法计算，保持原样] ...
        
        return attn_output, attn_weights, past_key_value