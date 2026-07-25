# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    is_torch_fx_available = lambda: False
from .configuration_deepseek import DeepseekV2Config
import torch.distributed as dist
import numpy as np

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # type: ignore
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa  # type: ignore


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV2Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekV2RMSNorm)


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV2MLP(
                            config, intermediate_size=config.moe_intermediate_size
                        )
                        if i >= self.ep_rank * self.experts_per_rank
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV2MLP(
                        config, intermediate_size=config.moe_intermediate_size
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _robust_normalize(t: torch.Tensor) -> torch.Tensor:
    """Outlier-robust normalize a [B, S] tensor to (0, 1) along the sequence dim.

    Min-max collapses under Massive Outliers (an activation outlier with L2 norm
    ~100x normal would saturate to 1.0 and crush every useful token toward 0,
    which threshold-eviction would then wrongly drop). We instead use a robust
    z-score built from the MEDIAN and MAD (median absolute deviation) — neither is
    dragged by extreme outliers — then squash with a sigmoid so outliers SATURATE
    near 1.0 while ordinary tokens keep a meaningful spread around 0.5.
    """
    med = t.median(dim=-1, keepdim=True).values
    mad = (t - med).abs().median(dim=-1, keepdim=True).values
    z = (t - med) / (1.4826 * mad + 1e-6)          # 1.4826: MAD->std consistency
    return torch.sigmoid(z)




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
        self.latent_eviction_coverage_floor  = 0.55  # min fraction of importance mass to retain
        self.latent_eviction_min_keep        = 0.20  # hard lower bound on mid-token keep ratio
        self.latent_eviction_max_keep        = 0.98  # hard upper bound (always evict >= 2%)
        # ---- Multi-layer committee scoring ----------------------------------
        # Layers in score_layers each compute importance scores independently.
        # decision_layer averages all committee scores before the final decision.
        # Averaging over 5 layers reduces per-layer noise (early layers are less
        # stable; late layers over-smooth).
        self.latent_eviction_score_layers    = [4, 6, 8, 12, 16]
        self.latent_eviction_decision_layer  = 16   # final aggregation layer (was 6)
        # ---- Soft eviction: Super Tokens ------------------------------------
        # Evicted tokens are NOT discarded; every pool_ratio consecutive evicted
        # tokens are merged into one super token and appended to the cache.
        #   content (latent c): importance-score weighted softmax average
        #   position (k_pe):    uniform average (RoPE phases must stay coherent)
        # cache layout after eviction: [sink | kept_mid | recent] + [super ...]
        self.latent_eviction_pool_ratio       = 4    # n_evicted // pool_ratio = n_super
        self.latent_eviction_pool_temperature = 0.1  # softmax temperature for content pooling
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
        # Long-sequence guard: full [n_mid, n_mid] scoring matrix is infeasible at
        # 32k+ prefill (32k × 16 heads × fp32 ≈ 68 GB → OOM).
        # Subsample K query rows instead; column-sum becomes an unbiased estimator.
        # Memory: O(K × n) vs O(n²).  32k @ K=512 ≈ 1 GB;  128k @ K=512 ≈ 4 GB.
        # Lower K to 128–256 if GPU memory is tight; higher K for less rank noise.
        self.latent_eviction_score_queries   = 512  # Max query rows used for scoring
        # ---- Eviction mode selector -----------------------------------------
        # "committee" : multi-layer committee scoring + soft eviction (default)
        # "streaming" : StreamingLLM — first n_sink + recent window
        # "h2o"       : Heavy-Hitter Oracle — per-layer attention-score ranking
        self.latent_eviction_mode               = "committee"
        self.latent_eviction_keep_ratio         = 0.50  # target keep fraction (streaming/h2o)
        self.latent_eviction_h2o_recent_ratio   = 0.10  # h2o: recent tokens always kept
        # -----------------------------------------------------------------------

        # ---- Chunked prefill control (set by DeepseekV2Model._chunked_forward) ----
        # _chunked_prefill_active: True while a chunked prefill is in progress.
        #   Suppresses eviction on intermediate chunks (1..N-1).
        # _chunked_prefill_final: True only for the LAST chunk. Triggers eviction
        #   with full-cached_latent scoring + retroactive layer trimming.
        # _chunked_prefill_total: total number of tokens across all chunks.
        #   Used by the final chunk to know the full sequence length for sink/recent.
        self._chunked_prefill_active = False
        self._chunked_prefill_final  = False
        self._chunked_prefill_total  = 0


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

    def _compute_importance_scores(
        self,
        compressed_kv: torch.Tensor,
        q_abs: Optional[torch.Tensor] = None,
        W_UV: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute query-aware × value-aware importance scores for mid tokens.

        Called once per committee layer.  Results are accumulated and averaged
        by the decision layer before _compute_keep_indices is invoked.

        Returns info_mid [B, n_mid], or None if there are no mid tokens.
        """
        bsz, seq_len, R = compressed_kv.shape
        device = compressed_kv.device

        n_sink   = min(self.latent_eviction_sink_tokens,   seq_len)
        n_recent = min(self.latent_eviction_recent_tokens, max(0, seq_len - n_sink))
        n_mid    = seq_len - n_sink - n_recent

        if n_mid <= 0:
            return None

        mid_latent = compressed_kv[:, n_sink : seq_len - n_recent, :]  # [B, n_mid, R]

        val_norm = None
        if W_UV is not None:
            # Compute val_norm one head at a time to bound peak memory.
            # The per-head val_proj tensor [B, 1, n_mid, v_head_dim] at 16k is
            # only ~4 MB.  Looping over heads is slower but keeps the committee
            # layers from accumulating large intermediate tensors.
            n_heads = W_UV.shape[0]
            val_norms: list = []
            for h in range(n_heads):
                # matmul: [B, 1, n_mid, R] × [1, v_head_dim, R]^T
                vp_h = torch.matmul(mid_latent.unsqueeze(1).float(),
                                    W_UV[h:h+1].float().transpose(-1, -2))
                val_norms.append(vp_h.squeeze(1).norm(dim=-1))  # [B, n_mid]
            val_norm = torch.stack(val_norms, dim=1).mean(dim=1)  # [B, n_mid]
            v_mean  = val_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            val_norm = val_norm / v_mean

        if q_abs is not None:
            # Score local chunk via causal mask — proven 50-70% compression
            # for single-chunk prefill. Uses local positions so causal mask
            # creates natural gradient across mid tokens.
            mid_local = compressed_kv[:, n_sink : seq_len - n_recent, :]
            K = min(self.latent_eviction_score_queries, q_abs.shape[2], n_mid)

            sample_pos  = torch.randperm(q_abs.shape[2], device=device)[:K].sort().values
            q_scored    = q_abs[:, :, sample_pos, :]

            raw_logits = torch.matmul(
                q_scored, mid_local.unsqueeze(1).transpose(-1, -2)
            ) * self.softmax_scale  # [B, H, K, n_mid]

            # Local-position causal mask
            row_pos = sample_pos  # 0-indexed within chunk
            causal_mask = torch.arange(n_mid, device=device).unsqueeze(0) > row_pos.unsqueeze(1)
            raw_logits = raw_logits.masked_fill(
                causal_mask[None, None], float("-inf"))
            attn_probs = torch.softmax(raw_logits, dim=-1, dtype=torch.float32)
            score   = attn_probs.sum(dim=2)                 # [B, H, n_mid]

            if val_norm is not None:
                info_mid = (score * val_norm.unsqueeze(1)).mean(dim=1)
            else:
                info_mid = score.mean(dim=1)
        else:
            info_stat = compute_latent_info_score(compressed_kv, window=self.latent_eviction_window)
            info_mid  = info_stat[:, n_sink : seq_len - n_recent].float()
            if val_norm is not None:
                info_mid = info_mid * val_norm

        return info_mid                                                # [B, n_mid]

    @staticmethod
    def _make_super_tokens(
        cached_latent:    torch.Tensor,   # [B, 1, seq_len, R]
        cached_kpe:       torch.Tensor,   # [B, 1, seq_len, kpe_dim]
        keep_indices:     torch.Tensor,   # [B, keep_k]  absolute positions
        info_mid:         torch.Tensor,   # [B, n_mid]
        n_sink:           int,
        n_recent:         int,
        pool_ratio:       int,
        pool_temperature: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Soft eviction: merge evicted mid tokens into super tokens via pooling.

        Content (latent c): importance-score weighted softmax average per chunk.
        Position (k_pe):    uniform average (RoPE phases cannot be importance-
                            weighted without distorting position encoding).

        Returns (super_c, super_kpe) each [B, 1, n_super, dim], or (None, None).
        """
        B, _, seq_len, R  = cached_latent.shape
        kpe_dim            = cached_kpe.shape[-1]
        n_mid              = seq_len - n_sink - n_recent

        if n_mid <= 0 or pool_ratio <= 0:
            return None, None

        dev       = cached_latent.device
        mid_range = torch.arange(n_sink, seq_len - n_recent, device=dev)  # [n_mid]

        is_kept = torch.zeros(seq_len, dtype=torch.bool, device=dev)
        is_kept[keep_indices[0]] = True                                # batch-1 inference

        evicted_mask = ~is_kept[mid_range]                             # [n_mid] bool
        evicted_abs  = mid_range[evicted_mask]                         # [n_evicted]
        n_evicted    = evicted_abs.shape[0]

        n_super = n_evicted // pool_ratio
        if n_super == 0:
            return None, None

        n_use           = n_super * pool_ratio
        evicted_abs_use = evicted_abs[:n_use]                          # [n_use]

        ev_c   = cached_latent[:, :, evicted_abs_use, :]               # [B, 1, n_use, R]
        ev_kpe = cached_kpe[:, :, evicted_abs_use, :]                  # [B, 1, n_use, kpe_dim]

        ev_mid_rel = evicted_abs_use - n_sink                          # [n_use] → mid-relative
        ev_scores  = info_mid[:, ev_mid_rel]                           # [B, n_use]

        c_chunks     = ev_c.squeeze(1).view(B, n_super, pool_ratio, R)
        kpe_chunks   = ev_kpe.squeeze(1).view(B, n_super, pool_ratio, kpe_dim)
        score_chunks = ev_scores.view(B, n_super, pool_ratio)

        # Importance-weighted content pooling
        weights  = torch.softmax(score_chunks / max(pool_temperature, 1e-4), dim=-1)
        super_c  = (c_chunks * weights.unsqueeze(-1)).sum(dim=2).unsqueeze(1)

        # Uniform position pooling (preserve spatial locality)
        super_kpe = kpe_chunks.mean(dim=2).unsqueeze(1)

        return super_c.to(cached_latent.dtype), super_kpe.to(cached_kpe.dtype)


                                               # [B, S]



    def _compute_keep_indices(
        self,
        compressed_kv: torch.Tensor,
        aggregated_info_mid: torch.Tensor,   # [B, n_mid] — averaged over committee layers
    ) -> torch.Tensor:
        """[核心优化点 1：Adaptive Budget from Multi-Layer Committee Scores]

        Receives pre-computed importance scores (averaged over all committee layers)
        and returns the token indices to keep.

        Budget 决策：肘部法（Knee Detection）+ 覆盖率下界
          · knee_k = argmax_k[ C(k) − k/n ]   （曲线距对角线最远点）
          · cov_k  = min k s.t. C(k) >= latent_eviction_coverage_floor
          · keep_k = max(knee_k, cov_k)

        无条件保护：
          · 前 latent_eviction_sink_tokens   个 token（attention sink）
          · 后 latent_eviction_recent_tokens 个 token（recency 偏差）

        Args:
            compressed_kv       : [B, S, kv_lora_rank]  归一化 latent
            aggregated_info_mid : [B, n_mid]  multi-layer averaged importance scores

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

        # ── 自适应 Budget（使用多层委员会聚合的 importance scores）──────────
        n_mid_keep = self._adaptive_budget(
            aggregated_info_mid,
            coverage_floor = self.latent_eviction_coverage_floor,
            min_keep       = self.latent_eviction_min_keep,
            max_keep       = self.latent_eviction_max_keep,
        )

        if n_mid_keep >= n_mid:
            keep_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1).clone()
            logger.debug(
                f"Latent eviction [layer {self.layer_idx}]: "
                f"chunk={seq_len}, no eviction (n_mid_keep={n_mid_keep} >= n_mid={n_mid})"
            )
            return keep_idx

        # ── Top-K 选中间段，拼合 Sink + 中间 + Recent ─────────────────────
        mid_order    = aggregated_info_mid.argsort(dim=-1, descending=True)[:, :n_mid_keep]
        mid_keep_idx = (mid_order + n_sink).sort(dim=-1).values

        sink_idx   = torch.arange(n_sink,              device=device).unsqueeze(0).expand(bsz, -1)
        recent_idx = torch.arange(seq_len - n_recent, seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        keep_idx   = torch.cat([sink_idx, mid_keep_idx, recent_idx], dim=1).sort(dim=-1).values

        evicted = seq_len - keep_idx.shape[1]
        logger.debug(
            f"Latent eviction [layer {self.layer_idx}] (multi-layer committee): "
            f"chunk={seq_len}, sink={n_sink}, recent={n_recent}, "
            f"mid_kept={n_mid_keep}/{n_mid}, evicted={evicted} ({evicted/seq_len*100:.1f}%)"
        )
        return keep_idx

    # ------------------------------------------------------------------
    # Alternative eviction modes (streaming / h2o)
    # ------------------------------------------------------------------

    def _streaming_keep_indices(
        self, seq_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        StreamingLLM keep-index set: first n_sink (attention sinks) +
        last window_size (recent) tokens.  Returns a 1-D sorted index
        tensor of shape [n_keep], or None if no eviction is needed.
        keep_ratio is read from self.latent_eviction_keep_ratio.
        """
        n_sink = min(self.latent_eviction_sink_tokens, seq_len)
        keep_ratio = getattr(self, 'latent_eviction_keep_ratio', 0.50)
        n_keep = max(n_sink + 1, int(seq_len * keep_ratio))
        if n_keep >= seq_len:
            return None
        window = max(1, n_keep - n_sink)
        sink_idx   = torch.arange(n_sink,              device=device)
        recent_idx = torch.arange(seq_len - window, seq_len, device=device)
        return torch.cat([sink_idx, recent_idx])   # already sorted

    def _h2o_keep_indices(
        self,
        q_abs:         torch.Tensor,   # [B, H, q_len, kv_lora_rank]
        q_pe:          torch.Tensor,   # [B, H, q_len, qk_rope_head_dim]
        cached_latent: torch.Tensor,   # [B, 1, S, kv_lora_rank]
        cached_kpe:    torch.Tensor,   # [B, 1, S, qk_rope_head_dim]
    ) -> Optional[torch.Tensor]:
        """
        H2O Heavy-Hitter Oracle keep-index set.

        Computes the full attention weight matrix for the prefill pass,
        accumulates scores by summing attention weights over all query
        positions (approximated with at most latent_eviction_score_queries
        query rows to bound memory), then keeps the top n_heavy tokens
        plus the most recent n_recent tokens unconditionally.

        Returns a 1-D sorted index tensor [n_keep] or None.
        """
        S = cached_latent.shape[2]
        keep_ratio   = getattr(self, 'latent_eviction_keep_ratio', 0.50)
        recent_ratio = getattr(self, 'latent_eviction_h2o_recent_ratio', 0.10)
        n_keep   = max(2, int(S * keep_ratio))
        n_recent = max(1, int(S * recent_ratio))
        n_recent = min(n_recent, n_keep - 1)
        if n_keep >= S:
            return None

        # Subsample from the END of the sequence (these queries have seen
        # the most context and produce the most representative scores).
        max_q = getattr(self, 'latent_eviction_score_queries', 512)
        q_len  = q_abs.shape[2]
        if q_len > max_q:
            q_start  = q_len - max_q
            q_abs_s  = q_abs[:, :, q_start:, :]
            q_pe_s   = q_pe[:,  :, q_start:, :]
        else:
            q_start  = 0
            q_abs_s  = q_abs
            q_pe_s   = q_pe

        # Attention logits [B, H, n_q, S]
        content = torch.matmul(q_abs_s, cached_latent.transpose(-1, -2))
        rope    = torch.matmul(q_pe_s,  cached_kpe.transpose(-1, -2))
        scores  = (content + rope) * self.softmax_scale

        # Causal mask: query at global position (q_start + i) can only attend
        # to keys 0 .. (q_start + i).
        n_q = q_abs_s.shape[2]
        row_pos = torch.arange(q_start, q_start + n_q,
                               device=scores.device).unsqueeze(1)   # [n_q, 1]
        col_pos = torch.arange(S, device=scores.device).unsqueeze(0)  # [1, S]
        causal_mask = torch.where(
            col_pos > row_pos,
            torch.full((1,), float('-inf'), device=scores.device, dtype=scores.dtype),
            torch.zeros((1,),               device=scores.device, dtype=scores.dtype),
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax → sum over query positions → mean over heads → [S]
        attn_w     = torch.softmax(scores.float(), dim=-1).to(q_abs.dtype)
        importance = attn_w.sum(dim=2).mean(dim=1)   # [B, S]
        importance = importance[0] if importance.shape[0] > 1 else importance.squeeze(0)

        # Protect recent tokens from being evicted
        importance[-n_recent:] = float('inf')

        _, keep_idx = importance.topk(n_keep)
        return keep_idx.sort().values   # maintain positional order

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
        # Do this head-by-head via matmul to stay under 16 MB per head at 16k,
        # instead of einsum("bhqd,hdr->bhqr") which broadcasts to a 5-D
        # intermediate [B,H,q_len,qk_nope_head_dim,R] → ~34 GB at 16k.
        q_abs_list = []
        for h in range(self.num_heads):
            qh = torch.matmul(q_nope[:, h:h+1, :, :].float(),
                              W_UK[h:h+1].float())
            q_abs_list.append(qh.to(q_nope.dtype))
        q_abs = torch.cat(q_abs_list, dim=1)
        del q_abs_list

        # ── Update cache: 存 latent（key_cache）和 k_pe（value_cache）───────
        # key_cache   ← c_kv_normed  [B, 1, S, kv_lora_rank=512]
        # value_cache ← k_pe         [B, 1, S, qk_rope_head_dim=64]
        # 两者合计 576 维/token，比展开 K/V 的 4096 维节省约 7×。
        c_kv_normed_4d = c_kv_normed.unsqueeze(1)  # [B, 1, q_len, kv_lora_rank]

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}

            # ---- Chunked-prefill awareness ----
            # _chunked_prefill_active: set by DeepseekV2Model._chunked_forward()
            #   on ALL chunks of a chunked prefill.
            # _chunked_prefill_final:  set only on the LAST chunk.
            # _chunked_prefill_total:  total tokens across all chunks.
            _chunked_active = getattr(self, '_chunked_prefill_active', False)
            _chunked_final  = getattr(self, '_chunked_prefill_final',  False)
            _chunked_total  = getattr(self, '_chunked_prefill_total',  0)

            # 必须在 update 前检查：update 后本层 cache 必然非空，检查失效
            # Chunked prefill: the final chunk must trigger eviction even though
            # past_key_value is non-empty (earlier chunks already populated it).
            _is_first_prefill = (
                q_len > 1
                and self.latent_eviction
                and (
                    past_key_value.get_seq_length(self.layer_idx) == 0
                    or _chunked_final  # final chunk of chunked prefill
                )
            )

            # Chunked prefill: suppress eviction on intermediate (non-final) chunks
            if _chunked_active and not _chunked_final:
                _is_first_prefill = False

            # Chunked prefill: suppress eviction on intermediate chunks.
            # Only the FINAL chunk triggers eviction (with full-cached-latent scoring).
            if _chunked_active and not _chunked_final:
                _is_first_prefill = False

            # update() 拼接 past + current，返回完整序列的 latent 和 k_pe
            # cached_latent : [B, 1, kv_seq_len, kv_lora_rank]
            # cached_kpe    : [B, 1, kv_seq_len, qk_rope_head_dim]
            cached_latent, cached_kpe = past_key_value.update(
                c_kv_normed_4d, k_pe, self.layer_idx, cache_kwargs
            )

            if _is_first_prefill:
                # [核心优化点 3：多层委员会决策 + 软驱逐 Super Tokens]
                #
                # 多层评分委员会:
                #   score_layers 中每一层独立计算 importance scores 并累积；
                #   decision_layer（最后一层）对所有层的分数取均值，得到更稳定的
                #   综合重要度，再调用 _compute_keep_indices 做最终驱逐决策。
                #
                # 软驱逐 Super Tokens:
                #   被驱逐的 mid tokens 按 pool_ratio 组做加权池化，生成 super tokens
                #   追加到 cache 末尾，供后续 decode 的 attention 使用。
                D            = self.latent_eviction_decision_layer
                score_layers = set(getattr(self, 'latent_eviction_score_layers', [D]))

                # ── 委员会层：计算本层 importance scores 并累积 ────────────────
                _eviction_mode = getattr(self, 'latent_eviction_mode', 'committee')
                if _eviction_mode == 'committee' and self.layer_idx in score_layers:
                    _info_this = self._compute_importance_scores(
                        c_kv_normed, q_abs=q_abs, W_UV=W_UV)
                    if _info_this is not None:
                        if not hasattr(past_key_value, '_evict_scores_list'):
                            past_key_value._evict_scores_list = []
                        past_key_value._evict_scores_list.append(_info_this.detach())

                keep_indices = None

                # ── 决策层：聚合委员会分数 → 驱逐 + super tokens ─────────────
                if _eviction_mode == 'committee' and self.layer_idx == D:
                    _scores_list = getattr(past_key_value, '_evict_scores_list', None)
                    if _scores_list:
                        _agg_info = torch.stack(_scores_list, dim=0).mean(dim=0)  # [B, n_mid]
                    else:
                        _agg_info = self._compute_importance_scores(
                            c_kv_normed, q_abs=q_abs, W_UV=W_UV)

                    if _agg_info is not None:
                        if _chunked_final:
                            _total_len  = cached_latent.shape[2]
                            _n_sink_d   = min(self.latent_eviction_sink_tokens,   _total_len)
                            _n_recent_d = min(self.latent_eviction_recent_tokens,
                                              max(0, _total_len - _n_sink_d))
                        else:
                            _n_sink_d   = min(self.latent_eviction_sink_tokens,   q_len)
                            _n_recent_d = min(self.latent_eviction_recent_tokens,
                                              max(0, q_len - _n_sink_d))

                        keep_indices = self._compute_keep_indices(c_kv_normed, _agg_info)
                        past_key_value.shared_keep_indices = keep_indices
                        past_key_value._shared_agg_info    = (_agg_info, _n_sink_d, _n_recent_d)

                        # ── 溯源修剪 Layer 0..D-1 + 追加 super tokens ──────────
                        _ret_idx = keep_indices.unsqueeze(1).unsqueeze(-1)  # [B,1,keep_k,1]
                        _pool_r  = getattr(self, 'latent_eviction_pool_ratio', 0)
                        _pool_t  = getattr(self, 'latent_eviction_pool_temperature', 0.1)

                        # 修剪 Layer 0..D（包括 decision layer）。
                        # D 层之后的层还没完成当前 chunk 的 cache update，
                        # keep_indices 里的全序列索引会越界。
                        for _prev in range(D + 1):
                            if _prev >= len(past_key_value.key_cache):
                                break
                            _pl = past_key_value.key_cache[_prev]
                            _pk = past_key_value.value_cache[_prev]
                            _tl = _pl.gather(2, _ret_idx.expand(-1, 1, -1, _pl.shape[-1]))
                            _tk = _pk.gather(2, _ret_idx.expand(-1, 1, -1, _pk.shape[-1]))
                            if _pool_r > 0:
                                _sc, _sk = self._make_super_tokens(
                                    _pl, _pk, keep_indices, _agg_info,
                                    _n_sink_d, _n_recent_d, _pool_r, _pool_t)
                                if _sc is not None:
                                    _tl = torch.cat([_tl, _sc],  dim=2)
                                    _tk = torch.cat([_tk, _sk],  dim=2)
                            past_key_value.key_cache[_prev]   = _tl
                            past_key_value.value_cache[_prev] = _tk
                        # Release the full-length storage referenced by _pl/_pk
                        del _pl, _pk, _tl, _tk, _ret_idx
                        if _pool_r > 0 and _sc is not None:
                            del _sc, _sk

                        # 清理临时累积分数
                        if hasattr(past_key_value, '_evict_scores_list'):
                            del past_key_value._evict_scores_list

                elif _eviction_mode == 'committee' and self.layer_idx > D:
                    keep_indices = getattr(past_key_value, "shared_keep_indices", None)

                # ── StreamingLLM / H2O: per-layer independent eviction ────────
                if _eviction_mode == 'streaming':
                    _s_idx = self._streaming_keep_indices(
                        cached_latent.shape[2], cached_latent.device)
                    if _s_idx is not None:
                        _B   = cached_latent.shape[0]
                        _sei = _s_idx.view(1, 1, -1, 1)   # [1, 1, n_keep, 1]
                        _tl  = cached_latent.gather(
                            2, _sei.expand(_B, 1, -1, cached_latent.shape[-1]))
                        _tk  = cached_kpe.gather(
                            2, _sei.expand(_B, 1, -1, cached_kpe.shape[-1]))
                        past_key_value.key_cache[self.layer_idx]   = _tl
                        past_key_value.value_cache[self.layer_idx] = _tk
                        # Do NOT reassign cached_latent/cached_kpe here.
                        # Current-layer attention still computes over the full prefill
                        # sequence (consistent with the pre-built causal attention_mask).
                        # Eviction only takes effect for subsequent decode steps.
                        if self.layer_idx == self.config.num_hidden_layers - 1:
                            past_key_value._seen_tokens = _tl.shape[2]
                elif _eviction_mode == 'h2o':
                    _h_idx = self._h2o_keep_indices(
                        q_abs, q_pe, cached_latent, cached_kpe)
                    if _h_idx is not None:
                        _B   = cached_latent.shape[0]
                        _hei = _h_idx.view(1, 1, -1, 1)   # [1, 1, n_keep, 1]
                        _tl  = cached_latent.gather(
                            2, _hei.expand(_B, 1, -1, cached_latent.shape[-1]))
                        _tk  = cached_kpe.gather(
                            2, _hei.expand(_B, 1, -1, cached_kpe.shape[-1]))
                        past_key_value.key_cache[self.layer_idx]   = _tl
                        past_key_value.value_cache[self.layer_idx] = _tk
                        # Same as streaming: do not touch cached_latent/cached_kpe
                        # so current-layer attention sees the full prefill sequence.
                        if self.layer_idx == self.config.num_hidden_layers - 1:
                            past_key_value._seen_tokens = _tl.shape[2]
                elif keep_indices is not None:
                    # [核心优化点 4：双轨同步切片（latent + k_pe）+ super tokens]
                    idx = keep_indices.unsqueeze(1).unsqueeze(-1)  # [B, 1, keep_k, 1]
                    trimmed_latent = cached_latent.gather(
                        2, idx.expand(-1, 1, -1, cached_latent.shape[-1])
                    )
                    trimmed_kpe = cached_kpe.gather(
                        2, idx.expand(-1, 1, -1, cached_kpe.shape[-1])
                    )

                    # 软驱逐：追加 super tokens 到当前层 cache
                    _pool_r = getattr(self, 'latent_eviction_pool_ratio', 0)
                    if _pool_r > 0:
                        _agg_t, _ns, _nr = getattr(
                            past_key_value, '_shared_agg_info',
                            (None, self.latent_eviction_sink_tokens,
                             self.latent_eviction_recent_tokens))
                        if _agg_t is not None:
                            _pool_t = getattr(self, 'latent_eviction_pool_temperature', 0.1)
                            _sc, _sk = self._make_super_tokens(
                                cached_latent, cached_kpe, keep_indices, _agg_t,
                                _ns, _nr, _pool_r, _pool_t)
                            if _sc is not None:
                                trimmed_latent = torch.cat([trimmed_latent, _sc],  dim=2)
                                trimmed_kpe    = torch.cat([trimmed_kpe,    _sk],  dim=2)

                    # 强制覆写当前层 cache
                    past_key_value.key_cache[self.layer_idx]   = trimmed_latent
                    past_key_value.value_cache[self.layer_idx] = trimmed_kpe

                    # [核心优化点 5：修复 DynamicCache 内置状态]
                    if self.layer_idx == self.config.num_hidden_layers - 1:
                        past_key_value._seen_tokens = trimmed_latent.shape[2]
        else:
            # 无 cache：只用当前 q_len 个 token
            cached_latent = c_kv_normed_4d  # [B, 1, q_len, kv_lora_rank]
            cached_kpe    = k_pe             # [B, 1, q_len, qk_rope_head_dim]

        # ── Chunked MLA attention (no O(S²) materialization) ────────────────
        # MLA decomposes Q·K^T = q_abs·latent^T + q_pe·k_pe^T.  Both terms
        # produce [B,H,q_len,kv_seq_len] — too large to materialise at 16k+.
        # We chunk Q into small pieces (CHUNK=16); each score slice is
        # [B,H,16,kv_seq_len] which stays easily in memory even at 64k.
        #
        # All einsum calls have been replaced with torch.matmul to avoid
        # 5-D broadcast intermediates that can reach tens of gigabytes.

        kv_seq_len = cached_latent.shape[2]  # actual (possibly evicted) cache length
        dev        = cached_latent.device
        CHUNK      = 16  # [B,H,16,S] at 32k bf16 ≈ 17 MB per score matrix

        outputs: list = []
        for _start in range(0, q_len, CHUNK):
            _end = min(_start + CHUNK, q_len)

            # ── content score  [B, H, chunk_sz, kv_seq_len]
            content_c = torch.matmul(
                q_abs[:, :, _start:_end, :],
                cached_latent.transpose(-1, -2),
            )
            # ── rope score  [B, H, chunk_sz, kv_seq_len]
            rope_c = torch.matmul(
                q_pe[:, :, _start:_end, :],
                cached_kpe.transpose(-1, -2),
            )
            scores_c = (content_c + rope_c) * self.softmax_scale
            del content_c, rope_c

            # ── Causal mask ──────────────────────────────────────────────
            if q_len > 1:
                col_idx = torch.arange(kv_seq_len, device=dev)
                row_idx = torch.arange(_start, _end, device=dev)
                causal  = row_idx.unsqueeze(1) >= col_idx.unsqueeze(0)
                scores_c = scores_c.masked_fill(
                    ~causal[None, None], float("-inf"))

            if attention_mask is not None:
                mask_c = attention_mask[:, :, _start:_end, :kv_seq_len]
                scores_c = scores_c + mask_c

            # ── softmax → weighted latent → value projection ────────────
            attn_c  = nn.functional.softmax(scores_c, dim=-1, dtype=torch.float32)
            attn_c  = attn_c.to(q_nope.dtype)
            del scores_c

            attn_c          = nn.functional.dropout(attn_c, p=self.attention_dropout, training=self.training)
            weighted_latent = torch.matmul(attn_c, cached_latent)           # [B,H,chunk,R]
            del attn_c

            # matmul avoids einsum's 5-D broadcast (up to 2 GB even for a small chunk)
            # weighted_latent: [B,H,chunk,R]  W_UV: [H,d_v,R]
            out_c = torch.matmul(weighted_latent, W_UV.transpose(-1, -2))   # [B,H,chunk,d_v]
            del weighted_latent

            # Move each chunk to CPU to cap GPU memory at one chunk's worth
            # of output tensors + one score matrix.
            outputs.append(out_c.cpu())

        attn_output = torch.cat(outputs, dim=2).to(dev)
        del outputs
        # attn_output: [B, H, q_len, v_head_dim]

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
    return info

# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV2
class DeepseekV2FlashAttention2(DeepseekV2Attention):
    """
    DeepseekV2 flash attention module. This module inherits from `DeepseekV2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekV2FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        # See eager forward: extend rotary tables to cover TRUE positions when
        # eviction shrinks the physical cache below the real sequence length.
        rotary_seq_len = kv_seq_len
        if self.latent_eviction and position_ids is not None:
            rotary_seq_len = max(rotary_seq_len, int(position_ids.max()) + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )
        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,
}


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


DeepseekV2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeepseekV2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DeepseekV2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _chunked_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        batch_size: int,
        seq_length: int,
        use_legacy_cache: bool,
    ):
        """Chunked prefill: split long sequence into 4K chunks.

        Each chunk flows through ALL decoder layers.  The KV cache accumulates
        via DynamicCache.update().  Eviction fires only on the final chunk.
        Peak memory is bounded at 4K-level intermediates regardless of total
        sequence length.
        """
        CHUNK_SIZE = 4096
        num_chunks = (seq_length + CHUNK_SIZE - 1) // CHUNK_SIZE

        # ---- Arm chunked-prefill flags on all attention layers ----
        for layer in self.layers:
            layer.self_attn._chunked_prefill_active = True
            layer.self_attn._chunked_prefill_final  = False
            layer.self_attn._chunked_prefill_total  = seq_length

        all_hidden_states = []
        next_decoder_cache = None

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end   = min(chunk_start + CHUNK_SIZE, seq_length)
            chunk_len   = chunk_end - chunk_start

            # ---- Mark final chunk for eviction trigger ----
            is_final = (chunk_idx == num_chunks - 1)
            if is_final:
                for layer in self.layers:
                    layer.self_attn._chunked_prefill_final = True

            # ---- Slice inputs for this chunk ----
            chunk_embeds = inputs_embeds[:, chunk_start:chunk_end, :]
            chunk_pos_ids = position_ids[:, chunk_start:chunk_end]

            # ---- Build per-chunk 4D causal mask ----
            # past_key_values_length = chunk_start (tokens already in cache)
            # Pass None as the padding mask — causal masking is sufficient
            # for chunked prefill. The original attention_mask may not match
            # the chunk shape (e.g. [1, 6144] vs [1, 4096]).
            chunk_attn_mask = _prepare_4d_causal_attention_mask(
                None,
                (batch_size, chunk_len),
                chunk_embeds,
                chunk_start,
            )

            # ---- Run all decoder layers ----
            hidden_states = chunk_embeds
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=chunk_attn_mask,
                    position_ids=chunk_pos_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = layer_outputs[
                        2 if output_attentions else 1
                    ]

            all_hidden_states.append(hidden_states)

            # Release chunk intermediates immediately
            del chunk_embeds, chunk_attn_mask, hidden_states

        # ---- Concatenate chunk outputs ----
        hidden_states = torch.cat(all_hidden_states, dim=1)
        del all_hidden_states

        # ---- Clean up chunked-prefill flags ----
        for layer in self.layers:
            layer.self_attn._chunked_prefill_active = False
            layer.self_attn._chunked_prefill_final  = False
            layer.self_attn._chunked_prefill_total  = 0

        # ---- Apply final layer norm ----
        hidden_states = self.norm(hidden_states)

        # ---- Signal CausalLM that chunked prefill was used ----
        # So it can skip full-sequence lm_head to avoid logits.float() OOM.
        self._did_chunked_prefill = True

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, None, None]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=None,
            attentions=None,
        )

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # ---- Chunked prefill: split long prefill into 4K chunks ----
        # Only active when:
        #   - eager attention (not flash_attn_2 — incompatible K/V layout)
        #   - use_cache=True
        #   - first prefill (past_key_values_length == 0, no existing cache)
        #   - sequence length > CHUNK_SIZE
        #   - batch_size == 1 (multi-batch would desync cache sizes)
        CHUNK_SIZE = 4096
        _do_chunk = (
            not self._use_flash_attention_2
            and use_cache
            and seq_length > CHUNK_SIZE
            and past_key_values_length == 0
            and batch_size == 1
        )

        if _do_chunk:
            return self._chunked_forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                batch_size=batch_size,
                seq_length=seq_length,
                use_legacy_cache=use_legacy_cache,
            )

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def configure_latent_eviction(
        self,
        enabled: bool = True,
        threshold: float = 0.3,
        window: int = 4,
        prefill_only: bool = True,
    ):
        """
        Enable or disable latent KV-cache eviction during chunked prefill.

        Eviction is threshold-based: each token's latent is scored by
        compute_latent_info_score (L2 norm + negative entropy + per-dim variance,
        minus neighbour redundancy). Tokens scoring BELOW `threshold` are dropped
        before the cache write. The number of evicted tokens adapts to the content
        of each chunk — it is NOT a fixed percentage.

        Args:
            enabled:   True to activate eviction, False to disable.
            threshold: Info-score cutoff in roughly [-0.5, 1.0]. Higher = more
                       aggressive eviction (keeps fewer tokens). Typical: 0.2 ~ 0.4.
            window:    Neighbour radius used for redundancy (duplicate) detection.
            prefill_only: If True (default, recommended for the validation phase),
                       eviction runs ONLY during the one-shot prefill; the decode
                       loop stays strictly append-only (no per-token scoring/slicing,
                       so tokens/s is not hurt). Measure quality first, optimise the
                       online sliding-window eviction later.

        Example::

            model.configure_latent_eviction(enabled=True, threshold=0.3)
            # ... run generation as usual ...
            model.configure_latent_eviction(enabled=False)   # restore default
        """
        # Model-level flag + true-length counter used by prepare_inputs_for_generation
        # to keep position_ids (true) and attention_mask length (physical) consistent.
        self._latent_eviction_active = bool(enabled)
        self._latent_true_seqlen = 0
        for layer in self.model.layers:
            attn = layer.self_attn
            attn.latent_eviction = enabled
            attn.latent_eviction_threshold = float(threshold)
            attn.latent_eviction_window = int(window)
            attn.latent_eviction_prefill_only = bool(prefill_only)

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # After chunked prefill, hidden_states can be 16K+.  lm_head produces
        # [B, 16K, vocab_size] which is ~3.4 GB in bf16 and ~6.7 GB after .float().
        # Since prefill logits are never consumed by downstream eval code (only
        # past_key_values matters), compute lm_head only on the last position.
        if getattr(self.model, '_did_chunked_prefill', False) and labels is None:
            logits = self.lm_head(hidden_states[:, -1:, :])
            self.model._did_chunked_prefill = False
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # ===== Latent-eviction-aware path =====
        # When eviction is on, the physical KV cache is SHORTER than the true
        # sequence length. HF's default machinery assumes cache_len == seq_len,
        # which would otherwise:
        #   (a) mis-trim input_ids via the (now smaller) seen_tokens,
        #   (b) build an attention_mask whose length (true) mismatches the physical
        #       KV length and crash the attention kernel, and
        #   (c) derive wrong position_ids.
        # We therefore fully synthesise the inputs here: feed only the genuinely
        # new token(s), supply TRUE absolute position_ids (for correct RoPE), and a
        # physical-length all-ones attention_mask (for shape-consistent kernels).
        if getattr(self, "_latent_eviction_active", False):
            bsz = input_ids.shape[0]
            is_prefill = past_key_values is None or (
                isinstance(past_key_values, Cache)
                and past_key_values.get_seq_length() == 0
            )
            if is_prefill:
                # Record true context length; let the prompt run through normally.
                self._latent_true_seqlen = input_ids.shape[1]
                position_ids = torch.arange(
                    input_ids.shape[1], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0)
                if inputs_embeds is not None:
                    model_inputs = {"inputs_embeds": inputs_embeds}
                else:
                    model_inputs = {"input_ids": input_ids}
                model_inputs.update(
                    {
                        "position_ids": position_ids,
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "attention_mask": attention_mask,
                    }
                )
                return model_inputs

            # ----- Decode step -----
            true_len = getattr(self, "_latent_true_seqlen", input_ids.shape[1] - 1)
            new = input_ids.shape[1] - true_len
            if new < 1:
                new = 1
            new_input_ids = input_ids[:, -new:]
            # TRUE absolute positions for the new token(s) -> correct RoPE phase.
            position_ids = (
                torch.arange(
                    true_len, true_len + new, dtype=torch.long, device=input_ids.device
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )
            # Physical-length all-ones mask -> matches the (compressed) KV cache, so
            # both the eager 4D-mask path and the flash unpad path stay shape-safe.
            physical_len = past_key_values.get_seq_length()
            phys_mask = torch.ones(
                bsz, physical_len + new, dtype=torch.long, device=input_ids.device
            )
            self._latent_true_seqlen = true_len + new
            return {
                "input_ids": new_input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": phys_mask,
            }
        # ===== Default path (unchanged) =====
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@add_start_docstrings(
    """
    The DeepseekV2 Model transformer with a sequence classification head on top (linear layer).

    [`DeepseekV2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2ForSequenceClassification(DeepseekV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
