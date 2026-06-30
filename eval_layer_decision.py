"""
eval_layer_decision.py — Latent Eviction Cross-Layer Decision Ablation Study
=============================================================================

Validates the "importance persistence" hypothesis in DeepSeek-V2-Lite (MLA):
  Can Layer-0 latent features safely proxy KV-cache eviction decisions
  for ALL transformer layers without meaningful PPL degradation?

Experimental Configurations
-----------------------------
  A  Baseline          — Dense KV cache (theoretical upper-bound PPL)
  B  Layer-0 Decision  — Layer 0 computes eviction indices; ALL layers share
                         them.  ← Core hypothesis validation
  C  Delayed Decision  — Layers 0-1 dense; Layer 2 computes for Layers 3+
                         ← Ablation: do features "settle" at Layer 0?
  D  Oracle            — Each layer decides independently via attention-mask
                         (no physical cache truncation) ← Theoretical floor

Eviction mechanism (eval-time simulation)
------------------------------------------
  Physical cache truncation is the production mechanism, but for sliding-
  window PPL evaluation we cannot use a growing KV cache.  Instead, evicted
  token positions are "silenced" in the attention logit matrix.  TWO subtle
  correctness requirements are enforced:

    (1) Diagonal safety — a token j that gets evicted was STILL able to attend
        to itself at the moment it was processed (it is dropped only AFTER, so
        future tokens i>j cannot see it).  We therefore add -1e9 to entry
        (i, j) ONLY when (j is evicted AND i > j); the diagonal i==j is never
        masked, preserving token j's own prediction.

    (2) Causality — the eviction decision for token j must NOT peek at any
        future token.  The redundancy penalty looks BACKWARD only (j vs j-1,
        j-2, …), via compute_latent_info_score_causal.

    (3) Fair compression — eviction uses a fixed Top-K KEEP_RATIO, so the
        physical compression rate is IDENTICAL across every config and every
        layer.  This removes the "did D just drop more tokens?" confound when
        comparing shared-decision (B) vs independent-oracle (D).

Metrics
-------
  · Perplexity (PPL)  on WikiText-2 test, sliding-window (stride = STRIDE)
  · KV Kept %         mean fraction of tokens NOT evicted across active layers

Output
------
  Markdown-formatted results table, suitable for direct paper insertion.

Usage
-----
    python eval_layer_decision.py
"""

import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Workspace & local imports ─────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

from modeling_deepseek import apply_rotary_pos_emb, _robust_normalize

# ═══════════════════════════════════════════════════════════════════════════════
# § 0  Experiment hyper-parameters
# ═══════════════════════════════════════════════════════════════════════════════

KEEP_RATIO         = 0.70   # Top-K fraction of tokens to KEEP. Fixed → IDENTICAL
                            # physical compression across ALL configs & layers, so
                            # B-vs-D differences reflect WHERE the decision is made,
                            # never HOW MANY tokens were dropped (fair ablation).
EVICTION_WINDOW    = 4      # Neighbour-redundancy radius (CAUSAL: looks back only)
MAX_SEQ_LEN        = 1024   # Context window per PPL chunk (tokens)
STRIDE             = 512    # Sliding-window stride  (tokens evaluated/step)
MAX_CHUNKS         = 80     # Cap on WikiText-2 chunks (None = full ~500 chunks)
DTYPE              = torch.bfloat16

# (id, human label, internal strategy key)
CONFIGS = [
    ("A", "Baseline",          "baseline"),
    ("B", "Layer-0 Decision",  "layer_0"),
    ("C", "Delayed Decision",  "layer_2"),
    ("D", "Oracle",            "oracle"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  WikiText-2 loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_wikitext2_test(tokenizer):
    """Load and tokenise the WikiText-2 test split."""
    try:
        from datasets import load_dataset
        ds   = load_dataset("wikitext", "wikitext-2-raw-v1",
                            split="test", trust_remote_code=True)
        text = "\n\n".join(x for x in ds["text"] if x.strip())
        print(f"  WikiText-2 test  : {len(text):,} chars")
    except Exception as exc:
        print(f"  [WARN] datasets unavailable ({exc}). Using built-in mini corpus.")
        text = (
            "The Transformer architecture introduced multi-head self-attention "
            "and feed-forward layers, enabling massively parallel sequence "
            "modelling and achieving state-of-the-art results across a wide "
            "range of natural language processing benchmarks. "
        ) * 3000

    enc = tokenizer(text, return_tensors="pt")
    print(f"  Tokenised length : {enc.input_ids.shape[1]:,} tokens")
    return enc


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Eviction masking logic
# ═══════════════════════════════════════════════════════════════════════════════

class EvictionState:
    """
    Shared mutable state across all layers within ONE forward pass.
    Automatically reset (evict_vector cleared) at layer_idx == 0.
    """
    def __init__(self):
        self.evict_vector: torch.Tensor | None = None   # [B, S]  1=evict, 0=keep
        self.kept_frac_shared: float = 1.0              # kept-fraction of shared decision
        # Per-layer kept-fraction list (populated inside patched forward).
        self.kept_fracs: list[float] = []


def compute_latent_info_score_causal(
    compressed_kv: torch.Tensor,
    window: int = 4,
    w_norm: float = 0.4,
    w_entropy: float = 0.3,
    w_variance: float = 0.3,
    w_redundancy: float = 0.5,
) -> torch.Tensor:
    """
    CAUSAL variant of compute_latent_info_score for honest PPL evaluation.

    Identical informativeness signals (L2 norm, negative entropy, per-dim
    variance), but the neighbour-redundancy penalty looks ONLY BACKWARD: token
    j's redundancy is its max cosine similarity to PAST neighbours j-1 … j-w.
    The production version also writes redundancy onto the earlier token (a
    look-AHEAD), which would leak future information into the decision for j and
    render the measured PPL meaningless. That back-write is removed here.

    Returns
        info : [B, S]  higher = more worth keeping
    """
    x = compressed_kv.float()
    bsz, seq_len, _ = x.shape

    norm        = x.norm(dim=-1)                                       # [B, S]
    prob        = F.softmax(x, dim=-1)
    entropy     = -(prob * (prob + 1e-10).log()).sum(dim=-1)          # [B, S]
    neg_entropy = -entropy
    variance    = x.var(dim=-1)                                        # [B, S]

    x_unit     = F.normalize(x, dim=-1)
    redundancy = torch.zeros(bsz, seq_len, device=x.device, dtype=x.dtype)
    for offset in range(1, max(1, window) + 1):
        if offset >= seq_len:
            break
        # cosine(j, j-offset); assign ONLY to the later token j (causal, past-only)
        sim = (x_unit[:, offset:, :] * x_unit[:, :-offset, :]).sum(dim=-1)  # [B, S-offset]
        redundancy[:, offset:] = torch.maximum(redundancy[:, offset:], sim)

    info = (
        w_norm * _robust_normalize(norm)
        + w_entropy * _robust_normalize(neg_entropy)
        + w_variance * _robust_normalize(variance)
        - w_redundancy * _robust_normalize(redundancy)
    )
    return info                                                        # [B, S]


def _build_evict_vector(c_kv: torch.Tensor):
    """
    Top-K eviction (FIXED compression rate, independent of layer/threshold).

    Keep the highest-scoring round(S * KEEP_RATIO) tokens; evict the rest. The
    evicted count is therefore deterministic and IDENTICAL for every config and
    every layer, so cross-config PPL gaps cannot be confounded by how many
    tokens were dropped.

    Returns
        evict     : [B, S] float  1.0 = evicted, 0.0 = kept
        kept_frac : float         keep_k / S
    """
    info          = compute_latent_info_score_causal(c_kv, window=EVICTION_WINDOW)  # [B,S]
    bsz, seq_len  = info.shape
    keep_k        = max(1, int(round(seq_len * KEEP_RATIO)))
    keep_idx      = info.argsort(dim=-1, descending=True)[:, :keep_k]   # [B, keep_k]
    evict         = torch.ones(bsz, seq_len, device=info.device, dtype=info.dtype)
    evict.scatter_(1, keep_idx, 0.0)                                    # kept → 0
    return evict, keep_k / seq_len


def _causal_evict_mask(evict: torch.Tensor, q_len: int) -> torch.Tensor:
    """
    Build an additive attention mask that simulates physical eviction WITHOUT
    breaking causality or the diagonal.

    Entry (i, j) receives -1e9 ONLY when:  evict[j] == 1  AND  i > j
      · i == j  is never masked → token j keeps its OWN self-attention (it was
        evicted only AFTER being processed).
      · i <  j  is already handled by the causal mask.

    Args
        evict  : [B, S]   1.0 = evicted
        q_len  : int      number of query rows (== S in this sliding-window eval)

    Returns
        mask   : [B, 1, q_len, S]  additive (0 or -1e9)
    """
    bsz, seq_len = evict.shape
    strict_lower = torch.tril(                                  # i > j → 1
        torch.ones(q_len, seq_len, device=evict.device, dtype=evict.dtype),
        diagonal=-1,
    )
    mask = evict[:, None, None, :] * strict_lower[None, None, :, :]   # [B,1,q_len,S]
    return mask * (-1e9)


def resolve_eviction(
    c_kv: torch.Tensor,
    layer_idx: int,
    strategy: str,
    state: EvictionState,
):
    """
    Determine the per-token eviction vector for the current layer.

    Handles cross-layer coordination: Layer 0 always resets shared state.

    Returns
        evict_vec : Tensor [B, S] (1=evict) or None
        kept_frac : float
    """
    # ── Reset shared state at the first layer of every forward pass ───────
    if layer_idx == 0:
        state.evict_vector = None

    # ── Config A: no eviction ─────────────────────────────────────────────
    if strategy == "baseline":
        return None, 1.0

    # ── Config B: Layer 0 decides → all layers share ──────────────────────
    if strategy == "layer_0":
        if layer_idx == 0:
            ev, kf = _build_evict_vector(c_kv)
            state.evict_vector     = ev
            state.kept_frac_shared = kf
        if state.evict_vector is None:
            return None, 1.0
        return state.evict_vector.to(c_kv.device), state.kept_frac_shared

    # ── Config C: layers 0-1 dense; layer 2 computes for layers 3+ ───────
    if strategy == "layer_2":
        if layer_idx < 2:
            return None, 1.0
        if layer_idx == 2:
            # Compute and store — but layer 2 itself stays dense
            ev, kf = _build_evict_vector(c_kv)
            state.evict_vector     = ev
            state.kept_frac_shared = kf
            return None, 1.0
        # layer >= 3
        if state.evict_vector is None:
            return None, 1.0
        return state.evict_vector.to(c_kv.device), state.kept_frac_shared

    # ── Config D: each layer decides independently ────────────────────────
    if strategy == "oracle":
        ev, kf = _build_evict_vector(c_kv)
        return ev, kf

    return None, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Patched attention forward
# ═══════════════════════════════════════════════════════════════════════════════

def make_eval_forward(layer_idx: int, strategy: str, state: EvictionState):
    """
    Returns a replacement `forward` for DeepseekV2Attention that:
      1. Performs the standard MLA eager attention computation.
      2. Optionally injects an eviction column mask before softmax.

    The patched forward is used for ALL configs (including Baseline) to
    ensure identical execution paths; for Baseline the mask is simply None.
    """

    def _fwd(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        kwargs.pop("padding_mask", None)
        bsz, q_len, _ = hidden_states.size()

        # ── Query projection ───────────────────────────────────────────────
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # ── KV projection ──────────────────────────────────────────────────
        raw     = self.kv_a_proj_with_mqa(hidden_states)
        c_kv, k_pe = torch.split(raw, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe    = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv      = (
            self.kv_b_proj(self.kv_a_layernorm(c_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ── RoPE ───────────────────────────────────────────────────────────
        if position_ids is None:
            position_ids = torch.arange(
                q_len, dtype=torch.long, device=hidden_states.device
            ).unsqueeze(0)
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        Q = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        Q[:, :, :, : self.qk_nope_head_dim] = q_nope
        Q[:, :, :, self.qk_nope_head_dim :] = q_pe

        K = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        K[:, :, :, : self.qk_nope_head_dim] = k_nope
        K[:, :, :, self.qk_nope_head_dim :] = k_pe

        # ── Attention logits ────────────────────────────────────────────────
        logits = torch.matmul(Q, K.transpose(2, 3)) * self.softmax_scale

        # Apply causal attention mask (shape [B, 1, q_len, q_len])
        if attention_mask is not None:
            logits = logits + attention_mask

        # ── Eviction (strategy-dependent, causal & diagonal-safe) ──────────
        evict_vec, kept_frac = resolve_eviction(c_kv, layer_idx, strategy, state)
        if evict_vec is not None:
            logits = logits + _causal_evict_mask(evict_vec, q_len)

        state.kept_fracs.append(kept_frac)

        # ── Softmax → dropout → weighted sum ───────────────────────────────
        attn = F.softmax(logits, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn = F.dropout(attn, p=self.attention_dropout, training=self.training)
        out  = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous()
        out = out.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        out = self.o_proj(out)

        return out, (attn if output_attentions else None), past_key_value

    return _fwd


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Sliding-window PPL evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_ppl(model, encodings, strategy: str, device: str):
    """
    Compute sliding-window perplexity and mean KV kept fraction.

    For each sliding window [begin, end]:
      · Feed the full window as context.
      · Loss is computed only on the NEW tokens (last `trg_len` positions).
      · This ensures every token's loss is computed exactly once.

    Returns
        ppl      : float   perplexity
        kept_avg : float   mean fraction of tokens kept (compression metric)
    """
    num_layers  = model.config.num_hidden_layers
    attn_layers = [layer.self_attn for layer in model.model.layers]
    state       = EvictionState()

    # ── Install patched forwards ────────────────────────────────────────────
    orig_fwds = []
    for idx, attn in enumerate(attn_layers):
        orig_fwds.append(attn.forward)
        attn.forward = types.MethodType(
            make_eval_forward(idx, strategy, state), attn
        )

    seq_len  = encodings.input_ids.shape[1]
    nlls     = []
    all_kept = []
    prev_end = 0
    n_chunks = 0

    total_steps = min(seq_len // STRIDE + 1, MAX_CHUNKS or 10_000)

    try:
        with tqdm(
            total=total_steps,
            desc=f"  [{strategy:<10}]",
            unit="chunk",
            leave=False,
        ) as pbar:
            for begin in range(0, seq_len, STRIDE):
                if MAX_CHUNKS and n_chunks >= MAX_CHUNKS:
                    break

                end     = min(begin + MAX_SEQ_LEN, seq_len)
                trg_len = end - prev_end          # number of NEW tokens this step

                ids = encodings.input_ids[:, begin:end].to(device)
                tgt = ids.clone()
                tgt[:, :-trg_len] = -100          # mask context; score only new tokens

                state.kept_fracs = []             # reset per-chunk accumulator

                with torch.no_grad():
                    loss = model(ids, labels=tgt).loss

                nlls.append(loss.detach().cpu() * trg_len)

                if state.kept_fracs:
                    all_kept.append(float(np.mean(state.kept_fracs)))

                prev_end  = end
                n_chunks += 1
                pbar.update(1)

                if end == seq_len:
                    break

    finally:
        # ── Restore original forwards ───────────────────────────────────────
        for attn, orig in zip(attn_layers, orig_fwds):
            attn.forward = orig

    ppl      = torch.exp(torch.stack(nlls).sum() / prev_end).item()
    kept_avg = float(np.mean(all_kept)) if all_kept else 1.0
    return ppl, kept_avg


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  Latent Eviction Cross-Layer Decision  —  Ablation Study")
    print("  DeepSeek-V2-Lite  ·  WikiText-2 test  ·  PPL + Compression")
    print("═" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"\n  device        = {device}\n"
        f"  dtype         = {DTYPE}\n"
        f"  keep_ratio    = {KEEP_RATIO}  (Top-K, fixed compression)\n"
        f"  window        = {EVICTION_WINDOW}  (causal, past-only)\n"
        f"  max_seq_len   = {MAX_SEQ_LEN}\n"
        f"  stride        = {STRIDE}\n"
        f"  max_chunks    = {MAX_CHUNKS}\n"
    )

    # ── Load tokeniser & model ─────────────────────────────────────────────
    print("[ Loading tokenizer … ]")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[ Loading model (attn_implementation=eager) … ]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="eager",   # ensures DeepseekV2Attention is used
    )
    model.eval()
    print(f"  num_hidden_layers = {model.config.num_hidden_layers}")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("\n[ Loading WikiText-2 test … ]")
    enc = load_wikitext2_test(tok)

    # ── Run all configurations ─────────────────────────────────────────────
    print("\n[ Running evaluations … ]\n")
    results: list[tuple] = []   # (id, label, strategy, ppl, kept, elapsed_s)

    for cfg_id, label, strategy in CONFIGS:
        print(f"  Config {cfg_id}  ·  {label}")
        t0            = time.time()
        ppl, kept_avg = evaluate_ppl(model, enc, strategy, device)
        elapsed       = time.time() - t0
        results.append((cfg_id, label, strategy, ppl, kept_avg, elapsed))
        print(
            f"    PPL = {ppl:.4f}  |  Kept = {kept_avg:.1%}  "
            f"|  Elapsed = {elapsed:.0f} s\n"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Print Markdown table
    # ═══════════════════════════════════════════════════════════════════════
    baseline_ppl = results[0][3]

    # Column widths tuned for readability in monospace / paper
    HDR = (
        "| Config | Description              |     PPL |  Kept% | Evicted% "
        "| Δ PPL vs A |  Time(s) |"
    )
    SEP = (
        "|:------:|:-------------------------|--------:|-------:|---------:"
        "|-----------:|---------:|"
    )

    print("\n" + "═" * 72)
    print("  RESULTS — Perplexity & KV Cache Compression (WikiText-2 test)")
    print("═" * 72 + "\n")
    print(HDR)
    print(SEP)

    for cfg_id, label, strategy, ppl, kept, elapsed in results:
        evicted = 1.0 - kept
        delta   = ppl - baseline_ppl
        if cfg_id == "A":
            delta_s = "—"
        else:
            delta_s = f"{delta:+.4f}"
        print(
            f"| {cfg_id:^6} | {label:<24} | {ppl:7.4f} | {kept:5.1%} "
            f"| {evicted:7.1%}  | {delta_s:>10}  | {elapsed:7.0f}  |"
        )

    # ── Experiment metadata footer ─────────────────────────────────────────
    print()
    print(
        f"> **Model**: DeepSeek-V2-Lite  |  "
        f"**Dataset**: WikiText-2 test  |  "
        f"**Keep ratio**: {KEEP_RATIO} (Top-K)  |  "
        f"**Window**: {EVICTION_WINDOW} (causal)  |  "
        f"**Seq len**: {MAX_SEQ_LEN}  |  "
        f"**Stride**: {STRIDE}  |  "
        f"**Chunks**: {MAX_CHUNKS}"
    )

    # ── Automatic interpretation ───────────────────────────────────────────
    _, _, _, ppl_b, kept_b, _ = results[1]
    _, _, _, ppl_c, kept_c, _ = results[2]
    _, _, _, ppl_d, kept_d, _ = results[3]

    delta_b  = ppl_b - baseline_ppl
    delta_c  = ppl_c - baseline_ppl
    delta_d  = ppl_d - baseline_ppl
    bc_gap   = abs(delta_b - delta_c)
    bd_gap   = abs(delta_b - delta_d)
    rel_bc   = bc_gap / max(abs(delta_b), 1e-6)

    print("\n" + "─" * 72)
    print("  KEY FINDINGS\n")
    print(f"  Δ PPL  A→B  (Layer-0 Decision vs Baseline) : {delta_b:+.4f}")
    print(f"  Δ PPL  A→C  (Delayed Decision vs Baseline) : {delta_c:+.4f}")
    print(f"  Δ PPL  A→D  (Oracle vs Baseline)           : {delta_d:+.4f}")
    print(f"  |B−C|  (Layer-0 vs Delayed)                : {bc_gap:.4f}")
    print(f"  |B−D|  (Layer-0 vs Oracle)                 : {bd_gap:.4f}")
    print()

    if rel_bc < 0.10:
        print(
            "  ✓  B ≈ C  (|B−C| < 10% of B's degradation) — "
            "features settle already at Layer 0; delaying yields no benefit."
        )
    else:
        print(
            f"  △  B and C differ by {bc_gap:.4f} PPL — "
            "there may be incremental benefit to using a later decision layer."
        )

    if bd_gap / max(abs(delta_b), 1e-6) < 0.15:
        print(
            "  ✓  B ≈ D  (Layer-0 within 15% of Oracle) — "
            "cross-layer shared decision approaches the independent-oracle ceiling."
        )
    else:
        print(
            f"  △  B lags D by {bd_gap:.4f} PPL — "
            "per-layer independent decisions provide measurable additional benefit."
        )

    print("─" * 72)


if __name__ == "__main__":
    main()
