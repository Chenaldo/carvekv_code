"""
configure_training.py — Calibrate compute_latent_info_score weights
====================================================================

Finds optimal (w_norm, w_entropy, w_variance, w_redundancy) for the
statistical token-importance scoring used in latent KV-cache eviction.

Why rank-correlation, not PPL?
-------------------------------
In the production forward path (q_abs available), these weights only
affect keep_k (how many tokens to retain), NOT which tokens are selected
(that is decided by the query-aware attention column-sum scoring).
Directly maximising the Spearman rank-correlation between the statistical
proxy and the query-aware oracle is therefore:
  · Faster:    one calibration pass vs. N_trials × full PPL evaluations
  · Cleaner:   optimises the right objective directly
  · Accurate:  higher correlation → better budget estimation → better PPL

Two calibration methods
------------------------
  NNLS  (fast, ~5 min):
    Non-negative least squares on (features, oracle_scores) pairs.
    Guaranteed globally optimal for the L2 objective.

  Optuna (thorough, ~20 min, optional):
    Bayesian optimisation over weight space maximising Spearman ρ.
    Useful when NNLS solution is on the boundary (a weight collapses to 0).
    Falls back to random search if optuna is not installed.

Usage
-----
    python configure_training.py                    # NNLS + Optuna
    python configure_training.py --method nnls      # NNLS only
    python configure_training.py --method optuna --n-trials 300
    python configure_training.py --n-seqs 100       # more calibration data

Output
------
    calibrated_weights.json   — weights + diagnostics
    Printed copy-paste snippet for compute_latent_info_score() defaults
"""

import os
import sys
import json
import types
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

# ─── Defaults ─────────────────────────────────────────────────────────────────
N_CALIB_SEQS   = 50     # calibration forward passes
MAX_SEQ_LEN    = 1024   # tokens per pass
DECISION_LAYER = 6      # layer whose features are calibrated (matches production)
DTYPE          = torch.bfloat16
SCORE_QUERIES  = 512    # subsampled queries for oracle column sums (OOM guard)


# ─── Signal helpers ───────────────────────────────────────────────────────────

def _robust_normalize(t: torch.Tensor) -> torch.Tensor:
    """Outlier-robust normalize [B, S] → (0,1) via median/MAD + sigmoid."""
    med = t.median(dim=-1, keepdim=True).values
    mad = (t - med).abs().median(dim=-1, keepdim=True).values
    return torch.sigmoid((t - med) / (1.4826 * mad + 1e-6))


def _statistical_features(c_kv: torch.Tensor, window: int = 4) -> torch.Tensor:
    """
    Compute the 4 normalized statistical feature columns used in
    compute_latent_info_score, suitable for NNLS fitting.

    Returns [B, S, 4]:
        col 0:  _robust_normalize(L2 norm)
        col 1:  _robust_normalize(negative entropy)
        col 2:  _robust_normalize(per-dim variance)
        col 3: -_robust_normalize(neighbour redundancy)   ← sign-flipped!

    col 3 is negated so that all four NNLS coefficients are ≥ 0 and map
    directly to (w_norm, w_entropy, w_variance, w_redundancy).
    The original formula is:
        info = w_norm*f0 + w_entropy*f1 + w_variance*f2 - w_redundancy*red
             = w @ [f0, f1, f2, -red]     with w = [w_norm, w_entropy, w_variance, w_redundancy]
    """
    x = c_kv.float()
    B, S, _ = x.shape

    # L2 norm — larger = richer representation
    norm = x.norm(dim=-1)                                       # [B, S]

    # Negative entropy — more peaked softmax = more specific token
    prob        = F.softmax(x, dim=-1)
    neg_entropy = (prob * (prob + 1e-10).log()).sum(dim=-1)     # Σ p·log(p) = -H

    # Per-dim variance — less flat = more informative
    variance = x.var(dim=-1)                                    # [B, S]

    # Neighbour redundancy — high cosine similarity to neighbours = less worth keeping
    x_unit     = F.normalize(x, dim=-1)
    redundancy = torch.zeros(B, S, device=x.device, dtype=x.dtype)
    for offset in range(1, max(1, window) + 1):
        if offset >= S:
            break
        sim = (x_unit[:, offset:] * x_unit[:, :-offset]).sum(dim=-1)
        redundancy[:, offset:]  = torch.maximum(redundancy[:, offset:],  sim)
        redundancy[:, :-offset] = torch.maximum(redundancy[:, :-offset], sim)

    f0 =  _robust_normalize(norm)
    f1 =  _robust_normalize(neg_entropy)
    f2 =  _robust_normalize(variance)
    f3 = -_robust_normalize(redundancy)    # negated → positive coefficient in NNLS

    return torch.stack([f0, f1, f2, f3], dim=-1)  # [B, S, 4]


def _oracle_scores(q_abs: torch.Tensor, c_kv: torch.Tensor,
                   W_UV: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Query-aware × Value-aware oracle importance at Layer D.

    Identical to the production _compute_keep_indices scoring:
        Importance[j] = (Σ_i prob_{i,j}) × ‖W_UV @ c_j‖₂

    Returns [B, S] scores, min-max normalised to [0, 1].
    """
    B, H, S, R = q_abs.shape
    K = min(SCORE_QUERIES, S)

    if K < S:
        sample_pos = torch.randperm(S, device=q_abs.device)[:K].sort().values
        q_sc  = q_abs[:, :, sample_pos, :]
        cmask = (torch.arange(S, device=q_abs.device).unsqueeze(0)
                 > sample_pos.unsqueeze(1))
    else:
        q_sc  = q_abs
        cmask = ~torch.tril(torch.ones(S, S, device=q_abs.device, dtype=torch.bool))

    logits = torch.matmul(q_sc, c_kv.unsqueeze(1).transpose(-1, -2)) * scale
    logits = logits.masked_fill(cmask[None, None], float("-inf"))
    probs   = torch.softmax(logits, dim=-1, dtype=torch.float32)
    col_sum = probs.sum(dim=2).mean(dim=1)                      # [B, S]

    # Value-aware weighting  ‖W_UV @ c_j‖₂ per token, averaged over heads
    val_proj = torch.einsum("bnr,hdr->bhnd", c_kv.float(), W_UV.float())
    val_norm = val_proj.norm(dim=-1).mean(dim=1)                # [B, S]
    v_mean   = val_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    val_norm = val_norm / v_mean

    oracle = col_sum * val_norm                                 # [B, S]

    # Min-max normalise to [0, 1] for comparable regression targets
    lo = oracle.min(dim=-1, keepdim=True).values
    hi = oracle.max(dim=-1, keepdim=True).values
    return (oracle - lo) / (hi - lo + 1e-6)


# ─── Data collection ─────────────────────────────────────────────────────────

def _load_corpus(tokenizer, n_tokens: int) -> torch.Tensor:
    """Load code corpus and return token ids [1, n_tokens]."""
    text = None

    try:
        from datasets import load_dataset
        ds = load_dataset("code-search-net/code_search_net", "python",
                          split="test", trust_remote_code=False)
        snippets = [ex["whole_func_string"] for ex in ds
                    if ex.get("whole_func_string", "").strip()]
        if snippets:
            raw  = "\n\n".join(snippets)
            text = raw[:n_tokens * 5]   # ~5 chars/token — truncate BEFORE tokenizing
            print(f"  Corpus : code-search-net Python ({len(snippets):,} functions)")
    except Exception as exc:
        print(f"  [WARN] code-search-net unavailable ({exc}); using built-in code")

    if text is None:
        base = (
            "def quicksort(arr):\n    if len(arr)<=1: return arr\n"
            "    pivot=arr[len(arr)//2]\n"
            "    l=[x for x in arr if x<pivot]\n"
            "    m=[x for x in arr if x==pivot]\n"
            "    r=[x for x in arr if x>pivot]\n"
            "    return quicksort(l)+m+quicksort(r)\n\n"
            "def binary_search(arr, t):\n    lo,hi=0,len(arr)-1\n"
            "    while lo<=hi:\n        m=(lo+hi)//2\n"
            "        if arr[m]==t: return m\n"
            "        elif arr[m]<t: lo=m+1\n        else: hi=m-1\n    return -1\n\n"
        )
        reps = (n_tokens * 4 // len(base)) + 2
        text = (base * reps)[: n_tokens * 4]
        print(f"  Corpus : built-in Python fallback")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Token indices sequence length")
        ids = tokenizer(text, return_tensors="pt").input_ids

    ids = ids[:, :n_tokens]
    print(f"  Tokens : {ids.shape[1]:,}")
    return ids


def collect_calibration_data(model, tokenizer, n_seqs: int, max_seq_len: int,
                              decision_layer: int, device: str):
    """
    Run n_seqs forward passes; at `decision_layer` capture:
      · statistical feature matrix  [B, S, 4]
      · oracle importance scores     [B, S]

    Returns:
        X : np.ndarray [N_tokens, 4]
        y : np.ndarray [N_tokens]
    """
    corpus = _load_corpus(tokenizer, n_seqs * max_seq_len + max_seq_len).to(device)

    attn     = model.model.layers[decision_layer].self_attn
    orig_fwd = attn.forward
    captured: dict = {}

    # ── Monkey-patch: capture features + oracle at decision_layer ────────────
    def _capture(self_obj, hidden_states, attention_mask=None, position_ids=None,
                 past_key_value=None, output_attentions=False, use_cache=False, **kw):
        kw.pop("padding_mask", None)
        with torch.no_grad():
            B, S, _ = hidden_states.size()

            if self_obj.q_lora_rank is None:
                q = self_obj.q_proj(hidden_states)
            else:
                q = self_obj.q_b_proj(
                    self_obj.q_a_layernorm(self_obj.q_a_proj(hidden_states)))
            q = q.view(B, S, self_obj.num_heads, self_obj.q_head_dim).transpose(1, 2)
            q_nope, _ = torch.split(
                q, [self_obj.qk_nope_head_dim, self_obj.qk_rope_head_dim], dim=-1)

            raw = self_obj.kv_a_proj_with_mqa(hidden_states)
            c_kv_raw, _ = torch.split(
                raw, [self_obj.kv_lora_rank, self_obj.qk_rope_head_dim], dim=-1)
            c_kv = self_obj.kv_a_layernorm(c_kv_raw)   # [B, S, R]

            W_kv = self_obj.kv_b_proj.weight.view(
                self_obj.num_heads,
                self_obj.qk_nope_head_dim + self_obj.v_head_dim,
                self_obj.kv_lora_rank,
            )
            W_UK = W_kv[:, :self_obj.qk_nope_head_dim, :]
            W_UV = W_kv[:, self_obj.qk_nope_head_dim:, :]
            q_abs = torch.einsum("bhqd,hdr->bhqr", q_nope, W_UK)

            captured["features"] = _statistical_features(c_kv).cpu()
            captured["oracle"]   = _oracle_scores(
                q_abs, c_kv, W_UV, self_obj.softmax_scale).cpu()

        return orig_fwd(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, **kw,
        )

    attn.forward = types.MethodType(_capture, attn)

    all_features, all_oracle = [], []
    n_done = 0
    total  = corpus.shape[1]

    try:
        for start in range(0, total - max_seq_len, max_seq_len):
            if n_done >= n_seqs:
                break
            ids = corpus[:, start : start + max_seq_len]
            with torch.no_grad():
                model(ids, use_cache=False)
            if "features" in captured:
                all_features.append(captured["features"].squeeze(0).numpy())
                all_oracle.append(  captured["oracle"].squeeze(0).numpy())
                n_done += 1
                print(f"\r  Collecting {n_done}/{n_seqs} sequences …", end="", flush=True)
    finally:
        attn.forward = orig_fwd

    print()
    X = np.concatenate(all_features, axis=0)   # [N_tokens, 4]
    y = np.concatenate(all_oracle,   axis=0)   # [N_tokens]
    print(f"  Dataset : {X.shape[0]:,} token samples (4 features each)\n")
    return X, y


# ─── Calibration methods ─────────────────────────────────────────────────────

def calibrate_nnls(X: np.ndarray, y: np.ndarray):
    """
    Non-negative least squares: min‖X w − y‖²  s.t. w ≥ 0.

    The globally optimal solution for the L2 objective.  Coefficients map
    directly to (w_norm, w_entropy, w_variance, w_redundancy) because col 3
    of X is already sign-flipped to −redundancy_norm.
    """
    w, _ = nnls(X, y)
    rho, _ = spearmanr(X @ w, y)
    return w, float(rho)


def calibrate_optuna(X: np.ndarray, y: np.ndarray, n_trials: int = 200):
    """
    Maximise Spearman rank-correlation via Bayesian optimisation.

    Falls back to random search if optuna is not installed.
    Useful when an NNLS coefficient collapses to 0 (boundary solution).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            w = np.array([
                trial.suggest_float("w_norm",       0.0, 3.0),
                trial.suggest_float("w_entropy",    0.0, 3.0),
                trial.suggest_float("w_variance",   0.0, 3.0),
                trial.suggest_float("w_redundancy", 0.0, 3.0),
            ])
            rho, _ = spearmanr(X @ w, y)
            return float(rho)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best = study.best_params
        w = np.array([best["w_norm"], best["w_entropy"],
                      best["w_variance"], best["w_redundancy"]])
        return w, float(study.best_value)

    except ImportError:
        print("  [INFO] optuna not installed → running random search (5 000 samples)")
        return _random_search(X, y)


def _random_search(X: np.ndarray, y: np.ndarray, n_samples: int = 5_000):
    rng = np.random.default_rng(42)
    best_rho, best_w = -1.0, np.array([0.4, 0.3, 0.3, 0.5])
    for _ in range(n_samples):
        w = rng.uniform(0.0, 3.0, 4)
        rho, _ = spearmanr(X @ w, y)
        if rho > best_rho:
            best_rho, best_w = rho, w.copy()
    return best_w, float(best_rho)


# ─── Diagnostic ───────────────────────────────────────────────────────────────

def feature_diagnostics(X: np.ndarray, y: np.ndarray):
    """Per-feature Spearman correlation with oracle (shows individual signal quality)."""
    labels = ["w_norm", "w_entropy", "w_variance", "w_redundancy"]
    print("  Per-feature Spearman ρ with oracle:")
    for i, lbl in enumerate(labels):
        sign  = "+" if i < 3 else "−"   # redundancy is a penalty
        rho, p = spearmanr(X[:, i], y)
        bar = "█" * int(abs(rho) * 20)
        print(f"    {sign}{lbl:<16}  ρ = {rho:+.4f}  {bar}")
    print()


# ─── Output helpers ───────────────────────────────────────────────────────────

def _fmt_weights(w: np.ndarray) -> str:
    labels = ["w_norm", "w_entropy", "w_variance", "w_redundancy"]
    return ", ".join(f"{l}={v:.4f}" for l, v in zip(labels, w))


def _print_result(method: str, w: np.ndarray, rho: float):
    labels = ["w_norm", "w_entropy", "w_variance", "w_redundancy"]
    print(f"  {method}  (Spearman ρ = {rho:.4f})")
    for l, v in zip(labels, w):
        bar = "▮" * int(v / 3.0 * 20)
        print(f"    {l:<16} = {v:.4f}  {bar}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate compute_latent_info_score weights")
    parser.add_argument(
        "--method", choices=["nnls", "optuna", "both"], default="both")
    parser.add_argument(
        "--n-seqs", type=int, default=N_CALIB_SEQS,
        help=f"Calibration sequences (default {N_CALIB_SEQS})")
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Optuna search trials (default 200)")
    parser.add_argument(
        "--decision-layer", type=int, default=DECISION_LAYER,
        help=f"Layer D for feature capture (default {DECISION_LAYER})")
    parser.add_argument(
        "--output", default="calibrated_weights.json")
    args = parser.parse_args()

    print("\n" + "═" * 68)
    print("  compute_latent_info_score — Weight Calibration")
    print(f"  Method: {args.method}  |  Seqs: {args.n_seqs}"
          f"  |  Decision Layer: {args.decision_layer}")
    print("═" * 68)
    print(
        "\n  Objective: maximise Spearman(statistical_score, oracle_score)\n"
        "  Oracle   : query-aware column-sum × Value-norm at Layer "
        f"{args.decision_layer}\n"
        "  This is faster and more principled than PPL-based optimisation\n"
        "  (weights only affect budget keep_k, not token ranking in production)\n"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[ Loading tokenizer … ]")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("[ Loading model … ]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    print(f"\n[ Collecting calibration data (Layer {args.decision_layer}) … ]")
    X, y = collect_calibration_data(
        model, tok, args.n_seqs, MAX_SEQ_LEN, args.decision_layer, device)

    # Diagnostic: individual feature quality
    feature_diagnostics(X, y)

    results = {}

    # Default weights as baseline
    w_default = np.array([0.4, 0.3, 0.3, 0.5])
    rho_default, _ = spearmanr(X @ w_default, y)
    print(f"  Baseline (default weights)  ρ = {rho_default:.4f}\n")

    if args.method in ("nnls", "both"):
        print("[ Method 1: NNLS … ]")
        w_nnls, rho_nnls = calibrate_nnls(X, y)
        results["nnls"] = {"weights": w_nnls.tolist(), "spearman_rho": rho_nnls}
        _print_result("NNLS", w_nnls, rho_nnls)

    if args.method in ("optuna", "both"):
        print(f"[ Method 2: Optuna ({args.n_trials} trials) … ]")
        w_opt, rho_opt = calibrate_optuna(X, y, n_trials=args.n_trials)
        results["optuna"] = {"weights": w_opt.tolist(), "spearman_rho": rho_opt}
        _print_result("Optuna", w_opt, rho_opt)

    # ── Pick best ─────────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["spearman_rho"])
    best_w    = np.array(results[best_name]["weights"])
    best_rho  = results[best_name]["spearman_rho"]
    improvement = best_rho - rho_default

    labels = ["w_norm", "w_entropy", "w_variance", "w_redundancy"]

    print("═" * 68)
    print(f"  BEST: {best_name}  (ρ = {best_rho:.4f}, Δ vs default = {improvement:+.4f})")
    print()
    for l, v in zip(labels, best_w):
        print(f"    {l:<16} = {v:.4f}")
    print("═" * 68)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "best_method":         best_name,
        "best_weights":        {l: float(v) for l, v in zip(labels, best_w)},
        "spearman_rho":        float(best_rho),
        "default_rho":         float(rho_default),
        "improvement":         float(improvement),
        "calibration_config":  {
            "n_seqs":          args.n_seqs,
            "max_seq_len":     MAX_SEQ_LEN,
            "decision_layer":  args.decision_layer,
            "model_dir":       MODEL_DIR,
        },
        "all_results": results,
        "usage": (
            "Paste the best_weights into compute_latent_info_score() defaults:\n"
            f"  {_fmt_weights(best_w)}"
        ),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {args.output}")

    print("\n  ── Copy-paste into compute_latent_info_score() ──────────────────")
    print(f"  def compute_latent_info_score(")
    print(f"      compressed_kv,")
    print(f"      window: int = 4,")
    print(f"      w_norm:       float = {best_w[0]:.4f},")
    print(f"      w_entropy:    float = {best_w[1]:.4f},")
    print(f"      w_variance:   float = {best_w[2]:.4f},")
    print(f"      w_redundancy: float = {best_w[3]:.4f},")
    print(f"  ) -> torch.Tensor:")
    print()


if __name__ == "__main__":
    main()
