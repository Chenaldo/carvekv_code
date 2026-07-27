"""
final_run.py  --  DeepSeek-Coder-V2-Lite Latent KV Eviction
             Comprehensive Evaluation Script
=======================================================================

Experiment matrix: 3 configurations x 6 test categories

Configurations
--------------
  baseline   eviction=False                  (full KV cache, no eviction)
  evict_L4   eviction=True, decision_layer=4 (early-layer, more aggressive)
  evict_L6   eviction=True, decision_layer=6 (recommended, balanced)

Test categories
---------------
  1. Suffix PPL     code-search-net/python → fineweb-edu(code) → built-in Python
                    (code-first; WikiText-2/C4 excluded — poor proxy for a
                    code model)
                    Method: prefill prefix (eviction triggers) -> compute NLL
                    on suffix given the post-eviction KV cache.
  2. NIAH           Needle-in-a-Haystack: integer constant buried in Python
                    code at 25%/50%/75% depth, context 2k/4k/8k tokens.
  3. Code recall    6 synthetic prompts with cross-reference variables;
                    does the model recall names defined far earlier?
  4. HumanEval      syntax-pass@1 on N problems (needs evalplus/human-eval).
  5. LongBench      multi-doc QA F1 (needs HuggingFace datasets).
  6. System         Peak VRAM, prefill latency, TTFT, cache compression
                    at 1k/2k/4k/8k contexts.

Usage
-----
    python final_run.py                     # full run
    python final_run.py --quick             # fast smoke test
    python final_run.py --skip-optional     # skip HumanEval + LongBench
    python final_run.py --niah-max 32768    # extend NIAH to 32k tokens
    python final_run.py --save out.json     # output path
"""

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache as _DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

# ── Experiment configurations ─────────────────────────────────────────────────
# Algorithm: Multi-layer committee scoring + Soft eviction (Super Tokens)
#
# Committee layers {4,6,8,12,16} each compute importance scores; layer 16
# averages all scores and makes the final keep/evict decision.
# Evicted tokens are pooled into super tokens (pool_ratio=4 → 1) and
# appended to the cache instead of being discarded.
#
# Ablation design:
#   baseline          — no eviction, measures raw model quality
#   committee_c90     — full algorithm, cov_floor=0.55 → adaptive ≤70% total compression
#                       (name kept for comparability; cov_floor was 0.90 before)
#   committee_c85     — more aggressive, cov_floor=0.45 → adaptive ≤70%, typically 55-65%
#   committee_hard_c90 — committee scoring only, NO soft eviction (pool_ratio=0)
#                       isolates the benefit of super tokens
#
# cov_floor tuning rationale (pool_ratio=4):
#   total_compress ≈ mid_keep + (1-mid_keep)/4 = 0.75×mid_keep + 0.25
#   Hard cap: MAX_KEEP=0.60 → max total = 0.60 + 0.10 = 0.70 (70%)
#   For adaptivity, cov_floor must be ≤ MAX_KEEP so knee detection dominates
#   and cov_k < 0.60×n_mid for at least some content distributions:
#     cov_floor=0.55 → uniform corpus: cov_k≈0.55→total≈66%; concentrated→less
#     cov_floor=0.45 → uniform corpus: cov_k≈0.45→total≈59%; concentrated→less
CONFIGS: List[Dict[str, Any]] = [
    {"name": "baseline",
     "eviction": False,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0},
    {"name": "committee_c90",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.55, "pool_ratio": 4},
    {"name": "committee_c85",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.45, "pool_ratio": 4},
    {"name": "committee_hard_c90",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.55, "pool_ratio": 0},
]

# ── Ablation configurations ───────────────────────────────────────────────────
# Each ablation isolates exactly ONE design choice vs the full committee_c90.
#
#   Ablation A  (soft eviction)    : committee_hard_c90 is already in CONFIGS;
#                                    reused here for the joint ablation table.
#   Ablation B  (adaptive budget)  : fixed_ratio_70  — knee disabled, exact 70%
#                                    compare vs committee_c90 to show knee value
#   Ablation C  (committee)        : single_L6 / single_L12
#                                    single layer, no averaging, same c90 floor
#   Ablation D  (cross-layer share): oracle_approx — score per layer independently
#                                    (approximated by decision_layer=6 with no
#                                    retroactive pruning; real Oracle is the PPL
#                                    eval's Config D in eval_layer_decision.py)
ABLATION_CONFIGS: List[Dict[str, Any]] = [
    # Reference: full algorithm and baseline
    {"name": "baseline",
     "eviction": False,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0},
    {"name": "committee_c90",           # full system (reference)
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.55, "pool_ratio": 4},
    # Ablation A: isolate soft eviction (super tokens)
    {"name": "hard_evict_c90",          # same as committee_hard_c90
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.55, "pool_ratio": 0},
    # Ablation B: isolate adaptive budget (knee detection)
    # min_keep == max_keep == 0.60 → mid-token keep fixed at 60%, bypasses knee
    # total ≈ 0.60 + 0.40/4 = 70%  (compare vs committee_c90 to show knee value)
    {"name": "fixed_ratio_70",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.55,
     "min_keep": 0.60,   "max_keep": 0.60,   "pool_ratio": 4},
    # Ablation C-1: isolate committee (single layer L6, same coverage floor)
    {"name": "single_L6",
     "eviction": True,   "decision_layer": 6,
     "score_layers": [6],                "cov_floor": 0.55, "pool_ratio": 4},
    # Ablation C-2: isolate committee (single layer L12)
    {"name": "single_L12",
     "eviction": True,   "decision_layer": 12,
     "score_layers": [12],               "cov_floor": 0.55, "pool_ratio": 4},
]

# Eviction hyper-parameters (defaults; per-config values override via cfg key)
# cov_floor must be ≤ MAX_KEEP so that the knee/cov budget is not always
# overridden by the hard cap, keeping the algorithm adaptive.
# Uniform corpus: cov_k ≈ cov_floor × n_mid; concentrated: less.
COVERAGE_FLOOR   = 0.55
MIN_KEEP         = 0.30
# MAX_KEEP: hard upper bound on mid-token keep fraction.
# Total compression ≈ MAX_KEEP + (1-MAX_KEEP)/pool_ratio
#   = 0.60 + 0.40/4 = 0.70  →  max 70% cache retention.
# Adaptive budget varies below this cap based on importance distribution.
MAX_KEEP         = 0.60
EVICTION_WINDOW  = 4
SCORE_LAYERS     = [4, 6, 8, 12, 16]   # committee layers
POOL_RATIO       = 4                    # n_evicted // pool_ratio = n_super
POOL_TEMPERATURE = 0.1                  # softmax temperature for content pooling
DTYPE            = torch.bfloat16

# Default experiment scale (overridden by --quick via args)
_DEFAULTS = dict(
    n_ppl_seqs   = 20,
    n_niah_seeds = 3,
    niah_lengths = [2048, 4096, 6144, 8192, 12288, 16384, 32768, 65536],
    niah_depths  = [0.25, 0.50, 0.75],
    n_humaneval  = 20,
    n_longbench  = 10,
    sys_lengths  = [1024, 2048, 4096, 6144, 8192, 16384, 32768, 65536],
)

# ── Baseline comparison configurations ───────────────────────────────────────
# Head-to-head comparison of our method vs StreamingLLM and H2O at equivalent
# ~50% cache retention.  committee_c90/c85 use the full algorithm (eviction
# handled inside modeling_deepseek_test.py).  streaming_llm / h2o run with
# eviction=False and apply post-hoc cache truncation after prefill via a
# cache_fn callback — no changes to the model forward pass required.
COMPARISON_CONFIGS: List[Dict[str, Any]] = [
    {"name": "baseline",
     "eviction": False, "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0},
    {"name": "committee_c90",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.55, "pool_ratio": 4},
    {"name": "committee_c85",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.45, "pool_ratio": 4},
    # StreamingLLM: keep first n_sink attention-sink tokens + recent sliding window.
    # keep_ratio=0.50 — real in-model prefill eviction, not post-hoc truncation.
    {"name": "streaming_llm",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0,
     "eviction_mode": "streaming", "keep_ratio": 0.50},
    # H2O Heavy-Hitter Oracle: keep top tokens by cumulative attention score
    # + recent window.  Real in-model prefill eviction using attention weights.
    {"name": "h2o",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0,
     "eviction_mode": "h2o", "keep_ratio": 0.50, "h2o_recent_ratio": 0.10},
]


def _get_cache_fn(cfg: dict):
    """Return a post-hoc cache truncator for StreamingLLM / H2O configs.
    Returns None for committee configs (their eviction is built into the model)."""
    mode = cfg.get("cache_mode")
    if mode == "streaming":
        n_sink     = cfg.get("n_sink", 4)
        keep_ratio = cfg.get("keep_ratio", 0.50)
        return lambda kv: _cache_streaming(kv, n_sink, keep_ratio)
    if mode == "h2o":
        keep_ratio   = cfg.get("keep_ratio", 0.50)
        recent_ratio = cfg.get("recent_ratio", 0.10)
        return lambda kv: _cache_h2o(kv, keep_ratio, recent_ratio)
    return None


def _cache_streaming(past_kv, n_sink: int, keep_ratio: float):
    """
    StreamingLLM-style post-hoc cache truncation.
    Keeps the first n_sink attention-sink tokens and the most recent
    (keep_ratio * S - n_sink) tokens; discards everything in between.
    Operates on DynamicCache key_cache / value_cache lists in-place.
    MLA layout: key_cache[i] = [B, 1, S, kv_lora_rank],
                value_cache[i] = [B, 1, S, qk_rope_head_dim].
    """
    if not hasattr(past_kv, "key_cache") or not past_kv.key_cache:
        return past_kv
    for i in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[i]    # [B, 1, S, D]
        v = past_kv.value_cache[i]
        S = k.shape[2]
        n_keep = max(n_sink + 1, int(S * keep_ratio))
        if n_keep >= S:
            continue
        window = max(1, n_keep - n_sink)
        past_kv.key_cache[i]   = torch.cat([k[:, :, :n_sink], k[:, :, -window:]], dim=2)
        past_kv.value_cache[i] = torch.cat([v[:, :, :n_sink], v[:, :, -window:]], dim=2)
    return past_kv


def _cache_h2o(past_kv, keep_ratio: float, recent_ratio: float = 0.10):
    """
    H2O-style post-hoc cache truncation.
    Approximates 'heavy-hitter' importance using the L2 norm of each token's
    compressed latent key vector (MLA stores c_kv_normed in key_cache, shape
    [B, 1, S, kv_lora_rank=512]).  Keeps the top (keep_ratio - recent_ratio)
    fraction by norm, plus the most recent (recent_ratio) fraction.
    Operates on DynamicCache key_cache / value_cache lists in-place.
    """
    if not hasattr(past_kv, "key_cache") or not past_kv.key_cache:
        return past_kv
    for i in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[i]    # [B, 1, S, D]
        v = past_kv.value_cache[i]
        S = k.shape[2]
        n_keep   = max(2, int(S * keep_ratio))
        n_recent = max(1, int(S * recent_ratio))
        n_recent = min(n_recent, n_keep - 1)
        if n_keep >= S:
            continue
        # Importance: mean L2 norm over heads dimension → [B, S]
        importance = k.float().norm(dim=-1).mean(dim=1)
        importance = importance[0] if importance.shape[0] > 1 else importance.squeeze(0)
        # Protect recent tokens so they are always kept
        importance[-n_recent:] = float("inf")
        _, keep_idx = importance.topk(n_keep)
        keep_idx = keep_idx.sort().values
        past_kv.key_cache[i]   = k[:, :, keep_idx, :]
        past_kv.value_cache[i] = v[:, :, keep_idx, :]
    return past_kv


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepSeek Latent Eviction -- Comprehensive Evaluation")
    p.add_argument("--quick",         action="store_true",
                   help="Reduce all subset sizes for a fast smoke test (~15 min)")
    p.add_argument("--skip-optional", action="store_true",
                   help="Skip HumanEval and LongBench (no extra packages needed)")
    p.add_argument("--niah-max",      type=int, default=16384,
                   help="Max NIAH context length (default 16384; 32768 for extended test)")
    p.add_argument("--save",          default="eval_results.json",
                   help="Output JSON path (default: eval_results.json)")
    p.add_argument("--ablation",      action="store_true",
                   help="Run ablation study configs (ABLATION_CONFIGS) instead of "
                        "main CONFIGS. Isolates super-token, knee-detection, and "
                        "committee contributions individually.")
    p.add_argument("--only",          default="",
                   help="Comma-separated list of experiment numbers to run "
                        "(e.g. --only 5,6 runs only LongBench and System metrics). "
                        "Valid values: 1=PPL, 2=NIAH, 3=CodeRecall, "
                        "4=HumanEval, 5=LongBench, 6=System. "
                        "Default: run all experiments.")
    p.add_argument("--comparison",    action="store_true",
                   help="Head-to-head comparison vs StreamingLLM and H2O baselines "
                        "(runs PPL + System metrics only by default; use --only to "
                        "override). Uses COMPARISON_CONFIGS and saves a combined PDF.")
    return p.parse_args()


def _scale(args: argparse.Namespace) -> dict:
    """Return experiment scale dict, reduced for --quick mode."""
    d = dict(_DEFAULTS)
    if args.quick:
        d.update(dict(
            n_ppl_seqs   = 5,
            n_niah_seeds = 2,
            niah_lengths = [2048, 4096],
            n_humaneval  = 5,
            n_longbench  = 3,
            sys_lengths  = [1024, 2048, 4096],
        ))
    d["niah_lengths"] = [l for l in d["niah_lengths"] if l <= args.niah_max]
    return d

# ── Model helpers ─────────────────────────────────────────────────────────────
def load_model() -> Tuple[Any, Any]:
    print("[ Loading tokenizer ... ]")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print("[ Loading model (device_map=auto, bfloat16) ... ]")
    # NOTE: 4-bit quantization is INCOMPATIBLE with the latent eviction logic
    # because modeling_deepseek_test.py calls .view() on kv_b_proj.weight, which
    # requires a dense tensor.  Packed NF4/INT8 weights fail with shape errors.
    # Instead we load in bfloat16 with device_map="auto" — the MoE architecture
    # only activates ~2.4B params per token, so GPU memory is manageable.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="eager",
        )
        print("[ Attention: eager ]")
    except (ValueError, NotImplementedError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="sdpa",
        )
        print("[ Attention: sdpa (eager not supported) ]")
    model.eval()
    return tok, model


def configure_model(model, cfg: dict) -> int:
    """
    Apply a config entry to all DeepseekV2Attention layers.
    Supports both legacy (single-layer) and new (committee + soft-eviction) API.
    Returns the number of layers configured.
    """
    n = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekV2Attention":
            continue
        module.latent_eviction                = cfg["eviction"]
        module.latent_eviction_decision_layer = cfg["decision_layer"]
        module.latent_eviction_window         = EVICTION_WINDOW
        # Knee-detection budget  (per-config min/max_keep override for ablations)
        module.latent_eviction_coverage_floor = cfg.get("cov_floor", COVERAGE_FLOOR)
        module.latent_eviction_min_keep       = cfg.get("min_keep",  MIN_KEEP)
        module.latent_eviction_max_keep       = cfg.get("max_keep",  MAX_KEEP)
        # Multi-layer committee scoring
        module.latent_eviction_score_layers   = cfg.get("score_layers", SCORE_LAYERS)
        # Soft eviction: super tokens (pool_ratio=0 disables)
        module.latent_eviction_pool_ratio       = cfg.get("pool_ratio", POOL_RATIO)
        module.latent_eviction_pool_temperature = cfg.get("pool_temperature", POOL_TEMPERATURE)
        # Eviction mode: committee (default) | streaming | h2o
        module.latent_eviction_mode             = cfg.get("eviction_mode", "committee")
        module.latent_eviction_keep_ratio       = cfg.get("keep_ratio", 0.50)
        module.latent_eviction_h2o_recent_ratio = cfg.get("h2o_recent_ratio", 0.10)
        n += 1
    return n


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _cache_len(past_kv) -> int:
    """Actual number of cached tokens after eviction."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "key_cache") and past_kv.key_cache:
        return past_kv.key_cache[0].shape[2]
    if isinstance(past_kv, (list, tuple)) and past_kv:
        return past_kv[0][0].shape[2]
    try:
        return past_kv.get_seq_length(0)
    except Exception:
        return 0


# ── Experiment 1: Suffix PPL ──────────────────────────────────────────────────
_BUILTIN_CORPUS = ("""\
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot  = arr[len(arr) // 2]
    left   = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right  = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class BinarySearchTree:
    class _Node:
        def __init__(self, key):
            self.key, self.left, self.right = key, None, None
    def __init__(self): self.root = None
    def insert(self, key): self.root = self._ins(self.root, key)
    def _ins(self, node, key):
        if node is None: return self._Node(key)
        if key < node.key: node.left  = self._ins(node.left,  key)
        elif key > node.key: node.right = self._ins(node.right, key)
        return node
    def inorder(self):
        out = []; self._io(self.root, out); return out
    def _io(self, n, out):
        if n: self._io(n.left, out); out.append(n.key); self._io(n.right, out)

def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    l, r = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    res, i, j = [], 0, 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]: res.append(l[i]); i += 1
        else:            res.append(r[j]); j += 1
    return res + l[i:] + r[j:]

def dijkstra(graph, start):
    import heapq
    dist = {n: float('inf') for n in graph}; dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph.get(u, {}).items():
            nd = dist[u] + w
            if nd < dist[v]: dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist

""" * 60)


def _load_ppl_corpus(tokenizer, max_tokens: int = 60000) -> torch.Tensor:
    """
    Load a code-focused corpus for PPL evaluation.

    DeepSeek-Coder is pretrained on code-heavy data, so natural-language
    corpora (WikiText-2, C4) are poor proxies for code-logic PPL degradation.
    Priority order (code-first):

      1. code-search-net/python  -- real Python functions, same source as
                                    configure_training.py; diverse, no padding
      2. HuggingFaceFW/fineweb-edu (code-filtered)  -- educational web text
                                    with at least 10% code-like lines
      3. Built-in _BUILTIN_CORPUS  -- hardcoded algorithms; always available,
                                    guaranteed deterministic fallback

    WikiText-2 and C4 are intentionally excluded: high natural-language PPL
    reflects language-model quality in general, not the code-path that is
    actually stressed by the latent eviction algorithm.
    """
    # ── Priority 1: code-search-net Python ────────────────────────────────
    try:
        from datasets import load_dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset("code-search-net/code_search_net", "python",
                              split="test", trust_remote_code=False, streaming=True)
        texts = []
        for ex in ds:
            fn = ex.get("whole_func_string", "").strip()
            if fn:
                texts.append(fn)
            if len("\n\n".join(texts)) > max_tokens * 5:
                break
        if texts:
            text = "\n\n".join(texts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ids = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=max_tokens).input_ids[0]
            print(f"  Corpus: code-search-net/python ({ids.shape[0]:,} tokens)")
            return ids
    except Exception:
        pass

    # ── Priority 2: FineWeb-Edu code-filtered ─────────────────────────────
    try:
        from datasets import load_dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", trust_remote_code=False, streaming=True)
        texts = []
        for ex in ds:
            t = ex.get("text", "").strip()
            lines      = t.splitlines()
            code_lines = sum(
                1 for ln in lines
                if any(kw in ln for kw in ("def ", "class ", "import ", "    ", "{"))
            )
            if len(lines) > 5 and code_lines / len(lines) >= 0.10:
                texts.append(t)
            if len("\n\n".join(texts)) > max_tokens * 5:
                break
        if texts:
            text = "\n\n".join(texts)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ids = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=max_tokens).input_ids[0]
            print(f"  Corpus: fineweb-edu code-filtered ({ids.shape[0]:,} tokens)")
            return ids
    except Exception:
        pass

    # ── Priority 3 (always succeeds): built-in Python corpus ─────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = tokenizer(_BUILTIN_CORPUS, return_tensors="pt",
                        truncation=True, max_length=max_tokens).input_ids[0]
    print(f"  Corpus: built-in Python ({ids.shape[0]:,} tokens)"
          "  [pip install datasets for a richer external corpus]")
    return ids


@torch.no_grad()
def _suffix_ppl_one(model, prefix_ids: torch.Tensor,
                    suffix_ids: torch.Tensor,
                    cache_fn=None) -> Optional[float]:
    """
    Prefill prefix (triggers eviction if enabled), then compute NLL on
    suffix using the post-eviction KV cache.

    Implementation notes
    --------------------
    * An explicit DynamicCache() is created and passed to the prefill call.
      This is required for transformers >= 4.47 where the model no longer
      auto-creates a cache internally when use_cache=True without an existing
      past_key_values argument.  It also guarantees that
      `past_key_value.get_seq_length() == 0` at the start, which is the
      condition that triggers latent eviction (_is_first_prefill).
    * The suffix scoring call uses use_cache=True (not False). Passing
      past_key_values with use_cache=False has undefined behaviour in
      some transformers versions and may cause shape mismatches.
    """
    dev       = _device(model)
    prefix_4d = prefix_ids.unsqueeze(0).to(dev)
    suffix_4d = suffix_ids.unsqueeze(0).to(dev)

    # ── Step 1: prefill  (eviction triggers here if enabled) ─────────────
    if _HAS_DYNAMIC_CACHE:
        # transformers >= 4.36: explicit empty cache ensures correct
        # _is_first_prefill detection and compatibility with 4.47+ API.
        init_cache = _DynamicCache()
        out_pre    = model(prefix_4d, past_key_values=init_cache, use_cache=True)
    else:
        out_pre = model(prefix_4d, use_cache=True)

    past_kv = out_pre.past_key_values
    del out_pre   # free logits / hidden states immediately

    # Apply optional post-hoc cache truncator (StreamingLLM / H2O baselines)
    if cache_fn is not None:
        past_kv = cache_fn(past_kv)

    # ── Step 2: score suffix given post-eviction cache ────────────────────
    # use_cache=True is intentional: avoids undefined behaviour when
    # past_key_values is provided alongside use_cache=False.
    out_suf = model(suffix_4d, past_key_values=past_kv,
                    labels=suffix_4d, use_cache=True)
    if out_suf.loss is None:
        return None
    return math.exp(min(out_suf.loss.item(), 20.0))


def run_ppl(model, tokenizer, args, scale: dict) -> Dict:
    print("\n" + "─" * 68)
    print("  [1/6] Suffix PPL  (prefix prefill -> post-eviction NLL on suffix)")
    print("─" * 68)

    n_seqs   = scale["n_ppl_seqs"]
    pref_len = 256 if args.quick else 512
    suf_len  = 64  if args.quick else 128
    corpus   = _load_ppl_corpus(
        tokenizer, max_tokens=(pref_len + suf_len) * n_seqs + suf_len)
    total    = corpus.shape[0]

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        cache_fn = _get_cache_fn(cfg)
        ppls = []
        for i in range(n_seqs):
            s = i * (pref_len + suf_len)
            if s + pref_len + suf_len > total:
                break
            ppl = _suffix_ppl_one(model,
                                   corpus[s:s + pref_len],
                                   corpus[s + pref_len:s + pref_len + suf_len],
                                   cache_fn=cache_fn)
            if ppl is not None:
                ppls.append(ppl)
        avg = float(np.mean(ppls)) if ppls else float("nan")
        print(f"  {cfg['name']:12}  suffix_ppl = {avg:7.3f}  (n={len(ppls)})")
        results[cfg["name"]] = {"suffix_ppl": avg, "n": len(ppls),
                                "pref_len": pref_len, "suf_len": suf_len}
    return results


# ── Experiment 2: NIAH ────────────────────────────────────────────────────────
# Filler code units for the haystack
_FILLER_UNITS = [
    "def fn_{i}(x): return x * {i} + {i} ** 2\n",
    "CONST_{i} = {i} * 17 + 3  # auto-gen constant\n",
    "class C_{i}:\n    val = {i}\n    def get(self): return self.val\n",
    "BUF_{i} = [0] * {i}  # placeholder\n",
    ("def proc_{i}(data):\n"
     "    return [v + {i} for v in data]\n"),
]


def _gen_filler(n_units: int) -> str:
    parts = []
    for i in range(n_units):
        unit = _FILLER_UNITS[i % len(_FILLER_UNITS)].format(i=i)
        parts.append(unit)
    return "".join(parts)


_FILLER_TOKENIZED: Optional[torch.Tensor] = None


def _get_filler_ids(tokenizer, max_tokens: int) -> torch.Tensor:
    global _FILLER_TOKENIZED
    if _FILLER_TOKENIZED is None or _FILLER_TOKENIZED.shape[0] < max_tokens:
        filler = _gen_filler(max_tokens // 2 + 500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _FILLER_TOKENIZED = tokenizer(
                filler, return_tensors="pt",
                truncation=True, max_length=max_tokens + 200,
            ).input_ids[0]
    return _FILLER_TOKENIZED[:max_tokens]


def _greedy_gen(model, tokenizer, input_ids: torch.Tensor,
                max_new_tokens: int) -> torch.Tensor:
    """
    Standardised greedy-decode wrapper.
    - Passes explicit attention_mask  (suppresses 'mask not set' warning)
    - Overrides temperature/top_p=1.0 (suppresses 'do_sample=False' warnings)
    - Sets pad_token_id               (suppresses 'pad_token_id not set' warning)
    - Filters deprecated-API warnings from the model cache internals
    """
    attn_mask   = torch.ones_like(input_ids)
    pad_tok_id  = tokenizer.eos_token_id or 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*seen_tokens.*")
        warnings.filterwarnings("ignore", message=".*get_max_cache.*")
        warnings.filterwarnings("ignore", message=".*do_sample.*")
        warnings.filterwarnings("ignore", message=".*attention mask.*")
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        return model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=pad_tok_id,
            use_cache=True,
        )


def _build_niah_prompt(tokenizer, target_len: int, needle_val: int,
                        depth: float, device: torch.device
                        ) -> Tuple[torch.Tensor, str]:
    """
    Build a token sequence of ~target_len with a secret integer constant
    buried at relative position `depth`.
    Returns (ids [1,N], str(needle_val)).
    """
    needle_str = f"\nSECRET_CONSTANT = {needle_val}\n"
    question_str = (
        "\n\nQuestion: What is the value of SECRET_CONSTANT?\n"
        "Answer: "
    )

    needle_ids = tokenizer(needle_str, return_tensors="pt").input_ids[0]
    question_ids = tokenizer(question_str, return_tensors="pt").input_ids[0]

    # Keep final token count near target_len to avoid accidental overflow effects.
    # Reserve space for needle + question, and fill remaining budget with code filler.
    reserved = needle_ids.shape[0] + question_ids.shape[0]
    filler_budget = max(8, target_len - reserved)
    filler = _get_filler_ids(tokenizer, filler_budget + 64)

    pre_len = max(1, int(filler_budget * depth))
    pre_len = min(pre_len, filler_budget - 1)
    post_len = max(1, filler_budget - pre_len)

    pre_ids = filler[:pre_len]
    post_ids = filler[pre_len: pre_len + post_len]
    ids = torch.cat([pre_ids, needle_ids, post_ids, question_ids]).unsqueeze(0)
    return ids.to(device), str(needle_val)


@torch.no_grad()
def _niah_one(model, tokenizer, ctx_ids: torch.Tensor,
              expected: str, max_new: int = 20) -> bool:
    gen = _greedy_gen(model, tokenizer, ctx_ids, max_new)
    answer = tokenizer.decode(
        gen[0, ctx_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    expected_norm = re.sub(r"[^0-9]", "", expected)
    answer_norm   = re.sub(r"[^0-9]", "", answer)
    return expected_norm in answer_norm


def run_niah(model, tokenizer, args, scale: dict) -> Dict:
    print("\n" + "─" * 68)
    print("  [2/6] NIAH  (needle in Python code haystack)")
    print("─" * 68)

    device  = _device(model)
    lengths = scale["niah_lengths"]
    depths  = scale["niah_depths"]
    needles = [73847, 51293, 88421, 62017, 39584][:scale["n_niah_seeds"]]

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        cfg_res: Dict[str, Any] = {}
        for ctx_len in lengths:
            correct = total = 0
            for depth in depths:
                for nv in needles:
                    try:
                        ctx, exp = _build_niah_prompt(
                            tokenizer, ctx_len, nv, depth, device)
                        hit = _niah_one(model, tokenizer, ctx, exp, max_new=48)
                        correct  += int(hit)
                    except Exception:
                        pass
                    total += 1
            recall = correct / max(total, 1)
            cfg_res[str(ctx_len)] = {"recall": recall, "correct": correct,
                                      "total": total}
            print(f"  {cfg['name']:12}  ctx={ctx_len:5d}  "
                  f"recall={recall:.0%}  ({correct}/{total})")
        results[cfg["name"]] = cfg_res
    return results


# ── Experiment 3: Code cross-reference recall ─────────────────────────────────
_LONG_FILLER = (
    "\n# --- auto-generated data processing utilities ---\n"
    "def normalize(x, lo, hi): return (x - lo) / (hi - lo + 1e-8)\n"
    "def clip(x, lo, hi): return max(lo, min(hi, x))\n"
    "def softmax(xs):\n"
    "    import math\n"
    "    ex = [math.exp(v) for v in xs]; s = sum(ex)\n"
    "    return [v/s for v in ex]\n"
    "\nclass RingBuffer:\n"
    "    def __init__(self, cap): self._d=[None]*cap; self._h=self._t=self._n=0; self._c=cap\n"
    "    def push(self, v):\n"
    "        self._d[self._t]=v; self._t=(self._t+1)%self._c\n"
    "        if self._n<self._c: self._n+=1\n"
    "        else: self._h=(self._h+1)%self._c\n"
    "    def pop(self):\n"
    "        if not self._n: raise IndexError\n"
    "        v=self._d[self._h]; self._h=(self._h+1)%self._c; self._n-=1; return v\n"
    "\nWEIGHTS=[0.1,0.2,0.3,0.15,0.25]; BIASES=[0.01,0.02,-0.01,0.005,0.0]\n"
    "LAYERS=[64,128,256,128,64]\n"
) * 4   # repeat to make the gap wider


# (prompt_template, expected_token, description)
_CODE_RECALL_TESTS: List[Tuple[str, str, str]] = [
    # 1. class constant
    (
        "class Config:\n"
        "    DATABASE_POOL_SIZE  = 42\n"
        "    MAX_RETRY_ATTEMPTS  = 7\n"
        "    REQUEST_TIMEOUT_SEC = 30.0\n"
        "    CACHE_TTL_SECONDS   = 3600\n"
        "    MAX_CONNECTIONS     = 128\n"
        "{FILLER}\n"
        "def create_connection_pool():\n"
        "    pool_size = Config.",
        "DATABASE_POOL_SIZE",
        "class constant",
    ),
    # 2. module-level secret
    (
        "ENCRYPTION_KEY = b\"s3cr3t_k3y_2024\"\n"
        "HASH_ALGORITHM  = \"sha256\"\n"
        "SALT_ROUNDS     = 12\n"
        "{FILLER}\n"
        "def encrypt_payload(data: bytes) -> bytes:\n"
        "    import hmac\n"
        "    return hmac.new(ENCRYPTION_KEY, data, HASH_ALGORITHM",
        "ENCRYPTION_KEY",
        "module secret",
    ),
    # 3. function recall
    (
        "def validate_email(addr: str) -> bool:\n"
        "    import re\n"
        "    return bool(re.match(r'^[\\w.+-]+@[\\w-]+\\.[\\w.]+$', addr))\n"
        "{FILLER}\n"
        "def register_user(username: str, email: str):\n"
        "    if not validate_email",
        "validate_email",
        "function recall",
    ),
    # 4. dataclass field
    (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class TrainingConfig:\n"
        "    learning_rate: float = 3e-4\n"
        "    batch_size:    int   = 32\n"
        "    num_epochs:    int   = 100\n"
        "    warmup_steps:  int   = 500\n"
        "    weight_decay:  float = 0.01\n"
        "{FILLER}\n"
        "def build_optimizer(model, cfg: TrainingConfig):\n"
        "    import torch.optim as optim\n"
        "    return optim.AdamW(model.parameters(), lr=cfg.learning_rate",
        "learning_rate",
        "dataclass field",
    ),
    # 5. exception class
    (
        "class DatabaseError(Exception): pass\n"
        "class ConnectionTimeoutError(DatabaseError):\n"
        "    def __init__(self, host, sec):\n"
        "        super().__init__(f'Timeout {host} after {sec}s')\n"
        "{FILLER}\n"
        "def connect(host, timeout=30.0):\n"
        "    import socket\n"
        "    try:\n"
        "        return socket.create_connection((host, 5432), timeout)\n"
        "    except socket.timeout:\n"
        "        raise ConnectionTimeout",
        "ConnectionTimeoutError",
        "exception class",
    ),
    # 6. protocol constant
    (
        "MAGIC_HEADER     = b\"\\xDE\\xAD\\xBE\\xEF\"\n"
        "PROTOCOL_VERSION = 3\n"
        "MAX_PACKET_SIZE  = 65535\n"
        "{FILLER}\n"
        "def encode_packet(payload: bytes) -> bytes:\n"
        "    length = len(payload).to_bytes(4, 'big')\n"
        "    return MAGIC_HEADER + bytes([PROTOCOL_VERSION]) + length",
        "MAGIC_HEADER",
        "protocol constant",
    ),
]


@torch.no_grad()
def _code_recall_one(model, tokenizer, prompt: str,
                      expected: str, max_new: int = 40) -> bool:
    dev  = _device(model)
    ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    gen  = _greedy_gen(model, tokenizer, ids, max_new)
    out  = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    return expected in out


def run_code_recall(model, tokenizer, args, scale: dict) -> Dict:
    print("\n" + "─" * 68)
    print("  [3/6] Code cross-reference recall")
    print("─" * 68)

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        correct  = 0
        details  = []
        for tmpl, expected, desc in _CODE_RECALL_TESTS:
            prompt = tmpl.replace("{FILLER}", _LONG_FILLER)
            try:
                hit = _code_recall_one(model, tokenizer, prompt, expected)
            except Exception:
                hit = False
            correct += int(hit)
            details.append({"desc": desc, "expected": expected, "hit": hit})

        n     = len(_CODE_RECALL_TESTS)
        score = correct / n
        print(f"  {cfg['name']:12}  {correct}/{n} = {score:.0%}")
        for d in details:
            mark = "OK" if d["hit"] else "XX"
            print(f"    [{mark}] {d['desc']:<30}  (expect '{d['expected']}')")
        results[cfg["name"]] = {"score": score, "n": n, "details": details}

    return results


# ── Experiment 4: HumanEval (optional) ───────────────────────────────────────
def run_humaneval(model, tokenizer, args, scale: dict) -> Optional[Dict]:
    print("\n" + "─" * 68)
    print("  [4/6] HumanEval syntax-pass@1  (optional)")
    print("─" * 68)

    # Try to load problems
    problems = None
    for loader_name, loader in [
        ("evalplus",         lambda: __import__("evalplus.data", fromlist=["get_human_eval_plus"]).get_human_eval_plus()),
        ("human_eval",       lambda: __import__("human_eval.data", fromlist=["read_problems"]).read_problems()),
        ("hf_humaneval",     lambda: {str(ex["task_id"]): ex
                                      for ex in __import__("datasets").load_dataset(
                                          "openai_humaneval", split="test",
                                          trust_remote_code=False)}),
    ]:
        try:
            raw = loader()
            problems = list(raw.items())
            print(f"  Source: {loader_name} ({len(problems)} problems)")
            break
        except Exception:
            pass

    if problems is None:
        print("  [SKIP] No HumanEval source found. "
              "Install via: pip install evalplus")
        return None

    n        = min(scale["n_humaneval"], len(problems))
    problems = problems[:n]

    def _get_prompt(task_id, prob) -> str:
        if isinstance(prob, dict):
            return prob.get("prompt", prob.get("text", ""))
        return str(prob)

    def _check_syntax(code: str) -> bool:
        import ast
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        passed = 0
        dev    = _device(model)

        for task_id, prob in problems:
            prompt = _get_prompt(task_id, prob)
            try:
                ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
                gen  = _greedy_gen(model, tokenizer, ids, 256)
                comp = tokenizer.decode(gen[0, ids.shape[1]:],
                                         skip_special_tokens=True)
                if _check_syntax(prompt + comp):
                    passed += 1
            except Exception:
                pass

        score = passed / n
        print(f"  {cfg['name']:12}  syntax-pass@1 = {score:.0%}  ({passed}/{n})")
        results[cfg["name"]] = {"syntax_pass_at_1": score, "n": n}

    return results


# ── Experiment 5: LongBench (optional) ───────────────────────────────────────
def run_longbench(model, tokenizer, args, scale: dict) -> Optional[Dict]:
    print("\n" + "─" * 68)
    print("  [5/6] LongBench multi-doc QA  (optional)")
    print("─" * 68)

    samples = None
    _LB_CONFIGS = ["multifieldqa_en", "hotpotqa", "2wikimqa", "qasper"]

    # Strategy 1: try standard parquet loading (no loading script needed)
    for cfg_name in _LB_CONFIGS:
        if samples is not None:
            break
        for load_kwargs in [
            # Try name= param without trust_remote_code
            {"path": "THUDM/LongBench", "name": cfg_name,
             "split": "test", "trust_remote_code": False},
            # Try direct parquet via HF hub URL
            {"path": "parquet",
             "data_files": f"hf://datasets/THUDM/LongBench/data/{cfg_name}-00000-of-00001.parquet",
             "split": "train"},
            {"path": "parquet",
             "data_files": f"hf://datasets/THUDM/LongBench/data/{cfg_name}_e-00000-of-00001.parquet",
             "split": "train"},
        ]:
            try:
                from datasets import load_dataset
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ds = load_dataset(**load_kwargs)
                candidates = list(ds)[:scale["n_longbench"]]
                if candidates:
                    samples = candidates
                    print(f"  Loaded LongBench {cfg_name} ({len(samples)} samples)")
                    break
            except Exception:
                continue

    if samples is None:
        print("  [SKIP] LongBench unavailable (all loading strategies failed).")
        print("         The THUDM/LongBench dataset no longer supports loading scripts.")
        print("         Check https://huggingface.co/datasets/THUDM/LongBench for ")
        print("         updated parquet file paths.")
        return None

    def _f1(pred: str, gold: str) -> float:
        pt = set(pred.lower().split())
        gt = set(gold.lower().split())
        if not pt or not gt:
            return 0.0
        inter = pt & gt
        p     = len(inter) / len(pt)
        r     = len(inter) / len(gt)
        return 2 * p * r / (p + r + 1e-8)

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        f1s = []
        dev = _device(model)

        for ex in samples:
            ctx      = ex.get("context", "")[:4000]
            question = ex.get("input", "")
            answers  = ex.get("answers", [""])
            gold     = answers[0] if isinstance(answers, list) else str(answers)
            prompt   = f"{ctx}\n\nQuestion: {question}\nAnswer:"
            try:
                ids  = tokenizer(prompt, return_tensors="pt",
                                  truncation=True, max_length=2048).input_ids.to(dev)
                gen  = _greedy_gen(model, tokenizer, ids, 64)
                pred = tokenizer.decode(gen[0, ids.shape[1]:],
                                         skip_special_tokens=True).strip()
                f1s.append(_f1(pred, gold))
            except Exception:
                f1s.append(0.0)

        avg = float(np.mean(f1s)) if f1s else 0.0
        print(f"  {cfg['name']:12}  avg F1 = {avg:.3f}  (n={len(f1s)})")
        results[cfg["name"]] = {"avg_f1": avg, "n": len(f1s)}

    return results


# ── Experiment 6: System metrics ──────────────────────────────────────────────
_SYS_CODE = ("""\
import os, sys, math, time, json, re
from collections import defaultdict, deque, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

def fibonacci(n):
    a, b = 0, 1
    for _ in range(n - 1): a, b = b, a + b
    return b if n > 0 else 0

class LRUCache:
    def __init__(self, cap):
        self.cap = cap; self.c = OrderedDict()
    def get(self, k):
        if k not in self.c: return -1
        self.c.move_to_end(k); return self.c[k]
    def put(self, k, v):
        if k in self.c: self.c.move_to_end(k)
        self.c[k] = v
        if len(self.c) > self.cap: self.c.popitem(last=False)

def tokenize(text, vocab):
    return [vocab.get(w, 0) for w in text.lower().split()]

CONST_A = 1.41421356; CONST_B = 2.71828182; CONST_C = 3.14159265

""" * 400)


def _sys_measure(model, tokenizer, n_tokens: int, cache_fn=None) -> Optional[Dict]:
    """Prefill n_tokens; return VRAM / TTFT / compression metrics."""
    dev = _device(model)
    if str(dev) == "cpu":
        print("    [SKIP] CPU-only: skipping VRAM metrics")
        return None

    # Tokenise with truncation to suppress length-overflow warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = tokenizer(
            _SYS_CODE, return_tensors="pt",
            truncation=True, max_length=n_tokens + 10,
        ).input_ids[:, :n_tokens].to(dev)
    actual = ids.shape[1]
    if actual < 128:
        return None

    attn_mask = torch.ones_like(ids)   # explicit mask prevents AssertionError

    # Prefill — pass explicit DynamicCache so eviction triggers (requires
    # past_key_value.get_seq_length() == 0 at first-prefill detection)
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        if _HAS_DYNAMIC_CACHE:
            out = model(ids, attention_mask=attn_mask,
                        past_key_values=_DynamicCache(), use_cache=True)
        else:
            out = model(ids, attention_mask=attn_mask, use_cache=True)
    torch.cuda.synchronize(dev)
    prefill_ms  = (time.perf_counter() - t0) * 1000
    peak_vram   = torch.cuda.max_memory_allocated(dev) / 1024**3

    # TTFT: decode one token from the post-eviction cache
    past_kv  = out.past_key_values
    # Apply optional post-hoc cache truncator (StreamingLLM / H2O baselines)
    if cache_fn is not None:
        past_kv = cache_fn(past_kv)
    kept     = _cache_len(past_kv)          # actual cache length after eviction

    # ── Actual KV-cache storage (after eviction) ──────────────────────────
    # This is the true memory occupied by the compressed latent KV tensors
    # once eviction has run.  Unlike peak_vram (which includes activations and
    # is dominated by the full-sequence prefill for all methods), kv_cache_gb
    # directly reflects how many tokens each algorithm chose to retain.
    kv_bytes = 0
    if hasattr(past_kv, "key_cache"):
        for _ck, _cv in zip(past_kv.key_cache, past_kv.value_cache):
            if _ck is not None:
                kv_bytes += _ck.numel() * _ck.element_size()
            if _cv is not None:
                kv_bytes += _cv.numel() * _cv.element_size()
    kv_cache_gb = kv_bytes / 1024 ** 3
    dummy    = ids[:, -1:]
    # mask size must be (batch, kept_cache + 1 new token), NOT original input length
    dummy_mask = torch.ones(ids.shape[0], kept + 1, device=dev, dtype=torch.long)
    torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    with torch.no_grad():
        model(dummy, past_key_values=past_kv,
              attention_mask=dummy_mask, use_cache=True)
    torch.cuda.synchronize(dev)
    ttft_ms = (time.perf_counter() - t1) * 1000

    compress = kept / actual if actual > 0 else 1.0

    del out, past_kv
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "input_len":      actual,
        "kept_len":       kept,
        "compress_ratio": compress,
        "prefill_ms":     prefill_ms,
        "ttft_ms":        ttft_ms,
        "peak_vram_gb":   peak_vram,
        "kv_cache_gb":    kv_cache_gb,
    }


def run_system(model, tokenizer, args, scale: dict) -> Dict:
    print("\n" + "─" * 68)
    print("  [6/6] System metrics  (KV Cache / TTFT / cache compression)")
    print("─" * 68)
    print(f"  {'Config':12}  {'ctx':>6}  {'compress':>8}  "
          f"{'prefill':>10}  {'ttft':>8}  {'kv_cache':>9}")
    print(f"  {'':12}  {'':>6}  {'':>8}  "
          f"{'(ms)':>10}  {'(ms)':>8}  {'(GB)':>9}")
    print("  " + "─" * 58)

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        cache_fn = _get_cache_fn(cfg)
        cfg_res: Dict[str, Any] = {}
        for n in scale["sys_lengths"]:
            # Clear GPU cache before each measurement to avoid OOM cascade
            gc.collect()
            torch.cuda.empty_cache()
            try:
                m = _sys_measure(model, tokenizer, n, cache_fn=cache_fn)
                if m:
                    cfg_res[str(n)] = m
                    print(
                        f"  {cfg['name']:12}  {n:6d}  "
                        f"{m['compress_ratio']:>8.1%}  "
                        f"{m['prefill_ms']:>10.0f}  "
                        f"{m['ttft_ms']:>8.1f}  "
                        f"{m['kv_cache_gb']:>9.3f}"
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"  {cfg['name']:12}  {n:6d}  OOM — OutOfMemoryError")
                gc.collect()
                torch.cuda.empty_cache()
                break   # larger lengths will also OOM for this config
            except RuntimeError:
                import traceback
                print(f"  {cfg['name']:12}  {n:6d}  RUNTIME ERROR → SEE BELOW:")
                traceback.print_exc()
                print(f"  --- end traceback ---")
                gc.collect()
                torch.cuda.empty_cache()
                break   # shape mismatch won't fix itself at longer lengths
            except Exception as e:
                print(f"  {cfg['name']:12}  {n:6d}  "
                      f"ERROR ({type(e).__name__}): {str(e)[:80]}")
                gc.collect()
                torch.cuda.empty_cache()
        results[cfg["name"]] = cfg_res

    return results


# ── System metrics chart ─────────────────────────────────────────────────────
def _plot_system_metrics(system_results: Dict, save_path: str = "system_metrics.pdf") -> None:
    """
    Single combined chart: all four metrics on one axes, normalized to baseline.
    X-axis: context lengths.  Each metric uses a distinct marker shape.
    Each config uses a distinct colour.  Y-axis: ratio vs baseline (1.0 = baseline).
    Requires matplotlib; gracefully skips if not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import matplotlib.lines as mlines
    except ImportError:
        print("  [SKIP chart] matplotlib not found.  "
              "Install with: pip install matplotlib")
        return

    # (data_key, legend_label, marker)
    METRIC_DEFS = [
        ("compress_ratio", "Compress Ratio",  "o"),   # circle
        ("prefill_ms",     "Prefill (ms)",    "^"),   # triangle
        ("kv_cache_gb",    "KV Cache (GB)",   "s"),   # square
        ("ttft_ms",        "TTFT (ms)",       "D"),   # diamond
    ]

    # Collect sorted context-length keys
    ctx_keys: set = set()
    for cfg_data in system_results.values():
        ctx_keys.update(cfg_data.keys())
    ctx_lengths = sorted(ctx_keys, key=lambda x: int(x) if str(x).isdigit() else 0)
    if not ctx_lengths:
        print("  [SKIP chart] No system data available.")
        return

    config_names = list(system_results.keys())
    baseline_data = system_results.get("baseline", {})
    x_pos    = list(range(len(ctx_lengths)))
    x_labels = [str(c) for c in ctx_lengths]
    # High-contrast palette: maximally distinct hues
    colors = [
        "#E63946",  # vivid red
        "#2196F3",  # bright blue
        "#4CAF50",  # green
        "#7B1FA2",  # purple
        "#9C27B0",  # purple
        "#00BCD4",  # cyan
        "#F06292",  # pink
        "#8BC34A",  # lime
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(
        "System Metrics — DeepSeek Latent Eviction  (normalized to baseline)",
        fontsize=13, fontweight="bold",
    )

    # Horizontal reference line at 1.0
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.text(len(ctx_lengths) - 0.52, 1.015, "baseline",
            color="#888888", fontsize=8, ha="right", va="bottom")

    for ci, cfg_name in enumerate(config_names):
        cfg_data  = system_results[cfg_name]
        linestyle = "--" if cfg_name == "baseline" else "-"
        for metric_key, _, marker in METRIC_DEFS:
            xs, ys = [], []
            for xi, ctx in enumerate(ctx_lengths):
                v = cfg_data.get(ctx, {}).get(metric_key)
                b = baseline_data.get(ctx, {}).get(metric_key)
                if v is not None and b is not None and float(b) != 0.0:
                    xs.append(xi)
                    ys.append(float(v) / float(b))
            if ys:
                ax.plot(xs, ys,
                        marker=marker, markersize=9, linewidth=1.8,
                        linestyle=linestyle,
                        color=colors[ci % len(colors)],
                        alpha=0.88)

    # ── Legend (two groups: configs by colour, metrics by marker) ─────────
    config_handles = [
        mlines.Line2D([], [], color=colors[ci % len(colors)], linewidth=2.2,
                      linestyle="--" if name == "baseline" else "-",
                      label=name)
        for ci, name in enumerate(config_names)
    ]
    metric_handles = [
        mlines.Line2D([], [], color="#333333", marker=mk, linestyle="None",
                      markersize=9, label=label)
        for _, label, mk in METRIC_DEFS
    ]
    leg_cfg = ax.legend(handles=config_handles, title="Config",
                        loc="lower left",  fontsize=8, title_fontsize=9,
                        framealpha=0.9)
    ax.add_artist(leg_cfg)
    ax.legend(handles=metric_handles,  title="Metric",
              loc="lower right", fontsize=8, title_fontsize=9,
              framealpha=0.9)

    ax.set_xlabel("Context Length (tokens)", fontsize=11)
    ax.set_ylabel("Ratio vs Baseline", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(True, linestyle="--", alpha=0.3)

    # Auto-fit y range: centre curves with padding, always include 1.0
    all_ys = []
    for ci, cfg_name in enumerate(config_names):
        cfg_data = system_results[cfg_name]
        for metric_key, _, _ in METRIC_DEFS:
            for ctx in ctx_lengths:
                v = cfg_data.get(ctx, {}).get(metric_key)
                b = baseline_data.get(ctx, {}).get(metric_key)
                if v is not None and b is not None and float(b) != 0.0:
                    all_ys.append(float(v) / float(b))
    if all_ys:
        data_min, data_max = min(all_ys), max(all_ys)
        span    = max(data_max - data_min, 0.1)
        pad     = span * 0.35
        y_lo    = max(0.0, data_min - pad)
        y_hi    = max(data_max + pad, 1.0 + pad * 0.5)
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  System metrics chart -> {save_path}")


# ── Comparison chart (Ours vs StreamingLLM vs H2O) ───────────────────────────
def _plot_comparison_chart(ppl_results: Dict, niah_results: Dict,
                            system_results: Dict,
                            save_path: str = "comparison.pdf") -> None:
    """
    1×3 comparison figure: Suffix-PPL | NIAH recall | Peak VRAM.

    All three panels compare baseline / committee_c90 / committee_c85 /
    streaming_llm / h2o at ~50% cache retention.

    panel [0]: bar chart — suffix PPL (lower = better)
    panel [1]: line chart — NIAH recall vs context length, averaged over depths
    panel [2]: line chart — peak VRAM (GB) vs context length
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        print("  [SKIP comparison chart] matplotlib not found. "
              "Install with: pip install matplotlib")
        return

    STYLES: Dict[str, tuple] = {
        "baseline":      ("#888888", "--", "Baseline"),
        "committee_c90": ("#2196F3", "-",  "Ours (c=0.90)"),
        "committee_c85": ("#00BCD4", "-",  "Ours (c=0.85)"),
        "streaming_llm": ("#FF9800", "-",  "StreamingLLM"),
        "h2o":           ("#E63946", "-",  "H2O"),
    }

    names = list(system_results.keys())

    # Sorted context lengths from system results
    ctx_keys: set = set()
    for d in system_results.values():
        ctx_keys.update(d.keys())
    ctx_lengths = sorted(ctx_keys, key=lambda x: int(x) if str(x).isdigit() else 0)
    x_pos    = list(range(len(ctx_lengths)))
    x_labels = [str(c) for c in ctx_lengths]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "KV Cache Eviction — Committee+SuperToken vs StreamingLLM vs H2O"
        "   ( ≈50% cache retention )",
        fontsize=12, fontweight="bold",
    )

    # ── [0] Suffix PPL bar chart ──────────────────────────────────────────
    ax0 = axes[0]
    ppl_names  = [n for n in names if n in ppl_results]
    ppl_vals   = [ppl_results[n].get("suffix_ppl", float("nan")) for n in ppl_names]
    bar_colors = [STYLES.get(n, ("#aaa", "-", n))[0] for n in ppl_names]
    bars = ax0.bar(range(len(ppl_names)), ppl_vals,
                   color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ax0.set_xticks(range(len(ppl_names)))
    ax0.set_xticklabels(
        [STYLES.get(n, (None, None, n))[2] for n in ppl_names],
        rotation=28, ha="right", fontsize=8,
    )
    ax0.set_ylabel("Suffix PPL", fontsize=9)
    ax0.set_title("Suffix Perplexity  (↓ better)", fontsize=9)
    ax0.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    finite_v = [v for v in ppl_vals if not math.isnan(v)]
    pad = (max(finite_v) - min(finite_v)) * 0.015 if len(finite_v) > 1 else 0.1
    for bar, val in zip(bars, ppl_vals):
        if not math.isnan(val):
            ax0.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + pad,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    # ── [1] NIAH recall line chart ────────────────────────────────────────
    ax1 = axes[1]
    niah_ctx = sorted(
        {ctx for nd in niah_results.values() for ctx in nd},
        key=lambda x: int(x) if str(x).isdigit() else 0
    )
    niah_x = list(range(len(niah_ctx)))
    for name in names:
        style = STYLES.get(name, ("#999", "-", name))
        nd = niah_results.get(name, {})
        xs, ys = [], []
        for xi, ctx in enumerate(niah_ctx):
            r = nd.get(ctx, {}).get("recall")
            if r is not None:
                xs.append(xi)
                ys.append(float(r))
        if ys:
            ax1.plot(xs, ys, color=style[0], linestyle=style[1],
                     marker="o", markersize=5, linewidth=1.8,
                     label=style[2], alpha=0.88)
    ax1.set_title("NIAH Recall  (↑ better)", fontsize=9)
    ax1.set_xticks(niah_x)
    ax1.set_xticklabels([str(c) for c in niah_ctx], rotation=30, fontsize=7)
    ax1.set_xlabel("Context Length (tokens)", fontsize=8)
    ax1.set_ylabel("Recall", fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.yaxis.set_major_formatter(
        __import__("matplotlib.ticker", fromlist=["FuncFormatter"])
        .FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.grid(True, linestyle="--", alpha=0.3)

    # ── [2] KV Cache Storage line chart ──────────────────────────────────
    ax2 = axes[2]
    for name in names:
        style = STYLES.get(name, ("#999", "-", name))
        xs, ys = [], []
        for xi, ctx in enumerate(ctx_lengths):
            v = system_results[name].get(ctx, {}).get("kv_cache_gb")
            if v is not None:
                xs.append(xi)
                ys.append(float(v))
        if ys:
            ax2.plot(xs, ys, color=style[0], linestyle=style[1],
                     marker="o", markersize=5, linewidth=1.8,
                     label=style[2], alpha=0.88)
    ax2.set_title("KV Cache Storage (GB)  (↓ better)", fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=30, fontsize=7)
    ax2.set_xlabel("Context Length (tokens)", fontsize=8)
    ax2.set_ylabel("KV Cache (GB)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # ── Shared legend ─────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color=STYLES[n][0], linestyle=STYLES[n][1],
                      linewidth=2.0, marker="o", markersize=5, label=STYLES[n][2])
        for n in STYLES if n in names
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8.5, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart -> {save_path}")


# ── Summary table ─────────────────────────────────────────────────────────────
def _fmt(v, spec=".3f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return format(float(v), spec)


def print_summary(all_results: Dict) -> None:
    names  = [c["name"] for c in CONFIGS]
    W      = 14
    total  = 30 + W * len(names)

    def row(label, vals):
        print(f"  {label:<28}" + "".join(f"  {str(v):>{W-2}}" for v in vals))

    print("\n\n" + "=" * total)
    print(f"  {'FINAL RESULTS':^{total - 4}}")
    print("=" * total)
    print(f"  {'Metric':<28}" + "".join(f"  {n:>{W-2}}" for n in names))
    print("-" * total)

    # 1. PPL
    if "ppl" in all_results:
        row("Suffix PPL",
            [_fmt(all_results["ppl"].get(n, {}).get("suffix_ppl")) for n in names])

    # 2. NIAH per context length (dynamic; supports custom grids such as 6k)
    if "niah" in all_results:
        ctx_keys = set()
        for n in names:
            ctx_keys.update(all_results["niah"].get(n, {}).keys())
        for ctx in sorted(ctx_keys, key=lambda x: int(x) if str(x).isdigit() else 10**9):
            vals = []
            for n in names:
                r   = all_results["niah"].get(n, {}).get(ctx, {})
                rec = r.get("recall")
                vals.append("N/A" if rec is None else f"{rec:.0%}")
            row(f"NIAH recall (ctx={ctx})", vals)

    # 3. Code recall
    if "code_recall" in all_results:
        row("Code x-ref recall",
            [_fmt(all_results["code_recall"].get(n, {}).get("score")) for n in names])

    # 4. HumanEval
    if all_results.get("humaneval"):
        row("HumanEval syntax-pass@1",
            [_fmt(all_results["humaneval"].get(n, {}).get("syntax_pass_at_1"))
             for n in names])

    # 5. LongBench
    if all_results.get("longbench"):
        row("LongBench F1",
            [_fmt(all_results["longbench"].get(n, {}).get("avg_f1")) for n in names])

    # 6. System @ 4096
    if "system" in all_results:
        for metric, label, spec in [
            ("compress_ratio", "Cache compression (4k)",  ".1%"),
            ("prefill_ms",     "Prefill latency (4k, ms)", ".0f"),
            ("ttft_ms",        "TTFT (4k, ms)",             ".1f"),
            ("kv_cache_gb",    "KV Cache (4k, GB)",         ".3f"),
        ]:
            vals = []
            for n in names:
                v = all_results["system"].get(n, {}).get("4096", {}).get(metric)
                vals.append("N/A" if v is None else format(float(v), spec))
            row(label, vals)

    print("=" * total)


# ── Main ──────────────────────────────────────────────────────────────────────

def _patch_config_auto_map(module_name: str) -> None:
    """
    Rewrite config.json auto_map to point at the given local module.
    AutoModelForCausalLM.from_pretrained(trust_remote_code=True) reads this
    file each call, so switching the module name here redirects which
    modeling_*.py is loaded next time the model is instantiated.
    """
    cfg_path = os.path.join(MODEL_DIR, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["auto_map"]["AutoModel"]          = f"{module_name}.DeepseekV2Model"
    cfg["auto_map"]["AutoModelForCausalLM"] = f"{module_name}.DeepseekV2ForCausalLM"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[ Model backend → {module_name} ]")


def _check_eviction_api(model) -> None:
    """Warn if the loaded model uses the old threshold-based eviction API."""
    for module in model.modules():
        if type(module).__name__ == "DeepseekV2Attention":
            has_new  = hasattr(module, "latent_eviction_coverage_floor")
            has_knee = hasattr(module, "_adaptive_budget")
            if has_new and has_knee:
                print("[OK] Knee-detection eviction API confirmed.")
            else:
                print("[WARN] Old threshold-based eviction API detected.")
                print("       Migrate modeling_deepseek_copy.py -> modeling_deepseek_test.py")
                print("       for knee-detection eviction before final evaluation.")
            break


def main() -> None:
    args  = parse_args()
    sc    = _scale(args)

    # ── Select config set: main evaluation vs ablation study ─────────────────
    global CONFIGS
    if args.comparison:
        CONFIGS = COMPARISON_CONFIGS
        mode_label = "Baseline Comparison (Ours vs StreamingLLM vs H2O)"
        if args.save == "eval_results.json":
            args.save = "eval_results_comparison.json"
    elif args.ablation:
        CONFIGS = ABLATION_CONFIGS
        mode_label = "Ablation Study"
        save_default = args.save.replace(".json", "_ablation.json")
        if args.save == "eval_results.json":   # user did not override --save
            args.save = save_default
    else:
        mode_label = "Main Evaluation"

    print("\n" + "=" * 68)
    print("  DeepSeek-Coder-V2-Lite  --  Latent Eviction Evaluation")
    print(f"  Mode       : {mode_label}")
    print(f"  Configs    : {[c['name'] for c in CONFIGS]}")
    print(f"  Quick mode : {args.quick}")
    print(f"  NIAH max   : {args.niah_max} tokens")
    if args.only:
        print(f"  Only exps  : {args.only}")
    print("=" * 68)

    # Parse --only into a set of ints; empty set = run all
    _only: set = set()
    if args.only.strip():
        for tok_str in args.only.split(","):
            tok_str = tok_str.strip()
            if tok_str.isdigit() and 1 <= int(tok_str) <= 6:
                _only.add(int(tok_str))
            else:
                print(f"  [WARN] Unknown experiment id '{tok_str}' in --only, skipping.")
    # Comparison mode defaults to PPL (1) + NIAH (2) + System (6) only
    if args.comparison and not _only:
        _only = {1, 2, 6}
        print("  [INFO] --comparison: auto-restricting to experiments 1 (PPL), "
              "2 (NIAH), and 6 (System). Use --only to override.")
    _run = lambda n: (not _only) or (n in _only)

    _needs_1_5 = any(_run(i) for i in range(1, 6))
    _needs_6   = _run(6)

    all_results: Dict[str, Any] = {}

    # ── Phase A: tests 1-5 — uses modeling_deepseek.py ──────────────────────
    if _needs_1_5:
        _patch_config_auto_map("modeling_deepseek")
        tok, model = load_model()
        _check_eviction_api(model)

        if _run(1):
            all_results["ppl"]         = run_ppl(model, tok, args, sc)
        if _run(2):
            all_results["niah"]        = run_niah(model, tok, args, sc)
        if _run(3):
            all_results["code_recall"] = run_code_recall(model, tok, args, sc)
        if not args.skip_optional:
            if _run(4):
                he = run_humaneval(model, tok, args, sc)
                if he:
                    all_results["humaneval"] = he
            if _run(5):
                lb = run_longbench(model, tok, args, sc)
                if lb:
                    all_results["longbench"] = lb

        if _needs_6:
            print("\n[ Unloading Phase-A model before Phase-B ... ]")
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # ── Phase B: test 6 — uses modeling_deepseek_test.py ─────────────────────
    if _needs_6:
        _patch_config_auto_map("modeling_deepseek_test")
        tok, model = load_model()
        _check_eviction_api(model)

        all_results["system"] = run_system(model, tok, args, sc)
        chart_path = args.save.replace(".json", "_system.pdf")
        _plot_system_metrics(all_results["system"], save_path=chart_path)

    # Generate combined comparison chart when running in --comparison mode
    if args.comparison and "ppl" in all_results and "system" in all_results:
        comp_chart = args.save.replace(".json", "_comparison.pdf")
        _plot_comparison_chart(
            all_results["ppl"],
            all_results.get("niah", {}),
            all_results["system"],
            save_path=comp_chart,
        )

    print_summary(all_results)

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {args.save}")
    print("Done.\n")


if __name__ == "__main__":
    main()
