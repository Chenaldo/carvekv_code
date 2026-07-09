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
#   committee_c90     — full new algorithm, coverage_floor=0.90
#   committee_c85     — more aggressive, coverage_floor=0.85 (more VRAM savings)
#   committee_hard_c90 — committee scoring only, NO soft eviction (pool_ratio=0)
#                       isolates the benefit of super tokens
CONFIGS: List[Dict[str, Any]] = [
    {"name": "baseline",
     "eviction": False,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0},
    {"name": "committee_c90",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.90, "pool_ratio": 4},
    {"name": "committee_c85",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.85, "pool_ratio": 4},
    {"name": "committee_hard_c90",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.90, "pool_ratio": 0},
]

# Eviction hyper-parameters (defaults; per-config values override via cfg key)
COVERAGE_FLOOR   = 0.90
MIN_KEEP         = 0.30
MAX_KEEP         = 0.98
EVICTION_WINDOW  = 4
SCORE_LAYERS     = [4, 6, 8, 12, 16]   # committee layers
POOL_RATIO       = 4                    # n_evicted // pool_ratio = n_super
POOL_TEMPERATURE = 0.1                  # softmax temperature for content pooling
DTYPE            = torch.bfloat16

# Default experiment scale (overridden by --quick via args)
_DEFAULTS = dict(
    n_ppl_seqs   = 20,
    n_niah_seeds = 3,
    niah_lengths = [2048, 4096, 8192],
    niah_depths  = [0.25, 0.50, 0.75],
    n_humaneval  = 20,
    n_longbench  = 10,
    sys_lengths  = [1024, 2048, 4096, 8192],
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepSeek Latent Eviction -- Comprehensive Evaluation")
    p.add_argument("--quick",         action="store_true",
                   help="Reduce all subset sizes for a fast smoke test (~15 min)")
    p.add_argument("--skip-optional", action="store_true",
                   help="Skip HumanEval and LongBench (no extra packages needed)")
    p.add_argument("--niah-max",      type=int, default=8192,
                   help="Max NIAH context length (default 8192; 32768 for full test)")
    p.add_argument("--save",          default="eval_results.json",
                   help="Output JSON path (default: eval_results.json)")
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
    print("[ Loading model (device_map=auto) ... ]")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="eager",
    )
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
        # Knee-detection budget
        module.latent_eviction_coverage_floor = cfg.get("cov_floor", COVERAGE_FLOOR)
        module.latent_eviction_min_keep       = MIN_KEEP
        module.latent_eviction_max_keep       = MAX_KEEP
        # Multi-layer committee scoring
        module.latent_eviction_score_layers   = cfg.get("score_layers", SCORE_LAYERS)
        # Soft eviction: super tokens (pool_ratio=0 disables)
        module.latent_eviction_pool_ratio       = cfg.get("pool_ratio", POOL_RATIO)
        module.latent_eviction_pool_temperature = cfg.get("pool_temperature", POOL_TEMPERATURE)
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
                    suffix_ids: torch.Tensor) -> Optional[float]:
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
        ppls = []
        for i in range(n_seqs):
            s = i * (pref_len + suf_len)
            if s + pref_len + suf_len > total:
                break
            ppl = _suffix_ppl_one(model,
                                   corpus[s:s + pref_len],
                                   corpus[s + pref_len:s + pref_len + suf_len])
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
    buried at relative position `depth`.  Returns (ids [1,N], str(needle_val)).
    """
    filler   = _get_filler_ids(tokenizer, target_len + 200)
    pre_len  = max(1, int(target_len * depth))
    post_len = max(1, target_len - pre_len - 40)

    pre_ids  = filler[:pre_len]
    post_ids = filler[pre_len: pre_len + post_len]

    needle_str   = f"\nSECRET_CONSTANT = {needle_val}  # injected needle\n"
    question_str = ("\n# ----------------------------------------\n"
                    f"# Q: What is the value of SECRET_CONSTANT?\n"
                    f"# A: SECRET_CONSTANT = ")

    needle_ids   = tokenizer(needle_str,   return_tensors="pt").input_ids[0]
    question_ids = tokenizer(question_str, return_tensors="pt").input_ids[0]

    ids = torch.cat([pre_ids, needle_ids, post_ids, question_ids]).unsqueeze(0)
    return ids.to(device), str(needle_val)


@torch.no_grad()
def _niah_one(model, tokenizer, ctx_ids: torch.Tensor,
              expected: str, max_new: int = 20) -> bool:
    gen    = _greedy_gen(model, tokenizer, ctx_ids, max_new)
    answer = tokenizer.decode(gen[0, ctx_ids.shape[1]:],
                               skip_special_tokens=True).strip()
    return expected in answer


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
                        hit       = _niah_one(model, tokenizer, ctx, exp)
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

    try:
        from datasets import load_dataset
        ds      = load_dataset("THUDM/LongBench", "multifieldqa_en",
                               split="test", trust_remote_code=True)
        samples = list(ds)[:scale["n_longbench"]]
        print(f"  Loaded LongBench multifieldqa_en ({len(samples)} samples)")
    except Exception as e:
        print(f"  [SKIP] LongBench unavailable: {e}")
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


def _sys_measure(model, tokenizer, n_tokens: int) -> Optional[Dict]:
    """Prefill n_tokens; return VRAM / TTFT / compression metrics."""
    dev = _device(model)
    if str(dev) == "cpu":
        print("    [SKIP] CPU-only: skipping VRAM metrics")
        return None

    ids = tokenizer(_SYS_CODE, return_tensors="pt").input_ids[:, :n_tokens].to(dev)
    actual = ids.shape[1]
    if actual < 128:
        return None

    # Prefill
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    torch.cuda.synchronize(dev)
    prefill_ms  = (time.perf_counter() - t0) * 1000
    peak_vram   = torch.cuda.max_memory_allocated(dev) / 1024**3

    # TTFT: decode one token from the post-eviction cache
    past_kv  = out.past_key_values
    dummy    = ids[:, -1:]
    torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    with torch.no_grad():
        model(dummy, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize(dev)
    ttft_ms = (time.perf_counter() - t1) * 1000

    kept     = _cache_len(past_kv)
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
    }


def run_system(model, tokenizer, args, scale: dict) -> Dict:
    print("\n" + "─" * 68)
    print("  [6/6] System metrics  (VRAM / TTFT / cache compression)")
    print("─" * 68)
    print(f"  {'Config':12}  {'ctx':>6}  {'compress':>8}  "
          f"{'prefill':>10}  {'ttft':>8}  {'vram':>7}")
    print(f"  {'':12}  {'':>6}  {'':>8}  "
          f"{'(ms)':>10}  {'(ms)':>8}  {'(GB)':>7}")
    print("  " + "─" * 56)

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        cfg_res: Dict[str, Any] = {}
        for n in scale["sys_lengths"]:
            try:
                m = _sys_measure(model, tokenizer, n)
                if m:
                    cfg_res[str(n)] = m
                    print(
                        f"  {cfg['name']:12}  {n:6d}  "
                        f"{m['compress_ratio']:>8.1%}  "
                        f"{m['prefill_ms']:>10.0f}  "
                        f"{m['ttft_ms']:>8.1f}  "
                        f"{m['peak_vram_gb']:>7.2f}"
                    )
            except Exception as e:
                print(f"  {cfg['name']:12}  {n:6d}  ERROR: {e}")
        results[cfg["name"]] = cfg_res

    return results


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

    # 2. NIAH per context length
    if "niah" in all_results:
        for ctx in ["2048", "4096", "8192"]:
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
            ("peak_vram_gb",   "Peak VRAM (4k, GB)",        ".2f"),
        ]:
            vals = []
            for n in names:
                v = all_results["system"].get(n, {}).get("4096", {}).get(metric)
                vals.append("N/A" if v is None else format(float(v), spec))
            row(label, vals)

    print("=" * total)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args  = parse_args()
    sc    = _scale(args)

    print("\n" + "=" * 68)
    print("  DeepSeek-Coder-V2-Lite  --  Latent Eviction Evaluation")
    print(f"  Configs    : {[c['name'] for c in CONFIGS]}")
    print(f"  Quick mode : {args.quick}")
    print(f"  NIAH max   : {args.niah_max} tokens")
    print("=" * 68)

    tok, model = load_model()

    # Check eviction API version
    for module in model.modules():
        if type(module).__name__ == "DeepseekV2Attention":
            has_new = hasattr(module, "latent_eviction_coverage_floor")
            has_knee = hasattr(module, "_adaptive_budget")
            if has_new and has_knee:
                print("[OK] Knee-detection eviction API confirmed.")
            else:
                print("[WARN] Old threshold-based eviction API detected.")
                print("       Migrate modeling_deepseek_copy.py -> modeling_deepseek.py")
                print("       for knee-detection eviction before final evaluation.")
            break

    all_results: Dict[str, Any] = {}

    all_results["ppl"]         = run_ppl(model, tok, args, sc)
    all_results["niah"]        = run_niah(model, tok, args, sc)
    all_results["code_recall"] = run_code_recall(model, tok, args, sc)

    if not args.skip_optional:
        he = run_humaneval(model, tok, args, sc)
        if he:
            all_results["humaneval"] = he
        lb = run_longbench(model, tok, args, sc)
        if lb:
            all_results["longbench"] = lb

    all_results["system"] = run_system(model, tok, args, sc)

    print_summary(all_results)

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results -> {args.save}")
    print("Done.\n")


if __name__ == "__main__":
    main()
