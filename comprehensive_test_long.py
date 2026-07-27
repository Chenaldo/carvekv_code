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
  6. System         KV Cache size, prefill latency, TTFT, cache compression
                    at 1k/2k/4k/8k/16k/32k/64k contexts.

Usage
-----
    python final_run.py                     # full run
    python final_run.py --quick             # fast smoke test
    python final_run.py --skip-optional     # skip HumanEval + LongBench
    python final_run.py --niah-max 65536    # extend NIAH up to 64k tokens
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

# ── Ablation configurations ───────────────────────────────────────────────────
ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {"name": "baseline",
     "eviction": False,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0},
    {"name": "committee_c90",           # full system (reference)
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.90, "pool_ratio": 4},
    {"name": "hard_evict_c90",          # same as committee_hard_c90
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.90, "pool_ratio": 0},
    {"name": "fixed_ratio_70",
     "eviction": True,   "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.90,
     "min_keep": 0.70,   "max_keep": 0.70,   "pool_ratio": 4},
    {"name": "single_L6",
     "eviction": True,   "decision_layer": 6,
     "score_layers": [6],                "cov_floor": 0.90, "pool_ratio": 4},
    {"name": "single_L12",
     "eviction": True,   "decision_layer": 12,
     "score_layers": [12],               "cov_floor": 0.90, "pool_ratio": 4},
]

# Eviction hyper-parameters
COVERAGE_FLOOR   = 0.90
MIN_KEEP         = 0.30
MAX_KEEP         = 0.98
EVICTION_WINDOW  = 4
SCORE_LAYERS     = [4, 6, 8, 12, 16]
POOL_RATIO       = 4
POOL_TEMPERATURE = 0.1
DTYPE            = torch.bfloat16

# Default experiment scale (Updated for 16k, 32k, 64k)
_DEFAULTS = dict(
    n_ppl_seqs   = 20,
    n_niah_seeds = 3,
    niah_lengths = [2048, 4096, 8192, 16384, 32768, 65536],
    niah_depths  = [0.25, 0.50, 0.75],
    n_humaneval  = 20,
    n_longbench  = 10,
    sys_lengths  = [2048, 4096, 8192, 16384, 32768, 65536],
)

# ── Baseline comparison configurations ───────────────────────────────────────
COMPARISON_CONFIGS: List[Dict[str, Any]] = [
    {"name": "baseline",
     "eviction": False, "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0},
    {"name": "committee_c90",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.90, "pool_ratio": 4},
    {"name": "committee_c85",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [4, 6, 8, 12, 16], "cov_floor": 0.85, "pool_ratio": 4},
    {"name": "streaming_llm",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0,
     "eviction_mode": "streaming", "keep_ratio": 0.50},
    {"name": "h2o",
     "eviction": True,  "decision_layer": 16,
     "score_layers": [16], "cov_floor": 1.00, "pool_ratio": 0,
     "eviction_mode": "h2o", "keep_ratio": 0.50, "h2o_recent_ratio": 0.10},
]


def _get_cache_fn(cfg: dict):
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
    if not hasattr(past_kv, "key_cache") or not past_kv.key_cache:
        return past_kv
    for i in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[i]
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
    if not hasattr(past_kv, "key_cache") or not past_kv.key_cache:
        return past_kv
    for i in range(len(past_kv.key_cache)):
        k = past_kv.key_cache[i]
        v = past_kv.value_cache[i]
        S = k.shape[2]
        n_keep   = max(2, int(S * keep_ratio))
        n_recent = max(1, int(S * recent_ratio))
        n_recent = min(n_recent, n_keep - 1)
        if n_keep >= S:
            continue
        importance = k.float().norm(dim=-1).mean(dim=1)
        importance = importance[0] if importance.shape[0] > 1 else importance.squeeze(0)
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
    p.add_argument("--niah-max",      type=int, default=65536,
                   help="Max NIAH context length (default 65536)")
    p.add_argument("--save",          default="eval_results.json",
                   help="Output JSON path (default: eval_results.json)")
    p.add_argument("--ablation",      action="store_true",
                   help="Run ablation study configs")
    p.add_argument("--only",          default="",
                   help="Comma-separated list of experiment numbers to run")
    p.add_argument("--comparison",    action="store_true",
                   help="Head-to-head comparison vs StreamingLLM and H2O baselines")
    return p.parse_args()


def _scale(args: argparse.Namespace) -> dict:
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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print("[ Attention: sdpa (memory-efficient) ]")
    except (ValueError, NotImplementedError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="eager",
        )
        print("[ Attention: eager (sdpa not supported by this model) ]")
    model.eval()
    return tok, model


def configure_model(model, cfg: dict) -> int:
    n = 0
    for module in model.modules():
        if type(module).__name__ != "DeepseekV2Attention":
            continue
        module.latent_eviction                = cfg["eviction"]
        module.latent_eviction_decision_layer = cfg["decision_layer"]
        module.latent_eviction_window         = EVICTION_WINDOW
        module.latent_eviction_coverage_floor = cfg.get("cov_floor", COVERAGE_FLOOR)
        module.latent_eviction_min_keep       = cfg.get("min_keep",  MIN_KEEP)
        module.latent_eviction_max_keep       = cfg.get("max_keep",  MAX_KEEP)
        module.latent_eviction_score_layers   = cfg.get("score_layers", SCORE_LAYERS)
        module.latent_eviction_pool_ratio       = cfg.get("pool_ratio", POOL_RATIO)
        module.latent_eviction_pool_temperature = cfg.get("pool_temperature", POOL_TEMPERATURE)
        module.latent_eviction_mode             = cfg.get("eviction_mode", "committee")
        module.latent_eviction_keep_ratio       = cfg.get("keep_ratio", 0.50)
        module.latent_eviction_h2o_recent_ratio = cfg.get("h2o_recent_ratio", 0.10)
        n += 1
    return n


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _cache_len(past_kv) -> int:
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
# ── (Unchanged, included for script continuity) ─────────────────────────────
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
    dev       = _device(model)
    prefix_4d = prefix_ids.unsqueeze(0).to(dev)
    suffix_4d = suffix_ids.unsqueeze(0).to(dev)

    if _HAS_DYNAMIC_CACHE:
        init_cache = _DynamicCache()
        out_pre    = model(prefix_4d, past_key_values=init_cache, use_cache=True)
    else:
        out_pre = model(prefix_4d, use_cache=True)

    past_kv = out_pre.past_key_values
    del out_pre

    if cache_fn is not None:
        past_kv = cache_fn(past_kv)

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
    needle_str = f"\nSECRET_CONSTANT = {needle_val}\n"
    question_str = (
        "\n\nQuestion: What is the value of SECRET_CONSTANT?\n"
        "Answer: "
    )

    needle_ids = tokenizer(needle_str, return_tensors="pt").input_ids[0]
    question_ids = tokenizer(question_str, return_tensors="pt").input_ids[0]
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
) * 4

_CODE_RECALL_TESTS: List[Tuple[str, str, str]] = [
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

    for cfg_name in _LB_CONFIGS:
        if samples is not None:
            break
        for load_kwargs in [
            {"path": "THUDM/LongBench", "name": cfg_name,
             "split": "test", "trust_remote_code": False},
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
    dev = _device(model)
    if str(dev) == "cpu":
        print("    [SKIP] CPU-only: skipping memory metrics")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ids = tokenizer(
            _SYS_CODE, return_tensors="pt",
            truncation=True, max_length=n_tokens + 10,
        ).input_ids[:, :n_tokens].to(dev)
    actual = ids.shape[1]
    if actual < 128:
        return None

    attn_mask = torch.ones_like(ids)
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

    past_kv  = out.past_key_values
    if cache_fn is not None:
        past_kv = cache_fn(past_kv)
    kept     = _cache_len(past_kv)

    kv_bytes = 0
    if hasattr(past_kv, "key_cache"):
        for _ck, _cv in zip(past_kv.key_cache, past_kv.value_cache):
            if _ck is not None:
                kv_bytes += _ck.numel() * _ck.element_size()
            if _cv is not None:
                kv_bytes += _cv.numel() * _cv.element_size()
    kv_cache_gb = kv_bytes / 1024 ** 3
    dummy    = ids[:, -1:]
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
        "kv_cache_gb":    kv_cache_gb,
    }

def run_system(model, tokenizer, args, scale: dict) -> Dict:
    print("\n" + "─" * 68)
    print("  [6/6] System metrics  (KV Cache Size / TTFT / cache compression)")
    print("─" * 68)
    print(f"  {'Config':12}  {'ctx':>6}  {'compress':>8}  "
          f"{'prefill':>10}  {'ttft':>8}  {'kv_cache':>9}")
    print(f"  {'':12}  {'':>6}  {'':>8}  "
          f"{'(ms)':>10}  {'(ms)':>8}  {'(GB)':>9}")
    print("  " + "─" * 56)

    results: Dict[str, Any] = {}
    for cfg in CONFIGS:
        configure_model(model, cfg)
        cache_fn = _get_cache_fn(cfg)
        cfg_res: Dict[str, Any] = {}
        for n in scale["sys_lengths"]:
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
                        f"{m['kv_cache_gb']:>9.2f}"
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom:
                msg = str(oom)[:60].replace("\n", " ")
                print(f"  {cfg['name']:12}  {n:6d}  OOM — {msg}")
                gc.collect()
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"  {cfg['name']:12}  {n:6d}  "
                      f"ERROR ({type(e).__name__}): {str(e)[:80]}")
                gc.collect()
                torch.cuda.empty_cache()
        results[cfg["name"]] = cfg_res

    return results

# ── System metrics chart ─────────────────────────────────────────────────────
def _plot_system_metrics(system_results: Dict, save_path: str = "system_metrics.pdf") -> None:
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

    METRIC_DEFS = [
        ("compress_ratio", "Compress Ratio", "o"),
        ("prefill_ms",     "Prefill (ms)",   "^"),
        ("kv_cache_gb",    "KV Cache (GB)",  "s"),
        ("ttft_ms",        "TTFT (ms)",      "D"),
    ]

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
    colors = [
        "#E63946", "#2196F3", "#4CAF50", "#7B1FA2",
        "#9C27B0", "#00BCD4", "#F06292", "#8BC34A",
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(
        "System Metrics — DeepSeek Latent Eviction  (normalized to baseline)",
        fontsize=13, fontweight="bold",
    )

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

    if "ppl" in all_results:
        row("Suffix PPL",
            [_fmt(all_results["ppl"].get(n, {}).get("suffix_ppl")) for n in names])

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

    if "code_recall" in all_results:
        row("Code x-ref recall",
            [_fmt(all_results["code_recall"].get(n, {}).get("score")) for n in names])

    if all_results.get("humaneval"):
        row("HumanEval syntax-pass@1",
            [_fmt(all_results["humaneval"].get(n, {}).get("syntax_pass_at_1"))
             for n in names])

    if all_results.get("longbench"):
        row("LongBench F1",
            [_fmt(all_results["longbench"].get(n, {}).get("avg_f1")) for n in names])

    if "system" in all_results:
        for metric, label, spec in [
            ("compress_ratio", "Cache compression (4k)",  ".1%"),
            ("prefill_ms",     "Prefill latency (4k, ms)", ".0f"),
            ("ttft_ms",        "TTFT (4k, ms)",             ".1f"),
            ("kv_cache_gb",    "KV Cache (4k, GB)",         ".2f"),
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
        if args.save == "eval_results.json":
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

    _only: set = set()
    if args.only.strip():
        for tok_str in args.only.split(","):
            tok_str = tok_str.strip()
            if tok_str.isdigit() and 1 <= int(tok_str) <= 6:
                _only.add(int(tok_str))
            else:
                print(f"  [WARN] Unknown experiment id '{tok_str}' in --only, skipping.")
    
    if args.comparison and not _only:
        _only = {1, 2, 6}
        print("  [INFO] --comparison: auto-restricting to experiments 1 (PPL), "
              "2 (NIAH), and 6 (System). Use --only to override.")
    _run = lambda n: (not _only) or (n in _only)

    tok, model = load_model()

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

    if _run(6):
        all_results["system"] = run_system(model, tok, args, sc)
        chart_path = args.save.replace(".json", "_system.pdf")
        _plot_system_metrics(all_results["system"], save_path=chart_path)

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
