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
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Workspace & local imports ─────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

# modeling_deepseek.py 使用相对导入，不能直接 import。
# 以下三个函数从 modeling_deepseek.py 逐字复制，均为纯 torch 实现，无外部依赖。

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _robust_normalize(t: torch.Tensor) -> torch.Tensor:
    """Outlier-robust normalize [B, S] → (0,1) via median/MAD + sigmoid."""
    med = t.median(dim=-1, keepdim=True).values
    mad = (t - med).abs().median(dim=-1, keepdim=True).values
    z = (t - med) / (1.4826 * mad + 1e-6)
    return torch.sigmoid(z)

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
MAX_CHUNKS         = 200     # Cap on code-corpus chunks (None = full corpus)
DTYPE              = torch.bfloat16

# ── Needle-in-a-Haystack (NIAH) settings ──────────────────────────────────────
NIAH_SEQ_LEN   = 1024   # Total NIAH context length (must fit MAX_SEQ_LEN)
NIAH_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]  # Needle depth as fraction of haystack
NIAH_RECALL_NLL_THRESHOLD = 3.0  # NLL below this → needle recalled

# Each needle:
#   id    — short label for output table
#   fact  — the comment line injected into the haystack
#   query — continuation prefix after the haystack; model must complete with `check`
#   check — the unique fragment that proves the model remembered the needle
NIAH_NEEDLES = [
    dict(id="auth_token",
         fact  = "# IMPORTANT: The authentication token is XKCD-ALPHA-5792.\n",
         query = "# Q: What is the authentication token?\n# A: XKCD-",
         check = "ALPHA"),
    dict(id="config_code",
         fact  = "# NOTE: The configuration code is CFG-DELTA-8831.\n",
         query = "# Q: What is the configuration code?\n# A: CFG-",
         check = "DELTA"),
]

# Python filler code used to build the haystack around each needle
_HAYSTACK_CODE = (
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left  = [x for x in arr if x < pivot]\n"
    "    mid   = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + mid + quicksort(right)\n\n"
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        m = (lo + hi) // 2\n"
    "        if arr[m] == target:   return m\n"
    "        elif arr[m] < target:  lo = m + 1\n"
    "        else:                  hi = m - 1\n"
    "    return -1\n\n"
    "def flatten(lst):\n"
    "    result = []\n"
    "    for item in lst:\n"
    "        if isinstance(item, list):\n"
    "            result.extend(flatten(item))\n"
    "        else:\n"
    "            result.append(item)\n"
    "    return result\n\n"
    "def count_words(text: str) -> int:\n"
    "    return len(text.split())\n\n"
)

# ── Generation-side PPL test samples ──────────────────────────────────────────
# Design criteria for discriminative Gen-PPL:
#   · Context ≥ 350 tokens  → 30% eviction removes ~100 meaningful tokens
#   · Unusual constants (0xC0FFEE42, WINDOW_S=2.718…, C1/C2=0.5) that cannot
#     be predicted from model memory — the model MUST read them from context
#   · Continuation heavily references context constants/helpers; any eviction
#     of those definitions causes measurable NLL increase
GEN_PPL_SAMPLES = [
    dict(
        context=(
            "# Async record pipeline — shard-weighted routing\n"
            "BUFFER_FLUSH_MS   = 750\n"
            "MAX_SHARDS        = 11\n"
            "REPLICAS          = 3\n"
            "CHECKSUM_SEED     = 0xC0FFEE42\n"
            "SCHEMA_VER        = '4.1.7'\n"
            "_WEIGHTS = [1.5, 0.8, 1.2, 0.6, 1.0, 1.3, 0.9, 1.1, 0.7, 1.4, 1.0]\n\n"
            "def _shard_for(key: bytes) -> int:\n"
            "    import zlib\n"
            "    crc = zlib.crc32(key, CHECKSUM_SEED) & 0xFFFFFFFF\n"
            "    total = sum(_WEIGHTS)\n"
            "    thresholds, acc = [], 0.0\n"
            "    for w in _WEIGHTS:\n"
            "        acc += w / total\n"
            "        thresholds.append(acc)\n"
            "    norm = crc / 0xFFFFFFFF\n"
            "    for i, t in enumerate(thresholds):\n"
            "        if norm < t:\n"
            "            return i\n"
            "    return MAX_SHARDS - 1\n\n"
            "class Rec:\n"
            "    __slots__ = ('key', 'val', 'ts', 'ver', 'shard')\n"
            "    def __init__(self, key: bytes, val: bytes, ts: int):\n"
            "        self.key   = key\n"
            "        self.val   = val\n"
            "        self.ts    = ts\n"
            "        self.ver   = SCHEMA_VER\n"
            "        self.shard = _shard_for(key)\n\n"
            "class FlushMode:\n"
            "    EAGER   = 'eager'\n"
            "    LAZY    = 'lazy'\n"
            "    BATCHED = 'batched'\n\n"
            "def _dup_check(key: bytes, seen: set) -> bool:\n"
            "    import hashlib\n"
            "    h = key + CHECKSUM_SEED.to_bytes(4, 'little')\n"
            "    digest = hashlib.blake2b(h, digest_size=8).digest()\n"
            "    if digest in seen:\n"
            "        return True\n"
            "    seen.add(digest)\n"
            "    return False\n\n"
            "def flush_records(\n"
            "    records: list,\n"
            "    mode: str = FlushMode.BATCHED,\n"
            "    max_age_ms: int = BUFFER_FLUSH_MS,\n"
            ") -> dict:\n"
            "    \"\"\"Flush records to shards; drop schema-mismatched and duplicate entries.\"\"\"\n"
        ),
        continuation=(
            "    import time\n"
            "    now_ms = int(time.monotonic() * 1000)\n"
            "    seen, shard_map, errors = set(), {}, []\n"
            "    for rec in records:\n"
            "        if rec.ver != SCHEMA_VER:\n"
            "            errors.append(('schema', rec.key))\n"
            "            continue\n"
            "        if _dup_check(rec.key, seen):\n"
            "            continue\n"
            "        if now_ms - rec.ts > max_age_ms:\n"
            "            errors.append(('stale', rec.key))\n"
            "            continue\n"
            "        shard_map.setdefault(rec.shard, []).append(rec)\n"
            "    flushed = 0\n"
            "    for sid, batch in shard_map.items():\n"
            "        w = _WEIGHTS[sid % len(_WEIGHTS)]\n"
            "        if mode == FlushMode.EAGER or len(batch) * w >= 1.0:\n"
            "            flushed += len(batch) * REPLICAS\n"
            "    return {'flushed': flushed, 'shards': len(shard_map), 'errors': errors}\n"
        ),
    ),
    dict(
        context=(
            "# Sliding-window rate limiter — unusual parameters\n"
            "import time, collections, threading\n\n"
            "WINDOW_S       = 2.718281828   # e seconds (non-standard window)\n"
            "MAX_PER_WINDOW = 137           # prime cap\n"
            "BURST_CAP      = 17\n"
            "PENALTY_BASE   = 0.05          # seconds\n"
            "PENALTY_SCALE  = 1.618         # golden ratio back-off\n"
            "MIN_GAP_S      = 0.05\n\n"
            "_state: dict    = {}   # {client_id: deque of timestamps}\n"
            "_penalties: dict = {}  # {client_id: penalty_until (monotonic)}\n"
            "_lock = threading.Lock()\n\n"
            "def _purge(client_id: str, now: float) -> None:\n"
            "    dq = _state.get(client_id)\n"
            "    if not dq:\n"
            "        return\n"
            "    cutoff = now - WINDOW_S\n"
            "    while dq and dq[0] < cutoff:\n"
            "        dq.popleft()\n\n"
            "def _next_penalty(client_id: str) -> float:\n"
            "    prev = _penalties.get(client_id, PENALTY_BASE)\n"
            "    new_p = max(prev * PENALTY_SCALE, MIN_GAP_S)\n"
            "    _penalties[client_id] = new_p\n"
            "    return new_p\n\n"
            "def _effective_limit() -> int:\n"
            "    return MAX_PER_WINDOW + BURST_CAP\n\n"
            "def _under_penalty(client_id: str, now: float) -> tuple:\n"
            "    until = _penalties.get(client_id)\n"
            "    if until and now < until:\n"
            "        return True, until - now\n"
            "    return False, 0.0\n\n"
            "def allow_request(client_id: str) -> dict:\n"
            "    \"\"\"Thread-safe sliding-window rate check. Returns allow/deny + metadata.\"\"\"\n"
        ),
        continuation=(
            "    now = time.monotonic()\n"
            "    with _lock:\n"
            "        penalised, wait = _under_penalty(client_id, now)\n"
            "        if penalised:\n"
            "            return {'allowed': False, 'retry_after': wait, 'reason': 'penalty'}\n"
            "        _purge(client_id, now)\n"
            "        dq = _state.setdefault(client_id, collections.deque())\n"
            "        if len(dq) >= _effective_limit():\n"
            "            delay = _next_penalty(client_id)\n"
            "            _penalties[client_id] = now + delay\n"
            "            return {'allowed': False, 'retry_after': delay, 'reason': 'rate_limit'}\n"
            "        dq.append(now)\n"
            "        _penalties.pop(client_id, None)\n"
            "        return {'allowed': True,\n"
            "                'remaining': _effective_limit() - len(dq),\n"
            "                'window_s': WINDOW_S}\n"
        ),
    ),
    dict(
        context=(
            "# Open-addressing hash table — quadratic probing with Robin Hood eviction\n"
            "INIT_CAP = 23       # initial prime capacity\n"
            "LOAD_HI  = 0.72     # grow threshold\n"
            "LOAD_LO  = 0.18     # shrink threshold\n"
            "C1, C2   = 0.5, 0.5 # probe polynomial coefficients\n"
            "_EMPTY   = object()\n"
            "_DELETED = object()\n\n"
            "def _probe(h: int, cap: int):\n"
            "    \"\"\"Quadratic probe sequence: h + C1*i + C2*i^2  (mod cap).\"\"\"\n"
            "    for i in range(cap):\n"
            "        yield int(h + C1 * i + C2 * i * i) % cap\n\n"
            "class RHTable:\n"
            "    \"\"\"Robin Hood open-addressing table with dynamic resizing.\"\"\"\n"
            "    __slots__ = ('_keys', '_vals', '_flags', '_size', '_cap')\n\n"
            "    def __init__(self, cap: int = INIT_CAP):\n"
            "        self._cap   = cap\n"
            "        self._keys  = [None] * cap\n"
            "        self._vals  = [None] * cap\n"
            "        self._flags = [_EMPTY] * cap\n"
            "        self._size  = 0\n\n"
            "    def _load(self) -> float:\n"
            "        return self._size / self._cap\n\n"
            "    def _rebuild(self, new_cap: int) -> None:\n"
            "        old = list(zip(self._flags, self._keys, self._vals))\n"
            "        self.__init__(new_cap)\n"
            "        for f, k, v in old:\n"
            "            if f is not _EMPTY and f is not _DELETED:\n"
            "                self.put(k, v)\n\n"
            "    def get(self, key, default=None):\n"
            "        h = hash(key) % self._cap\n"
            "        for idx in _probe(h, self._cap):\n"
            "            f = self._flags[idx]\n"
            "            if f is _EMPTY:\n"
            "                return default\n"
            "            if f is not _DELETED and self._keys[idx] == key:\n"
            "                return self._vals[idx]\n"
            "        return default\n\n"
            "    def put(self, key, value) -> None:\n"
            "        \"\"\"Insert/update key; resize when load factor crosses LOAD_HI or LOAD_LO.\"\"\"\n"
        ),
        continuation=(
            "        if self._load() >= LOAD_HI:\n"
            "            self._rebuild(self._cap * 2 + 1)\n"
            "        elif self._size > 0 and self._load() <= LOAD_LO:\n"
            "            self._rebuild(max(INIT_CAP, self._cap // 2 + 1))\n"
            "        h = hash(key) % self._cap\n"
            "        for idx in _probe(h, self._cap):\n"
            "            f = self._flags[idx]\n"
            "            if f is _EMPTY or f is _DELETED:\n"
            "                self._flags[idx] = True\n"
            "                self._keys[idx]  = key\n"
            "                self._vals[idx]  = value\n"
            "                self._size += 1\n"
            "                return\n"
            "            if self._keys[idx] == key:\n"
            "                self._vals[idx] = value\n"
            "                return\n"
            "        raise RuntimeError('probe exhausted, cap=%d' % self._cap)\n"
        ),
    ),
]

# Candidate decision layers to sweep (which layer computes the shared eviction vector)
SWEEP_LAYERS = [0, 1, 2, 4, 6, 8, 12, 16]

# (id, human label, internal strategy key)
CONFIGS = (
    [("A", "Baseline", "baseline")]
    + [(f"L{L}", f"Layer-{L} Decision", f"layer_{L}") for L in SWEEP_LAYERS]
    + [("D", "Oracle", "oracle")]
)


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Code corpus loading  (domain-appropriate PPL)
# ═══════════════════════════════════════════════════════════════════════════════

def load_code_corpus(tokenizer):
    """
    Load and tokenise a Python code corpus for domain-appropriate PPL.

    Why not WikiText-2?
      DeepSeek-Coder-V2-Lite is a code model; WikiText-2 is English prose.
      Using a Python corpus measures the metric that matters.

    Priority:
      1. code-search-net/code_search_net  (Python, Parquet, no auth needed)
      2. bigcode/the-stack-smol           (Python subset, Parquet)
      3. openai/openai_humaneval          (correct HF dataset name)
      4. Built-in _HAYSTACK_CODE          (zero-dependency fallback)

    All options avoid trust_remote_code (loading scripts no longer supported).
    Corpus size is capped to exactly what MAX_CHUNKS × STRIDE + MAX_SEQ_LEN tokens
    require, preventing the tokenizer's "sequence too long" warning.
    """
    # How many chars we actually need (≈ 3.5 chars / token for Python code)
    tokens_needed = (MAX_CHUNKS or 500) * STRIDE + MAX_SEQ_LEN
    MAX_CHARS = tokens_needed * 4   # conservative upper bound

    text = None

    # ── Try 1: code-search-net (Python, always public, Parquet) ───────────
    try:
        from datasets import load_dataset
        ds = load_dataset("code-search-net/code_search_net", "python",
                          split="test", trust_remote_code=False)
        snippets = [ex["whole_func_string"] for ex in ds
                    if ex.get("whole_func_string", "").strip()]
        if snippets:
            text = "\n\n".join(snippets)
            # Repeat if corpus is shorter than needed
            while len(text) < MAX_CHARS and len(snippets) > 0:
                text = text + "\n\n" + text
            text = text[:MAX_CHARS]
            print(f"  Code corpus (code-search-net Python) : {len(text):,} chars")
    except Exception as exc:
        print(f"  [WARN] code-search-net unavailable ({exc})")

    # ── Try 2: bigcode/the-stack-smol (Python, Parquet) ───────────────────
    if text is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python",
                              split="train", streaming=True)
            snippets, total_chars = [], 0
            for ex in ds:
                code = ex.get("content", "").strip()
                if code:
                    snippets.append(code)
                    total_chars += len(code)
                    if total_chars >= MAX_CHARS:
                        break
            if snippets:
                text = "\n\n".join(snippets)[:MAX_CHARS]
                print(f"  Code corpus (bigcode/the-stack-smol Python) : {len(text):,} chars")
        except Exception as exc:
            print(f"  [WARN] bigcode/the-stack-smol unavailable ({exc})")

    # ── Try 3: openai/openai_humaneval (correct HF name, Parquet) ─────────
    if text is None:
        for ds_name in ("openai/openai_humaneval", "evalplus/humanevalplus"):
            try:
                from datasets import load_dataset
                ds = load_dataset(ds_name, split="test")
                snippets = [ex["prompt"] + ex["canonical_solution"] for ex in ds]
                raw = "\n\n".join(snippets)
                # Repeat until we have enough
                reps = (MAX_CHARS // max(1, len(raw))) + 2
                text = (raw * reps)[:MAX_CHARS]
                print(f"  Code corpus ({ds_name} × {reps}) : {len(text):,} chars")
                break
            except Exception as exc:
                print(f"  [WARN] {ds_name} unavailable ({exc})")

    # ── Fallback: built-in Python code (zero dependencies) ────────────────
    if text is None:
        reps = (MAX_CHARS // max(1, len(_HAYSTACK_CODE))) + 2
        text = (_HAYSTACK_CODE * reps)[:MAX_CHARS]
        print(f"  Code corpus (built-in Python fallback) : {len(text):,} chars")

    # Tokenise; suppress the benign "sequence longer than model_max_length"
    # warning — the PPL eval only ever feeds 1024-token windows to the model.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Token indices sequence length is longer than",
        )
        enc = tokenizer(text, return_tensors="pt")
    actual = enc.input_ids.shape[1]
    # Safety: never pass more tokens than the sliding-window eval needs
    if actual > tokens_needed:
        enc.input_ids = enc.input_ids[:, :tokens_needed]
        actual = tokens_needed
    print(f"  Tokenised length : {actual:,} tokens  (capped at {tokens_needed})")
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


def _build_evict_vector_from_context(c_kv: torch.Tensor, context_len: int):
    """
    Build eviction vector using ONLY the first `context_len` tokens for scoring.

    Tokens at positions >= context_len (the continuation / decode tokens) receive
    evict=0 unconditionally — they are being generated and must never be masked out.

    This mirrors real KV-cache inference: only the prefill (context) is evicted;
    each autoregressive decode step appends to the surviving cache without eviction.

    Returns
        evict     : [B, seq_len]  1.0 = evicted (0 for positions >= context_len)
        kept_frac : float         keep_k / context_len
    """
    bsz, seq_len, _ = c_kv.shape
    c_ctx  = c_kv[:, :context_len, :]
    info   = compute_latent_info_score_causal(c_ctx, window=EVICTION_WINDOW)

    keep_k = max(1, int(round(context_len * KEEP_RATIO)))
    keep_idx = info.argsort(dim=-1, descending=True)[:, :keep_k]

    evict_ctx = torch.ones(bsz, context_len, device=info.device, dtype=info.dtype)
    evict_ctx.scatter_(1, keep_idx, 0.0)

    if seq_len > context_len:
        zeros = torch.zeros(bsz, seq_len - context_len,
                            device=evict_ctx.device, dtype=evict_ctx.dtype)
        evict = torch.cat([evict_ctx, zeros], dim=1)
    else:
        evict = evict_ctx

    return evict, keep_k / context_len


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
    context_len: int = None,
):
    """
    Determine the per-token eviction vector for the current layer.

    Handles cross-layer coordination: Layer 0 always resets shared state.

    context_len : if provided (gen-PPL mode), eviction scores are computed from
                  the first context_len tokens only; continuation tokens are
                  unconditionally kept (evict=0).

    Returns
        evict_vec : Tensor [B, S] (1=evict) or None
        kept_frac : float
    """
    def _evict(c):
        """Build eviction vector, respecting context_len if set."""
        if context_len is not None and context_len < c.shape[1]:
            return _build_evict_vector_from_context(c, context_len)
        return _build_evict_vector(c)
    # ── Reset shared state at the first layer of every forward pass ───────
    if layer_idx == 0:
        state.evict_vector = None

    # ── Config A: no eviction ─────────────────────────────────────────────
    if strategy == "baseline":
        return None, 1.0

    # ── Config layer_L: layer L (and all deeper layers) share one eviction vector
    #    · layer_idx < L  → dense (no eviction)
    #    · layer_idx == L → compute, cache, AND apply eviction
    #    · layer_idx > L  → reuse cached vector
    if strategy.startswith("layer_"):
        L = int(strategy.split("_", 1)[1])
        if layer_idx < L:
            return None, 1.0
        if layer_idx == L:
            ev, kf = _evict(c_kv)
            state.evict_vector     = ev
            state.kept_frac_shared = kf
            return ev, kf
        # layer_idx > L
        if state.evict_vector is None:
            return None, 1.0
        return state.evict_vector.to(c_kv.device), state.kept_frac_shared

    # ── Config D: each layer decides independently ────────────────────────
    if strategy == "oracle":
        ev, kf = _evict(c_kv)
        return ev, kf

    return None, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Patched attention forward
# ═══════════════════════════════════════════════════════════════════════════════

def make_eval_forward(layer_idx: int, strategy: str, state: EvictionState,
                      context_len: int = None):
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
        evict_vec, kept_frac = resolve_eviction(c_kv, layer_idx, strategy, state,
                                                context_len=context_len)
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
# § 5  Needle-in-a-Haystack (NIAH) evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_niah_input(tokenizer, needle: dict, position_frac: float, device: str):
    """
    Build one NIAH test input.

    Sequence layout:
        [haystack_pre]  [fact]  [haystack_post]  [query]  [check]

    Labels: -100 everywhere EXCEPT [check] tokens.
    Loss = NLL of predicting [check] given full context (teacher-forcing).
    A needle is "recalled" when NLL < NIAH_RECALL_NLL_THRESHOLD.

    Returns
        input_ids : LongTensor [1, seq_len]
        labels    : LongTensor [1, seq_len]   (-100 except check fragment)
        check_len : int
    """
    enc_kw = dict(return_tensors="pt", add_special_tokens=False)
    fact_ids  = tokenizer(needle["fact"],  **enc_kw).input_ids[0]
    query_ids = tokenizer(needle["query"], **enc_kw).input_ids[0]
    check_ids = tokenizer(needle["check"], **enc_kw).input_ids[0]

    overhead   = len(fact_ids) + len(query_ids) + len(check_ids)
    hay_target = max(32, NIAH_SEQ_LEN - overhead)

    hay_base = tokenizer(_HAYSTACK_CODE, **enc_kw).input_ids[0]
    reps     = (hay_target // max(1, len(hay_base))) + 2
    hay_long = hay_base.repeat(reps)[:hay_target]

    split    = int(len(hay_long) * position_frac)
    pre_hay  = hay_long[:split]
    post_hay = hay_long[split:]

    full = torch.cat([pre_hay, fact_ids, post_hay, query_ids, check_ids])
    full = full[:NIAH_SEQ_LEN].unsqueeze(0).to(device)

    seq_len     = full.shape[1]
    check_start = min(
        len(pre_hay) + len(fact_ids) + len(post_hay) + len(query_ids),
        seq_len - len(check_ids),
    )
    check_end = min(check_start + len(check_ids), seq_len)

    labels = torch.full_like(full, -100)
    labels[:, check_start:check_end] = full[:, check_start:check_end]

    return full, labels, int(check_end - check_start)


def evaluate_niah(model, tokenizer, device: str) -> dict:
    """
    Needle-in-a-Haystack evaluation across ALL eviction configs.

    For every (strategy × needle × position) triple:
      1. Build NIAH input (Python haystack + injected fact + question prefix)
      2. Run forward with the eviction-patched forward (same monkey-patch as PPL)
      3. Compute NLL of the `check` fragment via teacher-forcing

    Metric: lower NLL → model predicted the check tokens confidently → needle recalled.
    This directly validates whether eviction mis-kills critical tokens.

    Returns
        dict: (cfg_id, label, strategy) → list of (needle_id, position, nll)
    """
    all_results: dict = {}

    for cfg_id, label, strategy in CONFIGS:
        attn_layers = [layer.self_attn for layer in model.model.layers]
        state       = EvictionState()

        orig_fwds = []
        for idx, attn in enumerate(attn_layers):
            orig_fwds.append(attn.forward)
            attn.forward = types.MethodType(
                make_eval_forward(idx, strategy, state), attn
            )

        runs = []
        try:
            for needle in NIAH_NEEDLES:
                for pos in NIAH_POSITIONS:
                    state.kept_fracs = []
                    input_ids, labels, check_len = _build_niah_input(
                        tokenizer, needle, pos, device
                    )
                    with torch.no_grad():
                        loss = model(input_ids, labels=labels).loss
                    runs.append((needle["id"], pos, float(loss)))
        finally:
            for attn, orig in zip(attn_layers, orig_fwds):
                attn.forward = orig

        all_results[(cfg_id, label, strategy)] = runs
        mean_nll   = float(np.mean([r[2] for r in runs]))
        recall_pct = float(np.mean([r[2] < NIAH_RECALL_NLL_THRESHOLD
                                    for r in runs])) * 100
        print(f"    NIAH  mean NLL = {mean_nll:.3f}  |  Recall = {recall_pct:.0f}%\n")

    return all_results


def print_niah_results(niah_results: dict):
    """
    Print NIAH recall table.

    Rows   = eviction configs
    Columns = mean NLL, recall%, per-position NLL (averaged over needles)
    """
    print("\n" + "═" * 72)
    print("  NIAH — Needle Recall  (NLL of check fragment, lower = better)")
    print(f"  Recall = NLL < {NIAH_RECALL_NLL_THRESHOLD}  |  "
          f"Positions: {NIAH_POSITIONS}  |  "
          f"Needles: {[n['id'] for n in NIAH_NEEDLES]}")
    print("═" * 72 + "\n")

    pos_labels = [f"@{p:.1f}" for p in NIAH_POSITIONS]
    hdr  = f"| {'Config':<6} | {'Description':<22} | {'MeanNLL':>7} | {'Recall%':>7} |"
    hdr += "".join(f" {pl:>6} |" for pl in pos_labels)
    sep  = f"|{'-'*8}|{'-'*24}|{'-'*9}|{'-'*9}|"
    sep += "".join(f"{'-'*8}|" for _ in pos_labels)

    print(hdr)
    print(sep)

    baseline_nll = None
    for (cfg_id, label, strategy), runs in niah_results.items():
        mean_nll   = float(np.mean([r[2] for r in runs]))
        recall_pct = float(np.mean([r[2] < NIAH_RECALL_NLL_THRESHOLD
                                    for r in runs])) * 100
        if cfg_id == "A":
            baseline_nll = mean_nll

        # Per-position mean (average over needles)
        pos_nlls = {}
        for _, pos, nll in runs:
            pos_nlls.setdefault(pos, []).append(nll)
        pos_means = [float(np.mean(pos_nlls.get(p, [float("inf")])))
                     for p in NIAH_POSITIONS]

        row  = f"| {cfg_id:<6} | {label:<22} | {mean_nll:7.3f} | {recall_pct:6.0f}% |"
        row += "".join(f" {v:6.2f} |" for v in pos_means)
        print(row)

    print()
    if baseline_nll is not None:
        print(f"  Baseline mean NLL = {baseline_nll:.3f} (reference)")
    print(
        f"\n  Position interpretation:\n"
        f"    @0.1 = needle near start (easy recall)\n"
        f"    @0.5 = needle in middle  (medium difficulty)\n"
        f"    @0.9 = needle near end   (easy recall, recency-protected)\n"
        f"  High NLL at @0.3–@0.7 indicates mid-sequence eviction is mis-killing needles."
    )
    print("─" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# § 5b  Generation-side PPL evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_gen_ppl(model, tokenizer, device: str) -> dict:
    """
    Generation-side PPL: NLL of continuation tokens given an eviction-truncated prefill.

    Unlike sliding-window PPL (all tokens processed together in one shot), this
    evaluation directly mimics autoregressive decode:

        1. Tokenise (context, continuation) pairs.
        2. Build the full input [context][continuation].
        3. Run forward with context-ONLY eviction:
             · Eviction scores are computed on context tokens (prefill).
             · Continuation tokens are never evicted (decode is append-only).
        4. Compute NLL exclusively on continuation tokens (labels=-100 for context).

    This exposes failure modes that sliding-window PPL misses:
      e.g. evicting a function header makes its body unpredictable, even if
      global per-token PPL looks acceptable.

    Returns
        dict: (cfg_id, label, strategy) → dict with
              mean_nll, gen_ppl, per_sample_nlls
    """
    all_results: dict = {}

    for cfg_id, label, strategy in CONFIGS:
        attn_layers = [layer.self_attn for layer in model.model.layers]
        sample_nlls = []

        for s_idx, sample in enumerate(GEN_PPL_SAMPLES):
            enc_kw = dict(return_tensors="pt", add_special_tokens=False)
            ctx_ids  = tokenizer(sample["context"],      **enc_kw).input_ids[0]
            cont_ids = tokenizer(sample["continuation"], **enc_kw).input_ids[0]
            ctx_len  = len(ctx_ids)
            # Log context length on first config to verify discriminability
            if cfg_id == "A":
                print(f"    sample {s_idx+1}: ctx={ctx_len} tok, "
                      f"cont={len(cont_ids)} tok, "
                      f"evict≈{int(ctx_len * (1-KEEP_RATIO))} tok")

            full_ids = torch.cat([ctx_ids, cont_ids]).unsqueeze(0).to(device)

            # Labels: -100 for context (prefill), actual tokens for continuation
            labels = torch.full_like(full_ids, -100)
            cont_start = ctx_len
            cont_end   = min(ctx_len + len(cont_ids), full_ids.shape[1])
            labels[:, cont_start:cont_end] = full_ids[:, cont_start:cont_end]

            # Install patched forwards with context-aware eviction
            state = EvictionState()
            orig_fwds = []
            for idx, attn in enumerate(attn_layers):
                orig_fwds.append(attn.forward)
                attn.forward = types.MethodType(
                    make_eval_forward(idx, strategy, state, context_len=ctx_len),
                    attn,
                )

            try:
                with torch.no_grad():
                    loss = model(full_ids, labels=labels).loss
                sample_nlls.append(float(loss))
            finally:
                for attn, orig in zip(attn_layers, orig_fwds):
                    attn.forward = orig

        mean_nll = float(np.mean(sample_nlls)) if sample_nlls else float("inf")
        gen_ppl  = float(np.exp(mean_nll))
        all_results[(cfg_id, label, strategy)] = {
            "mean_nll": mean_nll, "gen_ppl": gen_ppl, "per_sample": sample_nlls
        }
        print(f"    Gen-PPL = {gen_ppl:.3f}  (mean NLL = {mean_nll:.3f})\n")

    return all_results


def print_gen_ppl_results(gen_results: dict):
    """Print generation-side PPL table."""
    print("\n" + "═" * 72)
    print("  Gen-PPL — Decode Quality After Prefill Eviction")
    print(f"  {len(GEN_PPL_SAMPLES)} Python function samples  |  "
          f"Context-only eviction (continuation tokens always kept)")
    print("═" * 72 + "\n")

    hdr = (f"| {'Config':<6} | {'Description':<22} | {'GenPPL':>8} | "
           f"{'MeanNLL':>7} | {'Δ GenPPL vs A':>13} |")
    sep = f"|{'-'*8}|{'-'*24}|{'-'*10}|{'-'*9}|{'-'*15}|"
    print(hdr)
    print(sep)

    baseline_ppl = None
    for (cfg_id, label, strategy), v in gen_results.items():
        if cfg_id == "A":
            baseline_ppl = v["gen_ppl"]

    for (cfg_id, label, strategy), v in gen_results.items():
        gp  = v["gen_ppl"]
        nll = v["mean_nll"]
        if cfg_id == "A" or baseline_ppl is None:
            delta_s = "—"
        else:
            delta_s = f"{gp - baseline_ppl:+.3f}"
        print(f"| {cfg_id:<6} | {label:<22} | {gp:8.3f} | {nll:7.3f} | {delta_s:>13} |")

    print()
    if baseline_ppl is not None:
        print(f"  Baseline Gen-PPL = {baseline_ppl:.3f} (reference)")
    print(
        "\n  Interpretation:\n"
        "    Gen-PPL ≈ Baseline → eviction does not hurt code generation\n"
        "    Gen-PPL >> Baseline → eviction degrades decode quality\n"
        "    Complements sliding-window PPL (which measures prefill, not decode)"
    )
    print("─" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  Latent Eviction Cross-Layer Decision  —  Ablation Study")
    print("  DeepSeek-V2-Lite  ·  Python Code Corpus  ·  PPL + NIAH")
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
    print("\n[ Loading Python code corpus … ]")
    enc = load_code_corpus(tok)

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
    print("  RESULTS — Perplexity & KV Cache Compression (Python Code Corpus)")
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
        f"**Dataset**: Python Code Corpus  |  "
        f"**Keep ratio**: {KEEP_RATIO} (Top-K)  |  "
        f"**Window**: {EVICTION_WINDOW} (causal)  |  "
        f"**Seq len**: {MAX_SEQ_LEN}  |  "
        f"**Stride**: {STRIDE}  |  "
        f"**Chunks**: {MAX_CHUNKS}"
    )

    # ── Automatic interpretation — layer-sweep summary ─────────────────────
    # results[0]  = Baseline (A)
    # results[1:-1] = sweep layers L0 … L_max
    # results[-1] = Oracle (D)
    ppl_oracle   = results[-1][3]
    delta_oracle = ppl_oracle - baseline_ppl
    sweep_results = results[1:-1]   # all layer_L configs

    SWEET_SPOT_TOL = 0.5   # gap to Oracle considered "good enough"

    print("\n" + "─" * 72)
    print("  KEY FINDINGS  —  Layer-Sweep Gap vs Oracle\n")
    print(f"  Baseline PPL (A) : {baseline_ppl:.4f}")
    print(f"  Oracle   PPL (D) : {ppl_oracle:.4f}  (Δ vs A = {delta_oracle:+.4f})\n")
    print(f"  {'Layer':>7}  {'PPL':>8}  {'Δ vs A':>9}  {'Gap vs Oracle':>14}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*14}")

    sweet_spot_layer = None
    for cfg_id, label, strategy, ppl, kept, elapsed in sweep_results:
        delta_vs_a    = ppl - baseline_ppl
        gap_vs_oracle = ppl - ppl_oracle
        marker = ""
        if gap_vs_oracle <= SWEET_SPOT_TOL and sweet_spot_layer is None:
            sweet_spot_layer = cfg_id
            marker = "  ← Sweet Spot ✓"
        print(
            f"  {cfg_id:>7}  {ppl:>8.4f}  {delta_vs_a:>+9.4f}  {gap_vs_oracle:>+14.4f}{marker}"
        )

    print()
    if sweet_spot_layer is not None:
        print(
            f"  ✓  Sweet Spot: {sweet_spot_layer} — earliest layer within "
            f"{SWEET_SPOT_TOL} PPL of Oracle.\n"
            f"     Features are sufficiently settled by this layer;"
            f" using it as the shared decision layer is recommended."
        )
    else:
        print(
            f"  △  No sweep layer reached within {SWEET_SPOT_TOL} PPL of Oracle.\n"
            f"     Consider sweeping deeper layers or increasing KEEP_RATIO."
        )

    print("─" * 72)

    # ── NIAH evaluation ────────────────────────────────────────────────────
    print("\n\n[ Running Needle-in-a-Haystack evaluation … ]")
    print(f"  {len(NIAH_NEEDLES)} needles × {len(NIAH_POSITIONS)} positions × "
          f"{len(CONFIGS)} configs = "
          f"{len(NIAH_NEEDLES) * len(NIAH_POSITIONS) * len(CONFIGS)} forward passes\n")
    niah_results = evaluate_niah(model, tok, device)
    print_niah_results(niah_results)

    # ── Generation-side PPL ────────────────────────────────────────────────
    print("\n\n[ Running generation-side PPL evaluation … ]")
    print(f"  {len(GEN_PPL_SAMPLES)} code samples × {len(CONFIGS)} configs "
          f"= {len(GEN_PPL_SAMPLES) * len(CONFIGS)} forward passes\n")
    gen_results = evaluate_gen_ppl(model, tok, device)
    print_gen_ppl_results(gen_results)


if __name__ == "__main__":
    main()
