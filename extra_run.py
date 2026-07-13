"""
extra_run.py  —  Cross-Layer Importance Persistence Analysis
=============================================================

Direct empirical validation of the "cross-layer persistence" property of
MLA latent-token importance scores.

Experiment 1 — Spearman Rank Correlation Matrix
  For each pair of committee layers (L4, L6, L8, L12, L16), computes the
  Spearman rank correlation rho of per-token importance scores, averaged
  over all 10 test sequences.  A high rho between distant layer pairs
  (e.g. L4 vs L16) directly validates the persistence claim.

  Also computes the Jaccard index of the top-K keep sets (K = 30% of
  n_mid) between every layer pair — the eviction-decision analog of rho.

  Outputs:
    cross_layer_correlation.pdf   5×5 Spearman heatmap
    cross_layer_jaccard.pdf        5×5 Jaccard@30% heatmap

Experiment 3 — Token Importance Trajectory Visualisation
  For three representative sequences (concentrated / medium / flat
  importance distribution), plots the within-layer percentile rank of
  selected tokens across all five committee layers.  Non-crossing, stable
  trajectories for the top / bottom groups confirm cross-layer persistence.

  Outputs:
    trajectory_{label}.pdf   (one file per sequence)
    trajectory_combined.pdf  (3-panel summary figure)
"""

import os
import sys
import warnings

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
# 10 layers spanning early (0–2), transition (4–8), mid (10–14), deep (16–20).
# Layers 0/1/2 are expected to correlate poorly with later layers (cf. Table 1);
# layer 4+ should exhibit the high-persistence block.
COMMITTEE_LAYERS  = [0, 1, 2, 4, 6, 8, 10, 12, 16, 20]
SCORE_QUERIES     = 512        # max query subsample (matches production)
SINK_TOKENS       = 4
RECENT_TOKENS     = 16
DTYPE             = torch.bfloat16
JACCARD_K_RATIO   = 0.30       # top-K fraction for Jaccard keep-set comparison

# Sequences shown in trajectory plots (must be labels in TEST_CASES)
TRAJECTORY_LABELS = ["quicksort", "binary_search", "repetitive_code"]

# ── 10 diverse test sequences (identical to score_visual.py) ─────────────────
TEST_CASES = [
    ("simple_utils", """
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def lerp(a, b, t):
    return a + (b - a) * t

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
"""),

    ("binary_search", """
def binary_search(arr, target):
    \"\"\"Standard iterative binary search.
    Returns index of target in sorted arr, or -1 if not found.
    \"\"\"
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def lower_bound(arr, target):
    \"\"\"First index where arr[i] >= target.\"\"\"
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def upper_bound(arr, target):
    \"\"\"First index where arr[i] > target.\"\"\"
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
"""),

    ("quicksort", """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot  = arr[len(arr) // 2]
    left   = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right  = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def partition(arr, lo, hi):
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1

def quicksort_inplace(arr, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        p = partition(arr, lo, hi)
        quicksort_inplace(arr, lo, p - 1)
        quicksort_inplace(arr, p + 1, hi)
    return arr
"""),

    ("bfs_graph", """
from collections import deque

def bfs(graph, start):
    \"\"\"Breadth-first traversal; returns nodes in visit order.\"\"\"
    visited = set()
    queue   = deque([start])
    order   = []
    visited.add(start)
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order

def shortest_path(graph, start, end):
    \"\"\"BFS shortest path between start and end. Returns list of nodes or [].\"\"\"
    if start == end:
        return [start]
    visited = {start: None}
    queue   = deque([start])
    while queue:
        node = queue.popleft()
        for nb in graph.get(node, []):
            if nb not in visited:
                visited[nb] = node
                if nb == end:
                    path, cur = [], end
                    while cur is not None:
                        path.append(cur)
                        cur = visited[cur]
                    return path[::-1]
                queue.append(nb)
    return []
"""),

    ("class_stack", """
class Stack:
    \"\"\"A simple stack using a Python list as backing storage.\"\"\"

    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(reversed(self._data))

    def __repr__(self):
        return f"Stack({self._data!r})"

    def clear(self):
        self._data.clear()
"""),

    ("repetitive_code", """
a0 = 0; a1 = 0; a2 = 0; a3 = 0; a4 = 0
b0 = 1; b1 = 1; b2 = 1; b3 = 1; b4 = 1
c0 = 0; c1 = 0; c2 = 0; c3 = 0; c4 = 0
d0 = 1; d1 = 1; d2 = 1; d3 = 1; d4 = 1
e0 = 0; e1 = 0; e2 = 0; e3 = 0; e4 = 0

row0 = [a0, a1, a2, a3, a4]
row1 = [b0, b1, b2, b3, b4]
row2 = [c0, c1, c2, c3, c4]
row3 = [d0, d1, d2, d3, d4]
row4 = [e0, e1, e2, e3, e4]

s0 = sum(row0); s1 = sum(row1); s2 = sum(row2)
s3 = sum(row3); s4 = sum(row4)

total = s0 + s1 + s2 + s3 + s4
total = s0 + s1 + s2 + s3 + s4
total = s0 + s1 + s2 + s3 + s4
total = s0 + s1 + s2 + s3 + s4

result = total * 1 + total * 1 + total * 1
result = total * 1 + total * 1 + total * 1
result = total * 1 + total * 1 + total * 1
"""),

    ("dp_lcs", """
def lcs(s1, s2):
    \"\"\"Longest Common Subsequence via DP. Returns the LCS string.\"\"\"
    m, n = len(s1), len(s2)
    dp   = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    result = []
    i, j   = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(result))

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
"""),

    ("matrix_ops", """
def mat_mul(A, B):
    \"\"\"Naive O(n^3) matrix multiplication.\"\"\"
    n, k, m = len(A), len(B), len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for p in range(k):
            if A[i][p] == 0:
                continue
            for j in range(m):
                C[i][j] += A[i][p] * B[p][j]
    return C

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_mul(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def trace(M):
    return sum(M[i][i] for i in range(min(len(M), len(M[0]))))
"""),

    ("decorator_cache", """
import functools
import time

def retry(max_attempts=3, delay=1.0, exceptions=(Exception,)):
    \"\"\"Retry a function up to max_attempts times on specified exceptions.\"\"\"
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))
            raise last_exc
        return wrapper
    return decorator

def memoize(func):
    \"\"\"Simple memoization using a dict keyed on positional args.\"\"\"
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    wrapper.cache_clear = cache.clear
    return wrapper

@memoize
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
"""),

    ("docstring_heavy", """
def process_data(records, key=None, reverse=False, limit=None):
    \"\"\"
    Process and sort a list of records.

    This function takes an iterable of records, optionally extracts a
    sort key from each record, sorts them in ascending or descending
    order, and returns up to `limit` records.

    Parameters
    ----------
    records : iterable
        The input records. Each element may be any comparable type,
        or a dict / object when `key` is specified.
    key : callable or None
        If provided, applied to each record to extract the sort key.
        Example: key=lambda r: r['score']
    reverse : bool, default False
        If True, sort in descending order.
    limit : int or None
        If provided, return only the first `limit` results after sorting.

    Returns
    -------
    list
        Sorted (and optionally limited) records.

    Raises
    ------
    TypeError
        If records are not comparable and no key is provided.

    Examples
    --------
    >>> process_data([3, 1, 2])
    [1, 2, 3]
    >>> process_data([{'v': 2}, {'v': 1}], key=lambda r: r['v'])
    [{'v': 1}, {'v': 2}]
    \"\"\"
    data = list(records)
    data.sort(key=key, reverse=reverse)
    if limit is not None:
        data = data[:limit]
    return data
"""),
]

# ── Multi-layer capture (single forward pass) ─────────────────────────────────

def _run_and_capture_all_layers(model, tokenizer, text, device, layer_indices):
    """
    Run ONE forward pass and simultaneously capture the importance-scoring
    tensors (c_kv, q_abs, W_UV, scale) at every layer in `layer_indices`.

    Returns
    -------
    captured : dict[int -> dict]
        {layer_idx: {"c_kv": Tensor, "q_abs": Tensor,
                     "W_UV": Tensor, "scale": float}}
        All tensors are CPU float32.
    n_tokens : int
    """
    captured      = {l: {} for l in layer_indices}
    orig_forwards = {}

    def _make_hook(attn_mod, layer_cap, orig_fwd):
        """Closure: captures one layer's tensors then calls the original forward."""
        def hooked(hidden_states, attention_mask=None, position_ids=None,
                   past_key_value=None, output_attentions=False,
                   use_cache=False, **kw):
            kw.pop("padding_mask", None)
            B, S, _ = hidden_states.shape
            with torch.no_grad():
                # ── Query ────────────────────────────────────────────────
                if attn_mod.q_lora_rank is None:
                    q = attn_mod.q_proj(hidden_states)
                else:
                    q = attn_mod.q_b_proj(
                        attn_mod.q_a_layernorm(
                            attn_mod.q_a_proj(hidden_states)))
                q = q.view(B, S, attn_mod.num_heads,
                           attn_mod.q_head_dim).transpose(1, 2)
                q_nope, _ = torch.split(
                    q,
                    [attn_mod.qk_nope_head_dim, attn_mod.qk_rope_head_dim],
                    dim=-1,
                )
                # ── KV latent ────────────────────────────────────────────
                raw = attn_mod.kv_a_proj_with_mqa(hidden_states)
                c_kv_raw, _ = torch.split(
                    raw,
                    [attn_mod.kv_lora_rank, attn_mod.qk_rope_head_dim],
                    dim=-1,
                )
                c_kv = attn_mod.kv_a_layernorm(c_kv_raw)   # [B, S, R]
                # ── Weight absorption ────────────────────────────────────
                W_kv = attn_mod.kv_b_proj.weight.view(
                    attn_mod.num_heads,
                    attn_mod.qk_nope_head_dim + attn_mod.v_head_dim,
                    attn_mod.kv_lora_rank,
                )
                W_UK = W_kv[:, : attn_mod.qk_nope_head_dim, :]
                W_UV = W_kv[:, attn_mod.qk_nope_head_dim :, :]
                q_abs = torch.einsum("bhqd,hdr->bhqr", q_nope, W_UK)

                layer_cap["c_kv"]  = c_kv.cpu().float()
                layer_cap["q_abs"] = q_abs.cpu().float()
                layer_cap["W_UV"]  = W_UV.cpu().float()
                layer_cap["scale"] = float(attn_mod.softmax_scale)

            return orig_fwd(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kw,
            )
        return hooked

    # Install hooks
    for l in layer_indices:
        attn = model.model.layers[l].self_attn
        orig_forwards[l] = attn.forward
        attn.forward = _make_hook(attn, captured[l], orig_forwards[l])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        n_tokens = int(ids.shape[1])
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        for l in layer_indices:
            model.model.layers[l].self_attn.forward = orig_forwards[l]

    return captured, n_tokens


# ── Importance scoring (mirrors production _compute_keep_indices) ─────────────

def _compute_importance(c_kv, q_abs, W_UV, scale, sample_pos=None):
    """
    Value-Aware × Query-Aware importance for mid tokens.

    Parameters
    ----------
    sample_pos : LongTensor or None
        If provided, use these fixed query indices instead of sampling
        randomly.  Pass the same tensor for all layers to eliminate
        Monte-Carlo variance from the correlation comparison.

    Returns
    -------
    importance : np.ndarray [n_mid]  or  None if n_mid <= 0
    n_sink, n_recent, n_mid : int
    """
    S      = c_kv.shape[1]
    n_sink   = min(SINK_TOKENS,   S)
    n_recent = min(RECENT_TOKENS, max(0, S - n_sink))
    n_mid    = S - n_sink - n_recent

    if n_mid <= 0:
        return None, n_sink, n_recent, 0

    mid_latent = c_kv[:, n_sink : S - n_recent, :]          # [1, n_mid, R]
    q_mid      = q_abs[:, :, n_sink : S - n_recent, :]       # [1, H, n_mid, R]

    # ── Column-sum (query-aware) ──────────────────────────────────────────
    K = min(SCORE_QUERIES, n_mid)
    if sample_pos is not None:
        # Use the caller-supplied fixed subsample
        q_scored    = q_mid[:, :, sample_pos, :]
        causal_mask = (
            torch.arange(n_mid).unsqueeze(0) > sample_pos.unsqueeze(1)
        )
    elif K < n_mid:
        sample_pos  = torch.randperm(n_mid)[:K].sort().values
        q_scored    = q_mid[:, :, sample_pos, :]
        causal_mask = (
            torch.arange(n_mid).unsqueeze(0) > sample_pos.unsqueeze(1)
        )
    else:
        q_scored    = q_mid
        causal_mask = ~torch.tril(
            torch.ones(n_mid, n_mid, dtype=torch.bool))

    logits = torch.matmul(q_scored,
                          mid_latent.transpose(-1, -2)) * scale
    logits = logits.masked_fill(causal_mask[None, None], float("-inf"))
    probs   = torch.softmax(logits, dim=-1, dtype=torch.float32)
    col_sum = probs.sum(dim=2).mean(dim=1)                    # [1, n_mid]

    # ── Value-aware weighting ─────────────────────────────────────────────
    val_proj = torch.einsum("bnr,hdr->bhnd",
                            mid_latent, W_UV)                 # [1,H,n_mid,d]
    val_norm = val_proj.norm(dim=-1).mean(dim=1)              # [1, n_mid]
    v_mean   = val_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    val_norm = val_norm / v_mean

    importance = (col_sum * val_norm).squeeze(0)              # [n_mid]
    return importance.numpy(), n_sink, n_recent, n_mid


# ── Collect scores at all committee layers for one sequence ───────────────────

def collect_all_layer_scores(model, tokenizer, text, device):
    """
    Run one forward pass and return per-layer importance scores.

    Returns
    -------
    dict[int -> np.ndarray[n_mid]]  or  None on failure
    also returns n_tokens (int)
    """
    captured, n_tokens = _run_and_capture_all_layers(
        model, tokenizer, text, device, COMMITTEE_LAYERS)

    # Choose a fixed query subsample shared across all layers to eliminate
    # Monte-Carlo variance from the cross-layer correlation.
    # Compute n_mid from the first successfully captured layer.
    first_cap = None
    for l in COMMITTEE_LAYERS:
        if captured[l]:
            first_cap = captured[l]
            break
    if first_cap is None:
        return None, n_tokens

    S        = first_cap["c_kv"].shape[1]
    n_sink   = min(SINK_TOKENS,   S)
    n_recent = min(RECENT_TOKENS, max(0, S - n_sink))
    n_mid    = S - n_sink - n_recent
    if n_mid <= 0:
        return None, n_tokens

    K = min(SCORE_QUERIES, n_mid)
    shared_sample = (torch.randperm(n_mid)[:K].sort().values
                     if K < n_mid else None)

    layer_scores = {}
    for l in COMMITTEE_LAYERS:
        cap = captured[l]
        if not cap:
            return None, n_tokens
        scores, _, _, nm = _compute_importance(
            cap["c_kv"], cap["q_abs"], cap["W_UV"], cap["scale"],
            sample_pos=shared_sample,
        )
        if scores is None or nm != n_mid:
            return None, n_tokens
        layer_scores[l] = scores

    return layer_scores, n_tokens


# ── Experiment 1: Spearman correlation + Jaccard keep-set overlap ─────────────

def run_experiment1(model, tokenizer, device):
    """
    Compute and visualise the cross-layer Spearman rank correlation matrix
    and the Jaccard@K keep-set overlap matrix.
    """
    n_l = len(COMMITTEE_LAYERS)
    spearman_acc = np.zeros((n_l, n_l))
    jaccard_acc  = np.zeros((n_l, n_l))
    n_valid      = 0

    print("\n" + "=" * 68)
    print("  Experiment 1: Cross-Layer Importance Rank Correlation")
    print("=" * 68)

    for label, text in TEST_CASES:
        print(f"  {label:<22} … ", end="", flush=True)
        result = collect_all_layer_scores(model, tokenizer, text, device)
        if result[0] is None:
            print("skipped")
            continue
        layer_scores, n_tokens = result
        n_mid = len(next(iter(layer_scores.values())))
        K_jac = max(1, int(round(n_mid * JACCARD_K_RATIO)))
        scores_list = [layer_scores[l] for l in COMMITTEE_LAYERS]

        for i in range(n_l):
            for j in range(n_l):
                rho, _ = stats.spearmanr(scores_list[i], scores_list[j])
                spearman_acc[i, j] += rho

                # Jaccard: top-K sets
                top_i = set(np.argsort(scores_list[i])[::-1][:K_jac])
                top_j = set(np.argsort(scores_list[j])[::-1][:K_jac])
                jac   = len(top_i & top_j) / len(top_i | top_j)
                jaccard_acc[i, j] += jac

        n_valid += 1
        print(f"n_mid={n_mid:4d}, n_tok={n_tokens:4d}  OK")

    if n_valid == 0:
        print("[ERROR] No valid sequences — cannot compute correlation.")
        return

    spearman_avg = spearman_acc / n_valid
    jaccard_avg  = jaccard_acc  / n_valid

    # ── Console table ─────────────────────────────────────────────────────
    labels = [f"L{l}" for l in COMMITTEE_LAYERS]
    w = 7
    print(f"\n  Spearman ρ (averaged over {n_valid} sequences):")
    print("        " + "".join(f"{lb:>{w}}" for lb in labels))
    for i, la in enumerate(labels):
        row = "  " + f"{la:<6}" + "".join(
            f"{spearman_avg[i,j]:>{w}.3f}" for j in range(n_l))
        print(row)

    print(f"\n  Jaccard@{int(JACCARD_K_RATIO*100)}% (averaged over {n_valid} sequences):")
    print("        " + "".join(f"{lb:>{w}}" for lb in labels))
    for i, la in enumerate(labels):
        row = "  " + f"{la:<6}" + "".join(
            f"{jaccard_avg[i,j]:>{w}.3f}" for j in range(n_l))
        print(row)

    # ── Plot heatmaps ─────────────────────────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap
    rg_cmap = LinearSegmentedColormap.from_list("rg", ["#d62728", "#2ca02c"])

    for matrix, title, fname, vmin in [
        (spearman_avg,
         f"Cross-Layer Spearman Rank Correlation\n"
         f"(averaged over {n_valid} code sequences)",
         "cross_layer_correlation.pdf", 0.0),
        (jaccard_avg,
         f"Cross-Layer Jaccard Keep-Set Overlap  "
         f"(top-{int(JACCARD_K_RATIO*100)}%)\n"
         f"(averaged over {n_valid} code sequences)",
         "cross_layer_jaccard.pdf", 0.0),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(matrix, vmin=vmin, vmax=1.0, cmap=rg_cmap,
                       aspect="equal")
        ax.set_xticks(range(n_l))
        ax.set_yticks(range(n_l))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=9, pad=8)
        fig.tight_layout()
        out = os.path.join(MODEL_DIR, fname)
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved → {fname}")

    return spearman_avg, jaccard_avg


# ── Experiment 3: Token importance trajectory visualisation ───────────────────

def _plot_trajectory(ax, layer_scores, label, n_tokens):
    """
    Draw one trajectory subplot on `ax`.

    For each layer, rank-normalises the importance scores (percentile rank
    in [0, 1]).  Selects the top-5, mid-5, and bottom-5 tokens by their
    rank at the first committee layer (L4), then plots each token's
    percentile rank across all committee layers.

    High-persistence → red (top) lines stay near 1, blue (bottom) lines
    stay near 0, with minimal crossings.
    """
    n_l    = len(COMMITTEE_LAYERS)
    n_mid  = len(layer_scores[COMMITTEE_LAYERS[0]])
    x_pos  = list(range(n_l))
    x_lbls = [f"L{l}" for l in COMMITTEE_LAYERS]

    # Percentile rank in [1/n, 1] for each layer
    pct = {}
    for l in COMMITTEE_LAYERS:
        pct[l] = stats.rankdata(layer_scores[l]) / n_mid

    # Select tokens by the first committee layer that is >= 4 (i.e. L4).
    # L0/L1/L2 are poor decision layers (cf. Table 1) and must not be used
    # as the selection reference, or mid-importance lines will be chaotic.
    ref_layer   = next((l for l in COMMITTEE_LAYERS if l >= 4), COMMITTEE_LAYERS[0])
    l4_pct      = pct[ref_layer]
    sorted_desc = np.argsort(l4_pct)[::-1]   # high → low

    n_sel = min(5, n_mid // 3)
    top_idx = sorted_desc[:n_sel]
    bot_idx = sorted_desc[-n_sel:]
    mid_start = max(0, (n_mid - n_sel) // 2)
    mid_idx   = sorted_desc[mid_start : mid_start + n_sel]

    cmap_r = plt.cm.Reds(np.linspace(0.5, 0.9, n_sel))
    cmap_b = plt.cm.Blues(np.linspace(0.5, 0.9, n_sel))
    cmap_g = plt.cm.Greys(np.linspace(0.45, 0.70, n_sel))

    for k in range(n_sel):
        y_top = [pct[l][top_idx[k]] for l in COMMITTEE_LAYERS]
        y_bot = [pct[l][bot_idx[k]] for l in COMMITTEE_LAYERS]
        y_mid = [pct[l][mid_idx[k]] for l in COMMITTEE_LAYERS]

        lbl_t = "High-importance" if k == 0 else "_"
        lbl_b = "Low-importance"  if k == 0 else "_"
        lbl_m = "Mid-importance"  if k == 0 else "_"

        ax.plot(x_pos, y_top, color=cmap_r[k], linewidth=1.8,
                marker="o", markersize=4, label=lbl_t)
        ax.plot(x_pos, y_bot, color=cmap_b[k], linewidth=1.8,
                marker="s", markersize=4, label=lbl_b)
        ax.plot(x_pos, y_mid, color=cmap_g[k], linewidth=1.2,
                linestyle="--", marker="^", markersize=3, label=lbl_m)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_lbls, fontsize=8)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Percentile rank (within layer)", fontsize=7.5)
    ax.set_xlabel("Committee layer", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)

    # Kendall τ between L4 (first useful layer) and all others.
    # Display mean and min to keep the title compact for 10-layer setups.
    ref_layer = next((l for l in COMMITTEE_LAYERS if l >= 4), COMMITTEE_LAYERS[0])
    ref_ranks = stats.rankdata(layer_scores[ref_layer])
    taus = [
        stats.kendalltau(ref_ranks, stats.rankdata(layer_scores[l])).statistic
        for l in COMMITTEE_LAYERS[1:]
    ]
    tau_detail = "  ".join(
        f"L{l}:{taus[k]:.2f}" for k, l in enumerate(COMMITTEE_LAYERS[1:])
    )
    ax.set_title(
        f"{label}  (n_tok={n_tokens}, n_mid={n_mid})\n"
        f"Kendall \u03c4 vs L{ref_layer} (first useful layer) — "
        f"mean {np.mean(taus):.2f}, min {np.min(taus):.2f}  [{tau_detail}]",
        fontsize=6.5, pad=4,
    )
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.7)


def run_experiment3(model, tokenizer, device):
    """
    For each sequence in TRAJECTORY_LABELS, collect per-layer scores and
    draw the token importance trajectory plot.
    """
    print("\n" + "=" * 68)
    print("  Experiment 3: Token Importance Trajectory Visualisation")
    print("=" * 68)

    records = []   # (label, layer_scores, n_tokens)
    for label, text in TEST_CASES:
        if label not in TRAJECTORY_LABELS:
            continue
        print(f"  {label:<22} … ", end="", flush=True)
        result = collect_all_layer_scores(model, tokenizer, text, device)
        if result[0] is None:
            print("skipped")
            continue
        layer_scores, n_tokens = result
        records.append((label, layer_scores, n_tokens))
        print(f"n_mid={len(layer_scores[COMMITTEE_LAYERS[0]]):4d}  OK")

        # Individual PDF
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_trajectory(ax, layer_scores, label, n_tokens)
        fig.suptitle(
            "Token Importance Trajectory across Committee Layers\n"
            "Red = high-importance (by L4), Blue = low, Grey = mid",
            fontsize=8.5,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(MODEL_DIR, f"trajectory_{label}.pdf")
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"              Saved → trajectory_{label}.pdf")

    if not records:
        print("[ERROR] No trajectory sequences collected.")
        return

    # Combined 3-panel PDF
    n_rec = len(records)
    fig, axes = plt.subplots(1, n_rec, figsize=(6.5 * n_rec, 5.0))
    if n_rec == 1:
        axes = [axes]
    for ax, (label, layer_scores, n_tokens) in zip(axes, records):
        _plot_trajectory(ax, layer_scores, label, n_tokens)

    ref_l_label = next((l for l in COMMITTEE_LAYERS if l >= 4), COMMITTEE_LAYERS[0])
    fig.suptitle(
        f"Token Importance Trajectories across Committee Layers "
        f"(token groups selected by L{ref_l_label} rank: Red=top, Blue=bottom, Grey=mid)\n"
        "Stable, non-crossing red/blue bands indicate cross-layer persistence.",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(MODEL_DIR, "trajectory_combined.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → trajectory_combined.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "═" * 68)
    print("  Cross-Layer Importance Persistence Analysis")
    print(f"  Committee layers : {COMMITTEE_LAYERS}")
    print(f"  Device           : {device}")
    print(f"  Score queries K  : {SCORE_QUERIES}")
    print(f"  Sink / Recent    : {SINK_TOKENS} / {RECENT_TOKENS}")
    print("═" * 68)

    print("\n[ Loading tokenizer … ]")
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

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    # ── Run experiments ───────────────────────────────────────────────────
    run_experiment1(model, tok, device)
    run_experiment3(model, tok, device)

    print("\n" + "═" * 68)
    print("  Output files:")
    for fname in [
        "cross_layer_correlation.pdf",
        "cross_layer_jaccard.pdf",
        "trajectory_quicksort.pdf",
        "trajectory_binary_search.pdf",
        "trajectory_repetitive_code.pdf",
        "trajectory_combined.pdf",
    ]:
        path = os.path.join(MODEL_DIR, fname)
        status = "✓" if os.path.exists(path) else "✗ (not produced)"
        print(f"  {status}  {fname}")
    print("═" * 68)


if __name__ == "__main__":
    main()
