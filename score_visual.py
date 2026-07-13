"""
score_visual.py — Query-Aware Importance Score Distribution Visualizer
======================================================================

Runs 10 diverse test sequences through the model, captures the
Value-Aware × Query-Aware importance scores at decision layer D=6
(same formula as _compute_keep_indices production path), and plots:

  score_distributions.png  — 2×5 grid: sorted score curve + cumulative
                             mass curve per test, with 70/80/90% coverage
                             markers
  score_overlay.png        — all 10 curves overlaid on a normalised axis
                             for cross-test comparison

Reading the plots
-----------------
  Blue line  : importance score sorted descending (normalised to [0,1])
  Orange dash: cumulative mass (fraction of total importance covered)
  Vertical dotted lines: keep-ratio required for 70 / 80 / 90 % coverage

  Steep blue curve → importance is concentrated in a few tokens → can
  evict aggressively with little loss.
  Flat blue curve  → importance is spread evenly → must keep most tokens.
"""

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")                          # headless / file output
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
DECISION_LAYER = 6
SCORE_QUERIES  = 512          # subsample K queries for OOM safety
SINK_TOKENS    = 4
RECENT_TOKENS  = 16
DTYPE          = torch.bfloat16
COVERAGE_MARKS = [0.70, 0.80, 0.90]
MARK_COLORS    = ["#2ca02c", "#ff7f0e", "#d62728"]   # green / orange / red

# ── 10 diverse test cases ─────────────────────────────────────────────────────
# Chosen to span different importance-concentration profiles:
#   repetitive_code / repeat_comment  → expected FLAT distribution
#   complex algorithms / decorators   → expected PEAKED distribution
TEST_CASES = [
    # 1 ── simple utilities (short, diverse content)
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

    # 2 ── binary search (classic, medium)
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

    # 3 ── quicksort (recursive, medium-long)
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

    # 4 ── BFS / shortest path (graph traversal)
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

    # 5 ── Stack class (OOP, multiple methods)
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

    # 6 ── REPETITIVE code (expected FLAT distribution)
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

    # 7 ── Dynamic programming LCS (algorithmic, with backtrack)
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

    # Backtrack
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

    # 8 ── Matrix operations (numeric, dense)
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

    # 9 ── Decorator + memoize (metaprogramming)
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

    # 10 ── Long docstring-heavy code (many comment tokens, few logic tokens)
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

# ── Score capture via forward hook ───────────────────────────────────────────

def _run_and_capture(model, tokenizer, text, device):
    """
    Forward-pass `text` through the model.  At DECISION_LAYER, captures:
        c_kv  : [1, S, kv_lora_rank]  normalised latent
        q_abs : [1, H, S, kv_lora_rank]  absorbed query
        W_UV  : [H, v_head_dim, kv_lora_rank]  Value projection
        scale : float  softmax scale factor

    Returns (c_kv, q_abs, W_UV, scale, n_tokens) all on CPU fp32,
    or (None, …, n_tokens) if the sequence is too short.
    """
    attn     = model.model.layers[DECISION_LAYER].self_attn
    orig_fwd = attn.forward
    captured: dict = {}

    def hooked_forward(hidden_states, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False,
                       use_cache=False, **kw):
        kw.pop("padding_mask", None)
        B, S, _ = hidden_states.shape

        with torch.no_grad():
            # Query
            if attn.q_lora_rank is None:
                q = attn.q_proj(hidden_states)
            else:
                q = attn.q_b_proj(
                    attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
            q = q.view(B, S, attn.num_heads, attn.q_head_dim).transpose(1, 2)
            q_nope, _ = torch.split(
                q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

            # KV latent
            raw = attn.kv_a_proj_with_mqa(hidden_states)
            c_kv_raw, _ = torch.split(
                raw, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
            c_kv = attn.kv_a_layernorm(c_kv_raw)              # [B, S, R]

            # Weight absorption
            W_kv = attn.kv_b_proj.weight.view(
                attn.num_heads,
                attn.qk_nope_head_dim + attn.v_head_dim,
                attn.kv_lora_rank,
            )
            W_UK = W_kv[:, : attn.qk_nope_head_dim, :]
            W_UV = W_kv[:, attn.qk_nope_head_dim :, :]
            q_abs = torch.einsum("bhqd,hdr->bhqr", q_nope, W_UK)

            captured["c_kv"]  = c_kv.cpu().float()
            captured["q_abs"] = q_abs.cpu().float()
            captured["W_UV"]  = W_UV.cpu().float()
            captured["scale"] = float(attn.softmax_scale)

        return orig_fwd(
            hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache, **kw,
        )

    attn.forward = hooked_forward
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        n_tokens = int(ids.shape[1])
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        attn.forward = orig_fwd

    if not captured:
        return None, None, None, None, n_tokens
    return (captured["c_kv"], captured["q_abs"],
            captured["W_UV"], captured["scale"], n_tokens)


def _compute_importance(c_kv, q_abs, W_UV, scale):
    """
    Value-Aware × Query-Aware importance for mid tokens.

    Mirrors _compute_keep_indices production formula:
        col_sum[j] = Σ_i prob_{i,j}   (causal attention column sum)
        val_norm[j] = mean_H(‖W_UV @ c_j‖₂)
        importance  = col_sum × val_norm

    Returns raw scores [n_mid] (numpy float32) and region metadata.
    """
    S = c_kv.shape[1]
    n_sink   = min(SINK_TOKENS,   S)
    n_recent = min(RECENT_TOKENS, max(0, S - n_sink))
    n_mid    = S - n_sink - n_recent

    if n_mid <= 0:
        return None, n_sink, n_recent, 0

    mid_latent = c_kv[:, n_sink : S - n_recent, :]          # [1, n_mid, R]
    q_mid      = q_abs[:, :, n_sink : S - n_recent, :]       # [1, H, n_mid, R]

    # ── Column-sum (query-aware) ──────────────────────────────────────────
    K = min(SCORE_QUERIES, n_mid)
    if K < n_mid:
        sample_pos  = torch.randperm(n_mid)[:K].sort().values
        q_scored    = q_mid[:, :, sample_pos, :]
        causal_mask = (
            torch.arange(n_mid).unsqueeze(0) > sample_pos.unsqueeze(1)
        )                                                     # [K, n_mid]
    else:
        q_scored    = q_mid
        causal_mask = ~torch.tril(
            torch.ones(n_mid, n_mid, dtype=torch.bool))      # [n_mid, n_mid]

    logits = torch.matmul(q_scored,
                          mid_latent.transpose(-1, -2)) * scale
    logits = logits.masked_fill(causal_mask[None, None], float("-inf"))
    probs   = torch.softmax(logits, dim=-1, dtype=torch.float32)
    col_sum = probs.sum(dim=2).mean(dim=1)                    # [1, n_mid]

    # ── Value-aware weighting ─────────────────────────────────────────────
    val_proj = torch.einsum("bnr,hdr->bhnd",
                            mid_latent, W_UV)                 # [1, H, n_mid, d]
    val_norm = val_proj.norm(dim=-1).mean(dim=1)              # [1, n_mid]
    v_mean   = val_norm.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    val_norm = val_norm / v_mean

    importance = (col_sum * val_norm).squeeze(0)              # [n_mid]
    return importance.numpy(), n_sink, n_recent, n_mid


# ── Plotting ─────────────────────────────────────────────────────────────────

def _build_record(label, n_tokens, scores_raw, n_mid):
    """
    Given raw importance scores [n_mid], build a plotting record with:
      scores_norm  : sorted descending, normalised to [0,1]
      cumsum       : cumulative mass fraction (shifted to non-negative first)
      keep_ratio   : {cov: ratio} for each coverage mark
    """
    scores_desc = np.sort(scores_raw)[::-1].copy()

    s_min, s_max = scores_desc.min(), scores_desc.max()
    scores_norm  = (scores_desc - s_min) / max(s_max - s_min, 1e-8)

    # Cumulative mass (shift to ≥0 before normalising)
    shifted = scores_desc - scores_desc.min()
    total   = shifted.sum()
    cumsum  = np.cumsum(shifted) / max(total, 1e-8)

    keep_ratio = {}
    for cov in COVERAGE_MARKS:
        idx = int(np.searchsorted(cumsum, cov))
        idx = min(idx, n_mid - 1)
        keep_ratio[cov] = (idx + 1) / n_mid

    return dict(
        label       = label,
        n_tokens    = n_tokens,
        n_mid       = n_mid,
        scores_norm = scores_norm,
        cumsum      = cumsum,
        keep_ratio  = keep_ratio,
    )


def _subplot_test(ax, r, idx):
    """Draw one test subplot (sorted score + cumulative mass + coverage marks)."""
    scores = r["scores_norm"]
    cumsum = r["cumsum"]
    x      = np.linspace(0, 1, len(scores))

    # Score curve (left axis)
    ax.fill_between(x, scores, alpha=0.15, color="#1f77b4")
    ax.plot(x, scores, color="#1f77b4", linewidth=1.4, label="score")
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("importance (norm)", fontsize=6.5, color="#1f77b4")
    ax.tick_params(axis="y", labelsize=6, colors="#1f77b4")
    ax.tick_params(axis="x", labelsize=6)

    # Cumulative mass (right axis)
    ax2 = ax.twinx()
    ax2.plot(x, cumsum, color="#ff7f0e", linewidth=1.4,
             linestyle="--", label="cum. mass")
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("cum. mass", fontsize=6.5, color="#ff7f0e")
    ax2.tick_params(axis="y", labelsize=6, colors="#ff7f0e")

    # Coverage markers
    cov_lines = []
    for cov, col in zip(COVERAGE_MARKS, MARK_COLORS):
        kr  = r["keep_ratio"][cov]
        ax.axvline(kr, color=col, linewidth=0.9, linestyle=":")
        ax2.axhline(cov, color=col, linewidth=0.5, linestyle=":", alpha=0.6)
        cov_lines.append(f"{int(cov*100)}%→{kr*100:.0f}%keep")

    ax.set_xlim(0, 1)
    ax.set_xlabel("rank / n_mid →", fontsize=6)
    ax.set_title(
        f"{r['label']}  (n_tok={r['n_tokens']}, n_mid={r['n_mid']})\n"
        + "  ".join(cov_lines),
        fontsize=6.8, pad=3,
    )


def plot_grid(results):
    """2×5 grid, one subplot per test."""
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 9))
    axes = axes.flatten()

    for i, r in enumerate(results):
        _subplot_test(axes[i], r, i)

    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Query-Aware (Value×Attention) Importance Score Distribution — sorted descending\n"
        "Blue fill = normalised score  │  Orange dashed = cumulative mass  "
        "│  Dotted verticals = keep-ratio for 70% / 80% / 90% coverage",
        fontsize=9, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(MODEL_DIR, "score_distributions.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → score_distributions.pdf")


def plot_overlay(results):
    """Side-by-side overlay: sorted scores (left) + cumulative mass (right)."""
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap("tab10")

    for i, r in enumerate(results):
        col   = cmap(i % 10)
        x     = np.linspace(0, 1, len(r["scores_norm"]))
        label = f"{r['label']} (n_mid={r['n_mid']})"
        ax_s.plot(x, r["scores_norm"], color=col, linewidth=1.3,
                  alpha=0.85, label=label)
        ax_c.plot(x, r["cumsum"],      color=col, linewidth=1.3,
                  alpha=0.85, label=label)

    # Reference lines on cumulative plot
    for cov, col in zip(COVERAGE_MARKS, MARK_COLORS):
        ax_c.axhline(cov, color=col, linewidth=1.1, linestyle="--",
                     label=f"{int(cov*100)}% coverage")

    for ax, title, ylabel in [
        (ax_s,
         "Sorted Importance Scores — all tests\n(x = relative rank, 0 = most important)",
         "Normalised importance"),
        (ax_c,
         "Cumulative Importance Mass — all tests\n(x = fraction of tokens kept)",
         "Fraction of total importance covered"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Relative token rank (0 → 1)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="upper right" if ax is ax_s else "lower right")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    ax_s.set_ylim(-0.05, 1.05)
    ax_c.set_ylim(0, 1.05)

    fig.suptitle(
        "Steep curve = concentrated importance (evict aggressively)\n"
        "Flat curve  = spread importance (evict conservatively)",
        fontsize=9,
    )
    fig.tight_layout()
    out = os.path.join(MODEL_DIR, "score_overlay.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → score_overlay.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "═" * 68)
    print("  Importance Score Visualizer")
    print(f"  Decision layer : D={DECISION_LAYER}  │  Device: {device}")
    print(f"  Score queries  : K={SCORE_QUERIES}   │  Sink={SINK_TOKENS}, Recent={RECENT_TOKENS}")
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
    print()

    results = []
    for label, text in TEST_CASES:
        print(f"  [{len(results)+1:02d}/10] {label:<20} … ", end="", flush=True)

        c_kv, q_abs, W_UV, scale, n_tokens = _run_and_capture(
            model, tok, text, device)

        if c_kv is None:
            print(f"skipped (n_tokens={n_tokens})")
            continue

        scores_raw, n_sink, n_recent, n_mid = _compute_importance(
            c_kv, q_abs, W_UV, scale)

        if scores_raw is None:
            print(f"skipped (n_mid=0, n_tokens={n_tokens})")
            continue

        r = _build_record(label, n_tokens, scores_raw, n_mid)
        results.append(r)

        kr_str = "  │  ".join(
            f"cov{int(c*100)}%→keep{r['keep_ratio'][c]*100:.0f}%"
            for c in COVERAGE_MARKS
        )
        print(f"n_tokens={n_tokens:4d}, n_mid={n_mid:4d}  │  {kr_str}")

    if not results:
        print("\n[ERROR] No results collected.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print(f"  {'Test':<22}  {'n_mid':>6}  "
          f"{'70% cov':>8}  {'80% cov':>8}  {'90% cov':>8}")
    print("─" * 68)
    for r in results:
        print(
            f"  {r['label']:<22}  {r['n_mid']:>6}  "
            f"{r['keep_ratio'][0.70]*100:>7.1f}%  "
            f"{r['keep_ratio'][0.80]*100:>7.1f}%  "
            f"{r['keep_ratio'][0.90]*100:>7.1f}%"
        )
    print("═" * 68)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[ Plotting {len(results)} tests … ]")
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass   # older matplotlib — use default

    plot_grid(results)
    plot_overlay(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
