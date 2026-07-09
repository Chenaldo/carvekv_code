#!/usr/bin/env python3
"""
fix_modeling.py  --  Restore DeepseekV2Attention.forward as a proper class method.

Problem
-------
compute_latent_info_score() is a 0-indent (module-level) function that was
inserted into the middle of DeepseekV2Attention's method definitions.
Python ends the class at that point, so _compute_keep_indices and forward
end up as DEAD CODE inside the function (4-indent after "return info").
Result: DeepseekV2Attention has no forward() -> _forward_unimplemented() crash.

Fix
---
Move the dead-code block (everything after "return info" inside the function,
up to the next 0-indent line) to BEFORE compute_latent_info_score.
Those methods then sit at 4-indent inside the class body -- correctly.

Usage
-----
    python3 fix_modeling.py                        # default path
    python3 fix_modeling.py /path/to/modeling.py   # explicit path

After the fix, clear the HuggingFace module cache and re-run:
    rm -rf ~/.cache/huggingface/modules/transformers_modules/
    python final_run.py --quick
"""

import ast
import re
import shutil
import sys
from pathlib import Path

# ── Path resolution ───────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    PATH = Path(sys.argv[1])
else:
    PATH = Path(__file__).parent / "modeling_deepseek.py"

if not PATH.exists():
    sys.exit(f"[ERROR] File not found: {PATH}")

print(f"Target: {PATH}")

# ── Helper: check DeepseekV2Attention methods via AST ─────────────────────────
def get_attn_methods(src: str):
    tree = ast.parse(src)
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "DeepseekV2Attention":
            return [
                n.name for n in cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    return None

# ── Load source ───────────────────────────────────────────────────────────────
src = PATH.read_text(encoding="utf-8")

methods_before = get_attn_methods(src)
if methods_before is None:
    sys.exit("[ERROR] DeepseekV2Attention class not found in the file.")

print(f"DeepseekV2Attention methods (before): {methods_before}")

if "forward" in methods_before:
    print("[OK] forward is already a proper class method. No fix needed.")
    sys.exit(0)

print("[!] forward is NOT a class method. Applying fix ...")

# ── Locate compute_latent_info_score (must be at column 0) ───────────────────
fn_m = re.search(r"^def compute_latent_info_score\(", src, re.MULTILINE)
if not fn_m:
    sys.exit("[ERROR] compute_latent_info_score not found at module level.")

fn_start = fn_m.start()
print(f"compute_latent_info_score starts at char {fn_start} "
      f"(line ~{src[:fn_start].count(chr(10)) + 1})")

# ── Locate "    return info" (the real last statement of the function) ────────
# Search only within the function body (from fn_start onward).
return_m = re.search(r"^    return info", src[fn_start:], re.MULTILINE)
if not return_m:
    sys.exit("[ERROR] 'return info' not found inside compute_latent_info_score.")

# Position just after the newline that terminates the return line.
return_line_end = fn_start + return_m.end()
# Consume the trailing newline if present.
if return_line_end < len(src) and src[return_line_end] == "\n":
    return_line_end += 1

print(f"'return info' ends at char {return_line_end} "
      f"(line ~{src[:return_line_end].count(chr(10)) + 1})")

# ── Find the end of the dead-code section (next 0-indent non-blank line) ──────
after_return = src[return_line_end:]
next0_m = re.search(r"^[^\s\n]", after_return, re.MULTILINE)

if next0_m:
    dead_end = return_line_end + next0_m.start()
    dead_code = src[return_line_end:dead_end]
    print(f"Dead-code section: {len(dead_code.splitlines())} lines "
          f"(chars {return_line_end}-{dead_end})")
else:
    # No 0-indent line found: all remaining text is dead code.
    dead_end = len(src)
    dead_code = after_return
    print(f"Dead-code section runs to end of file: {len(dead_code.splitlines())} lines")

if not dead_code.strip():
    sys.exit("[ERROR] Dead-code section is empty — nothing to move.")

# Show first few lines for verification.
print("First 4 lines of dead-code block:")
for line in dead_code.splitlines()[:4]:
    print(f"    {line!r}")

# ── Build new source ──────────────────────────────────────────────────────────
# Layout:
#   [everything before compute_latent_info_score]
#   + [dead_code block]           <- moves here (still 4-indent, now inside class)
#   + [compute_latent_info_score up to and including return info\n]
#   + [rest of file after dead_code]
before_fn  = src[:fn_start]
the_fn     = src[fn_start:return_line_end]   # real function body + return line
after_dead = src[dead_end:]                  # rest of file

new_src = before_fn + dead_code + the_fn + after_dead

# ── Syntax check ──────────────────────────────────────────────────────────────
try:
    ast.parse(new_src)
    print("Syntax check: PASSED")
except SyntaxError as exc:
    print(f"Syntax check FAILED: {exc}")
    print("Not saving. Please inspect the file manually.")
    sys.exit(1)

# ── Verify forward is now a class method ─────────────────────────────────────
methods_after = get_attn_methods(new_src)
print(f"DeepseekV2Attention methods (after):  {methods_after}")

if "forward" not in (methods_after or []):
    print("[ERROR] Verification failed: forward is still not a class method.")
    print("The file structure may be more complex. Manual fix required.")
    sys.exit(1)

# ── Save (with backup) ────────────────────────────────────────────────────────
backup = PATH.with_suffix(".py.bak")
shutil.copy2(PATH, backup)
print(f"Backup saved to: {backup}")

PATH.write_text(new_src, encoding="utf-8")
print(f"[OK] Fixed and saved to: {PATH}")

print()
print("Next steps:")
print("  1. rm -rf ~/.cache/huggingface/modules/transformers_modules/")
print("  2. python final_run.py --quick")
