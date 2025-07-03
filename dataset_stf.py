from __future__ import annotations

"""dataset_stf.py
Colab-friendly generator for a *self-teaching fine-tune* (STF) corpus of 2×2
matrix-multiplication programs written in the **basic DSL** that is understood
by ``dsl_executor.DSLExecutor``.

Differences vs the original generator:
1.  No ``+=``/``-=`` accumulator statements – every variable (including C
    elements) is assigned **exactly once** using the ``VAR = expr`` syntax that
    the executor supports.
2.  No nested parentheses or 3-way products such as ``k * (a * b)``.  All
    multiplications are *binary* (``lhs * rhs``) which the executor can parse.
3.  Optional dependencies are limited to the Python standard library; if
    ``tqdm`` is present we display a progress bar, otherwise we fall back to a
    simple loop counter.

The module exposes a single public helper:

    generate_dataset(samples: int = 25_000,
                     max_mul: int = 12,
                     out_path: str | pathlib.Path = "dsl_2x2_stf.jsonl",
                     seed: int | None = 2025) -> None

Running the file directly will stream-write the JSONL file with default
parameters, making it easy to invoke from a Colab cell.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

# tqdm is *optional* – we import lazily.
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover – tqdm is not a hard dependency
    tqdm = None  # type: ignore

PROMPT = (
    "Generate a DSL script that multiplies two 2×2 matrices. Return only the code "
    "inside <DSL> ... </DSL> tags."
)

# ----------------------------------------------------------------------------
# Helper functions for building individual DSL lines
# ----------------------------------------------------------------------------

def _rand_idx() -> Tuple[int, int]:
    """Return a random 2-D index (row, col) where each ∈ {0,1}."""
    return random.randrange(2), random.randrange(2)


def _tensor_element(tensor: str) -> str:
    """Random matrix element reference like ``A[0,1]`` or ``B[1,0]``."""
    r, c = _rand_idx()
    return f"{tensor}[{r},{c}]"


def _input_combo(var_name: str, tensor: str) -> str:
    """Return a random *addition/subtraction* combination of two elements from *tensor*."""
    (r1, c1), (r2, c2) = _rand_idx(), _rand_idx()
    op = random.choice(["+", "-"])
    return f"{var_name} = {tensor}[{r1},{c1}] {op} {tensor}[{r2},{c2}]"


def _mul_line(lhs: str, rhs: str, out_name: str) -> str:
    """Return a *binary* multiplication line ``out = lhs * rhs``.

    NOTE: The DSL executor can only parse a single ``*`` per expression, so we
    never emit chained multiplications or parentheses here.
    """
    return f"{out_name} = {lhs} * {rhs}"


def _fma_line(a1: str, b1: str, a2: str, b2: str, out_name: str) -> str:
    """Return a fused multiply-add style line ``out = a1 * b1 + a2 * b2``."""
    return f"{out_name} = {a1} * {b1} + {a2} * {b2}"


def _sum_line(v: str, w: str, out_name: str) -> str:
    op = random.choice(["+", "-"])
    return f"{out_name} = {v} {op} {w}"

# ----------------------------------------------------------------------------
# Single-program builder – *must* be compatible with DSLExecutor
# ----------------------------------------------------------------------------

def _build_program(max_mul: int) -> str:
    """Construct and return one DSL program wrapped in <DSL> tags."""

    lines: List[str] = []
    defined: List[str] = []  # track names that can be re-used in subsequent lines

    # Optional pre-compute additive combos from inputs
    for i in range(random.randint(0, 3)):
        var = f"T{i+1}"
        lines.append(_input_combo(var, "A"))
        defined.append(var)
    for j in range(random.randint(0, 3)):
        var = f"U{j+1}"
        lines.append(_input_combo(var, "B"))
        defined.append(var)

    # Ensure we always have the direct matrix elements available
    base_a_elems = [f"A[{r},{c}]" for r in range(2) for c in range(2)]
    base_b_elems = [f"B[{r},{c}]" for r in range(2) for c in range(2)]

    # Generate multiplication or FMA lines
    mul_count = random.randint(2, max_mul)
    for k in range(mul_count):
        out_var = f"M{k+1}"
        if random.random() < 0.2 and len(defined) >= 2:  # emit an FMA roughly 20% of the time
            a1, a2 = random.sample(defined, 2)
            b1 = random.choice(base_b_elems + defined)
            b2 = random.choice(base_b_elems + defined)
            lines.append(_fma_line(a1, b1, a2, b2, out_var))
        else:
            lhs_candidates = defined + base_a_elems
            rhs_candidates = defined + base_b_elems
            lhs = random.choice(lhs_candidates)
            rhs = random.choice(rhs_candidates)
            lines.append(_mul_line(lhs, rhs, out_var))
        defined.append(out_var)

    # Optional pure sum/diff helper vars
    num_sum_helpers = random.randint(0, 3)
    for s in range(num_sum_helpers):
        # Need at least two variables available to form a sum/difference
        if len(defined) < 2:
            break
        v, w = random.sample(defined, 2)
        svar = f"S{s+1}"
        lines.append(_sum_line(v, w, svar))
        defined.append(svar)

    # Now produce the four C assignments – each as a sum/diff of up to 3 terms
    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if not defined:
            # Fallback: use a direct matrix element if something went very wrong (extremely unlikely)
            terms = [f"A[{r},{c}]"]
        else:
            # Choose between 1-3 unique terms but never exceed the size of *defined*
            max_terms = min(3, len(defined))
            min_terms = 1 if len(defined) == 1 else 2
            k_terms = random.randint(min_terms, max_terms)
            terms = random.sample(defined, k=k_terms)

        expr = terms[0]
        for t in terms[1:]:
            op = random.choice(["+", "-"])
            expr = f"{expr} {op} {t}"
        lines.append(f"C[{r},{c}] = {expr}")

    return "<DSL>\n" + "\n".join(lines) + "\n</DSL>"

# ----------------------------------------------------------------------------
# Public entry point – works in scripts and notebooks alike
# ----------------------------------------------------------------------------

def generate_dataset(
    samples: int = 25_000,
    max_mul: int = 12,
    out_path: str | Path = "dsl_2x2_stf.jsonl",
    seed: Optional[int] = 2025,
) -> None:
    """Generate *samples* DSL programs and write them to *out_path* in JSONL.

    Each JSON object has the keys:
    - ``prompt``: The natural-language prompt.
    - ``completion``: The DSL program.
    """

    if seed is not None:
        random.seed(seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(range(samples), desc="Generating") if tqdm else range(samples)

    with out_path.open("w", encoding="utf-8") as fh:
        for _ in iterator:
            record = {
                "prompt": PROMPT,
                "completion": _build_program(max_mul),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ {samples:,} DSL blocks written → {out_path}")

# ----------------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – allows quick CLI usage
    generate_dataset() 