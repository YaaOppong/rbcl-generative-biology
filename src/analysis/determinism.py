"""Check that two generation runs agree on the prompts they share.

Replaces the `repro_check` that lived in scripts/run_partA_protocol.py. That one
regenerated base rows and compared them against Part A's stored 1500 nt corpus,
to license REUSING those stored rows as Part B's control. The premise is gone:
the base arm is now regenerated in every run at 1800 nt, so a comparison against
the stored 1500 nt rows would fail by construction rather than tell you anything.

The half worth keeping is the question underneath it -- given the same prompt and
the same seed, does a separate job produce the same sequence? Everything that
compares arms generated in different jobs depends on the answer being yes.

It is not guaranteed a priori. Batched autoregressive decoding is not
bit-identical across batch composition, so a level or replicate subset that
happened to land on different batch boundaries would diverge. src.generate.runner
groups batches by token budget and each level x replicate block is exactly 120
prompts = 30 whole batches, so subsets align -- but that is a property of this
prompt corpus, not a law, and a regenerated corpus could silently break it.

Usage:
    python -m src.analysis.determinism \
        results/generate_a/<run>/part_a/generated_base.csv \
        results/generate_b/<run>/part_b/generated_base.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compare(a: Path, b: Path, key: str = "prompt_id", col: str = "seq_nt") -> dict:
    """Compare two generation CSVs on the rows they have in common."""
    da = pd.read_csv(a).set_index(key)
    db = pd.read_csv(b).set_index(key)
    shared = da.index.intersection(db.index)
    if len(shared) == 0:
        return {"shared": 0, "identical": None,
                "note": "no prompts in common -- different levels or replicates"}
    left, right = da.loc[shared, col], db.loc[shared, col]
    same = (left == right)
    differing = [str(i) for i in shared[~same]][:5]
    return {"shared": len(shared), "identical": int(same.sum()),
            "differing": int((~same).sum()), "examples": differing,
            "byte_identical": bool(same.all())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left", type=Path)
    ap.add_argument("right", type=Path)
    ap.add_argument("--column", default="seq_nt")
    args = ap.parse_args()
    r = compare(args.left, args.right, col=args.column)
    for k, v in r.items():
        print(f"  {k}: {v}")
    if r.get("byte_identical") is False:
        raise SystemExit("runs disagree on shared prompts -- cross-run comparison "
                         "of arms is not licensed until this is understood")


if __name__ == "__main__":
    main()
