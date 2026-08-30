"""Active-site integrity of Part A's generations, per sequence and per residue.

Rebuilds the two tables deleted on 2026-08-30 as unreproducible. Everything this
needs is now committed: the generated corpus, and data/spinacia_P00875.fasta
(UniProt P00875, fetched 2026-08-30). Scoring is src.eval.active_site, which
maps each translation onto the reference by ALIGNMENT rather than by indexing --
a single indel would otherwise report catalytic loss where there is only a frame
offset.

TWO POPULATIONS, and conflating them is the error this module exists to avoid:

  scored     translation >= 100 aa. Long enough for the alignment to mean
             something; reported per sequence.
  scorable   translation reaches residue 380, the last of the 11. Only these
             can be scored on the FULL set, so only these enter the per-residue
             table. A generation that stops at residue 200 has not lost
             His327 -- it never reached it.

The residue ROLE labels are carried over verbatim from the original table
(recovered from git history at commit 27559dc) so this rebuild does not silently
re-annotate the biology.

Usage:
    python -m src.analysis.part_a_active_site \
        --corpus data/part_a_generated_corpus.csv --level L1_donor_90 --tag l1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis_l1 import score
from src.eval.active_site import CATALYTIC, _aligner, load_reference, score_sequence

OUT = Path("results/part_a/tables")
REFERENCE = Path("data/spinacia_P00875.fasta")

# Verbatim from the pre-deletion table; not re-derived here.
ROLES = {
    175: "active site (Lys, RuBP)", 177: "active site (Lys)",
    201: "carbamylated Lys (Mg²⁺, activation)",
    203: "Mg²⁺ coordination (Asp)", 204: "Mg²⁺ coordination (Glu)",
    294: "active site (His)", 295: "active site (Arg, RuBP P1)",
    327: "active site (His)", 334: "active site (Lys, loop 6)",
    379: "active site (Ser)", 380: "active site (Gly)",
}


def score_corpus(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Per-sequence active-site scores. One aligner, reused: building a
    PairwiseAligner per call dominates the runtime otherwise."""
    aligner = _aligner()
    rows = []
    for r in df.itertuples():
        cds = r.cds_recomputed
        if not isinstance(cds, str):
            continue
        s = score_sequence(cds, reference, aligner=aligner)
        if s is None:          # translation under 100 aa
            continue
        rows.append({"prompt_id": r.prompt_id, "tax_group": r.tax_group,
                     "level_name": r.level_name, "n_correct": s["n_correct"],
                     "identity": s["identity"], "aa_len": s["aa_len"],
                     "n_covered": s["n_covered"], "scorable": s["scorable"],
                     "full_length": r.full_length,
                     **{f"hit_{p}": s["hits"][p] for p in sorted(CATALYTIC)}})
    return pd.DataFrame(rows)


def per_residue(scored: pd.DataFrame) -> pd.DataFrame:
    """Correct-residue fraction, over BOTH defensible populations.

    The pre-deletion table reported one population without naming it: its n=174
    is the count of full-length L1 generations under the corpus's stored
    `full_length` column. That is a narrower set than `scorable`, because a
    full-length CDS (>=1400 nt) necessarily reaches residue 380 while a long but
    broken ORF can reach it too. The two answer different questions --

      full_length  among generations that produced a complete CDS, are the
                   catalytic residues right? (the biologically meaningful one)
      scorable     among generations long enough to reach all 11 residues,
                   are they right? (includes broken ORFs that ran long)

    -- and the gap between them is itself informative, so both are emitted
    rather than one being chosen silently.
    """
    rows = []
    for label, sub in (("full_length", scored[scored.full_length]),
                       ("scorable", scored[scored.scorable])):
        for pos, aa in sorted(CATALYTIC.items()):
            n_ok = int((sub[f"hit_{pos}"] == aa).sum())
            rows.append({"population": label, "pos": pos, "aa": aa,
                         "correct": n_ok, "n": len(sub),
                         "frac": n_ok / len(sub) if len(sub) else float("nan"),
                         "role": ROLES[pos]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--level", default="L1_donor_90",
                    help="prompt level to score, or 'all'")
    ap.add_argument("--tag", default="l1", help="output filename prefix")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ref = load_reference(str(args.reference))
    df = score(pd.read_csv(args.corpus))
    if args.level != "all":
        df = df[df.level_name == args.level]
    if df.empty:
        raise SystemExit(f"no rows at level {args.level!r}")

    scored = score_corpus(df, ref)
    resid = per_residue(scored)
    args.out.mkdir(parents=True, exist_ok=True)
    base_cols = ["prompt_id", "tax_group", "n_correct", "identity", "aa_len",
                 "n_covered", "scorable", "full_length"]
    for name, table, cols in (("base", scored, base_cols),
                              ("per_residue", resid, None)):
        p = args.out / f"{args.tag}_active_site_{name}.csv"
        (table[cols] if cols else table).to_csv(p, index=False, float_format="%.4f")
        print(f"  wrote {p}  ({len(table)} rows)")
    print(f"  scored={len(scored)}  scorable(reach residue 380)={int(scored.scorable.sum())}")


if __name__ == "__main__":
    main()
