"""Regenerate Part A's tables from committed inputs.

Every table this module writes is the output of a command run against data in
this repository. Tables that could only be produced from inputs the repository
does not hold -- the natural rbcL corpus, a structure, ESM-2 embeddings -- were
deleted rather than shipped as artefacts nobody can rebuild; see
results/part_a/README.md.

Scoring goes through src.analysis_l1.score, which calls Part A's own
src.eval.metrics.is_full_length. Never reimplement the predicate here: a first
attempt elsewhere in this project used `cds_len >= 1000` and reported the L1 base
rate as 63.9% against the published 48.3%, because full_length is four conditions
and not a length floor.

NOTE ON THE PUBLISHED NUMBERS. The corpus carries a stored `full_length` column
written by a scorer that is not in this repository, and it disagrees with
is_full_length on 16 of 1,800 rows -- always in the same direction, stored False
where recomputed is True. These tables report the RECOMPUTED value, because it is
the one a reader can reproduce. That moves the curve by up to 1.7 points at a
level; `full_length_recount_discrepancies.csv` lists every affected row.

Usage:
    # tables describing the committed 1,500 nt Part A corpus
    python -m src.analysis.part_a_tables --corpus data/part_a_generated_corpus.csv

    # titration from a generation run (any total length)
    python -m src.analysis.part_a_tables \
        --generated results/generate_a/<run>/part_a/generated_base.csv \
        --tag 1800nt_rep0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis_l1 import score

OUT = Path("results/part_a/tables")


def titration(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate per prompt level: the R6 curve."""
    return df.groupby("level_name").agg(
        prefix_nt=("prefix_nt", "first"),
        n=("full_length", "size"),
        full_length=("full_length", "mean"),
        read_through=("read_through", "mean"),
        median_cds_len=("cds_len_recomputed", "median"),
    )


def titration_by_clade(df: pd.DataFrame) -> pd.DataFrame:
    """The same curve split by clade -- where R6's algal collapse is visible."""
    return df.pivot_table(index="tax_group", columns="level_name",
                          values="full_length", aggfunc="mean")


def generation_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Per-level QC over the whole corpus: R6's headline table."""
    return df.groupby("level_name").agg(
        n=("full_length", "size"),
        no_inframe_stop=("read_through", "sum"),
        full_length=("full_length", "sum"),
        median_cds_len=("cds_len_recomputed", "median"),
        pass_rate=("full_length", "mean"),
    ).round(4)


def interference(df: pd.DataFrame) -> pd.DataFrame:
    """Pass rate by clade x level, plus the L0->L1 drop that R6 turns on.

    Sorted by that drop, so the clades the short prompt destroys come first.
    """
    piv = df.pivot_table(index="tax_group", columns="level_name",
                         values="full_length", aggfunc="mean")
    if {"L0_shared_seed", "L1_donor_90"} <= set(piv.columns):
        piv["delta_L0_L1"] = piv["L1_donor_90"] - piv["L0_shared_seed"]
        piv = piv.sort_values("delta_L0_L1")
    piv["n_per_level"] = df.groupby("tax_group").size() // df.level_name.nunique()
    return piv


def recount_discrepancies(raw: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Rows where the corpus's stored `full_length` disagrees with is_full_length.

    Reported for every level, not just L1: the disagreement is not confined to
    one level and it runs one way at all of them, which is why these tables use
    the recomputed value.
    """
    if "full_length" not in raw:
        return pd.DataFrame()
    stored = raw.full_length.astype(str).str.lower().eq("true").to_numpy()
    mask = stored != scored.full_length.to_numpy()
    out = pd.DataFrame({
        "prompt_id": scored.prompt_id[mask],
        "level_name": scored.level_name[mask],
        "tax_group": scored.tax_group[mask],
        "stored_cds_len": raw.cds_len[mask] if "cds_len" in raw else None,
        "recomputed_cds_len": scored.cds_len_recomputed[mask],
        "stored_full_length": stored[mask],
        "recomputed_full_length": scored.full_length[mask],
    })
    return out.sort_values(["level_name", "prompt_id"])



def _write(table: pd.DataFrame, path: Path, index: bool = True) -> None:
    table.to_csv(path, float_format="%.4f", index=index)
    print(f"  wrote {path}  ({len(table)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path,
                    help="the committed Part A corpus, for the QC tables")
    ap.add_argument("--generated", type=Path,
                    help="a generated_base.csv from scripts/generate_ab.py")
    ap.add_argument("--tag", help="suffix for --generated outputs, e.g. 1800nt_rep0")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not (args.corpus or args.generated):
        ap.error("give at least one of --corpus, --generated")
    if args.generated and not args.tag:
        ap.error("--generated needs --tag")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.corpus:
        raw = pd.read_csv(args.corpus)
        df = score(raw)
        _write(generation_qc(df), args.out / "generation_qc.csv")
        _write(interference(df), args.out / "l1_interference.csv")
        disc = recount_discrepancies(raw, df)
        _write(disc, args.out / "full_length_recount_discrepancies.csv", index=False)

    if args.generated:
        df = score(pd.read_csv(args.generated))
        _write(titration(df), args.out / f"titration_{args.tag}.csv")
        _write(titration_by_clade(df), args.out / f"titration_by_clade_{args.tag}.csv")



if __name__ == "__main__":
    main()
