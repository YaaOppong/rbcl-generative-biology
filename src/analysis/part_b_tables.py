"""Regenerate Part B's comparison tables from a committed multi-arm run.

Same motivation as part_a_tables: a table nobody can regenerate is an assertion
with a filename. Every number Part B reports should come out of this command.

Three tables, and the second and third are the ones that carry the argument:

  pass_rate_by_clade   the headline, per clade per arm
  paired               one row per donor per arm, with 32-mer containment, so
                       the paired tests and the memorisation control can both be
                       recomputed without rerunning generation
  transfer             pass rate against the number of records that arm's corpus
                       actually contained for that clade -- the table that shows
                       clades are fixed whether or not they were trained on

Containment is measured against each arm's OWN training corpus, and the base arm
is included deliberately: rbcL is conserved enough that base scores ~0.39 above
the 90% threshold against a corpus it never saw, so only the difference from base
is attributable to finetuning.

Usage:
    python -m src.analysis.part_b_tables \
        --run results/generate_b/<run>/part_b --tag l1_1800nt
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from src.analysis_l1 import score
from src.eval.memorisation import containment, corpus_kmers

OUT = Path("results/part_b")
CORPORA = {"all_fullcds": "data/all_fullcds.jsonl",
           "all_fullcds_atg": "data/all_fullcds_atg.jsonl",
           "b1_sparse_clade": "data/b1.jsonl"}

# The corpora and the evaluation set do NOT share a clade vocabulary. The corpora
# split diatoms out as their own clade; the evaluation donors group them under
# SAR/other protist. Unmapped, `counts.get(clade, 0)` returns 0 and the transfer
# table reports SAR/other protist as an untrained clade while b1_sparse_clade in
# fact holds 523 diatom records -- the same phylum (Ochrophyta) as the SAR
# evaluation donors. That error reached docs/DESIGN.md before it was caught, so
# the mapping is explicit here and unmapped labels are reported, not swallowed.
CLADE_ALIASES = {"Diatoms": "SAR/other protist"}


def _require(path: str, why: str) -> None:
    """The corpora are gitignored (sequences are fetched, not committed), so on a
    fresh clone they are absent. Missing them used to mean the containment and
    transfer tables were silently omitted -- no error, just gone. Say so instead."""
    raise SystemExit(
        f"{why} needs {path}, which is not present.\n"
        "The corpora are rebuilt from committed manifests, not committed themselves:\n"
        "  python -m src.data.build_dataset --arm B1_sparse_clade --out data/b1.jsonl\n"
        "See data/README.md for the other arms.")


def load_arms(run: Path) -> dict[str, pd.DataFrame]:
    arms = {}
    for csv in sorted(run.glob("generated_*.csv")):
        arms[csv.stem.replace("generated_", "")] = score(pd.read_csv(csv))
    if not arms:
        raise SystemExit(f"no generated_*.csv under {run}")
    return arms


def containment_table(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every arm against every corpus, because the number is reference-dependent.

    Containment only means something next to a baseline: rbcL is conserved
    enough that BASE scores 0.386 above the 90% threshold against a corpus it
    never saw, so a finetuned arm's 0.525 is not 0.525 worth of copying. But the
    baseline itself moves with the reference -- base scores 0.494 against
    all_fullcds and 0.091 against the narrower b1_sparse_clade corpus. Reporting
    one number per arm therefore hides which yardstick was used, which is how an
    earlier version of this analysis produced a figure that did not match the
    prose. Every pair is emitted instead.
    """
    rows = []
    missing = [p for p in CORPORA.values() if not Path(p).exists()]
    if missing:
        _require(missing[0], "the containment table")
    for corpus_name, path in CORPORA.items():
        with open(path) as fh:
            ref = corpus_kmers(json.loads(line)["sequence"] for line in fh)
        for arm, df in arms.items():
            cds = df.cds_recomputed.dropna()
            if cds.empty:
                continue
            c = pd.Series([containment(x, ref) for x in cds])
            rows.append({"arm": arm, "corpus": corpus_name, "n": len(c),
                         "mean": c.mean(), "median": c.median(),
                         "frac_over_90pct": (c > 0.9).mean(),
                         "own_corpus": arm == corpus_name})
    return pd.DataFrame(rows)


def add_own_containment(arms: dict[str, pd.DataFrame]) -> None:
    """Per-row containment against each arm's OWN training corpus.

    Base has no own corpus, so it gets no value here -- use containment_table
    for the baseline, where the reference is named explicitly.
    """
    for arm, df in arms.items():
        path = CORPORA.get(arm)
        if path is None or not Path(path).exists():
            continue
        with open(path) as fh:
            ref = corpus_kmers(json.loads(line)["sequence"] for line in fh)
        df["containment_own_corpus"] = [containment(c, ref) if pd.notna(c) else None
                                        for c in df.cds_recomputed]


def transfer_table(arms: dict[str, pd.DataFrame], arm: str) -> pd.DataFrame | None:
    """Pass rate per clade beside that clade's record count in the arm's corpus."""
    path = CORPORA.get(arm)
    if path is None or arm not in arms or "base" not in arms:
        return None
    if not Path(path).exists():
        _require(path, f"the transfer table for {arm!r}")
    with open(path) as fh:
        raw = collections.Counter(json.loads(line)["clade"] for line in fh)
    counts: collections.Counter = collections.Counter()
    for clade, n in raw.items():
        counts[CLADE_ALIASES.get(clade, clade)] += n
    base, ft = arms["base"], arms[arm]
    # A corpus label with no evaluation counterpart is either a clade nobody
    # generated from (fine) or a vocabulary mismatch that would silently read as
    # zero records (not fine). Name them so the difference is a judgement call
    # someone makes, rather than one the join makes by omission.
    unmapped = {c: n for c, n in counts.items()
                if c not in set(base.tax_group.unique())}
    if unmapped:
        print(f"  NOTE: {arm} corpus labels with no evaluation counterpart "
              f"(not scored in the transfer table): {unmapped}")
    rows = []
    for clade, g in base.groupby("tax_group"):
        rows.append({"tax_group": clade, "n_donors": len(g),
                     "records_in_corpus": counts.get(clade, 0),
                     "base": g.full_length.mean(),
                     arm: ft[ft.tax_group == clade].full_length.mean()})
    return pd.DataFrame(rows).sort_values(["records_in_corpus", "base"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path,
                    help="a part_b/ directory holding generated_<arm>.csv files")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--transfer-arm", default="b1_sparse_clade",
                    help="arm whose corpus coverage the transfer table is keyed on")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    arms = load_arms(args.run)
    add_own_containment(arms)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = []
    for arm, df in arms.items():
        d = df.copy(); d["arm"] = arm; frames.append(d)
    both = pd.concat(frames)

    p = both.pivot_table(index="tax_group", columns="arm", values="full_length",
                         aggfunc="mean")
    p.to_csv(args.out / f"{args.tag}_pass_rate_by_clade.csv", float_format="%.4f")
    cols = ["prompt_id", "donor_acc", "tax_group", "arm", "full_length",
            "read_through", "cds_len_recomputed"]
    if "containment_own_corpus" in both:
        cols.append("containment_own_corpus")
    ct = containment_table(arms)
    if not ct.empty:
        ct.to_csv(args.out / f"{args.tag}_containment.csv", index=False,
                  float_format="%.4f")
    both[cols].to_csv(args.out / f"{args.tag}_paired.csv", index=False,
                      float_format="%.4f")
    t = transfer_table(arms, args.transfer_arm)
    if t is not None:
        t.to_csv(args.out / f"{args.tag}_transfer.csv", index=False,
                 float_format="%.4f")
    for f in sorted(args.out.glob(f"{args.tag}_*.csv")):
        print(f"  wrote {f}")


if __name__ == "__main__":
    main()
