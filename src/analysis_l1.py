"""Analyse an L1 finetuned run against Part A's published L1 baseline.

Kept as a file, not a pasted cell: it must run again for each further prefix
level, and re-pasting it is how two arms drift apart.

HISTORICAL NOTE: this module was written when Part A's stored 1500 nt base rows
were to be REUSED as Part B's control, with a repro_check licensing that reuse.
That is no longer how it works. The base arm is regenerated inside every run at
1800 nt, on the same prompts and seeds as the adapter arms, so the control is
paired within the job rather than borrowed across protocols. Use
src.analysis.determinism to check that two runs agree on shared prompts -- they
do, byte-for-byte, on the 120 L1 donors generated in separate jobs.

IMPORTANT: scoring calls src.eval.metrics.is_full_length -- Part A's own
function -- rather than reimplementing it. A first version of this module
used `cds_len >= 1000` and reported the base L1 rate as 63.9% against Part
A's published 48.3%, because full_length is FOUR conditions, not a length
floor: 1400 <= len <= 1550, len divisible by 3, terminal stop present, and
no internal stop. Any reimplementation here is a silent comparability bug.

The predicate is applied to the CDS (truncated at the first in-frame stop), NOT
to the raw 1500 nt generation. On the raw sequence it gives a 0.28% pass rate
against Part A's published 48.33%, so the CDS reading is the one Part A used.

Known discrepancy, 16 rows across the four donor levels (L1:4, L2:4, L3:6,
L4:2; L0 is clean): Part A's stored `full_length` is False for sequences whose
own stored `cds_nt`/`cds_len` satisfy every condition, so that column disagrees
with the data beside it. All 16 run in the SAME direction -- stored False,
recomputed True -- which is not the signature of a random stale write, and no
additional condition separates them from rows the column scores True: the four
L1 rows were checked directly and carry an ATG start, a terminal stop, no
internal stop and no ambiguous bases. The cause is not established.

Per-level effect on the published curve:

    L0 0.9722 -> 0.9722    L1 0.4833 -> 0.4944    L2 0.8417 -> 0.8528
    L3 0.9667 -> 0.9833    L4 0.9750 -> 0.9806

results/part_a/tables/full_length_recount_discrepancies.csv lists the four L1
rows only; it predates the all-level recount.

Consequence for comparing arms: score BOTH arms with this module. Never pair
Part A's stored column for the base arm against recomputed values for the
finetuned arm, or ~1% of any difference is an artefact of which scorer ran.
"""
import pandas as pd

from src.eval.metrics import STOPS, is_full_length


def cds_of(seq):
    """Truncate at the first in-frame stop, INCLUSIVE. Returns None when no
    in-frame stop exists, matching Part A's NaN cds_len for read-through."""
    s = str(seq).upper()
    for i in range(0, len(s) - 2, 3):
        if s[i:i + 3] in STOPS:
            return s[:i + 3]
    return None


def score(df, seq_col="seq_nt"):
    """Recompute metrics from the sequences, via Part A's own predicate."""
    out = df.copy()
    # Built with list comprehensions, not Series.map: map coerces None to NaN and
    # then passes the float back into the lambda. Read-through (no in-frame stop
    # in 1500 nt) is a real outcome here, so it must stay explicit.
    cds = [cds_of(s) for s in out[seq_col]]
    out["cds_recomputed"] = cds
    out["cds_len_recomputed"] = [len(c) if c is not None else None for c in cds]
    out["read_through"] = [c is None for c in cds]
    out["full_length"] = [is_full_length(c) if c is not None else False for c in cds]
    out["n_internal_stops"] = [
        0 if c is None else sum(1 for i in range(0, len(c) - 3, 3)
                                if c[i:i + 3] in STOPS)
        for c in cds
    ]
    out["acgt_frac"] = out[seq_col].map(
        lambda s: sum(ch in "ACGT" for ch in str(s).upper()) / max(1, len(str(s))))
    return out


def compare(ft_csv, part_a_csv, level="L1_donor_90"):
    ft = score(pd.read_csv(ft_csv))
    pa = pd.read_csv(part_a_csv)
    base = score(pa[pa.level_name == level])
    ft["arm"], base["arm"] = "finetuned", "base"
    both = pd.concat([base, ft], ignore_index=True)

    overall = both.groupby("arm").agg(
        n=("prompt_id", "size"),
        full_length=("full_length", "mean"),
        read_through=("read_through", "mean"),
        median_cds=("cds_len_recomputed", "median"),
        mean_internal_stops=("n_internal_stops", "mean"),
        acgt=("acgt_frac", "mean"),
    ).round(4)

    # Paired on prompt -- what the shared seeds buy. Unpaired would be weaker.
    pair = base[["prompt_id", "full_length", "cds_len_recomputed"]].merge(
        ft[["prompt_id", "full_length", "cds_len_recomputed"]],
        on="prompt_id", suffixes=("_base", "_ft"))
    pair["gained"] = (~pair.full_length_base) & pair.full_length_ft
    pair["lost"] = pair.full_length_base & (~pair.full_length_ft)

    by_clade = both.pivot_table(index="tax_group", columns="arm",
                                values="full_length", aggfunc="mean")
    by_clade["n"] = base.groupby("tax_group").size()
    return both, overall, pair, by_clade.sort_values("base")