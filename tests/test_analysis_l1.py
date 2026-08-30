"""Tests for L1 scoring against Part A's baseline.

The load-bearing property is that `score` reproduces Part A's published
`full_length` column. If it drifts, every base-vs-finetuned comparison silently
mixes two different definitions and the difference is partly an artefact.

A first version of this module scored `cds_len >= 1000` and put the base L1 rate
at 63.9% against Part A's published 48.3% -- a 15-point error that would have
been read as the finetune underperforming.
"""

import pandas as pd
import pytest

from src.analysis_l1 import cds_of, score
from src.eval.metrics import is_full_length


def test_full_length_needs_all_four_conditions():
    """Length in range, divisible by 3, terminal stop, no internal stop."""
    body = "ATG" + "GCT" * 474          # 1425 nt, in frame, no stops
    assert is_full_length(body + "TAA")           # 1428, terminal stop
    assert not is_full_length(body)               # no terminal stop
    assert not is_full_length("ATG" + "GCT" * 100 + "TAA")   # too short
    assert not is_full_length(body + "TAA" + "G")            # not divisible by 3
    assert not is_full_length("ATG" + "TAA" + "GCT" * 473 + "TAA")  # internal stop


def test_cds_of_returns_none_on_read_through():
    """No in-frame stop is a real outcome, not a zero-length CDS.

    Part A records NaN cds_len for these; conflating them with short CDSs would
    move them into the wrong side of the length filter.
    """
    assert cds_of("ATG" + "GCT" * 10) is None
    assert cds_of("ATGGCTTAAGGG") == "ATGGCTTAA"      # inclusive of the stop
    assert cds_of("ATGTAAGCTTAA") == "ATGTAA"         # first stop wins


def test_score_marks_read_through_not_full_length():
    df = pd.DataFrame({"seq_nt": ["ATG" + "GCT" * 499,          # read-through
                                  "ATG" + "GCT" * 474 + "TAA"]})  # clean 1428
    out = score(df)
    assert out.read_through.tolist() == [True, False]
    assert out.full_length.tolist() == [False, True]
    assert out.cds_len_recomputed.tolist()[1] == 1428


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/part_a_generated_corpus.csv").exists(),
    reason="Part A corpus not staged locally",
)
def test_score_reproduces_part_a_published_column():
    """Agreement must stay >=98% on L1, with the 4 known stale rows excluded.

    Not 100%: Part A's stored full_length is False for four sequences whose own
    cds_nt satisfies every condition. Those are documented in
    results/part_a/tables/full_length_recount_discrepancies.csv.
    """
    pa = pd.read_csv("data/part_a_generated_corpus.csv")
    l1 = pa[pa.level_name == "L1_donor_90"]
    sc = score(l1)
    agreement = (sc.full_length.values == l1.full_length.astype(bool).values).mean()
    assert agreement >= 0.98, f"scoring drifted from Part A: {agreement:.4f}"
    # read-through count must match Part A's NaN cds_len exactly
    assert sc.read_through.sum() == l1.cds_len.isna().sum()
