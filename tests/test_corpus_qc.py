"""Quality gates on the training corpus.

The corpus these replace was 68.3% complete by the same predicate used to score
generations: 23% of its sequences never terminated, and Diatoms terminated in
only 45% of records. Training on non-terminating sequences while measuring
whether generations terminate makes the training data teach the failure mode
being scored.

Every gate is a rejection, not a repair. A corpus is cheaper to shrink than a
result is to caveat.
"""
from pathlib import Path as _Path

import pytest

from src.data.build_dataset import is_complete_cds, is_unambiguous
from src.eval.metrics import is_full_length

BODY = "AAA" * 475          # 1,425 nt of in-frame sense codons


def test_a_complete_cds_passes():
    assert is_complete_cds("ATG" + BODY + "TAA")


def test_the_gate_is_the_evaluation_predicate():
    """Training data and scoring must agree on 'complete', or the corpus teaches
    one target while the metric rewards another."""
    for seq in ("ATG" + BODY + "TAA", "ATG" + BODY, "ATGAAA", "ATG" + BODY + "TAATAA"):
        assert is_complete_cds(seq) == is_full_length(seq.upper())


def test_no_terminal_stop_is_rejected():
    """The 23% that made the old corpus teach read-through."""
    assert not is_complete_cds("ATG" + BODY + "AAA")


def test_internal_stop_is_rejected():
    assert not is_complete_cds("ATG" + "AAA" * 200 + "TAA" + "AAA" * 274 + "TAA")


def test_out_of_frame_length_is_rejected():
    assert not is_complete_cds("ATG" + BODY + "TA")


def test_a_fragment_is_rejected():
    """Barcode submissions covering the middle of the gene."""
    assert not is_complete_cds("ATG" + "AAA" * 200 + "TAA")


def test_lowercase_is_handled():
    assert is_complete_cds(("ATG" + BODY + "TAA").lower())


def test_ambiguity_is_rejected():
    assert is_unambiguous("ACGT" * 10)
    assert not is_unambiguous("ACGT" * 10 + "N")
    assert not is_unambiguous("ACGT" * 10 + "R")     # IUPAC purine


def test_no_start_codon_gate():
    """DELIBERATE, and the reason belongs in a test because it looks like an
    omission. Requiring ATG is confounded with clade: Red algae are 17.2% ATG
    and Brown algae 17.7% (5'-partial barcode submissions with a codon_start
    offset) against 96-100% for Eudicots, Mosses and Monocots. Gating on it
    would cut the algal share from 36.9% to 17.3% and re-import the land-plant
    bias the corpus exists to correct.

    A 5'-truncated but otherwise complete CDS still carries its clade's codon
    usage and composition, which is what the finetune needs from it.
    """
    five_prime_partial = "GCT" + BODY + "TAA"        # complete, but no ATG
    assert is_complete_cds(five_prime_partial), \
        "a 5'-partial record must still pass; gating on ATG is clade-confounded"


@pytest.mark.parametrize("seq,ok", [
    ("ATG" + BODY + "TAG", True),
    ("ATG" + BODY + "TGA", True),
    ("ATG" + "AAA" * 600 + "TAA", False),           # 1,806 nt, beyond natural range
])
def test_stop_codons_and_length_bounds(seq, ok):
    assert is_complete_cds(seq) is ok


def test_containment_is_reported_per_arm_and_corpus():
    """Containment is meaningless without naming its reference. Base scores 0.386
    against all_fullcds and 0.000 against the narrower b1 corpus, so a single
    number per arm hides which yardstick was used -- which produced a reading
    that reversed once the pairs were made explicit: b1_sparse_clade appears to
    copy least on raw figures and copies MOST on lift over base."""
    src = _Path("src/analysis/part_b_tables.py").read_text()
    assert "def containment_table" in src
    assert "own_corpus" in src, "the diagonal is not identifiable in the output"
    assert "for corpus_name, path in CORPORA.items()" in src, \
        "containment is not computed against every corpus"


def test_transfer_table_tolerates_a_missing_arm():
    """A base-only run must not crash the table generator."""
    from src.analysis.part_b_tables import transfer_table
    assert transfer_table({"base": None}, "b1_sparse_clade") is None
