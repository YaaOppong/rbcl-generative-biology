"""Tests for the active-site endpoint.

Two properties are load-bearing:

1. **Alignment, not indexing.** An indel upstream of the active site must not be
   reported as catalytic loss. This is the failure mode metrics.py warns about.
2. **Out of reach is not absent.** A generation too short to contain Gly380
   must be flagged unscorable, not scored 0/11. The first Part B run generated
   300 nt and scored 0/11 in both arms -- purely an artefact of length.
"""

from pathlib import Path

import pytest

from src.eval.active_site import (
    CATALYTIC,
    MIN_COVERED_NT,
    load_reference,
    score_sequence,
)

REF_FASTA = Path("data/spinacia_P00875.fasta")
pytestmark = pytest.mark.skipif(
    not REF_FASTA.exists(), reason="P00875 reference not staged"
)


@pytest.fixture(scope="module")
def ref():
    return load_reference(str(REF_FASTA))


def test_reference_has_every_catalytic_residue_where_claimed(ref):
    """The 11 positions must hold the stated residues in P00875 itself.

    If this fails, the position table and the reference disagree and every
    downstream score is meaningless.
    """
    assert len(ref) == 475
    for pos, aa in CATALYTIC.items():
        assert ref[pos - 1] == aa, f"P00875 position {pos} is {ref[pos-1]}, not {aa}"


def test_reference_scores_perfectly_against_itself(ref):
    r = score_sequence(ref, ref, is_protein=True)
    assert r["n_correct"] == len(CATALYTIC)
    assert r["scorable"] is True
    assert r["identity"] == pytest.approx(1.0)


def test_upstream_indel_does_not_destroy_the_active_site(ref):
    """A 1-residue deletion at position 10 shifts every later index by one.

    Naive indexing would report near-total catalytic loss; alignment must not.
    """
    shifted = ref[:9] + ref[10:]
    r = score_sequence(shifted, ref, is_protein=True)
    assert r["n_correct"] == len(CATALYTIC), (
        f"alignment failed to absorb an upstream indel: {r['n_correct']}/11")


def test_short_sequence_is_flagged_unscorable_not_zero(ref):
    """300 nt reaches ~residue 100 -- no catalytic residue is in range."""
    short = ref[:100]
    r = score_sequence(short, ref, is_protein=True)
    assert r["n_covered"] == 0
    assert r["scorable"] is False
    assert r["n_correct"] == 0   # zero because unreachable, and scorable says so


def test_min_covered_nt_matches_the_last_residue():
    assert MIN_COVERED_NT == max(CATALYTIC) * 3 == 1140


def test_real_substitution_is_detected(ref):
    """Mutating the carbamylated Lys201 must register as a loss."""
    mutated = ref[:200] + "A" + ref[201:]
    r = score_sequence(mutated, ref, is_protein=True)
    assert r["hits"][201] == "A"
    assert r["n_correct"] == len(CATALYTIC) - 1
