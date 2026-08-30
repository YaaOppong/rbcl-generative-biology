"""Tests for CDS extraction and frame validation.

The codon_start bug these cover was silent and clade-structured: naive FASTA
extraction left ~45% of algal records out of frame while land plants were
unaffected, putting a confound directly on the primary endpoint (algal pass
rate). Regression coverage matters here.
"""
# Fixtures are COMPUTED, not hand-written. Hand-picked repeat sequences
# (e.g. "GCT" * n) are stop-free in all three frames, which makes them useless
# for frame tests: infer_frame correctly rejects them as ambiguous. These
# builders generate sequences with a verified frame structure and assert it, so a
# fixture can never silently stop exercising the behaviour under test.
import random

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from src.data.build_dataset import base_acc, extract_cds, infer_frame, is_in_frame

SENSE_CODONS = [
    a + b + c
    for a in "ACGT"
    for b in "ACGT"
    for c in "ACGT"
    if str(Seq(a + b + c).translate()) != "*"
]


def clean_frames(seq: str, min_len: int = 1000) -> list[int]:
    """Offsets at which seq translates with no internal stop codon."""
    out = []
    for off in (0, 1, 2):
        sub = seq[off:]
        sub = sub[: len(sub) // 3 * 3]
        if len(sub) >= min_len and str(Seq(sub).translate())[:-1].count("*") == 0:
            out.append(off)
    return out


def unique_frame_cds(n_codons: int = 479, seed: int = 7) -> str:
    """A CDS-like sequence that is stop-free in frame 0 ONLY.

    Mirrors real rbcL: ~1.4 kb, and shifting the frame introduces stops. The
    assertion guarantees the fixture exercises frame recovery rather than
    passing trivially.
    """
    rng = random.Random(seed)
    for _ in range(500):
        body = "ATG" + "".join(rng.choice(SENSE_CODONS) for _ in range(n_codons)) + "TAA"
        if clean_frames(body) == [0]:
            return body
    raise AssertionError("could not build a unique-frame fixture")


ORF = unique_frame_cds()
AMBIGUOUS = "GCT" * 400  # stop-free in all three frames

def gb_record(seq: str, codon_start: int = 1, gene: str = "rbcL") -> SeqRecord:
    rec = SeqRecord(Seq(seq), id="XX000001.1")
    rec.features = [
        SeqFeature(
            FeatureLocation(0, len(seq)),
            type="CDS",
            qualifiers={"gene": [gene], "codon_start": [str(codon_start)]},
        )
    ]
    return rec


def test_codon_start_offset_is_applied():
    """codon_start=3 means the first complete codon begins at base 3."""
    padded = "GG" + ORF
    got = extract_cds(gb_record(padded, codon_start=3))
    assert got == ORF[: len(ORF) // 3 * 3]
    assert len(got) % 3 == 0
    assert str(Seq(got).translate())[:-1].count("*") == 0


def test_codon_start_two():
    got = extract_cds(gb_record("G" + ORF, codon_start=2))
    assert got.startswith("ATG")


def test_trailing_partial_codon_is_trimmed():
    got = extract_cds(gb_record(ORF + "AT"))
    assert len(got) % 3 == 0


def test_ignoring_codon_start_would_produce_internal_stops():
    """Guards the specific failure mode: the naive path must be worse.

    If this test ever passes trivially the fixture has stopped exercising the
    bug, so it asserts on the naive extraction directly.
    """
    seq = "GG" + ORF
    naive = str(Seq(seq[: len(seq) // 3 * 3]).translate())
    fixed = str(Seq(extract_cds(gb_record(seq, codon_start=3))).translate())
    assert naive[:-1].count("*") > 0
    assert fixed[:-1].count("*") == 0


def test_falls_back_to_frame_inference_without_annotation():
    """CDS annotation availability is clade-structured: in the manifest audit,
    100% of red algal and diatom records carried a CDS feature but only 4% of
    eudicot records did. Requiring annotation would silently discard whole
    clades -- the same bias shape the frame fix exists to remove."""
    rec = SeqRecord(Seq(ORF), id="XX000002.1")
    rec.features = []
    got = extract_cds(rec)
    assert got is not None
    assert is_in_frame(got)
    assert got == ORF[: len(ORF) // 3 * 3]


def test_annotation_disagreeing_with_sequence_falls_back():
    """A codon_start that does not yield a clean frame must not be trusted.

    Here the annotation claims codon_start=2 on a sequence whose real frame is 0.
    Honouring it blindly writes an out-of-frame training record; the extractor
    must detect the internal stops and fall back to inference.
    """
    rec = gb_record(ORF, codon_start=2)  # wrong: shifts frame 0 into stops
    got = extract_cds(rec)
    assert got is not None, "should have recovered via inference"
    assert is_in_frame(got)
    assert got == ORF[: len(ORF) // 3 * 3]


def test_infer_frame_rejects_ambiguous_records():
    """If more than one frame is stop-free the record is ambiguous -- reject
    rather than guess, since a wrong guess writes an out-of-frame sequence."""
    assert len(clean_frames(AMBIGUOUS)) > 1  # fixture really is ambiguous
    assert infer_frame(AMBIGUOUS) is None


def test_infer_frame_recovers_single_clean_frame():
    """A record padded by two bases: inference must find frame 2."""
    padded = "GG" + ORF
    assert clean_frames(padded) == [2]  # fixture precondition
    got = infer_frame(padded)
    assert got is not None and is_in_frame(got)
    assert got == ORF[: len(ORF) // 3 * 3]


def test_infer_frame_rejects_short_records():
    """Below MIN_LEN an rbcL record is a fragment, and frame inference on a short
    sequence is unreliable -- too few codons for a stop to appear by chance."""
    assert infer_frame(ORF[:300]) is None


def test_prefers_named_gene_over_other_cds():
    rec = gb_record(ORF, gene="rbcL")
    rec.features.append(
        SeqFeature(
            FeatureLocation(0, 9),
            type="CDS",
            qualifiers={"gene": ["psbA"], "codon_start": ["1"]},
        )
    )
    assert extract_cds(rec, gene="rbcL") == ORF[: len(ORF) // 3 * 3]


def test_frame_gate_rejects_shifted_and_nonmultiple():
    assert is_in_frame(ORF[: len(ORF) // 3 * 3])
    assert not is_in_frame(ORF[1:])          # frame shift -> internal stops
    assert not is_in_frame(ORF[:-1])         # not a multiple of three
    assert not is_in_frame("")


def test_frame_gate_tolerates_rare_ambiguity_but_not_runs():
    clean = "ATG" + "GCT" * 40
    assert is_in_frame(clean)
    assert is_in_frame(clean[:-3] + "GNT")               # 1 ambiguous base
    assert not is_in_frame("N" * 30 + clean[30:])        # 25% ambiguous


def test_frame_gate_allows_partial_ends():
    """Barcode records rarely start at ATG or end at a stop -- only 166/300 of
    the B1 sample begin with ATG. The gate must not require either."""
    assert is_in_frame("GCT" * 50)


def test_base_acc_strips_version():
    assert base_acc("PZ367540.1") == "PZ367540"
    assert base_acc("PZ367540") == "PZ367540"


# ---------------------------------------------------------------------------
# Evaluation-donor exclusion. The manifest's heldout_donor_species column had
# 24 of the 120 donors marked False in the B2_balanced pool, so they were
# trained on -- their own target sequences included. The column has no
# generating script in this repo, so it cannot be regenerated or audited; the
# filter below is derived from prompts_corpus.csv instead and these tests are
# what keep it honest.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path

import pandas as _pd
import pytest as _pytest

from src.data import build_dataset as _bd


def test_evaluation_donors_are_resolved_from_the_prompt_corpus():
    donors = _bd.evaluation_donors()
    assert len(donors) == 120, "Part A generates from 120 donors"
    assert all("." not in d for d in donors), "accessions must be versionless"


def test_missing_prompt_corpus_raises_rather_than_excluding_nothing():
    """A silent no-op here is exactly how the 24 donors got into training."""
    with _pytest.raises(FileNotFoundError):
        _bd.evaluation_donors(_Path("data/does_not_exist.csv"))


def test_no_evaluation_donor_survives_the_filter_in_either_arm():
    """The property that matters, asserted against the real manifest: after both
    leakage filters, no row is an evaluation donor."""
    donors = _bd.evaluation_donors()
    man = _pd.read_csv(_bd.MANIFEST)
    for arm in ("B1_sparse_clade", "B2_balanced"):
        pool = man[man.arm == arm]
        kept = pool[~pool.heldout_donor_species]
        base = kept.acc.astype(str).str.split(".").str[0]
        after = set(base[~base.isin(donors)]) & donors
        assert not after, f"{arm}: {len(after)} evaluation donors survive the filter"


def test_the_species_flag_alone_is_insufficient():
    """Pins the defect this filter exists for. If the manifest is ever
    regenerated correctly this test should be deleted, not weakened."""
    donors = _bd.evaluation_donors()
    man = _pd.read_csv(_bd.MANIFEST)
    b2 = man[man.arm == "B2_balanced"]
    trainable = b2[~b2.heldout_donor_species]
    leaked = set(trainable.acc.astype(str).str.split(".").str[0]) & donors
    assert len(leaked) == 24, (
        f"expected the known 24 donors slipping past the species flag, got {len(leaked)}")


def test_clade_structure_survives_the_exclusion():
    """The corpus is structured by clade and must stay that way: dropping the
    donors must not remove a clade or materially reshape the mix."""
    donors = _bd.evaluation_donors()
    man = _pd.read_csv(_bd.MANIFEST)
    b2 = man[(man.arm == "B2_balanced") & (~man.heldout_donor_species)]
    base = b2.acc.astype(str).str.split(".").str[0]
    after = b2[~base.isin(donors)]
    before_c, after_c = b2.clade.value_counts(), after.clade.value_counts()
    assert set(before_c.index) == set(after_c.index), "a clade was emptied entirely"
    before_p = before_c / before_c.sum() * 100
    after_p = after_c / after_c.sum() * 100
    worst = (after_p - before_p).abs().max()
    assert worst < 1.0, f"clade composition shifted by {worst:.2f} pp"


def test_every_build_path_applies_the_taxid_holdout():
    """The species-level holdout must be enforced by taxid on EVERY path, not
    just the new one. Three donor species survived into the sparse-clade corpus
    with the accession filter and the manifest flag both in place, because their
    other records carry heldout_donor_species=False under different accessions.
    """
    src = _Path("src/data/build_dataset.py").read_text()
    assert src.count("donor_species_taxids()") >= 2, \
        "a build path is not applying the taxid holdout"
    i_legacy = src.index("def build(\n")
    assert "donor_species_taxids()" in src[i_legacy:], \
        "the legacy arm path does not apply the taxid holdout"
