"""The memorisation control.

The 52% figure quoted in the READMEs had no code behind it -- it could not be
recomputed, checked, or applied to a new run. These tests pin the behaviour of
the implementation that replaces it.
"""
import random

from src.eval.memorisation import K, containment, corpus_kmers, kmers

TRAIN = ["ATGGCACCTGATTACGAAACCAAAGATACTGATATCTTGGCAGCATTCCGAGTAACTCCTCAACCTGGA" * 3]


def test_a_verbatim_copy_is_fully_contained():
    ref = corpus_kmers(TRAIN)
    assert containment(TRAIN[0], ref) == 1.0


def test_unrelated_sequence_is_not_contained():
    ref = corpus_kmers(TRAIN)
    assert containment("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT", ref) == 0.0


def test_a_mosaic_of_two_records_scores_high_without_a_near_neighbour():
    """Why containment rather than nearest-neighbour identity: a sequence spliced
    from two training records is copied, but resembles neither.

    At rbcL length only the ~31 k-mers straddling the seam are novel, so a
    two-piece mosaic scores ~0.98. The seam cost is why this test uses realistic
    lengths: on a 114 nt toy it would score 0.64 and look like a weak signal.
    """
    # non-periodic: a repeated motif has few DISTINCT k-mers, so the seam would
    # dominate the set and the test would measure the fixture, not the method
    rng = random.Random(0)
    a = "".join(rng.choice("ACGT") for _ in range(750))
    b = "".join(rng.choice("ACGT") for _ in range(750))
    ref = corpus_kmers([a, b])
    mosaic = a[:len(a) // 2] + b[:len(b) // 2]
    assert len(mosaic) > 700, "the point is that the seam is negligible at length"
    assert containment(mosaic, ref) > 0.9


def test_half_copied_scores_near_half():
    ref = corpus_kmers(TRAIN)
    novel = "".join("ACGT"[(i * 7 + i // 3) % 4] for i in range(len(TRAIN[0]) // 2))
    c = containment(TRAIN[0][:len(TRAIN[0]) // 2] + novel, ref)
    assert 0.3 < c < 0.75, c


def test_short_sequences_do_not_report_novelty():
    """No k-mer means no evidence, not evidence of novelty."""
    assert containment("ACGT", corpus_kmers(TRAIN)) == 0.0
    assert kmers("ACGT") == set()


def test_k_is_long_enough_that_chance_sharing_is_negligible():
    assert K == 32
