"""How much of a generated sequence is copied from the training corpus.

A finetuned model that reproduces training sequences verbatim will score
perfectly on every structural metric in this repo -- length, frame, stop codons,
active-site residues -- while having generated nothing. So no generation result
from a finetuned arm is interpretable without this control beside it.

The measure is k-mer containment: the fraction of a sequence's k-mers that occur
anywhere in the training corpus. k=32 because it is long enough that sharing one
by chance is negligible (4^32 ~ 1.8e19 against a corpus of ~5.4e6 k-mers) and
short enough to catch a copied fragment rather than only a copied record.

Containment, not identity to the nearest neighbour: a sequence assembled from
two training records in halves is ~100% contained while having no close
neighbour, and that is still copying.

Read the score as evidence about the sequence, not a verdict:
  ~1.0  every fragment occurs in training. Recall, not synthesis.
  ~0.0  no shared fragment.
  between  a mosaic -- worth looking at directly rather than thresholding.

The base arm belongs in the same table as a reference: Evo 2's pretraining
included RefSeq plastid genomes, so some containment is expected without any
finetuning at all, and only the DIFFERENCE between arms is attributable to it.
"""
from __future__ import annotations

K = 32


def kmers(seq: str, k: int = K) -> set[int]:
    """Hashed k-mers of one sequence. Hashes, because the corpus holds millions
    and the strings themselves are an order of magnitude more memory."""
    s = str(seq).upper()
    return {hash(s[i:i + k]) for i in range(len(s) - k + 1)}


def corpus_kmers(sequences, k: int = K) -> set[int]:
    out: set[int] = set()
    for s in sequences:
        out |= kmers(s, k)
    return out


def containment(seq: str, reference: set[int], k: int = K) -> float:
    """Fraction of this sequence's k-mers that occur in the reference set.

    Returns 0.0 for a sequence shorter than k: no k-mer is evidence of nothing
    copied, not evidence of novelty, and the caller should exclude such rows.
    """
    ks = kmers(seq, k)
    if not ks:
        return 0.0
    return len(ks & reference) / len(ks)
