"""Active-site integrity: does a generated rbcL keep Rubisco catalytic?

The third endpoint in the harness, and the one that decides whether "valid CDS"
means anything biologically. A sequence can pass every length/frame/stop check
and still encode a dead enzyme if the catalytic residues are wrong.

The 11 positions are the RuBP-binding lysines, the carbamylated Lys201 that
activates the enzyme, the Asp/Glu pair coordinating the catalytic Mg2+, and the
loop-6 and downstream site residues. Numbering follows Spinacia oleracea P00875.

PROVENANCE. This set was selected as the near-invariant subset of the active
site in a conservation analysis over a natural rbcL alignment. That analysis was
run by code that was never committed, and its table was deleted on 2026-08-30
rather than shipped unreproducibly (results/part_a/README.md). The residue
identities below are checkable against P00875 and the Rubisco literature
independently of it, but the 99.5-100% conservation figure that motivated the
cut is NOT reproducible from this repository and is not asserted here.

Mapping is by ALIGNMENT, never by direct indexing. metrics.active_site_conserved
warns about this and it is not hypothetical: a single indel shifts every
downstream index, so naive indexing reports catalytic loss where there is only a
frame offset.

MINIMUM LENGTH. The last of the 11 residues (Gly380) ends at nucleotide 1140, so
a generation shorter than that cannot be scored on the full set -- it is not that
the residues are absent, it is that they are out of reach. The first Part B run
generated 300 nt and scored 0/11 across every sequence in both arms; that is an
artefact of protocol, not a finding about the model. `min_covered_nt` makes the
constraint explicit so it cannot be misread again.
"""
from __future__ import annotations

from Bio import Align
from Bio.Seq import Seq

# Spinacia oleracea P00875 1-based residue -> expected amino acid.
CATALYTIC = {175: "K", 177: "K", 201: "K", 203: "D", 204: "E",
             294: "H", 295: "R", 327: "H", 334: "K", 379: "S", 380: "G"}

MIN_COVERED_NT = max(CATALYTIC) * 3   # 1140: shorter cannot reach all 11


def _aligner():
    a = Align.PairwiseAligner(scoring="blastp")
    a.mode = "global"
    a.open_gap_score, a.extend_gap_score = -11, -1
    return a


def load_reference(fasta_path: str) -> str:
    """P00875 protein sequence from a FASTA (data/spinacia_P00875.fasta)."""
    with open(fasta_path) as fh:
        lines = [ln.strip() for ln in fh if not ln.startswith(">")]
    return "".join(lines)


def score_sequence(seq, reference: str, is_protein: bool = False,
                   aligner=None) -> dict | None:
    """Map onto the reference by alignment and read the catalytic columns.

    Returns None when the translation is too short to be meaningful (<100 aa).
    `covered` reports how many of the 11 positions the sequence is long enough
    to reach, so an out-of-reach residue is never scored as a substitution.
    """
    aligner = aligner or _aligner()
    q = seq if is_protein else str(Seq(str(seq).upper()).translate(to_stop=True))
    if len(q) < 100:
        return None
    aln = aligner.align(reference, q)[0]
    mapped: dict[int, str] = {}
    for (rs, re_), (qs, _qe) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(re_ - rs):
            mapped[rs + k] = q[qs + k]
    hits = {p: mapped.get(p - 1) for p in CATALYTIC}
    covered = sum(1 for p in CATALYTIC if p <= len(q))
    return {
        "aa_len": len(q),
        "identity": sum(1 for a, b in zip(aln[0], aln[1])
                        if a == b and a != "-") / len(reference),
        "hits": hits,
        "n_correct": sum(1 for p, aa in CATALYTIC.items() if hits[p] == aa),
        "n_covered": covered,
        "scorable": covered == len(CATALYTIC),
    }
