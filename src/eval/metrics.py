"""Evaluation metrics for generated rbcL sequences.

The primary endpoint is full-length pass rate by clade at a 90 nt prompt --
the exact cell where base Evo 2 scored 0.00 in all four algal groups.
Secondary endpoints guard against the two ways a finetune can look good
without being good: memorisation, and loss of catalytic constraint.
"""
from __future__ import annotations

from Bio.Seq import Seq

STOPS = {"TAA", "TAG", "TGA"}


def is_full_length(nt: str, min_len: int = 1400, max_len: int = 1550) -> bool:
    """Length in range, in frame, single terminal stop, no internal stop."""
    nt = nt.upper()
    if not (min_len <= len(nt) <= max_len) or len(nt) % 3:
        return False
    codons = [nt[i : i + 3] for i in range(0, len(nt), 3)]
    return codons[-1] in STOPS and not any(c in STOPS for c in codons[:-1])


def translate(nt: str) -> str:
    return str(Seq(nt).translate(to_stop=True))


def pass_rate_by_clade(records: list[dict]) -> dict[str, float]:
    """records: [{clade, sequence}, ...] -> {clade: fraction full-length}."""
    agg: dict[str, list[bool]] = {}
    for r in records:
        agg.setdefault(r["clade"], []).append(is_full_length(r["sequence"]))
    return {k: sum(v) / len(v) for k, v in sorted(agg.items())}


def nearest_identity(query: str, references: list[str]) -> float:
    """Max ungapped protein identity of query against references.

    A finetuned model whose output identity against its OWN training set
    exceeds the identity natural rbcL sequences show to each other has
    memorised rather than generalised -- a result to report, not a bug to hide.

    That natural baseline was measured once but its table was deleted on
    2026-08-30 as unreproducible (results/part_a/README.md), so no numeric
    threshold is quoted here. Re-derive it before using this function as a
    memorisation test; src.eval.memorisation.containment is the control that
    Part B actually reports, and it needs no such baseline.

    NOTE: this is an UNGAPPED comparison from index 0, so a single indel shifts
    every downstream position and collapses the score. src.eval.active_site
    aligns instead, and should be preferred where an indel is plausible.
    """
    q = translate(query)
    best = 0.0
    for ref in references:
        r = translate(ref)
        n = min(len(q), len(r))
        if not n:
            continue
        same = sum(a == b for a, b in zip(q[:n], r[:n]))
        best = max(best, same / max(len(q), len(r)))
    return best


def active_site_conserved(nt: str, site_positions: list[int], reference_aa: str) -> float:
    """Fraction of active-site residues matching the reference.

    site_positions are 1-based residue indices into the Spinacia oleracea
    (P00875) numbering; 35 structure-derived positions within 5 A of the
    ligand + Mg2+ site. Requires the caller to have mapped the query into
    reference coordinates -- ungapped comparison of unequal-length sequences
    produces frame-offset artifacts, not divergence. See DESIGN.md.
    """
    aa = translate(nt)
    hits = 0
    for pos in site_positions:
        i = pos - 1
        if i < len(aa) and i < len(reference_aa) and aa[i] == reference_aa[i]:
            hits += 1
    return hits / len(site_positions)
