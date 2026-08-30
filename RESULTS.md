# Results

Every result in this repository, with the table it comes from and the command
that rebuilds it. Method for all of it is in **[METHODS.md](METHODS.md)**; this
document states *what was found*, not *how*.

> ### Maintenance
>
> **This document is the single source of truth for findings.** The README
> summarises the headline and links here; it carries no results tables of its
> own, because two statements of one number means one of them goes stale.
>
> **Update this file in the same commit as any change to:** a results table
> under `results/`; a generation, training or analysis run; a scoring or endpoint
> definition that moves a number; the set of results that are reproducible; or
> anything withdrawn or restored.
>
> Every number here must be reproducible by a committed command. If a number
> changes, change it here first, then anywhere else it appears. A number in this
> file that no command reproduces is the defect this document exists to prevent.
> See also METHODS.md's maintenance note.

---

## Contents

- [1. Headline](#1-headline)
- [2. Part A — generation without finetuning](#2-part-a--generation-without-finetuning)
- [3. Part B — finetuning](#3-part-b--finetuning)
- [4. Memorisation](#4-memorisation)
- [5. Reproducibility results](#5-reproducibility-results)
- [6. Data-quality findings](#6-data-quality-findings)
- [7. Withdrawn results](#7-withdrawn-results)

---

## 1. Headline

Finetuning repairs Part A's failure completely — at a 90 nt prompt, full-length
generation goes **0.4917 → 0.9833**, paired across 120 donors, exact McNemar
p = 5.4e-17. Every clade where base Evo 2 scores zero reaches 1.000.

But it does **not** repair it by supplying missing coverage. An adapter trained
on a corpus containing no green algae at all takes green algae from 0.000 to
1.000, and across the 66 donors from clades genuinely absent from that corpus,
0.712 → 0.970 (p = 7.6e-05). Three corpora differing three-fold in size and
four-fold in clade breadth give identical results to four decimal places.

So the failure is not a knowledge gap about particular lineages. Base Evo 2 has
the sequence knowledge; a short lineage-specific prompt destroys its ability to
commit to a coherent output, and any in-domain finetuning restores it.

**This reading is testable and the test has not been run** — an arm trained on
land plants only and evaluated on algae would falsify it.

## 2. Part A — generation without finetuning

1,800 sequences: 120 donors × 5 prompt levels × 3 replicates, base Evo 2, 1,500 nt
budget. Plus a 2026-08-25 rerun of all five levels at 1,800 nt, replicate 0 only.

Rebuild: `python -m src.analysis.part_a_tables --corpus data/part_a_generated_corpus.csv`

### R6 — short prompts are worse than no prompt

The central result. Pass rate is **non-monotonic** in prompt length.

`generation_qc.csv` (1,500 nt, n=360/level):

| level | prefix | n | read-through | full-length | pass rate |
|---|---|---|---|---|---|
| `L0_shared_seed` | 30 nt | 360 | 0 | 350 | **0.9722** |
| `L1_donor_90` | 90 nt | 360 | 25 | 178 | **0.4944** |
| `L2_donor_210` | 210 nt | 360 | 8 | 307 | 0.8528 |
| `L3_donor_450` | 450 nt | 360 | 3 | 354 | 0.9833 |
| `L4_donor_900` | 900 nt | 360 | 3 | 353 | 0.9806 |

**Pooled across all levels: 0.8567 full-length** (this is R3).

Reproduced on a fresh run at 1,800 nt, replicate 0 (`titration_1800nt_rep0.csv`,
n=120/level): 0.9917 / 0.4917 / 0.8667 / 0.9667 / 0.9833 — **maximum deviation
0.020**. The 1,500 nt ceiling censored 6.9% of L1 generations; removing it moves
L1 by 0.3 points, so the censoring artefact was real and immaterial.

### R6 by clade — the collapse is clade-structured

`l1_interference.csv` (1,500 nt), sorted by the L0→L1 drop:

| clade | L0 | L1 | L2 | L3 | L4 | Δ L0→L1 | n/level |
|---|---|---|---|---|---|---|---|
| Brown algae | 1.000 | **0.000** | 0.571 | 0.952 | 0.810 | −1.000 | 21 |
| Other green | 1.000 | **0.000** | 1.000 | 0.833 | 1.000 | −1.000 | 6 |
| Red algae | 0.970 | **0.000** | 0.576 | 0.970 | 1.000 | −0.970 | 33 |
| Green algae | 0.967 | **0.000** | 0.667 | 1.000 | 0.967 | −0.967 | 30 |
| SAR/other protist | 1.000 | 0.091 | 0.697 | 1.000 | 0.939 | −0.909 | 33 |
| Mosses | 0.952 | 0.167 | 1.000 | 1.000 | 1.000 | −0.786 | 42 |
| Ferns | 1.000 | 0.733 | 0.933 | 0.933 | 1.000 | −0.267 | 15 |
| Liverworts | 0.909 | 0.697 | 0.879 | 0.970 | 1.000 | −0.212 | 33 |
| Conifers | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | −0.167 | 6 |
| Monocots | 1.000 | 0.909 | 1.000 | 1.000 | 1.000 | −0.091 | 33 |
| Eudicots | 0.988 | 0.901 | 0.988 | 0.988 | 1.000 | −0.086 | 81 |
| Other angiosperms | 0.926 | 0.963 | 0.852 | 1.000 | 1.000 | +0.037 | 27 |

Pass rate is **zero in four algal groups** at 90 nt, and the failure is
all-or-nothing per donor. Land plants are barely affected.

**This is not a prompt artefact.** All 360 L1 prompts translate stop-free in
frame 0 and all 360 begin with ATG
([METHODS §4.3](METHODS.md#43-prompt-integrity)).

**L0 is not a neutral reference.** The shared 30 nt seed is a land-plant start
region, so the L0 baseline is confounded in a direction that flatters it.

### R2a — active-site integrity holds where generation terminates

`l1_active_site_per_residue.csv`. Eleven catalytic residues, mapped onto
*Spinacia oleracea* P00875 by alignment. All eleven match the reference exactly.

| population | n | correct-residue fraction |
|---|---|---|
| full-length L1 generations | 178 | **0.9663 – 0.9944** |
| all generations reaching residue 380 | 193 | 0.9067 – 0.9378 |

Lowest in both populations is His327; highest is Lys201/Glu204 (full-length) and
Lys201/Glu204 (scorable). The gap between populations is informative: long but
broken ORFs lose catalytic residues that terminating ones keep.

Rebuild: `python -m src.analysis.part_a_active_site --corpus data/part_a_generated_corpus.csv --level L1_donor_90 --tag l1`

## 3. Part B — finetuning

One paired generation run, L1 (90 nt prompt), 1,800 nt budget, 120 donors,
replicate 0, four arms. Rebuild:
`python -m src.analysis.part_b_tables --run results/generate_b/<run>/part_b --tag l1_1800nt`

### Overall

| arm | full-length | read-through | vs base, paired |
|---|---|---|---|
| base | 0.4917 | 0.0500 | — |
| `all_fullcds` | **0.9833** | 0.0000 | 60 fail→pass, 1 pass→fail, p = 5.4e-17 |
| `all_fullcds_atg` | **0.9833** | 0.0083 | 61 / 2, p = 4.4e-16 |
| `b1_sparse_clade` | **0.9833** | 0.0083 | 60 / 1, p = 5.4e-17 |

**The three corpora are indistinguishable to four decimal places**, despite
differing three-fold in size (2,370 – 6,946 records) and four-fold in clade
breadth (5 – 19 clades).

The six residual failures fall on four distinct donors, two of them shared
between arms (`MK806439.1` fails in both `all_fullcds` arms, `PX744018.1` in
`all_fullcds` and `b1_sparse_clade`). At n=6 that is too few to call stochastic
or structural either way.

**Pass rate has no headroom left once the failure is repaired**, so it cannot
arbitrate between corpus designs. The start-codon question is *not* answered.

### By clade

`l1_1800nt_pass_rate_by_clade.csv`:

| clade | base | `all_fullcds` | `all_fullcds_atg` | `b1_sparse_clade` |
|---|---|---|---|---|
| Green algae | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Other green | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Red algae | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| Mosses | 0.0714 | 1.0000 | 1.0000 | 1.0000 |
| Brown algae | 0.1429 | 1.0000 | 1.0000 | 1.0000 |
| SAR/other protist | 0.1818 | 1.0000 | 1.0000 | 1.0000 |
| Ferns | 0.6000 | 0.8000 | 0.8000 | 1.0000 |
| Liverworts | 0.7273 | 1.0000 | 1.0000 | 1.0000 |
| Eudicots | 0.8519 | 0.9630 | 1.0000 | 0.9630 |
| Monocots | 0.9091 | 1.0000 | 0.9091 | 0.9091 |
| Conifers | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Other angiosperms | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**No arm regresses in angiosperms**, so the flattening cost the design worried
about did not appear at L1. It remains untested at levels where base succeeds.

### Transfer to clades absent from the corpus

`l1_1800nt_transfer.csv`. `b1_sparse_clade` trains on Red algae, Diatoms,
Mosses, Liverworts and Brown algae only.

| clade | records in b1 | n donors | base | `b1_sparse_clade` |
|---|---|---|---|---|
| Green algae | 0 | 10 | 0.0000 | 1.0000 |
| Other green | 0 | 2 | 0.0000 | 1.0000 |
| Ferns | 0 | 5 | 0.6000 | 1.0000 |
| Eudicots | 0 | 27 | 0.8519 | 0.9630 |
| Monocots | 0 | 11 | 0.9091 | 0.9091 |
| Conifers | 0 | 2 | 1.0000 | 1.0000 |
| Other angiosperms | 0 | 9 | 1.0000 | 1.0000 |

Pooled over these **66 donors**: 0.712 → 0.970, 18 fail→pass against 1, exact
McNemar **p = 7.6e-05**.

> **SAR/other protist is excluded from that pool** despite reading
> `records_in_corpus = 0`. The corpus labels diatoms as their own clade, and
> those 523 records are the same phylum (Ochrophyta) as the SAR evaluation
> donors, so the clade is covered rather than absent. Counting it would inflate
> the pool to 77 donors and the result to p = 2.2e-07.

## 4. Memorisation

`l1_1800nt_containment.csv` — 32-mer containment, **every arm against every
corpus**, because the measure is reference-dependent.

| arm | vs `all_fullcds` | vs `all_fullcds_atg` | vs `b1_sparse_clade` |
|---|---|---|---|
| base | 0.386 | 0.351 | 0.000 |
| `all_fullcds` | **0.525** | 0.467 | 0.167 |
| `all_fullcds_atg` | 0.504 | **0.496** | 0.143 |
| `b1_sparse_clade` | 0.185 | 0.177 | **0.168** |

*(fraction of generations with >90% of their 32-mers in the reference corpus;
bold = arm's own training corpus)*

**Base scores 0.386 against a corpus it never saw.** *rbcL* is conserved enough
that a correct novel sequence necessarily shares 32-mers with natural records, so
roughly that much of any arm's containment is conservation, not copying.

**Read as lift over base on the same corpus, never raw:**

| arm | own corpus | base, same corpus | lift |
|---|---|---|---|
| `all_fullcds` | 0.525 | 0.386 | +0.139 |
| `all_fullcds_atg` | 0.496 | 0.351 | +0.145 |
| `b1_sparse_clade` | 0.168 | 0.000 | **+0.168** |

On that basis **`b1_sparse_clade` memorises most, not least** — its low raw 0.168
merely reflects a narrower reference. Mechanistically unsurprising: 2,370 records
across five clades are easier to reproduce than 6,946 across nineteen.

This is **the one endpoint on which the three corpora visibly differ**, and it
differs in the direction that matters: the smallest corpus buys the same pass
rate at the highest copying cost.

## 5. Reproducibility results

**Byte-level determinism.** The base arm was generated independently in two runs,
on different days, from different commits, in different sandboxes. On the 120
shared L1 prompts, **all 120 sequences are byte-identical**
(`src/analysis/determinism.py`). This is what licenses the paired comparisons
above.

**Table regeneration.** All seven Part A tables and all four Part B tables
regenerate **byte-identically** from committed inputs.

**Scorer discrepancy.** The Part A corpus carries a stored `full_length` column
whose scorer is not in this repository. It disagrees with `is_full_length` on
**16 of 1,800 rows** — L1:4, L2:4, L3:6, L4:2, L0 clean — every one in the same
direction (stored `False`, recomputed `True`). Cause not established; all tables
report the recomputed value. Effect per level ≤ 1.7 points, no conclusion
changed. Rows listed in `full_length_recount_discrepancies.csv`.

**Sampling.** `top_k=4` restricts sampling to the four highest-probability
tokens, usually but not always the four nucleotides: 17 of 1,800 Part A
sequences contain a literal space, in every case *after* the terminal stop, so no
scored CDS is affected. Present in the base arm of the reruns (1/120 and 5/600)
and **absent from all three finetuned arms** (0/120 each).

## 6. Data-quality findings

### Frame recovery in algal barcode records

`frame_audit_by_clade.csv`. Naive FASTA extraction leaves roughly half of algal
*rbcL* records out of frame while land plants are largely unaffected, because
barcode submissions are partial CDS carrying a `codon_start` offset that FASTA
does not expose.

| clade | n | % with CDS annotation | % in frame, naive | % in frame, fixed |
|---|---|---|---|---|
| Conifers | 150 | 100.0 | 26.7 | 98.7 |
| Ferns | 150 | 88.7 | 26.7 | 99.3 |
| Diatoms | 150 | 100.0 | 34.7 | 100.0 |
| Magnoliids | 150 | 72.0 | 41.3 | 98.0 |
| Red algae | 150 | 100.0 | 52.7 | 100.0 |
| Brown algae | 150 | 100.0 | 59.3 | 100.0 |
| Eudicots | 150 | **4.0** | 86.7 | 87.3 |
| Green algae | 150 | 100.0 | 97.3 | 98.0 |
| Mosses | 150 | 98.7 | 98.7 | 98.7 |

Since the primary endpoint is algal pass rate, that confound would have sat
directly on the result.

**CDS annotation is itself clade-structured** — 100% of red algal and diatom
records carry a CDS feature against **4%** of eudicot records — so requiring
annotation would silently discard whole clades. Hence the stop-codon frame
inference fallback.

### Evo 2's plastid exposure

`og2_organelle_classification.csv`. The OpenGenome2 organelle partition:

| class | records | total nt | % records | % nt |
|---|---|---|---|---|
| photosynthetic plastid | 14,608 | 2.24e9 | 45.3 | 79.4 |
| mitochondrion | 17,613 | 5.81e8 | 54.6 | 20.6 |
| apicoplast (*rbcL*-less) | 17 | 573,797 | 0.05 | 0.02 |
| unclassified | 2 | 62,164 | 0.01 | 0.00 |

All 32,240 organelle accessions carry `NC_` prefixes — **100% RefSeq, zero
primary submissions**, while barcode *rbcL* records carry primary accessions
exclusively. Implied *rbcL* exposure ≈ 22 M tokens, **2.4 ppm** of the 9.3 T
trained.

## 7. Withdrawn results

**Withdrawn 2026-08-30:** R1 (natural novelty baseline), R2b (active-site
conservation across natural sequences and its agreement with structure), R4
(lineage pull), R5 (seed lineage bias) and the kinetics oracle, together with
the eleven tables backing them.

Each was produced by code that was never committed, from inputs this repository
does not hold. **They are not cited anywhere here and should not be cited from
here.** Full account, including what is recoverable and what is not:
[METHODS §12](METHODS.md#12-what-is-not-reproducible).

**Restored 2026-08-30:** R2a, above — validated against the deleted original
before it was removed, reproducing all eleven of its per-residue counts exactly.

---

## What these results do not establish

1. **Single training seed (0) per corpus** — between-corpus comparisons are
   confounded with training noise.
2. **The endpoint saturates** at 0.9833, so corpus designs cannot be ranked.
3. **The endpoint is largely syntactic** — applied to a CDS, the full-length
   predicate reduces to a length window. R2a is the only structural endpoint.
4. **One prompt level for Part B** (L1, replicate 0). Whether finetuning costs
   anything where base already succeeds is untested.
5. **One model.** Nothing separates Evo 2's behaviour from genomic language
   models generally.
6. **The falsifying experiment is unrun** — land plants only, evaluated on algae.
