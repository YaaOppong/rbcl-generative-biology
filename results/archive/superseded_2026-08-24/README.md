# Superseded, 2026-08-24 — do not cite

Two distinct reasons, and they are not interchangeable.

## Orphaned: produced by an adapter that no longer exists

| file | why |
|---|---|
| `l1_1800nt_pass_rate_by_clade.csv` | Part B at L1: base 0.492 → finetuned 0.992, every algal clade 0.000 → 1.000, exact McNemar p = 2.7e-17. |
| `l1_1800nt_paired_with_memorisation.csv` | The same run, paired per donor, with 32-mer containment against the training corpus. |

The result was real and internally sound — both arms scored with Part A's own
`is_full_length`, the adapter verified by sha256 before generation, all 266 LoRA
sites confirmed covered. It is set aside because the **corpus** behind it fails
the QC now in force: 68.3% complete-CDS, 23% of training sequences never
terminating, and 24 evaluation donors inside the training set. The adapter
(`6320b079…`) has been deleted from the weights volume, so these numbers cannot
be regenerated or extended.

The design flaw they exposed is worth keeping in view: training data that never
terminates, used to train a model whose termination is the scored endpoint.

## Not reproducible: numbers no query recovers

| file | why |
|---|---|
| `clade_representation.csv` | Its `rbcl_fullcds` counts do not match GenBank. Red algae recorded 2,338, returns 3,316; Eudicots 6,462 → 7,227; Mosses 780 → 811. Neither `"complete cds"[Title]` (445), `NOT UNVERIFIED[Title]` (3,313) nor a 2026 date cut (3,310) recovers the recorded value. The `headroom` column derives from those counts and inherits the problem. |
| `clade_coverage_gap.csv` | Reports `matches_corpus_query` from a query that was never recorded. Also documents a RETMAX=8000 ordering cap that lost whole clades — diatoms, lycophytes and cycads appear as "retrieved 0.0". |
| `b1_dataset_composition.csv` | Composition of a corpus built under the retired manifest, before the complete-CDS gate and the taxid holdout. |

The reproducible half of that table survives: `evo2_seen` comes from
`src/data/og2_audit.py`, which runs (`--verify-partition`). The GenBank half is
now produced by `src/data/discover_accessions.py`, which states its query,
records it beside the output, and pages through the history server rather than a
bare retmax.

**Retained here rather than deleted**, following this repository's convention for
superseded work: a wrong number with its reason attached is evidence about the
pipeline; a deleted one is a gap someone re-derives later.
