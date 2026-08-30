# Part B results — training-composition analysis only

> **Figures removed, 2026-08-26.** No code in this repository generated them, so
> they could not be checked, updated, or regenerated against current data —
> the same defect that retired the accession manifest and the per-clade GenBank
> counts. The underlying tables listed below are reproducible and remain. Figures
> should be re-made by a committed script that reads those tables.

**The finetune has not run.** Nothing here is a generation outcome.

| file | shows |
|---|---|
| `clade_representation.csv` | the numbers behind it |
| `within_clade_diversity.csv` | records per species — headroom is distinct species, not resequencing |
| `og2_plastid_composition.csv` | OpenGenome2 organelle accession composition: 99.7% `NC_` (rest `NW_`/`NT_` RefSeq WGS), zero primary-submission accessions |
| `og2_organelle_summary.csv` | plastid / mitochondrial split |
| `og2_organelle_classification.csv` | organelle records by class, with apicoplasts separated from *rbcL*-bearing plastids |
| `og2_organelle_classified.csv` | per-record class assignment (32,240 rows) |
| `og2_dataset_card.md` | OpenGenome2 dataset card as fetched, for the partition list |
| `clade_coverage_gap.csv` | our own retrieval cap dropping records — the same curation-policy failure mode |

When the run happens, this directory takes `b{1,2}_seed{0,1,2}/` subdirectories
holding `config.resolved.json`, `history.json`, `adapter_best.pt` (gitignored)
and the B1/B2 × clade pass-rate matrix.

## Frame-quality audit (data-pipeline finding, not a generation result)

`frame_audit_by_clade.csv` · `frame_audit_records.csv`
Reproduce: `python -m src.data.frame_audit --per-clade 150`

Building the Part B dataset surfaced a defect severe enough to have invalidated
the primary endpoint, so it is reported rather than quietly fixed.

**What went wrong.** Barcode *rbcL* submissions are overwhelmingly *partial* CDS.
GenBank records the offset of the first complete codon in a `codon_start`
qualifier on the CDS feature. FASTA does not carry that qualifier, and
Biopython's `feature.extract()` does not apply it. Any pipeline that fetches
FASTA and reads from base 1 therefore puts a large fraction of records out of
frame — silently, since an out-of-frame nucleotide sequence is still a valid
string. In the sampled manifest 409 of the 1769 records carrying a CDS feature (23%)
have
`codon_start` > 1.

**Why it mattered.** The error is not uniform across the tree. Under naive
extraction, land-plant clades were 87–99% frame-correct while the algal and
gymnosperm clades were 27–63%:

| clade | naive | corrected |
|---|---|---|
| Conifers | 26.7% | 98.7% |
| Ferns | 26.7% | 99.3% |
| Diatoms | 34.7% | 100.0% |
| Red algae | 52.7% | 100.0% |
| Brown algae | 59.3% | 100.0% |
| Mosses | 98.7% | 98.7% |

Part B's primary endpoint is whether finetuning raises the pass rate on
*under-represented algal clades*. A training corpus whose algal half is ~50%
out of frame while its land-plant half is clean would have produced a clade-
structured difference that looks exactly like the hypothesised effect. The
confound sits directly on the result being measured.

**The fix, and a second bias it exposed.** The pipeline now fetches GenBank
flatfile, reads the annotated CDS, and applies `codon_start`. That alone was not
enough: CDS *annotation availability* is also clade-structured, and in the
opposite direction — 100% of red algal and diatom records carry a CDS feature
but only 4% of eudicot records do. Requiring annotation would have discarded
most eudicots, reproducing the same class of bias with a different sign. The
extractor therefore falls back to inferring the frame by stop-codon search,
accepting a frame only when it is *uniquely* stop-free and rejecting ambiguous
records rather than guessing. Overall frame correctness: 63.5% naive → 84.0%
annotation-only → **95.9%** with fallback, and no clade regresses (minimum
per-clade change +0.0 pp). Annotation coverage does not predict naive frame
correctness (Pearson r = −0.15), confirming the two biases are independent.

`build_dataset.py` additionally *gates* on frame validity: an out-of-frame record
is rejected and counted in the provenance block (`rejected_out_of_frame`) rather
than repaired, so this failure cannot recur silently. Regression tests cover the
`codon_start` offset, the disagreeing-annotation fallback, and ambiguous-record
rejection; their fixtures are computed and self-verifying, because hand-written
repeat sequences are stop-free in all three frames and cannot exercise a frame
bug at all.

## Regenerating these tables

```bash
python -m src.analysis.part_b_tables \
    --run results/generate_b/<run>/part_b --tag l1_1800nt
```

Reads the committed `generated_<arm>.csv` files and re-derives every number:
pass rate per clade per arm, the paired per-donor table with 32-mer containment,
and the transfer table that puts each clade's pass rate beside the number of
records that arm's corpus actually held for it.

Scoring goes through `src.analysis_l1.score`, which calls Part A's own
`is_full_length`. The tables not listed here predate this script and have no
generating code — see the repository README's caveats.
