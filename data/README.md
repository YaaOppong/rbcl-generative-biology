# Data

**Sequences are not committed to this repository.** These files hold accession
manifests; `src/data/build_dataset.py` fetches sequences from NCBI GenBank at
build time. Two reasons: it keeps the repository small, and it keeps provenance
auditable — every training sequence traces to a GenBank accession with its
submitter attribution intact.

| file | rows | contents |
|---|---|---|
| `finetune_accessions.csv` | 11,619 | candidate pool for both arms: accession, taxid, organism, clade, CDS length, donor-species flag |
| `finetune_design.csv` | 29 | per-arm × clade record / species / nucleotide counts after the holdout |
| `excluded_donor_species.csv` | 510 | records excluded because their species appears in the evaluation donor panel |

## The species-level holdout

Evaluation uses 120 donor sequences. All 120 appear in the candidate pool by
accession — and their *species* contribute 510 further records, 223 of them
inside the B1 pool. Excluding the 120 accessions alone would still train on the
exact lineages being evaluated, under different accession numbers. The
`heldout_donor_species` column drives exclusion at species level.

## Clade assignment

`clade` is assigned from NCBI taxonomic lineage. Records without a resolvable
clade are retained in the manifest but excluded from both arms.

## Attribution

Sequence data from NCBI GenBank and RefSeq. Barcode submissions carry
collection locality and voucher metadata contributed by their original
submitters. If you reuse these manifests, cite GenBank and preserve the
accession chain.
