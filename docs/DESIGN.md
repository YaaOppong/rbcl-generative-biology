# Design rationale

Written before the Part B run, so the predictions below are pre-registered
rather than retrofitted. Every number was read back from a saved table.

## 1. The observation Part B explains

Part A generated 1,800 sequences: 120 donors × 5 prompt levels (0, 90, 210,
450, 900 nt of the donor's own 5' CDS) × 3 replicates. Full-length pass rate
by level:

| level | prompt | pass rate |
|---|---|---|
| L0 | none (30 nt shared seed) | 0.972 |
| L1 | 90 nt | **0.483** |
| L2 | 210 nt | 0.842 |
| L3 | 450 nt | 0.967 |
| L4 | 900 nt | 0.975 |

Non-monotonic, and the dip is clade-structured: zero pass rate in all four
algal groups at L1, near-baseline in land plants. Per donor the outcome is
all-or-nothing — a donor either passes in all three replicates or none, which
is why three seeds are mandatory downstream. A single seed cannot distinguish a
real shift from sampling.

L0 is not a clean reference. The shared 30 nt seed is a land-plant start
region, giving land-plant lineage accuracy 0.44 against algal 0.00. The
"unconditioned" baseline carries a lineage prior.

## 2. Part B — the coverage hypothesis

### Why coverage is a plausible cause

Evo 2's organelle training set is 32,240 accessions, every one prefixed `NC_`:
100% RefSeq, zero GenBank submissions.

| | count | OG2 share |
|---|---|---|
| RefSeq plastid genomes | 14,798 | 98.7% |
| GenBank plastid genomes | 66,789 | 22.2% |
| GenBank *rbcL* records | 402,723 | 3.67% |

RefSeq is one genome per species, curated, taxonomically thin. GenBank barcode
submissions are population-level. The model saw a near-complete sweep of the
former and almost none of the latter — a gap created by curation policy.

### Headroom is clade-structured and the aggregate conceals it

Full-CDS *rbcL* records in GenBank per RefSeq plastid genome, within clade:

| clade | records | genomes seen | headroom | novel species | novel/genome |
|---|---|---|---|---|---|
| Mosses | 780 | 51 | **15.3×** | 382 | **7.49** |
| Diatoms | 1,266 | 98 | **12.9×** | 461 | 4.70 |
| Liverworts | 448 | 44 | **10.2×** | 229 | 5.20 |
| Red algae | 2,338 | 232 | **10.1×** | 690 | 2.97 |
| Brown algae | 354 | 68 | **5.2×** | 145 | 2.13 |
| Eudicots | 6,462 | 9,141 | 0.7× | **4,303** | 0.47 |
| Monocots | 1,862 | 3,218 | 0.6× | 912 | 0.28 |

Globally the ratio is 1.03× — aggregate parity. Per clade it spans 0.5× to
15.3×. Simpson's-paradox structure in a training set, and the reason a global
coverage statistic cannot locate where a finetune would help.

Headroom is not redundancy: records per species runs 1.28 (eudicots) to 3.06
(red algae), so moss headroom is ~396 distinct species rather than repeated
resequencing of a few. 90.8% of the focal five's 2,101 species are absent from
Evo 2's plastid training.

### Why two arms

Eudicots offer 4,303 novel species against 1,907 for the focal five combined —
2.3× more new sequence in absolute terms despite sub-parity headroom. Relative
increment and absolute pool answer different questions, so a sparse-clade-only
finetune deliberately discards the larger pool. B2 is therefore scientifically
necessary, not a control.

| arm | selection | records | species | tokens/epoch |
|---|---|---|---|---|
| B1 | five clades with ≥5× headroom | 4,963 | 2,059 | 7.14 M |
| B2 | all clades, capped at 780 each | 6,146 | 2,964 | 8.81 M |

780 is the focal-five median record count. The cap pulls eudicots 6,462 → 752
and monocots 1,862 → 773. That flattening *is* the intervention.

Both arms are small — 7–9 M tokens per epoch against Evo 2's pretraining corpus
is a rounding error. This is adaptation, not retraining, and the expected effect
size should be stated as modest before the run rather than after.

### Leakage control

Evaluation uses 120 donor sequences. All 120 appear in the GenBank candidate
pool by accession, and their species contribute 510 further records overall —
223 inside the B1 pool (4.3%), spanning 42 species. Excluding accessions is not
enough: the same species under a different accession still leaks the lineage
being evaluated. The holdout operates at species level.

Validation splits for early stopping are stratified by clade, so early stopping
is not driven by whichever clade dominates the arm.

### Reading-frame integrity — resolved

A confound found while building the dataset and fixed before any GPU time, kept
here because it is the kind that survives review unnoticed. Barcode *rbcL*
submissions are partial CDS whose first complete codon is given by GenBank's
`codon_start` qualifier; FASTA does not carry it, and `feature.extract()` does
not apply it. Naive extraction therefore left 37% of records out of frame —
**and the error was clade-structured in the direction of the primary endpoint**
(algae and gymnosperms 27–63% frame-correct, land plants 87–99%). A corpus with
that structure would have produced a clade-specific difference indistinguishable
from the hypothesised finetuning effect.

Fixed by reading annotated CDS with `codon_start` applied, falling back to
unique-stop-free frame inference where annotation is absent — necessary because
annotation coverage is *also* clade-structured, in the opposite direction (100%
of red algal records annotated vs 4% of eudicots), so requiring annotation would
have reintroduced the same bias with the sign flipped. 63.5% → 95.9%
frame-correct, no clade regressing. `build_dataset.py` now rejects out-of-frame
records and counts them in its provenance block. Full result:
`results/part_b/README.md`.

### Model scale — RESOLVED: 7B

**Decision: every scientific arm runs at `evo2_7b`.** The earlier choice of 1B
is reversed.

The NVIDIA recipe is published for Evo2-1B; Part A generated with 7B
(`generate_corpus.py` hardcodes `Evo2("evo2_7b")`). **A 1B finetune cannot be
compared against 7B baselines** — Part A's whole baseline, the 1,800 sequences
and the per-clade failure rates Part B is measured against, is 7B, so a 1B
finetune entangles the finetuning effect with model scale.

Why 1B was chosen first, and why that was wrong: the reasoning was cost (10.5 vs
41.8 GPU-h) with Part A's evaluation re-run at 1B to restore the control. But
that re-run is most of the saving, so the gap is much smaller than the headline
suggests — and both figures are unvalidated estimates. A later throughput
estimate in this project was wrong by an order of magnitude, so neither number
should be trusted until measured. Trading a matched control for an unverified
saving is the wrong trade. The 1B path was method convenience — NVIDIA's
tutorial happens to use 1B — not a scientific argument, and `Evo2LoRA` works at
either scale regardless (adapter targets verified to transfer unchanged).

Empirically decisive: 7B has run successfully (weights cached, one completed
finetune with real numbers), while **1B has never once loaded** — every attempt
fell back to 7B via the fp8 gate, latterly because the patched config was
written to `/tmp` while evo2 reads it package-relative. That bug is fixed but
still untested, so the "cheaper" path remains unproven while the expensive one
is proven.

`demo_small.yaml` stays at `evo2_1b_base` deliberately: it is a smoke-test
config, explicitly not a scientific arm, and is the right place to answer the
separate question of whether 1B loads at all now.

The adapter config transfers across both scales without edits: `evo2-1b-8k.yml`
and `evo2-7b-1m.yml` differ only in width, depth (25 vs 32 layers) and attention
layer indices, and both build `StripedHyena2` from the same two block classes via
`get_block()`. The shipped `target_modules` resolve at both depths (104 injection
sites at 1B, 133 at 7B) with no layer skipped and the tuple-returning
`blocks.N.projections` never matched. Pinned by `tests/test_lora.py`.

**Dependency — 1B needs an fp8 workaround.** Both configs set
`use_fp8_input_projections: true`, which requires Transformer Engine. evo2 0.5.5
falls back to bf16 projections when TE is absent, but the fallback is gated on a
literal `"7b"` substring in the model name or config path (`is_7b_model` in
`evo2/models.py`); every other checkpoint raises `ImportError` instead. The
GPU environment used here deliberately omits TE — its import-time CUDA init hangs
during image save, and TE 1.13 caps flash-attn at ≤2.6.3. So 1B must be loaded
from a config copy with the flag set to false, taking the same numerical path as
the maintainers' own 7B fallback. **Untested as of this writing**; a GPU smoke
test gates the decision, and 7B (already working, weights cached) is the fallback
if the patch misbehaves. Note `import evo2` requires a live CUDA driver, so this
cannot be checked on CPU.

### Train/validation split — measured, not assumed

"Held out" is ambiguous for a barcode locus, and the ambiguity is quantitative.
After the exact-duplicate gate, conspecific *rbcL* records are **median 99.26%
identical** (56.6% of pairs >=99%, n=1,400) against **88.68%** for
different-species pairs within the same clade (0.3% >=99%, n=1,494). Under a
record-level split, **71% of B1 validation records had their species in
training**, so validation loss was substantially a memorisation readout.

A pure species-disjoint split fixes leakage but overcorrects: 1,516 of 2,079
species are singletons (42% of records), and holding out whole species strands
~20% of species from the gradient — real taxonomic diversity lost for a
hypothesis that is *about* taxonomic structure.

Both are therefore reported, from one 20% hold-back:

| set | construction | measures | drives early stopping |
|---|---|---|---|
| `val_novel` (10.1%) | whole species held out | generalisation to an unseen species | yes |
| `val_seen` (10.0%) | one record from species remaining in training | fit on familiar species | no |

Their per-epoch difference is logged as `memorisation_gap`. This turns the
design question into a measured quantity: if the gap is small, conspecific
leakage never mattered on this corpus. Species reaching training rises from 80%
(species-disjoint) to **89%**. Evidence: `b1_dataset_composition.csv`, archived 2026-08-24 (it was built
under the retired manifest, before the complete-CDS gate and the taxid holdout).

`val_seen` is capped by donor availability — one record per multi-record species
that stays in training — so the realised hold-back is `<= val_fraction`, never
more (20.1% on B1). Under-holding is safe; over-holding would silently shrink
the training set.

Two implementation bugs were caught in development and are pinned by
`tests/test_split.py`: placing largest species first **inverted the split** (80%
validation against a 10% target), and sorting by size after shuffling made the
seed inert — which would have made the three-seed replicates identical while
appearing to vary.

### The B1 corpus as built

3,607 unique CDS / 2,079 species / 5.15 M nt, from 4,963 manifest rows after the
species-level donor holdout: 152 unmatched at GenBank, 2 below the 1,000 nt
floor, 0 rejected out of frame, and **1,202 exact duplicates removed (25%)** —
the same species sequenced by different labs over the same conserved locus.
Duplicates contribute no signal; they upweight a sequence in the gradient by
sequencing effort, which is a curation artifact.

Composition is uneven by design — "sparse-clade" names *which* clades were
chosen (the five with >=5x coverage headroom), not that they are equally
represented: Red algae 42.8%, Diatoms 28.7%, Mosses 12.6%, Liverworts 8.9%,
Brown algae 7.0%. The headroom ranking is close to the reverse of the record
ranking, and any clade-level reading of the result has to account for that.

### Pre-registered predictions

| outcome | interpretation |
|---|---|
| B1 recovers algal L1 pass, B2 partially | coverage deficit — supports the curation-gap account |
| Neither recovers | architectural or prompt-length limit; coverage is not the cause |
| B1 recovers, B2 regresses in angiosperms | flattening has a real cost; quantify the trade |
| Both recover equally | clade identity of finetune data is irrelevant — generic domain adaptation |

Primary endpoint: full-length pass rate by clade at L1 (90 nt), the exact cell
where the base model scored zero in four algal groups. Secondary: lineage pull,
novelty against training, active-site constraint satisfaction.


### Outcome, 2026-08-25 — scored against the predictions above

**"Both recover equally" is what happened**, and its registered interpretation
was *clade identity of finetune data is irrelevant — generic domain adaptation*.
That reading therefore stands as pre-registered rather than post-hoc.

Three corpora, one paired run at L1, 120 donors each:

| arm | records | clades | algal | full-length |
|---|---|---|---|---|
| base | — | — | — | 0.4917 |
| `all_fullcds` | 6,946 | 19 | 38.2% | 0.9833 |
| `all_fullcds_atg` | 2,937 | 17 | 22.7% | 0.9833 |
| `b1_sparse_clade` | 2,370 | 5 | 75.3% | 0.9833 |

Identical to four decimal places. The residual 2/120 failures per arm land on
four distinct donors, two of them shared between arms (`MK806439.1` fails in both
`all_fullcds` arms, `PX744018.1` in `all_fullcds` and `b1_sparse_clade`). At n=6
that is too few to separate stochastic from structural in either direction.

The stronger form of the result is that the sparse-clade arm recovers clades its
corpus does not contain at all. `b1_sparse_clade` holds zero Green algae, Other
green or Ferns records; those clades go 0.000 → 1.000, 0.000 → 1.000 and
0.600 → 1.000. Pooled across the 66 donors from clades absent from its corpus:
18 fail→pass against 1, exact McNemar p = 7.6e-05. SAR/other protist is NOT in
that pool despite its `records_in_corpus = 0`: the corpus labels diatoms as their
own clade, and those 523 records are the same phylum (Ochrophyta) as the SAR
evaluation donors, so the clade is covered rather than absent.

So the curation-gap account, which motivated Part B, is **not** supported. The
model does not need to have seen a lineage to generate it; a 90 nt prompt breaks
something that any in-domain finetuning repairs.

**What this outcome does not establish.** Pass rate saturates at 0.983, so it
cannot rank corpus designs — the start-codon question is unanswered, not
answered negatively. Every adapter is a single training seed, so between-corpus
differences on any endpoint are confounded with training noise. And the
falsifying experiment is unrun: a corpus of land plants only, evaluated on
algae. If that also recovers algal generation, "generic domain adaptation" is
established; if it does not, the effect is phylogenetically bounded after all
and this outcome would need re-reading.

**Predictions that did not occur:** no arm regressed in angiosperms (Eudicots
0.852 → 0.963–1.000, Monocots 0.909 → 0.909–1.000), so the flattening cost this
design worried about did not appear at L1. It remains untested at the prompt
levels where base already succeeds.

### Known risks

1. **Resolved: OG2 partition audit.** The premise holds. `arcinstitute/opengenome2`
   exposes ten nucleotide partitions (`gtdb_v220`, `metagenomes`,
   `ncbi_eukaryotic_genomes`, `eukaryotic_genic_windows`, `mrna`, `mrna_splice`,
   `ncrna`, `transcripts`, `promoters`, `plasmids_phage`, `organelles`). There is
   no barcode or marker-gene partition. Plastid content enters only through
   `fasta/organelles/organelle_sequences.fasta.gz`, whose headers we enumerated:
   32,240 records, of which 14,625 are plastid *sensu lato*, 17,613
   mitochondrial, and 2 unclassifiable from title. Of the plastids, 17 are
   apicoplasts — the relict, non-photosynthetic plastids of apicomplexans, which
   have lost *rbcL* entirely — leaving **14,608 *rbcL*-bearing photosynthetic
   plastid genomes (2,236 Mb, 79.4% of organelle nucleotides)**. Note for anyone
   re-deriving this: a naive `chloroplast|plastid` title match also returns
   14,608, but it is not the same set — it wrongly includes 2 apicoplasts labelled
   `organelle: plastid:apicoplast` and wrongly excludes one `plastome` and one
   `cyanelle` record. The two errors happen to cancel, so agreement in the count
   is not evidence of agreement in the set.
   Accessions are 32,140 `NC_`, 97 `NW_`,
   3 `NT_` — all RefSeq-curated, and **zero** primary-submission accessions.
   Since barcode *rbcL* submissions carry primary accessions exclusively, none are
   present. A range read of the partition's first 4 MB recovered 511 headers, all
   511 in our enumeration and in identical order, confirming the enumeration is
   the partition rather than a proxy for it. Implied exposure: plastid is ~2.38 B
   of the organelle partition's 3 B tokens (effective dataloader weight 0.40%
   phase 1, 0.20% phase 2), and *rbcL* occupies ~0.94% of a mean 153 kb plastid
   genome, giving ~22 M *rbcL* tokens — 2.4 × 10⁻⁶ of the 9.3 T trained.
   Caveat: `eukaryotic_genic_windows` and `ncbi_eukaryotic_genomes` derive from
   nuclear assemblies, which can carry plastid-derived contigs (NUMT/NUPT
   analogues); this is unquantified and would only raise exposure, so the headroom
   figures are upper bounds on novelty, not lower.
2. **Resolved: generative-LoRA path — and it found a live defect.** Verified
   against `ArcInstitute/evo2` → `Zymrael/vortex` StripedHyena source. Every
   target name the configs previously carried (`attn_qkv`, `attn_out`, `mlp_in`,
   `mlp_out`, `hyena_mixer`) is absent from the real model, so `apply_lora` would
   have raised on first contact with real weights — the guard worked, but the
   config was wrong. Real names: attention blocks expose
   `inner_mha_cls.{Wqkv, out_proj}`; every block exposes `mlp.{l1, l2, l3}` (gated
   MLP, `l1`/`l2` up, `l3` down); hyena blocks expose `out_filter_dense`. All are
   `nn.Linear`. `blocks.N.projections` is a `TELinear`, which on both the
   TransformerEngine and pure-PyTorch branches subclasses `nn.Module` rather than
   `nn.Linear` and returns `(out, bias)` rather than a tensor — so it was
   unreachable by the injection filter *and* would have corrupted the forward pass
   if forced. It is now excluded by name and a requested non-`nn.Linear` target
   raises instead of being silently skipped, since under-adapting relative to the
   config is unfalsifiable after the fact. Covered by
   `tests/test_lora.py::test_shipped_targets_match_striped_hyena_layout` and three
   siblings, against a mock mirroring the real block layout.
   **Superseded — a first-party training path exists and we now use it.** The
   claim above ("the training path remains third-party rather than a supported
   first-party API") was wrong, and was written from a blog post rather than from
   the framework. `NVIDIA/bionemo-framework` ships `recipes/evo2_megatron` with
   `Evo2LoRA`, a LoRA variant on the Megatron Bridge PEFT stack
   (`--lora-finetune --lora-dim --lora-alpha --lora-dropout
   --lora-target-modules`, plus `--lora-skip-freeze-modules` for selectively
   fully-trainable modules; a module matching both lists raises `ValueError`).
   It ships `examples/lora-fine-tuning-tutorial.ipynb`, which finetunes the **1B**
   checkpoint — our chosen scale — with a head-only baseline. Its rank/alpha
   defaults (16/32) match the configs here.

   Training therefore moves to `Evo2LoRA`. Consequences:

   * **Module names differ, and both are correct.** BioNeMo targets Megatron
     names (`linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`,
     `dense_projection`); our own implementation targets Arc/vortex names
     (`inner_mha_cls.Wqkv`, `inner_mha_cls.out_proj`, `mlp.l1/l2/l3`,
     `out_filter_dense`). Same architecture, two module hierarchies, reached via
     different checkpoint formats.
   * **The harness survives**, which was the blocking objection. `recipes/
     evo2_megatron` ships `evo2_convert_vortex_to_mbridge` *and*
     `evo2_export_mbridge_to_vortex`, so we train in Megatron and export back to
     a Vortex `.pt`. Part A's evaluation path reads the finetuned model in the
     same format as the base model, keeping the harness byte-identical across
     parts as the standing constraint requires.
   * **`src/train/lora.py` is retained as a CPU-testable reference**, not the
     production path. It is what the suite exercises without a GPU, and it caught
     a real defect (every target name in the original configs was absent from the
     real model). It is no longer what trains the released adapters.
   * **The fp8/Transformer-Engine workaround becomes unnecessary** for training:
     Megatron handles precision natively (`--mixed-precision-recipe`). The
     workaround in `src/evo2_loader.py` is still needed for *inference* through
     Arc's `Evo2` class, which is how generation and scoring run.

   Data path: `preprocess_evo2` converts FASTA to Megatron indexed-binary form,
   and it performs its own **random, record-level** train/valid/test split. Handing
   it one FASTA with a nonzero `valid_split` would silently discard the
   species-disjoint design and reinstate the leakage measured above. So
   `src/train/bionemo_export.py` writes each split as its own FASTA with
   `train_split: 1.0`, giving BioNeMo no splitting decision to make, and emits the
   `taxonomy_data` lineage map so finetuning uses the same taxonomic conditioning
   format Evo 2 was pretrained with. `random_lineage_dropout` is pinned to 0.0
   (upstream default 0.1) because clade conditioning is the variable Part B
   manipulates. Pinned by `tests/test_bionemo_export.py`.
3. **Memorisation.** ~5,000 sequences of one gene, ~1,438 nt each, three
   epochs. The most likely outcome is close reproduction of training sequences.
   Detection: novelty of generated output against the *finetuning* set
   specifically. In practice this is measured by 32-mer containment
   (`src/eval/memorisation.py`), reported per arm against every corpus in
   `results/part_b/l1_1800nt_containment.csv`, with the base arm as the
   conservation floor. The natural-identity baseline that this risk was
   originally written against is no longer in the repository.
4. **Catastrophic forgetting on B2.** The cap discards 88% of available eudicot
   records. Angiosperm degradation is interpretable, but only against a
   base-model control at matched scale — risk 5 again.
5. **Adapter target names.** Module naming differs between Evo 2 releases;
   `apply_lora` raises rather than silently wrapping nothing.

## 3. Standing constraints

- No simulated or placeholder data in any analysis path.
- All replicates from one identical replicable method. A baseline generated
  under a different framework is a confound, not a baseline.
- Numbers read back from saved tables before being written down.
- Evaluation harness identical across parts, or the comparison is meaningless.
