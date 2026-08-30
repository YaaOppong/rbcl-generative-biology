# Methods

Exhaustive description of every procedure in this repository: what was done, with
which parameters, by which code, and what each step's output is. Numbers quoted
here are reproduced by the commands in [§11](#11-reproduction).

**Findings are in [RESULTS.md](RESULTS.md).** This document is *how*; that one is
*what was found*. The README carries neither.

Where a procedure is **not** reproducible from this repository, it says so
explicitly rather than being omitted ([§12](#12-what-is-not-reproducible)).

> ### Maintenance
>
> **This document is the single source of truth for method**, as
> [RESULTS.md](RESULTS.md) is for findings. The README carries neither and links
> to both, because two descriptions of one procedure means one of them goes
> stale.
>
> **Update this file in the same commit as any change to:** a parameter or
> config value; a QC gate or predicate; the prompt protocol, sampling settings
> or batching; the LoRA placement, hyperparameters or split; any scoring or
> endpoint definition; the set of tables that can be regenerated; or anything
> that moves between §11 (reproducible) and §12 (not reproducible).
>
> If a number quoted here changes, change it here first, then anywhere else it
> appears. A number in this file that no command reproduces is the defect this
> document exists to prevent.

---

## 1. Design

Two parts, each varying one factor against a fixed harness.

| part | varies | held fixed | status |
|---|---|---|---|
| **A** — generation | prompt length | base model, no finetuning | complete |
| **B** — finetuning | training corpus | prompt protocol from A | run at L1 |

The observation motivating B is Part A's R6: full-length pass rate
is **non-monotonic** in prompt length, and a 90 nt authentic prompt is worse than
a 30 nt generic seed.

## 2. Model and environment

- **Model:** Evo 2 7B (`evo2_7b`), `ArcInstitute/evo2`, StripedHyena
  architecture via `Zymrael/vortex`. 7B rather than 1B because Part A's entire
  baseline is 7B; a 1B finetune would confound the finetuning effect with model
  scale.
- **Tokeniser:** `CharLevelTokenizer(512)` — character-level over a 512-symbol
  vocabulary.
- **Precision:** bf16, as loaded by vortex.
- **Hardware:** NVIDIA H100 80GB (recorded per run in `run_summary.json`).
- **Checkpoint loading:** `src/evo2_loader.py`. Evo 2's fp8-projection fallback
  is gated on a literal `"7b"` substring in the model name, so non-7B
  checkpoints require a patched config with `use_fp8_input_projections: false`.
  That patch is applied only when required and is recorded in each run's
  `load_info.fp8_patch_applied`. For every run reported here the flag is
  `false` — the 7B path needed no patch.
- **CPU-only components** (all scoring, all analysis, all tests) require no GPU.

## 3. Data

### 3.1 Accession discovery

`src/data/discover_accessions.py` queries NCBI Entrez per clade:

```
rbcL[Gene] AND <taxon>[Organism] AND 1400:1500[SLEN]
```

`SLEN` is the length of the **record**, not the CDS. This deliberately selects
standalone *rbcL* submissions (barcode records) rather than whole plastid genomes
containing *rbcL* — the latter being precisely what Evo 2 already trained on
(§9). The range brackets the natural CDS range (longest observed 1,497 nt) with
room for short flanks.

26 clades are queried by NCBI taxon name (`Eudicots`→`eudicotyledons`,
`Red algae`→`Rhodophyta`, `Diatoms`→`Bacillariophyta`, …; full map in the
module). Output: `data/rbcl_fullcds_accessions.csv` — **17,205 records across
10,770 species**, with versioned accessions, taxid, title, clade and record
length. The query and per-clade counts are recorded in the accompanying
`.provenance.json`.

Accessions are **versioned** (`OP237583.1`). NCBI serves specific versions on
efetch, so this manifest is an exact, re-fetchable specification of the corpus.

### 3.2 Sequence retrieval and CDS extraction

`src/data/build_dataset.py`. Sequences are **not committed**; they are fetched
from GenBank at build time so provenance stays auditable.

- Fetched in batches of 200, up to 4 attempts per batch with backoff 3/6/12 s.
  Retries are per batch, so a success is never re-fetched.
- Joins are keyed on the **unversioned** accession on both sides, because efetch
  may echo a newer version than the manifest records.

**CDS extraction** (`extract_cds`) — two failure modes, both silent if unhandled:

1. **`codon_start`.** GenBank annotates partial CDS features with a
   `codon_start` qualifier giving the offset of the first complete codon (1, 2
   or 3). Biopython's `feature.extract()` returns the raw location span and does
   **not** apply it. Ignoring it leaves the sequence out of frame — in the B1
   sample, `codon_start=3` records translate to 38 internal stops instead of 0.
2. **Trailing partial codon.** After applying the offset the sequence is trimmed
   to a multiple of three.

The annotation is trusted only if it yields a clean reading frame; otherwise the
record falls through to `infer_frame`.

**Frame inference** (`infer_frame`) recovers the frame of unannotated records by
stop-codon search, accepting a frame only if it is **uniquely** stop-free. Two
clean frames means the record is ambiguous and is rejected rather than guessed
at. This is necessary because CDS annotation is itself clade-structured: 100% of
red algal and diatom records carry a CDS feature against 4% of eudicot records,
so requiring annotation would silently discard whole clades.

> **This is a reported finding, not just plumbing.** Naive FASTA extraction
> leaves roughly half of algal *rbcL* records out of frame while land plants are
> unaffected, because barcode submissions are partial CDS carrying a
> `codon_start` offset that FASTA does not expose. Since the primary endpoint is
> algal pass rate, that confound would have sat directly on the result.
> Quantified in `results/part_b/frame_audit_by_clade.csv`
> (`src/data/frame_audit.py`).

### 3.3 Quality gates

Every gate is a **rejection**, not a repair.

| gate | rule | rationale |
|---|---|---|
| complete CDS | `src.eval.metrics.is_full_length` | the *same* predicate that scores generations (§6.2) |
| ambiguity | any non-ACGT base rejects the record (`AMBIGUOUS_MAX = 0.0`) | |
| exact duplicates | removed | *rbcL* is a barcode locus; the same species recurs |
| start codon | **not** applied by default | confounded with clade: Red algae 17.2% ATG, Brown algae 17.7%, against 96–100% for Eudicots/Mosses/Monocots. Requiring it roughly halves the algal share |

The start-codon gate is not omitted on principle — it is made an **experimental
arm** (§3.5) so its effect is measured rather than assumed.

### 3.4 Leakage control

The 120 evaluation donors must not appear in training. Exclusion is applied at
**three** levels, each a floor under the last:

1. **Species flag** (`heldout_donor_species`) — the legacy manifest column.
   Insufficient on its own: it was wrong for 24 donors, which were consequently
   trained on.
2. **Accession** — base accessions of all 120 donors, read at build time from
   `data/prompts_corpus.csv` rather than trusting a precomputed column.
3. **Species taxid** — `data/evaluation_donor_taxids.csv`. This is the control
   the flag was meant to be: a donor's species appears in GenBank under other
   accessions, and those carry `heldout_donor_species=False` and a different
   accession, so they survive filters 1 and 2.

Rejection counts per build are recorded in each corpus's `.provenance.json`.

### 3.5 Training corpora

Three arms, differing only in inclusion rule:

| arm | file | records | species | clades | rule |
|---|---|---|---|---|---|
| `all_fullcds` | `data/all_fullcds.jsonl` | 6,946 | — | 19 | every full-CDS *rbcL* record; no cap, no clade selection |
| `all_fullcds_atg` | `data/all_fullcds_atg.jsonl` | 2,937 | — | 17 | as above **plus** an intact ATG start codon |
| `b1_sparse_clade` | `data/b1.jsonl` | 2,370 | 1,493 | 5 | the five clades with ≥5× GenBank-to-RefSeq headroom |

`b1_sparse_clade` composition: Red algae 1,043 · Diatoms 523 · Mosses 431 ·
Brown algae 221 · Liverworts 152.

`all_fullcds_atg` is the paired arm to `all_fullcds`; their **only** difference
is the start-codon gate, so the comparison isolates that one decision. Records
the gate removes are 5′-truncated by a median of 12 codons and at most 24; the
earliest scored active-site residue is 175, so nothing catalytic lies in the
missing region.

## 4. Evaluation prompts

### 4.1 Design

**120 donor accessions × 5 prompt levels × 3 replicates = 1,800 prompts.**

| level | prefix | source |
|---|---|---|
| `L0_shared_seed` | 30 nt | one seed shared by all donors |
| `L1_donor_90` | 90 nt | donor's own 5′ sequence |
| `L2_donor_210` | 210 nt | ” |
| `L3_donor_450` | 450 nt | ” |
| `L4_donor_900` | 900 nt | ” |

Donors span 12 taxonomic groups and carry a `split` label (99 train / 13 val / 8
test donors).

**L0 is not a neutral reference.** The shared 30 nt seed
(`ATGTCACCACAAACAGAGACTAAAGCAAGT`) is a land-plant start region. This is
recorded as a limitation of the no-prompt baseline, not treated as a control.

### 4.2 Provenance of the prompt file

`data/prompts_corpus.csv` was **reconstructed** (commit `638c055`); the original
was never committed. Every field was recovered from the Part A generated corpus.
The reconstruction is verifiable and verified:

- all 1,800 prompts are an **exact prefix** of their own generated sequence;
- all 1,800 stored seeds match the corpus metadata exactly;
- prefixes are **nested-consistent** across L1→L4 for all 120 donors (1,080/1,080
  pairs), i.e. each shorter prefix is a prefix of the next longer one.

### 4.3 Prompt integrity

All 360 L1 prompts translate stop-free in frame 0 and all 360 begin with ATG. At
30 codons a wrong frame would show a stop roughly 75% of the time, so this is
strong evidence the prompts are in frame. **The L1 collapse (§R6) is therefore
not a frame artefact or a start-codon artefact of the prompts.**

## 5. Generation

`src/generate/runner.py` — one procedure, shared by every part. Parts differ in
their **arms**, never in how a sequence is produced.

### 5.1 Sampling parameters

| parameter | value | note |
|---|---|---|
| temperature | 0.7 | explicit in Part A |
| `top_k` | 4 | `Evo2.generate`'s signature default, inherited not tuned |
| `top_p` | 1.0 | ” |
| batch size | 4 | uniform across levels, replicates and arms |
| total length | 1,500 nt (Part A) / 1,800 nt (reruns) | see §5.3 |

`top_k=4` was **not chosen by us**. It is the reference implementation's default
and is defensible only because it is held constant across every arm. It should
never be reported as a tuned hyperparameter. Its effect is to restrict sampling
to the four highest-probability tokens of a 512-symbol vocabulary — usually, but
not always, the four nucleotides: 17 of 1,800 Part A sequences contain a literal
space, in every case *after* the terminal stop, so no scored CDS is affected. The
same behaviour appears in the base arm of the 1,800 nt reruns (1/120 and 5/600)
and in **none** of the three finetuned arms (0/120 each).

### 5.2 Batching and pairing

Batches are **homogeneous in token budget** and derived deterministically from
the prompt list alone. Two properties the pairing depends on:

- mixing token budgets within a batch changes the padding, and batched
  autoregressive decoding is not bit-identical across padding;
- a deterministic order means a donor lands in the **same batch, at the same
  position, in every arm**.

The RNG is reseeded per batch from the batch's *first* prompt
(`torch.manual_seed`, `torch.cuda.manual_seed_all`). Arms are therefore paired on
prompt, seed and batch position within a single run.

Each arm loads the checkpoint fresh and releases it afterwards, so no adapter
state carries between arms.

### 5.3 Total length and censoring

Part A used a 1,500 nt total — only 33 nt (2%) above the longest natural *rbcL*
CDS (1,467 nt). A generation running slightly long is therefore **censored**
rather than failed: 39/1,800 hit the budget with no stop codon and were scored
not-full-length, indistinguishable from frame breakage. At L1 that was 25
sequences.

The 1,800 nt reruns give 23% headroom, so a read-through there means the model
genuinely failed to terminate. Effect of removing the ceiling at L1: **0.3
points** (0.4944 → 0.4917) — the artefact was real and immaterial.

### 5.4 Determinism

The base arm was generated independently in two runs on different days, from
different commits, in different sandboxes. On the 120 shared L1 prompts, **all
120 sequences are byte-identical**. Checked by `src/analysis/determinism.py`.

This is what licenses the paired comparison in §7.

## 6. Scoring

### 6.1 CDS recovery

`cds_of` truncates the generation at the **first in-frame stop codon,
inclusive**. No in-frame stop → read-through, recorded as such (not as a short
CDS). Decoding is autoregressive, so read-through past the stop cannot influence
the CDS preceding it.

### 6.2 Full-length predicate

`src.eval.metrics.is_full_length` — four conditions:

1. `1400 ≤ len ≤ 1550`
2. `len % 3 == 0`
3. terminal codon is a stop
4. no internal stop

Applied to the **CDS** (not the raw generation), conditions 2–4 are satisfied by
construction, so the operative test is the length window. Applied to the raw
1,500 nt generation it gives a 0.28% pass rate against Part A's published
48.33%, confirming the CDS reading is the one Part A used.

This predicate is used **identically** to gate the training corpus (§3.3) and to
score generations. It is never reimplemented — an early reimplementation using
`cds_len >= 1000` reported the L1 base rate as 63.9% against the published 48.3%.

### 6.3 Stored vs recomputed

`data/part_a_generated_corpus.csv` carries a stored `full_length` column written
by a scorer that is **not in this repository**. It disagrees with
`is_full_length` on **16 of 1,800 rows** (L1:4, L2:4, L3:6, L4:2; L0 clean),
every one in the same direction — stored `False`, recomputed `True`. The four L1
rows were checked directly and carry an ATG start, a terminal stop, no internal
stop and no ambiguous bases, so no additional condition explains them. The cause
is not established.

All tables report the **recomputed** value, because it is the one a reader can
reproduce. Per-level effect:

| level | stored | recomputed |
|---|---|---|
| L0 | 0.9722 | 0.9722 |
| L1 | 0.4833 | 0.4944 |
| L2 | 0.8417 | 0.8528 |
| L3 | 0.9667 | 0.9833 |
| L4 | 0.9750 | 0.9806 |

Every affected row is listed in
`results/part_a/tables/full_length_recount_discrepancies.csv`.

## 7. LoRA finetuning

`src/train/lora.py`, `src/train/lora_finetune.py`.

### 7.1 Adapter placement

A frozen `nn.Linear` is wrapped with a trainable low-rank update:
`y = W₀x + (B·A·dropout(x))·(α/r)`. `A` is Kaiming-uniform initialised; **`B` is
zero-initialised**, so the adapted model starts exactly at the base model.

Target modules, verified against `ArcInstitute/evo2` → `Zymrael/vortex`
StripedHyena source:

```
inner_mha_cls.Wqkv, inner_mha_cls.out_proj,   # attention blocks
mlp.l1, mlp.l2, mlp.l3,                       # gated MLP (l1/l2 up, l3 down)
out_filter_dense                              # hyena blocks
```

`blocks.N.projections` is **deliberately excluded**: it is a `TELinear`, which
subclasses `nn.Module` (not `nn.Linear`) and returns `(out, bias)` rather than a
tensor, so wrapping it would add a tensor to a tuple.

Two guards, both raising rather than warning: a target that matches **zero**
modules raises (module names differ between Evo 2 releases), and a target that
matches a **non-`nn.Linear`** raises rather than being skipped — silently
ignoring it would yield a finetune with fewer adapters than the config claims.

`make_trainable` (`src/evo2_loader.py`) is called **before** `apply_lora`.
Vortex loads weights inside `torch.inference_mode()`, marking every parameter an
*inference tensor*; those can never participate in autograd, and a training
forward pass dies inside vortex's own RMSNorm before any adapter code is reached.
Cloning each parameter out of inference mode produces normal tensors.

### 7.2 Hyperparameters

| parameter | value |
|---|---|
| LoRA rank / α / dropout | 16 / 32 / 0.05 |
| optimiser | AdamW, lr 1.0e-4 |
| schedule | OneCycleLR, `pct_start` = 0.03 |
| epochs | 3 |
| batch size / grad accumulation | 4 / 8 (effective 32) |
| max sequence length | 1,600 nt (headroom over the ~1,500 nt CDS for BOS/EOS) |
| seed | 0 |
| early stopping patience | 2 |

**Single seed per corpus.** Every between-corpus comparison is therefore
confounded with training noise. This is a known limitation, not an oversight.

### 7.3 Validation split

Three-way, clade-stratified (`three_way_split`):

- **`val_novel`** — whole species held out; no species appears in training. This
  is the honest generalisation measure.
- **`val_seen`** — single records pulled from species that **remain** in
  training. Measures fit on familiar species; the gap against `val_novel`
  quantifies conspecific leakage.
- **train** — everything else.

`val_fraction = 0.2` is the total held back; `novel_share = 0.5` splits it evenly
(10% + 10%). `val_seen` is capped by donor availability — it can take only one
record from a multi-record species — so the realised total is ≤ `val_fraction`,
never more. Species are assigned to `val_novel` in a shuffled order rather than
by record count, which would skew it toward less-sequenced species.

A per-run split manifest records counts, species lists per split, per-clade
breakdown, and `leaked_species_into_val_novel` (which warns if non-zero).

### 7.4 Loss

Next-token cross-entropy over **unpadded positions only**. Validation loss is
token-weighted by `mask[:, 1:].sum()` — the number of positions the loss actually
averaged over. Weighting by `mask.sum()` instead would over-weight short
sequences, biasing early stopping toward whichever batch composition holds them.

Validation loss is evaluated **before any weight update**, so the epoch −1 value
is on the record. Since `B` is zero-initialised this is exactly the base model's
loss on the same held-out records and tokenisation.

### 7.5 Adapter verification at generation time

Before any GPU work, `resolve_arms` resolves each named adapter, computes its
sha256 with the *same* function that recorded it at training time, and compares
against `--expect-sha`. A mismatch, or an unstated sha, raises `SystemExit`.

At load time `load_state_dict(..., strict=False)` is checked in **both**
directions:

- **unexpected keys** → the adapter carries tensors this model has no slot for;
  it does not fit and nothing can be concluded from the arm;
- **missing keys** → the model has LoRA sites the adapter does not cover. Those
  keep their initialisation, and `B` is zero, so each uncovered site is the
  **identity**. The arm would run partly or entirely as base while being labelled
  finetuned, and would still produce a clean CSV and a plausible pass rate.

Both raise. Coverage is recorded in `run_summary.json` under `adapter_load`.

## 8. Secondary endpoints

### 8.1 Memorisation — k-mer containment

`src/eval/memorisation.py`. The fraction of a generation's **32-mers** that occur
anywhere in a training corpus.

- *k*=32 because sharing one by chance is negligible (4³² ≈ 1.8×10¹⁹ against a
  corpus of ~5.4×10⁶ k-mers) while remaining short enough to catch a copied
  fragment rather than only a copied record.
- **Containment, not nearest-neighbour identity:** a sequence assembled from two
  training records in halves is ~100% contained while having no close neighbour,
  and that is still copying.
- Sequences shorter than *k* return 0.0 and should be excluded by the caller.

**Containment is reference-dependent, and that is reported rather than hidden.**
`results/part_b/l1_1800nt_containment.csv` gives **every arm against every
corpus**. The base arm is included deliberately as the conservation floor: base
never saw any finetuning corpus, yet scores 0.386 above the 90% threshold against
`all_fullcds` and 0.000 against the narrower `b1_sparse_clade` corpus. *rbcL* is
conserved enough that a correct novel sequence necessarily shares 32-mers with
natural records, so only the **difference from base, against the same reference**,
is attributable to finetuning.

### 8.2 Active-site integrity

`src/eval/active_site.py`, `src/analysis/part_a_active_site.py`.

Eleven catalytic positions in *Spinacia oleracea* P00875 numbering: the
RuBP-binding lysines (175, 177), the carbamylated Lys201, the Mg²⁺-coordinating
Asp203/Glu204, and 294, 295, 327, 334, 379, 380. All eleven match the committed
P00875 reference exactly.

- **Mapping is by ALIGNMENT, never by direct indexing.** A single indel shifts
  every downstream index, so naive indexing reports catalytic loss where there is
  only a frame offset. Global `PairwiseAligner`, BLASTP scoring, gap open −11 /
  extend −1.
- **Minimum length matters.** Gly380 ends at nucleotide 1,140, so a shorter
  generation cannot be scored on the full set — the residues are not absent, they
  are out of reach. `n_covered` and `scorable` make this explicit. (The first
  Part B run generated 300 nt and scored 0/11 across every sequence in both arms;
  that was protocol, not biology.)
- Translations under 100 aa are not scored.

Two populations are reported, because they answer different questions:
**`full_length`** (complete CDS — the biologically meaningful set) and
**`scorable`** (long enough to reach residue 380, including long broken ORFs).

## 9. Training-data exposure audit

`src/data/og2_audit.py`. Every headroom claim rests on Evo 2's *rbcL* exposure
being limited to whole plastid genomes with no barcode partition. Established
from the dataset rather than assumed:

- `arcinstitute/opengenome2` has no barcode or marker-gene partition; plastid
  sequence enters only via `fasta/organelles/`.
- That partition is RefSeq-only: all 32,240 organelle accessions carry `NC_`
  prefixes, **zero primary submissions**. Barcode *rbcL* records carry primary
  accessions exclusively.
- Of 32,240 organelle records, 14,608 are *rbcL*-bearing photosynthetic plastid
  genomes. 17 are apicoplasts, which have lost *rbcL* along with photosynthesis
  and are excluded — a plain `chloroplast|plastid` match returns the right
  *count* only by cancelling two errors.
- Implied *rbcL* exposure ≈ 22 M tokens, **2.4 ppm** of the 9.3 T trained.

Verify with `python -m src.data.og2_audit --verify-partition` (network).

## 10. Provenance and run tracking

- Every run writes `provenance.json` (git commit, dirty flag, input sha256s,
  config, sampling parameters) and `run_summary.json` (timings, per-arm row
  counts, adapter coverage, GPU, torch version).
- Runs harvest into `results/<arm>/<UTC>_<commit>/`, **never** a fixed path: a
  shared output directory means a rerun destroys the previous run's record.
- `RUNS.md` is an append-only, machine-written ledger; one row per run carrying
  the sandbox id that joins this repo's record to the compute provider's log. It
  is written **before** harvest, so a recovery failure cannot suppress the row.
- Generated sequences are mirrored to the weights volume as each arm completes,
  because the sandbox tar harvest needs a live sandbox and three sandboxes in
  this project exited before it ran.
- Adapters are identified by sha256 throughout (§7.5).

## 11. Reproduction

```bash
pip install -e ".[dev]"        # needs Python >= 3.10
pytest -q                      # 158 CPU-only; 202 with torch installed, as CI runs it

# --- data (network) ---
python -m src.data.discover_accessions --out data/rbcl_fullcds_accessions.csv
python -m src.data.build_dataset --arm B1_sparse_clade --out data/b1.jsonl

# --- Part A tables (no network, no GPU) ---
python -m src.analysis.part_a_tables --corpus data/part_a_generated_corpus.csv
python -m src.analysis.part_a_tables --exposure
python -m src.analysis.part_a_tables \
    --generated results/generate_a/<run>/part_a/generated_base.csv --tag 1800nt_rep0
python -m src.analysis.part_a_active_site \
    --corpus data/part_a_generated_corpus.csv --level L1_donor_90 --tag l1

# --- Part B tables (needs data/*.jsonl rebuilt first) ---
python -m src.analysis.part_b_tables \
    --run results/generate_b/<run>/part_b --tag l1_1800nt

# --- GPU ---
python scripts/run_on_modal.py --configs configs/b1_sparse_clade.yaml
python scripts/run_generate_on_modal.py --part A
python scripts/run_generate_on_modal.py --part B --arms base,b1_sparse_clade \
    --expect-sha b1_sparse_clade=<sha256>
```

All eight Part A tables and all four Part B tables regenerate **byte-identically**
from committed inputs.

## 12. What is *not* reproducible

Stated so it cannot be mistaken for an omission.

**Deleted 2026-08-30 — eleven Part A tables**, with results **R1** (natural
novelty baseline), **R2b** (active-site conservation across natural sequences and
its agreement with structure), **R4** (lineage pull), **R5** (seed lineage bias)
and the **kinetics oracle**. Each was produced by code that was never committed,
from inputs this repository does not hold: an alignment of the natural *rbcL*
corpus, a PDB structure, ESM-2 embeddings, and published kinetic measurements.

The **inputs** are obtainable — `data/rbcl_fullcds_accessions.csv` pins 17,205
versioned accessions and the exact query. What was lost is the **procedure**:
which structure and chains defined the 5 Å cut, what alignment produced the
conservation columns, how the kinetics set was filtered to n=183 species. Those
choices would have to be re-made and documented; the numbers would not land
exactly on the originals.

Two further items:

- **`rbcl_like`** — a column in the Part A corpus that no code produces,
  consumes or defines. Continuous on [0,1]; 953/1,800 rows exactly 1.0; 17 empty.
  Treat as absent. Anything depending on it (the three-frame `frameshift_audit`)
  is unrecoverable, not merely unrebuilt.
- **Part A's stored derived columns** (`cds_nt`, `cds_len`, `protein`,
  `full_length`) — the scorer is not committed. Unlike `rbcl_like` they are
  re-derivable, and `src/analysis_l1.py` is canonical where the two disagree
  (§6.3).

**Restored 2026-08-30:** `l1_active_site_base` and `l1_active_site_per_residue`,
which needed only UniProt P00875 (now committed) plus already-tested code.
Validated against the deleted originals before they were removed: on all 285
shared sequences `aa_len` and `n_correct` are identical and `identity` differs by
0.00e+00, and restricting the per-residue table to the original's population
reproduces all eleven of its counts exactly.

## 13. Known limitations

1. **Single training seed per corpus** — between-corpus differences are
   confounded with training noise (§7.2).
2. **The primary endpoint saturates.** All three finetuned arms reach 0.9833 at
   L1, so the design cannot rank corpus designs; the start-codon question is
   unanswered, not answered negatively.
3. **The endpoint is syntactic.** Applied to a CDS, `is_full_length` reduces to a
   length window (§6.2). Active-site integrity (§8.2) is the only structural
   endpoint.
4. **One model.** Nothing here separates behaviour of Evo 2 from behaviour of
   genomic language models generally.
5. **The falsifying experiment is unrun** — a land-plants-only corpus evaluated
   on algae. Without it, "generic domain adaptation" is inference, not result.
6. **Part A at 1,800 nt is replicate 0 only** (n=120/level) against the 1,500 nt
   curve's n=360.
7. **`nearest_identity`** compares ungapped from index 0, so a single indel
   collapses the score. Prefer the alignment-based `active_site` path.
