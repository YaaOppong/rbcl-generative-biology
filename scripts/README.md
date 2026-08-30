# scripts/

Entry points. These are **not** tests — `tests/` holds those, and everything in
it is named `test_*.py` and runs under pytest. Anything here either spends money
or writes results.

All of them assume the **repository root** as the working directory:

```bash
python scripts/check_modal_setup.py
```

## Current

One script trains, one generates. Parts A and B share the generator because the
base arm is identical work in both: Part A is `--arms base`, Part B adds
adapters. Two entry points would regenerate the same rows from the same seeds
and give the protocol somewhere to drift.

| script | what it does | cost |
|---|---|---|
| `check_modal_setup.py` | Verifies the pinned image, cached weights, required local inputs, and — with `--part B --arms <tags>` — that the adapters a run needs are actually on the volume. `--transport` additionally round-trips a file through a CPU-only sandbox. | ~5 s (`--transport`: ~30 s CPU) |
| `run_on_modal.py` | **Training launcher.** Runs `finetune.py` on your own Modal account, one config or several, and prints the adapter fingerprints. Does not generate. | ~30 min H100 per config |
| `run_generate_on_modal.py` | **Generation launcher.** `--part A` or `--part B`; dispatches the generator below into a sandbox with the checkpoint cache and adapter volume. | ~25 min H100 per arm |
| `finetune.py` | **Training only.** LoRA-finetunes Evo 2 7B, writes the adapter three ways plus the full training record, and stops. Idempotent: exits early if the adapter already exists (`--force` to retrain). | ~30 min H100 |
| `generate_ab.py` | **Generation, both parts.** `--arms base` is Part A (no finetuning); adding adapter tags makes it Part B, and `base,b1_sparse_clade,b2_balanced` is one paired run. The part is derived from the arms, not declared. Refuses any adapter whose sha256 was not stated. | ~25 min H100 per arm per 120 prompts |

Everything about *how* a sequence is decoded — batch size, per-batch seeding,
stop-codon truncation, the CSV schema — lives in `src/generate/runner.py`, once.
The parts declare only what they vary. A batch size or truncation rule that
differed between parts would be indistinguishable from a real effect, so it is
not something an entry point is allowed to restate.

**After any Modal SDK upgrade, run `check_modal_setup.py --transport`.** Modal
retired the Sandbox filesystem API (`sb.open`) that this launcher originally used
to move files; it now fails with `FAILED_PRECONDITION` on first contact. Files go
through `exec` stdin/stdout instead. The failure mode is a launch that looks
healthy right until the upload, so the round-trip is worth 30 s of CPU before a
paid GPU run.

Every launched run harvests into `results/<arm>/<UTC>_<commit>/`, never a fixed
path, with `results/<arm>/latest` pointing at the newest. That is not a
preference: a shared output directory means a rerun destroys the previous run's
record including its failure, which cost this project three tracebacks once and
silently overwrote a completed training record again before it was fixed.

Both launchers share `src/modal_job.py`: it packs the tree, exports `CS_JOB_ID`
and the commit (`CS_GIT_STATE` — a packed repo has no `.git`, so without it every
remote result records `commit: null`), harvests `out/` **even when the job
failed**, and appends one row to `RUNS.md` carrying the sandbox id that joins
this repo's record to Modal's own expiring log. The pinned image, volume and
adapter filename live in `src/modal_env.py`, once — a truncated copy of the image
id already cost this project a commit.

### The flow

```bash
python scripts/check_modal_setup.py                  # free, 5 s

# 1. train, once. Prints the sha256 in generate_ab.py's own --expect-sha form.
python scripts/run_on_modal.py --configs configs/b2_balanced.yaml

# 2. Part A: the base-model control at the same total length. No adapter exists
#    as far as this script is concerned.
python scripts/run_generate_on_modal.py --part A --levels L1_donor_90 --replicates 0

# 3. Part B: as often as the protocol changes, against those exact weights.
python scripts/run_generate_on_modal.py --part B --arms base,b2_balanced \
    --expect-sha b2_balanced=<sha256>
```

The generator runs standalone too (`python scripts/generate_ab.py`), which is what
the launcher invokes inside the sandbox. Locally they need the weights volume
mounted at `/weights`, so in practice you launch them.

Part B's second question — whether the already-represented clades have to be in
the finetuning corpus to keep the model flat across clades — is two adapters, so
it is two arms of *one* generation run:

```bash
python scripts/run_on_modal.py \
    --configs configs/b1_sparse_clade.yaml,configs/b2_balanced.yaml
python scripts/run_generate_on_modal.py --part B \
    --arms base,b1_sparse_clade,b2_balanced \
    --expect-sha b1_sparse_clade=<sha1>,b2_balanced=<sha2>
```

Run those on separate days and the comparison is no longer paired on prompt,
seed and batch position. Run them as arms and it is.

### Why training and generation are separate

Coupling them charged ~30 min of retraining to every generation experiment, so
changing a prompt protocol meant retraining an identical model. Worse, the two
were only ever comparable by accident: nothing checked that a later generation
used the same weights. Now `--expect-sha` does, and it fires before any
checkpoint loads — a mismatch costs a second, not a GPU-hour. `docs/RUN_TRACKING.md`
records what the training record contains and why.

`--expect-sha` is **required** for every adapter arm. `--allow-unverified-adapter`
exists for genuine iteration and records `adapter_verified: false`; it mirrors
`provenance.py`'s dirty-tree rule, where the unsafe thing stays possible but has
to be asked for by name.

## Superseded — removed at defaece

Three entry points were deleted rather than kept as dead code. **None produced
any current result**: `scripts/finetune.py` and `scripts/generate_ab.py` account
for every committed run record in `results/`. All three remain in git history —
`git show defaece:scripts/<name>` retrieves any of them.

| script | what it was | why it went |
|---|---|---|
| `run_arm.py` | Trained and generated in one process; produced the archived preliminary result (300 new tokens from a 90 nt prefix, single replicate). | The only superseded script named in a committed result's provenance: `results/archive/2026-08-21T11-10-04_a317084_preliminary/provenance.json` records `"entrypoint": "run_arm.py"`. That archive is marked do-not-cite, and the reference resolves through git history. |
| `run_partA_protocol.py` | Trained and generated in one process at Part A's protocol. Never produced a committed result. | Superseded by the train/generate split, then by `generate_ab.py`. Its one unported piece, `repro_check`, is now `src/analysis/determinism.py` — reframed, because the original licensed reusing Part A's stored 1500 nt rows as a control and the base arm is regenerated inside every run instead. |
| `portfolio_demo.py` | Demo finetune against `configs/demo_small.yaml`. Never produced a committed result. | That config states in its own first line that it is not a scientific arm. |
