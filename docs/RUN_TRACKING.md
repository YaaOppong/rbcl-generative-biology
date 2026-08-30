# Tracking what ran

## The training record

Training is one job (`scripts/finetune.py`, launched by
`scripts/run_on_modal.py`) and generation is another (`scripts/generate_ab.py`,
which covers both paper parts). That split is what lets
a prompt protocol change without paying ~30 min of retraining — but it moves
weight on the training record, since a later generation run cannot re-derive
what training did. Anything not written down at training time is lost.

A finetune run writes:

| file | why it has to be there |
|---|---|
| `run_summary.json` | model, GPU, seed, wall time, per-epoch losses, and `adapter.sha256` |
| `adapter_manifest.json` | every tensor name, shape and dtype in the adapter |
| `split_manifest.json` | **which species** went to train / val_seen / val_novel, per clade — membership, not just counts |
| `adapter_config.json` | trainable vs total parameters, injection-site count, the resolved LoRA and optimizer settings |
| `config.used.yaml` | the config as run, including any CLI override |
| `provenance.json` | commit, dirty flag, package versions, input checksums |
| `history.json` | per-epoch `val_seen` / `val_novel` and the memorisation gap |

The adapter itself is written three ways — persistent volume, harvested copy,
and manifest — because the first adapter produced in this project was lost when
its sandbox expired.

**`adapter.sha256` is what makes the split safe.** `generate_ab.py --expect-sha
<tag>=<sha>` refuses to run when the adapter on the volume does not match, and
the check happens before any checkpoint loads. Without it, "the finetuned arm"
is a claim about weights nobody verified; with it, a mismatch costs a second,
not a GPU-hour. The fingerprint is **required** for every adapter arm;
`--allow-unverified-adapter` waives it and records `adapter_verified: false`.

Runs launched on Modal land in `results/<arm>/<UTC>_<commit>/<tag>/`, and
`results/<arm>/latest` symlinks to the newest. `run_on_modal.py` prints the
fingerprints at the end of a training run, already formatted as
`generate_ab.py`'s `--expect-sha` argument. Retyping a sha256 by eye
is exactly the step where the wrong weights get blessed.

A generation run records, per arm: the adapter tag, its path on the volume, its
sha256, whether that sha was verified, and which config supplied the LoRA shape
the adapter was rebuilt with. Part B's arms are named adapters rather than a
base/finetuned pair, so `generated_b1_sparse_clade.csv` and
`generated_b2_balanced.csv` from one run are paired on prompt, seed and batch
position.

Two deliberate choices in that record:

- **Split membership, not just counts.** Counts cannot answer "was this species
  held out?". Re-deriving the split later would silently give a different answer
  if the split code or the corpus changed.
- **Cross-clade species leakage is recorded, not asserted.** The split is
  species-disjoint *per clade*, so a species assigned to two clades in the source
  metadata can legitimately appear in training for one and be held out in the
  other. That is a corpus data-quality question, and it must not kill a
  half-hour GPU run — it goes in `split_manifest.json` where analysis can see it.



Every result in this repo should answer three questions without asking anyone:
**which code produced it, from which data, and did it work.** This is how.

## What a run writes

Runs land in `results/<arm>/<UTC-timestamp>_<short-sha>/` — never a fixed path.
Three B2 attempts once shared one `out_dir` and each destroyed the previous
one's failure record; the tracebacks survived only because the remote sandboxes
happened to still be warm.

| file | tracked | why |
|---|---|---|
| `provenance.json` | yes | commit, dirty flag, package versions, GPU, input sha256, job id |
| `config.used.yaml` | yes | the config as resolved, not as written |
| `run_summary.json` | yes | timings, checkpoint chosen, eval metrics |
| `history.json` | yes | per-epoch losses incl. the epoch −1 baseline |
| `error.txt` | yes | traceback, when the run failed |
| `generations.csv` | yes | the sequences and their QC |
| `adapter_*.pt` | **no** | megabytes of binary; reference by checksum |
| `*.log`, `stdout.txt` | **no** | progress bars; regenerable from the provider |
| `latest` symlink | **no** | convenience only |

`data/*.jsonl` is gitignored, so **the input `sha256` in `provenance.json` is the
only anchor from a result back to its exact corpus bytes.** Do not add
provenance files to `.gitignore`.

`error.txt` is tracked deliberately. Three failed runs produced four real fixes
(`Evo2.__init__` has no `config_path`; adapters must inherit the base layer's
device and dtype; the installed evo2 has no `use_kernels` argument; vortex loads
weights under `torch.inference_mode()` so parameters must be cloned before
autograd). That trail is evidence, not clutter.

## The dirty-tree rule

`provenance.write()` **refuses to run on a modified working tree** unless you
pass `--allow-dirty`, which records `dirty: true` and the modified file list.

The reason is narrow: a result stamped with a commit SHA, produced from code that
differs from that commit, points a future reader at code that never ran. That is
worse than no SHA at all. `--allow-dirty` exists for genuine iteration — it just
labels the run honestly.

## The ledger

`RUNS.md` is append-only, one row per run: timestamp, arm, commit, dirty flag,
provider job id, outcome, one-line note. It is the human index over the run
directories.

The `job_id` bridges two records: the provider keeps its own log of intent and
stdout, outside git and expiring, while the repo keeps the durable record.
Without the id you cannot join them.

## Conventions worth keeping

- **Tag any commit whose numbers you cite.** `git tag b2-final <sha>` before a
  number goes in the README or manuscript.
- **Push before a GPU run.** Cheap, and it means "what ran" is on the remote,
  not only on the laptop that launched it.
- **Commit the failure too.** A run that failed is a result about the code.
## Known gaps — harden before releasing this as a package

**Neither of these affected the research run.** The three adapters
(`all_fullcds`, `all_fullcds_atg`, `b1_sparse_clade`) each trained to completion:
`train_exit=0`, three epochs, monotonically decreasing `val_novel`, and full
records. The sandbox died *after* all the work had finished and the records were
already on the volume, so it cost the tar harvest and nothing else.

They are recorded here, unfixed, for two reasons. Fixing them mid-experiment
would mean the code that produced the published adapters is not the code that
was reviewed. And both are about robustness for *someone else* running this
pipeline unattended, which is a release concern rather than a result concern.

### 1. The ledger row is written too late

`modal_job.launch` appends to `RUNS.md` in a `finally` block, but the launcher
recovers records from the volume *after* `launch` returns. A failure in recovery
therefore loses the ledger row for a run that actually succeeded, which is
exactly what happened on 2026-08-24 (see `RUNS.md`).

**Fix:** write the ledger row as soon as the job's exit status is known, before
any recovery or reporting. The row records that a job ran and how it ended;
nothing downstream of that should be able to suppress it.

### 2. Sandboxes exit 124 well inside their timeout

The 2026-08-24 training run was given 4.27 h, did ~1.8 h of work, and its
sandbox exited 124 at about 2 h 13 min -- after all three adapters had trained
and their records were safely on the volume, but before the tar harvest. The
2026-08-24 run before it died the same way at a 2.66 h budget.

124 is the conventional `timeout` exit status, so something is enforcing a limit
that is not the sandbox timeout we set. Until it is understood, treat the tar
harvest as unreliable and the volume as the durable path -- which is now how the
code is written.

**Fix:** identify the limit (Modal-side sandbox cap, the `sleep` entrypoint, or
an image-level constraint) before scheduling any job expected to exceed ~2 h.

This one is not purely a release concern: the generation grid is 4.5-8.3 h and
would hit the same wall. Generation writes its output incrementally and harvests
at the end, so unlike training it has no volume-backed durable copy -- a sandbox
dying at 2 h would lose the sequences produced up to that point. Either the
limit is understood first, or generation is split into jobs that each finish
comfortably inside it.
