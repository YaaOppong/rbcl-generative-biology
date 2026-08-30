"""Tests for the standalone training record and the train/generate boundary.

Training is one job (scripts/finetune.py, launched by scripts/run_on_modal.py)
and generation is another (scripts/generate_ab.py, which covers both paper
parts -- the base arm is the same work in each, so the part is a property of
which arms are requested). That makes the training record
load-bearing in a way it was not when one script did both: a later generation run
cannot re-derive what training did, so anything not written down is lost.

The load-bearing properties:

1. **The adapter is fingerprinted.** generate_ab.py refuses to run when an
   adapter's sha256 does not match what finetune.py recorded. Without the
   fingerprint the refusal is unenforceable and "the finetuned arm" becomes a
   claim about weights nobody checked.
2. **The split is persisted, not printed.** Which species were held out must be
   readable from the record, not recovered by re-running the split -- which
   silently changes if the split code or the corpus changes.
3. **The gate fires before the GPU.** A mismatch must be caught before any
   model loads, or the check costs real money to fail.
4. **The part is derived, not declared.** Part A is defined by the absence of
   finetuning, so it is read off the arms -- a run cannot be labelled as the
   part it is not.
"""

import ast
from pathlib import Path

FINETUNE = Path("scripts/finetune.py")
GENERATE = Path("scripts/generate_ab.py")
LAUNCHER = Path("scripts/run_on_modal.py")
RUNNER = Path("src/generate/runner.py")
TRAINER = Path("src/train/lora_finetune.py")

GENERATORS = (GENERATE,)


def _src(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text()


def _code(p: Path) -> str:
    """Source with the module docstring and comments stripped.

    Absence tests have to read code, not prose. These scripts document the
    commands that come NEXT -- generate_ab.py's usage line lives in the
    launcher's docstring on purpose -- and a grep over the whole file cannot
    tell an instruction to the reader from an instruction to the machine.
    """
    tree = ast.parse(_src(p))
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body = tree.body[1:]
    return ast.unparse(tree)


def _commands_in(p: Path, func: str) -> list[str]:
    """String literals naming a repo script inside one function.

    Scoped to a function because the launcher legitimately PRINTS the generate
    command it hands off to -- that lives in report_shas, and printing a command
    is not running one. What must never appear is a generation command inside
    the function that builds the remote job.
    """
    tree = ast.parse(_src(p))
    target = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == func), None)
    assert target is not None, f"{p} has no {func}()"
    return [n.value for n in ast.walk(target)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and "scripts/" in n.value]


def test_finetune_records_adapter_fingerprint():
    s = _src(FINETUNE)
    # provenance.file_digest, not a local hashlib call: generate_ab.py checks the
    # adapter through the same function, so the two cannot disagree on the hash.
    assert "provenance.file_digest" in s, "adapter is not fingerprinted"
    assert '"n_tensors"' in s and '"n_params"' in s
    assert "adapter_manifest.json" in s


def test_both_sides_of_the_gate_hash_the_same_way():
    assert "provenance.file_digest" in _src(RUNNER), \
        "generation computes the fingerprint its own way"


def test_finetune_persists_the_adapter_three_ways():
    """Volume + harvested copy + manifest. The first adapter in this project was
    lost to sandbox expiry; one copy is not enough."""
    s = _src(FINETUNE)
    assert "modal_env.adapter_path" in s, "no persistent volume copy"
    assert 'OUT / "adapter_best.pt"' in s, "no harvested copy"
    assert "adapter_manifest.json" in s, "no manifest"


def test_finetune_does_not_generate():
    """The whole point of the split: no generation in the training script."""
    s = _src(FINETUNE)
    for token in ("mdl.generate", ".generate(", "prompts_corpus", "generated_"):
        assert token not in s, f"finetune.py still generates ({token})"


def test_launcher_only_trains():
    """run_on_modal.py produces weights and stops. If it can also generate, the
    ~30 min retrain is back on the price of every generation experiment."""
    assert "--stage" not in _code(LAUNCHER), \
        "the launcher still has a stage switch; generation is not its job"
    cmds = _commands_in(LAUNCHER, "main")
    assert cmds, "no remote commands found -- has the launcher been restructured?"
    for c in cmds:
        assert "scripts/finetune.py" in c, f"launcher runs something else: {c}"
        assert "generate" not in c, f"launcher still generates: {c}"


def test_launcher_prints_the_sha_handoff():
    """The sha256 is how a generation run proves which weights it used, so the
    launcher must surface it rather than leaving it buried in the record."""
    s = _src(LAUNCHER)
    assert "expect-sha" in s, "launcher does not report the adapter fingerprint"
    assert "run_summary.json" in s


def test_launcher_gives_each_config_its_own_out_dir():
    """Two configs sharing out/ means the second run_summary.json overwrites the
    first -- and the record is the deliverable of a training job."""
    s = _src(LAUNCHER)
    assert "--out out/" in s


def test_generators_do_not_train():
    """No training OPERATIONS. Importing load_config from the trainer module is
    fine and necessary -- generate_ab.py needs the same LoraConfig dataclass to
    rebuild the adapter shape before loading weights into it. What must be absent
    is anything that updates weights."""
    for p in GENERATORS:
        s = _code(p)
        for token in ("loss.backward", "optimizer", ".step()", "history.json",
                      "lora_finetune.main", "src.train.lora_finetune --"):
            assert token not in s, f"{p.name} still trains ({token})"


def test_the_part_is_derived_from_the_arms():
    """Part A is generation WITHOUT finetuning. Deriving it from the arms means a
    base-only run cannot be recorded as Part B, or the reverse, whatever a flag
    says. It also defaults to base: asking for nothing gets you no adapter."""
    s = _code(GENERATE)
    assert 'PART = \'B\' if ADAPTER_ARMS else \'A\'' in s.replace('"', "'"), \
        "the part is declared rather than derived"
    assert "'part': PART" in s.replace('"', "'"), "the derived part is not recorded"
    assert "default='base'" in s.replace('"', "'"), "arms do not default to base"


def test_part_b_exposes_the_gate_on_its_command_line():
    """The gate's behaviour is tested in tests/test_generation_protocol.py
    against runner.resolve_arms. What must be true HERE is that the entry point
    actually offers it and passes the flags through."""
    s = _src(GENERATE)
    assert "--expect-sha" in s
    assert "--allow-unverified-adapter" in s
    assert "expect_sha=args.expect_sha" in s, "the flag is never passed to the gate"
    assert "allow_unverified=args.allow_unverified_adapter" in s, \
        "the waiver flag is never read"


def test_sha_gate_precedes_any_model_load():
    """A mismatch must be caught before the checkpoint loads, not after -- the
    check is worth a second, not a GPU-hour."""
    s = _src(GENERATE)
    i_gate = s.index("runner.resolve_arms")
    i_load = s.index("runner.resolve_checkpoint")
    assert i_gate < i_load, "sha gate runs after the checkpoint is resolved"


def test_part_b_arms_are_named_adapters():
    """Part B compares several corpora at once -- all_fullcds against its
    start-codon-gated twin, and against the sparse-clade arm -- and those are
    only comparable when run together. Two arms hardcoded as base/finetuned
    cannot express that."""
    s = _src(GENERATE)
    assert "b1_sparse_clade" in s and "all_fullcds" in s
    assert 'generated_{arm}.csv' in s, "arm name does not reach the output filename"
    r = _src(RUNNER)
    assert "duplicate arms" in r, "a repeated arm would silently overwrite its csv"
    assert "finetuned" in r, "the pre-split arm name is no longer accepted"


def test_generators_share_one_decoding_procedure():
    """The batching, seeding and CSV schema live in src/generate/runner.py. If a
    part redeclares them, the arms can drift apart in a way that is
    indistinguishable from a real effect."""
    for p in GENERATORS:
        s = _code(p)
        assert "from src.generate import runner" in s
        for token in ("BATCH_SIZE =", "TEMPERATURE =", "TOP_K =", "TOP_P =",
                      "def cds_of", "manual_seed"):
            assert token not in s, f"{p.name} redeclares the protocol ({token})"


def test_trainer_persists_the_split():
    s = _src(TRAINER)
    assert "split_manifest.json" in s, "split composition is not persisted"
    assert '"val_novel"' in s and '"per_clade"' in s
    # membership, not just counts -- counts cannot answer "was this species held out"
    assert '"species"' in s, "manifest records counts but not membership"


def test_trainer_records_what_was_adapted():
    s = _src(TRAINER)
    assert "adapter_config.json" in s
    assert "trainable_params" in s and "total_params" in s


def test_leakage_is_recorded_not_asserted():
    """A cross-clade taxid is a corpus data-quality issue, not grounds for
    killing a half-hour GPU run mid-flight."""
    s = _src(TRAINER)
    i = s.index("leaked_species_into_val_novel")
    window = s[i:i + 1200]
    assert "assert split_manifest" not in window, "leakage kills the run"
    assert "WARNING" in window, "leakage is silent"


def test_scripts_are_syntactically_valid():
    for p in (FINETUNE, GENERATE, LAUNCHER, RUNNER):
        ast.parse(_src(p))


# ---------------------------------------------------------------------------
# Record durability. Three adapters once survived a sandbox timeout while their
# loss curves, split manifests and provenance died with the container: the
# adapter was written to the persistent volume, the record was not. A model whose
# training cannot be documented is not a result.
# ---------------------------------------------------------------------------


def test_the_record_is_persisted_to_the_volume():
    s = _code(FINETUNE)
    assert "records" in s and "WEIGHTS_MOUNT" in s, \
        "the record never reaches persistent storage"
    assert s.count("persist_record()") >= 3, \
        "the record is not persisted on every exit path"


def test_the_record_is_persisted_on_failure_too():
    """A run that failed is a result about the code, and its traceback is the
    part worth keeping."""
    s = _src(FINETUNE)
    for stage in ("no checkpoint loaded", "training failed"):
        i = s.index(stage)
        assert "persist_record()" in s[max(0, i - 400):i], \
            f"the {stage!r} path does not persist its record"


def test_a_copy_failure_never_kills_a_completed_run():
    s = _src(FINETUNE)
    i = s.index("def persist_record()")
    body = s[i:i + 1600]
    assert "except OSError" in body and "WARNING" in body, \
        "a failed copy would abort a finished training run"


def test_the_launcher_stops_watching_a_dead_sandbox():
    """Iterating proc.stdout blocks and a dead sandbox does not close the stream:
    one run hung eighty minutes after its sandbox expired, so a dead job looked
    like a running one. A deadline alone is insufficient -- it would still wait
    hours after an early death -- so liveness is polled."""
    s = _src(Path("src/modal_job.py"))
    assert "sb.poll()" in s, "sandbox liveness is never checked while streaming"
    assert "threading" in s, "the blocking reader is not isolated"
    i = s.index("def stream_until")
    assert "deadline_s" in s[i:i + 900]


def test_records_are_recovered_from_the_volume_not_only_the_sandbox():
    """The tar harvest needs a live sandbox. The volume does not."""
    s = _src(Path("src/modal_job.py"))
    assert "def fetch_volume_records" in s
    launcher = _src(LAUNCHER)
    assert "fetch_volume_records" in launcher, \
        "the launcher never recovers records from the volume"
