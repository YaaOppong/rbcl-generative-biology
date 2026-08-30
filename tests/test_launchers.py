"""The two Modal launchers and the plumbing they share (src/modal_job.py).

There are two launchers -- one for weights, one for sequences -- and everything
they have in common is the part that has already lost work: an adapter died with
an unharvested sandbox, and a truncated copy of the image id needed its own
commit to fix (daee32b). So the shared pieces are asserted here rather than
trusted to stay in step.

No Modal calls: `modal` is imported inside launch(), so this runs in the CPU
suite alongside everything else.
"""
import ast
import json
from pathlib import Path

from src import modal_env, modal_job

TRAIN = Path("scripts/run_on_modal.py")
GENERATE = Path("scripts/run_generate_on_modal.py")
PREFLIGHT = Path("scripts/check_modal_setup.py")
LAUNCHERS = (TRAIN, GENERATE)


def _src(p: Path) -> str:
    assert p.exists(), f"{p} missing"
    return p.read_text()


def test_the_pinned_environment_has_one_definition():
    """A second copy of the image id is how it got truncated the first time."""
    for const in (modal_env.IMAGE_ID, modal_env.VOLUME):
        hits = [p for p in Path().rglob("*.py")
                if ".git" not in p.parts and const in p.read_text()
                and p != Path("src/modal_env.py")]
        assert not hits, f"{const} is restated in {hits}"
    assert modal_env.IMAGE_ID.startswith("im-")


def test_both_launchers_share_the_plumbing():
    """Packing, the CS_* exports, harvesting and the ledger row live in one
    place; a launcher that reimplements them will drift."""
    for p in LAUNCHERS:
        s = _src(p)
        assert "modal_job.launch" in s, f"{p.name} does not use the shared launcher"
        for token in ("Sandbox.create", "tarfile", "Volume.from_name"):
            assert token not in s, f"{p.name} reimplements the plumbing ({token})"


def test_the_job_hands_down_its_identity():
    """CS_JOB_ID joins this repo's record to Modal's expiring log; CS_GIT_STATE
    is how a sandbox with no .git still records the commit it ran. Both are read
    by src/provenance.py, so the launcher must pass them into the job."""
    env = modal_job.job_env("sb-test")
    assert env["CS_JOB_ID"] == "sb-test"
    assert json.loads(env["CS_GIT_STATE"])["commit"]
    for k, v in modal_env.hf_env().items():
        assert env[k] == v, "the checkpoint cache is not pinned"


def test_transfer_does_not_use_the_retired_filesystem_api():
    """Modal's Sandbox filesystem API (sb.open) fails server-side with
    FAILED_PRECONDITION -- it is what the first version of this launcher died
    on. Files move through exec's stdin/stdout instead."""
    s = _src(Path("src/modal_job.py"))
    assert "sb.open(" not in s, "the retired sandbox filesystem API is back"
    assert 'sb.exec("bash", "-lc", f"cat > {remote}"' in s, "upload is not via exec"
    assert "tar czf -" in s, "harvest is not via exec"
    assert "text=False" in s, "binary streams are required for a tarball"
    assert "stdin.write_eof" in s, "an unterminated stdin hangs the remote cat"


def test_upload_chunk_stays_under_modals_stdin_buffer():
    """modal.io_streams caps the stdin buffer at 2 MiB and raises BufferError on
    a write that would exceed it. A 4 MiB chunk failed on the first write of the
    first real run -- and the transport probe missed it because its payload fit
    in one write."""
    assert modal_job.CHUNK < 2 * 1024 * 1024


def test_stages_are_chained_on_success_only():
    """`&&`, never `;`: a later stage must not run on a machine whose earlier
    stage died, or the harvested record looks like a clean run."""
    s = _src(Path("src/modal_job.py"))
    assert '" && ".join(commands)' in s
    assert '"; ".join' not in s


def test_harvest_survives_a_failed_run():
    """error.txt is tracked deliberately -- a failed run is a result about the
    code, so it has to come home too."""
    s = _src(Path("src/modal_job.py"))
    i_rc = s.index("rc = stream_until")
    i_harvest = s.index("harvest(sb, out_dir)", i_rc)
    assert i_rc < i_harvest, "harvest runs before the job finishes"
    # the call must not sit inside a success branch -- check the statement itself,
    # not a substring scan of the region, which false-matched on `if rc == 0`
    line = next(ln for ln in s[i_harvest - 200:i_harvest + 60].splitlines()
                if "harvest(sb, out_dir)" in ln)
    assert line.strip().startswith("harvest("), \
        f"harvest is conditional: {line.strip()!r}"


def test_git_is_not_packed_but_the_commit_still_travels():
    assert ".git" in modal_job.SKIP
    assert "CS_GIT_STATE" in _src(Path("src/modal_job.py"))


def test_each_run_gets_its_own_directory():
    """A fixed out_dir means a rerun destroys the previous run's record, failure
    included -- three B2 attempts once shared one and each wiped the last. The
    second retrain in this project overwrote the first's record in place before
    this was fixed."""
    for p in LAUNCHERS:
        s = _src(p)
        assert "provenance.run_dir(" in s, f"{p.name} harvests to a fixed path"
        assert 'results" / "modal' not in s, f"{p.name} still uses the shared directory"


def test_harvest_extracts_into_the_run_directory():
    s = _src(Path("src/modal_job.py"))
    assert "--strip-components=1" in s, "the remote out/ wrapper is not stripped"
    assert "point_latest" in s, "no latest pointer for the newest run"


def test_every_launched_job_lands_in_the_ledger():
    """RUNS.md is the human index over run directories, and the sandbox id is
    the only thing that joins it to Modal's own log."""
    s = _src(Path("src/modal_job.py"))
    assert s.count("append_ledger") >= 2, \
        "there is no fallback path for a launch that never reaches an exit status"
    # primary write happens as soon as rc is known, BEFORE harvest and recovery;
    # the finally block is the fallback for a launch that dies before that.
    i_rc = s.index("rc = stream_until")
    assert s.index("append_ledger", i_rc) < s.index("harvest(sb, out_dir)", i_rc)
    assert "finally:" in s, "no fallback for a crashed launch"


def test_train_launcher_reports_the_sha_in_the_generators_own_syntax():
    """The handoff has to be paste-ready: retyping a sha256 by eye is exactly
    where the wrong weights get blessed."""
    s = _src(TRAIN)
    assert "--expect-sha" in s
    assert "run_generate_on_modal.py --part B" in s


def test_generation_budget_scales_with_the_workload():
    """A flat per-arm timeout budgeted 45 min for a full Part A titration, which
    needs six hours. A sandbox killed at its timeout cannot be harvested, so the
    undersized budget does not truncate the run -- it loses all of it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gen_launcher", GENERATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    small, large = mod.budget(240), mod.budget(1800)
    assert large > 6 * 3600, "a full Part A titration is not given six hours"
    assert large > small * 5, "the budget does not scale with the sequence count"
    assert small > 40 * 60, "no headroom over the measured rate"


def test_generate_launcher_keeps_the_parts_apart():
    """One script now serves both parts, so the launcher is what stops --part A
    being handed adapter arms -- Part A is the base model by definition."""
    s = _src(GENERATE)
    assert 'args.part == "A" and args.expect_sha' in s
    assert 'args.arms.strip() != "base"' in s
    assert 'args.part == "B" and not args.arms' in s, "part B without arms is silent"
    assert "generate_ab.py" in s and "generate_a.py" not in s


def test_preflight_checks_the_adapter_a_part_b_run_needs():
    s = _src(PREFLIGHT)
    assert "modal_env.adapter_path" in s
    assert "MISSING" in s


def test_launcher_scripts_are_syntactically_valid():
    for p in (TRAIN, GENERATE, PREFLIGHT, Path("src/modal_job.py")):
        ast.parse(_src(p))


def test_generated_sequences_are_mirrored_to_the_volume():
    """The sandbox tar harvest needs a live container, and one died before it ran
    -- taking a level's generated sequences with it. Training survived the same
    failure because finetune.py mirrors its record to the volume as it goes.
    Generation now does the same, so the volume is the durable copy and the tar
    is the convenience."""
    r = _src(Path("src/generate/runner.py"))
    assert "GENERATION_DIR" in r and "def mirror_to_volume" in r
    assert r.index("mirror_to_volume(out_csv.parent") < r.index("def mirror_to_volume"), \
        "sequences are not mirrored as each arm completes"
    assert "except OSError" in r, "a copy failure could end a producing run"
    s = _src(GENERATE)
    assert "fetch_volume_records" in s and 'prefix="generations"' in s, \
        "the launcher never recovers sequences from the volume"


def test_the_ledger_row_is_written_before_recovery():
    """The row records that a job ran and how it ended; nothing downstream should
    suppress it. A recovery failure once lost the row for a run that succeeded."""
    s = _src(Path("src/modal_job.py"))
    i_rc = s.index("rc = stream_until")
    i_ledger = s.index("append_ledger", i_rc)
    i_harvest = s.index("harvest(sb, out_dir)", i_rc)
    assert i_ledger < i_harvest, "the ledger row is still written after the harvest"
    assert "ledger_written" in s, "a crashed launch would record no attempt at all"
