"""Tests for run provenance.

The load-bearing property is the refusal: a dirty tree must not silently produce
a result stamped with a commit SHA, because that points a future reader at code
that never ran. Everything else here is bookkeeping.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from src.provenance import (
    DirtyTreeError,
    append_ledger,
    file_digest,
    git_state,
    record_failure,
    run_dir,
    write,
)


def _repo(tmp_path):
    """A real git repo -- git_state shells out, so a fake would test nothing."""
    r = tmp_path / "repo"
    r.mkdir()
    for cmd in (["init", "-q"], ["config", "user.email", "t@t"],
                ["config", "user.name", "t"]):
        subprocess.run(["git", *cmd], cwd=r, check=True, capture_output=True)
    (r / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=r, check=True,
                   capture_output=True)
    return r


def test_clean_tree_records_the_commit(tmp_path):
    r = _repo(tmp_path)
    prov = write(tmp_path / "out", config={"lr": 1e-4}, repo=r)
    assert prov["git"]["dirty"] is False
    assert len(prov["git"]["commit"]) == 40
    assert prov["config_sha256"]
    assert json.loads((tmp_path / "out" / "provenance.json").read_text())["git"]["commit"]


def test_dirty_tree_refuses_by_default(tmp_path):
    """The central guarantee: no silent SHA on modified code."""
    r = _repo(tmp_path)
    (r / "a.py").write_text("x = 2\n")
    with pytest.raises(DirtyTreeError, match="not be attributable"):
        write(tmp_path / "out", repo=r)


def test_dirty_tree_runs_when_explicitly_allowed_and_says_so(tmp_path):
    r = _repo(tmp_path)
    (r / "a.py").write_text("x = 2\n")
    prov = write(tmp_path / "out", repo=r, allow_dirty=True)
    assert prov["git"]["dirty"] is True
    assert prov["git"]["dirty_file_count"] == 1
    assert "a.py" in prov["git"]["dirty_files"]


def test_dirty_file_count_is_not_truncated(tmp_path):
    """Names are capped at 20; the COUNT must still be true."""
    r = _repo(tmp_path)
    for i in range(25):
        (r / f"f{i}.py").write_text(f"y = {i}\n")
    st = git_state(r)
    assert st["dirty_file_count"] == 25
    assert len(st["dirty_files"]) == 20


def test_provenance_is_written_before_work_so_crashes_stay_attributable(tmp_path):
    r = _repo(tmp_path)
    out = tmp_path / "out"
    write(out, repo=r)
    record_failure(out, "Traceback ...\nRuntimeError: boom", stage="train")
    assert (out / "provenance.json").exists()
    assert "RuntimeError: boom" in (out / "error.txt").read_text()
    assert "stage: train" in (out / "error.txt").read_text()


def test_run_dir_is_unique_per_commit_and_time(tmp_path):
    r = _repo(tmp_path)
    d1 = run_dir(tmp_path / "results", "b2", repo=r)
    assert d1.parent.name == "b2"
    sha = git_state(r)["commit_short"]
    assert d1.name.endswith(f"_{sha}")
    # a second commit changes the directory, so runs cannot collide
    (r / "b.py").write_text("z = 3\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "second"], cwd=r, check=True,
                   capture_output=True)
    assert run_dir(tmp_path / "results", "b2", repo=r).name.split("_")[-1] != sha


def test_input_digest_detects_a_changed_corpus(tmp_path):
    f = tmp_path / "corpus.jsonl"
    f.write_text('{"a": 1}\n')
    d1 = file_digest(f)
    f.write_text('{"a": 2}\n')
    assert file_digest(f)["sha256"] != d1["sha256"]
    assert file_digest(tmp_path / "missing.jsonl") is None


def test_large_input_skips_hash_but_records_size(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"0" * 2048)
    d = file_digest(f, limit_mb=0)
    assert d["sha256"] is None
    assert d["size_bytes"] == 2048
    assert "sha256_skipped_reason" in d


def test_ledger_appends_without_clobbering(tmp_path):
    led = tmp_path / "RUNS.md"
    append_ledger(led, {"utc": "t1", "arm": "B2", "outcome": "failed"})
    append_ledger(led, {"utc": "t2", "arm": "B2", "outcome": "ok"})
    body = led.read_text()
    assert body.count("| B2 |") == 2
    assert "t1" in body and "t2" in body
    assert body.index("t1") < body.index("t2")


def test_outside_a_checkout_reports_honestly(tmp_path):
    st = git_state(tmp_path)  # not a repo
    assert st["in_git_checkout"] is False
    assert st["commit"] is None
    # and a run there is not blocked -- there is no tree to be dirty
    prov = write(tmp_path / "out", repo=tmp_path)
    assert prov["git"]["commit"] is None