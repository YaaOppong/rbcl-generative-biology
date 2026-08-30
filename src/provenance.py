"""Record what code produced a result, and refuse to lie about it.

Motivation, from this project's own history: three B2 finetune attempts wrote to
the same fixed `out_dir`, so each overwrote the previous one's failure record.
The tracebacks that produced four real fixes survived only because the remote
sandboxes happened to still be warm. And no run recorded the commit it ran from,
so a result was attributable only by the accident of a clean tree.

Two rules follow, and they are the whole point of this module:

1. Every run writes `provenance.json` BEFORE doing any work -- so a crashed run
   is still attributable.
2. A dirty working tree is recorded as dirty, and by default refuses to run.
   An unqualified SHA on a modified tree is a false provenance claim, which is
   worse than no claim: it points a future reader at code that never ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Packages whose version materially changes the numbers. Recorded per run
# because a silent upgrade is otherwise invisible in the result.
TRACKED_PACKAGES = ("evo2", "torch", "vortex", "transformer_engine", "flash_attn",
                    "numpy", "biopython")


class DirtyTreeError(RuntimeError):
    """Raised when a run is attempted on a modified working tree."""


def _git(*args: str, repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(repo) if repo else None,
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def inherited_git_state() -> dict | None:
    """Git state handed down by a launcher, via CS_GIT_STATE (JSON).

    A remote job runs from an unpacked tarball with no `.git`, so it cannot see
    its own commit -- every remote result would record `commit: null`, which is
    exactly the attribution hole this module exists to close. The launcher knows
    the commit (it is standing in the checkout), so it passes it down. Marked
    `source: launcher` because it is a claim about the machine that dispatched
    the job, not one the job verified for itself.
    """
    raw = os.environ.get("CS_GIT_STATE")
    if not raw:
        return None
    try:
        state = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(state, dict) or not state.get("commit"):
        return None
    # Defaults for every key write() reads, so a partial or older launcher
    # payload degrades to a recorded gap rather than a KeyError mid-run.
    base = {"commit": None, "commit_short": None, "branch": None, "describe": None,
            "remote": None, "dirty": False, "dirty_file_count": 0, "dirty_files": []}
    return {**base, **state, "in_git_checkout": False, "source": "launcher"}


def git_state(repo: Path | None = None) -> dict:
    """Commit, dirt, and branch. Every field may be None outside a checkout.

    `dirty_files` is truncated but its COUNT is not -- a reader needs to know
    the true scale of divergence, not just the first few names.
    """
    status = _git("status", "--porcelain", repo=repo)
    if status is None:
        inherited = inherited_git_state()
        if inherited is not None:
            return inherited
    dirty_lines = [ln for ln in (status or "").splitlines() if ln.strip()]
    return {
        "commit": _git("rev-parse", "HEAD", repo=repo),
        "commit_short": _git("rev-parse", "--short", "HEAD", repo=repo),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "describe": _git("describe", "--tags", "--always", "--dirty", repo=repo),
        "remote": _git("config", "--get", "remote.origin.url", repo=repo),
        "dirty": bool(dirty_lines),
        "dirty_file_count": len(dirty_lines),
        # Split off the 2-char status code, not a fixed slice: `git status
        # --porcelain` emits " M path" for modified but "?? path" for untracked,
        # and a blind ln[3:] silently truncated the first character of every
        # modified filename ("a.py" -> ".py").
        "dirty_files": [ln[2:].strip() for ln in dirty_lines[:20]],
        "in_git_checkout": status is not None,
    }


def package_versions(names=TRACKED_PACKAGES) -> dict:
    import importlib.metadata as md

    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None  # not installed in this environment; recorded as such
    return out


def file_digest(path: str | Path, limit_mb: int = 512) -> dict | None:
    """SHA-256 of an input file, so a silently regenerated corpus is detectable.

    Skipped above `limit_mb` -- hashing adapter weights or a multi-GB dataset
    costs more than the provenance is worth; size and mtime still pin it.
    """
    p = Path(path)
    if not p.exists():
        return None
    size = p.stat().st_size
    rec = {"path": str(p), "size_bytes": size,
           "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
    if size <= limit_mb * 1024 * 1024:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        rec["sha256"] = h.hexdigest()
    else:
        rec["sha256"] = None
        rec["sha256_skipped_reason"] = f"larger than {limit_mb} MB"
    return rec


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {"cuda_available": True, "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda}
    except Exception as e:  # noqa: BLE001
        # Deliberately blind: probing the GPU must never be what fails a run.
        # Torch raises a wide and version-dependent range here (driver mismatch,
        # RuntimeError from triton, ImportError, OSError), and the correct
        # response to all of them is to record the failure and carry on.
        return {"cuda_available": None, "error": str(e)[:200]}


def run_dir(base: str | Path, tag: str, repo: Path | None = None) -> Path:
    """`<base>/<tag>/<UTC timestamp>_<short sha>` -- runs never collide.

    A fixed out_dir means a rerun destroys the previous run's record, including
    its failure. Keying on time and commit keeps every attempt.
    """
    # git_state, not _git: inside a remote sandbox the sha comes from the
    # launcher, and a run directory named "nogit" is not traceable to anything.
    sha = git_state(repo).get("commit_short") or "nogit"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return Path(base) / tag / f"{stamp}_{sha}"


def write(
    out_dir: str | Path,
    *,
    config: dict | None = None,
    inputs: list[str | Path] | None = None,
    extra: dict | None = None,
    repo: Path | None = None,
    allow_dirty: bool = False,
) -> dict:
    """Write provenance.json into out_dir. Call BEFORE the work starts.

    Raises DirtyTreeError on a modified tree unless allow_dirty=True, in which
    case `dirty: true` is recorded and the run is honestly labelled.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    git = git_state(repo)

    if git["dirty"] and not allow_dirty:
        raise DirtyTreeError(
            f"{git['dirty_file_count']} modified file(s); results would not be "
            f"attributable to commit {git['commit_short']}. Commit them, or pass "
            "allow_dirty=True to record the run as dirty."
        )

    prov = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "config": config,
        "config_sha256": (
            hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if config is not None else None
        ),
        "inputs": [file_digest(p) for p in (inputs or [])],
        "packages": package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu_info(),
        # Links the repo record to the provider's own job record, which lives
        # outside git and expires.
        "compute_job_id": os.environ.get("CS_JOB_ID"),
        "hostname_masked": True,
        **(extra or {}),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def record_failure(out_dir: str | Path, exc_text: str, stage: str = "unknown") -> None:
    """Persist a traceback next to the run that produced it.

    Failures are evidence. Four fixes in this project came out of three failed
    runs, and those tracebacks were nearly lost to an overwritten out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "error.txt").write_text(
        f"stage: {stage}\nutc: {datetime.now(timezone.utc).isoformat()}\n\n{exc_text}"
    )


def append_ledger(ledger: str | Path, row: dict) -> None:
    """Append one row to a human-readable RUNS.md table (created if absent)."""
    p = Path(ledger)
    cols = ["utc", "arm", "commit", "dirty", "job_id", "outcome", "note"]
    if not p.exists():
        p.write_text(
            "# Run ledger\n\nAppend-only. One row per run; `results/<arm>/<utc>_<sha>/`\n"
            "holds that run's provenance.json, config, metrics, and error.txt.\n\n"
            "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
        )
    with open(p, "a") as fh:
        fh.write("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n")
