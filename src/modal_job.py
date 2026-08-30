"""Dispatch a job to a Modal sandbox, and bring its record home.

Extracted from run_on_modal.py because there are now two launchers -- one for
training, one for generation -- and the plumbing they share is the part that has
already lost work once: an adapter died with an unharvested sandbox, and the
commit a remote run came from was recorded as null because the packed repo has
no `.git`.

What this module guarantees, so neither launcher has to remember it:

  * the repo is packed WITHOUT .git or caches, so the tarball stays small
  * CS_JOB_ID and CS_GIT_STATE reach the job as environment variables, which is
    how provenance.json in the sandbox learns the sandbox id and the commit it
    is running (see src/provenance.py -- both are read there, not here)
  * out/ is harvested even when the job fails, because a traceback is a result
  * one row per job lands in RUNS.md, carrying the sandbox id that joins the
    repo's record to Modal's own expiring log

Why a sandbox rather than a Modal function: the image is pinned by id and the
work is a shell pipeline over a packed repo, not a decorated Python entry point.

FILE TRANSFER, and why it looks like this: Modal's Sandbox filesystem API
(`sb.open`) is no longer supported server-side -- it fails with
FAILED_PRECONDITION on first contact, which is what the previous version of this
code did. Files move through `exec` instead: bytes into `cat > file` on stdin,
and out through `tar czf -` on stdout with `text=False`. That is the documented
path and it needs no volume commit semantics in either direction.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from src import modal_env, provenance

REPO = Path(__file__).resolve().parent.parent

# .git is excluded (large, and the commit travels via CS_GIT_STATE instead);
# data/bionemo is a regenerable multi-GB export.
# results/ is excluded wholesale: harvested records and tarballs from previous
# runs would otherwise grow every subsequent upload by the size of every run
# before it, and the sandbox never reads them.
SKIP = {".git", "__pycache__", ".venv", "data/bionemo", ".pytest_cache",
        ".ruff_cache", "results"}
SKIP_SUFFIX = {".pyc", ".gz"}


def pack_repo(dest: Path, repo: Path = REPO) -> Path:
    """Tar the working tree as it stands -- including uncommitted edits.

    That is deliberate and it is why the run is recorded as dirty: what runs
    remotely is this tree, not the commit it resembles.
    """
    with tarfile.open(dest, "w:gz") as tf:
        for p in sorted(repo.rglob("*")):
            rel = p.relative_to(repo)
            if any(str(rel).startswith(s) or s in rel.parts for s in SKIP):
                continue
            if p.suffix in SKIP_SUFFIX or not p.is_file():
                continue
            tf.add(p, arcname=f"repo/{rel}")
    return dest


def job_env(job_id: str, repo: Path = REPO) -> dict:
    """Environment that lets the remote run record itself honestly.

    Passed to `exec(env=...)` rather than shell-exported: the git state is JSON
    and would otherwise have to survive a round of shell quoting for no reason.
    """
    return {"CS_JOB_ID": job_id,
            "CS_GIT_STATE": json.dumps(provenance.git_state(repo),
                                       separators=(",", ":")),
            **modal_env.hf_env()}


# 1 MiB per write. modal.io_streams caps the stdin buffer at MAX_BUFFER_SIZE =
# 2 MiB and raises BufferError when a write would exceed it, so the chunk must
# stay comfortably under that -- drain() empties the buffer between writes.
# Do not raise this to 2 MiB or above.
CHUNK = 1024 * 1024


def upload(sb, local: Path, remote: str) -> None:
    """Stream a local file to `remote` inside the sandbox, through stdin."""
    proc = sb.exec("bash", "-lc", f"cat > {remote}", text=False)
    with open(local, "rb") as fh:
        while chunk := fh.read(CHUNK):
            proc.stdin.write(chunk)
            proc.stdin.drain()
    proc.stdin.write_eof()
    proc.stdin.drain()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"upload of {local} failed: exit {proc.returncode}")


def launch(commands: list[str], *, arm: str, out_dir: Path, gpu: str = "H100",
           timeout: int = 3600, detach: bool = False, repo: Path = REPO,
           ledger: str | Path | None = "RUNS.md", note: str = "") -> int:
    """Run `commands` in sequence in one sandbox. Returns the exit code.

    Commands are joined with `&&`, never `;`: a later stage must not run on a
    machine whose earlier stage died, because the harvested record would then
    look like a clean run.
    """
    import modal  # imported here so the CPU test suite need not have it

    tarball = pack_repo(repo.parent / "repo_run.tar.gz", repo)
    print(f"packed {tarball} ({tarball.stat().st_size/1e6:.1f} MB)")

    app = modal.App.lookup(modal_env.APP_NAME, create_if_missing=True)
    img = modal.Image.from_id(modal_env.IMAGE_ID)
    vol = modal.Volume.from_name(modal_env.VOLUME, create_if_missing=True)

    sb = modal.Sandbox.create(
        "bash", "-lc", f"sleep {timeout}",
        image=img, app=app, gpu=gpu,
        volumes={str(modal_env.WEIGHTS_MOUNT): vol}, timeout=timeout,
        workdir="/work",
    )
    print(f"sandbox {sb.object_id} — dashboard: https://modal.com/apps")
    row = {"utc": datetime.now(timezone.utc).isoformat(), "arm": arm,
           "job_id": sb.object_id, "note": note}
    point_latest(out_dir)
    git = provenance.git_state(repo)
    row |= {"commit": git["commit_short"], "dirty": git["dirty"]}
    rc = -1
    ledger_written = False
    try:
        upload(sb, tarball, "/work/repo.tar.gz")

        cmd = ("cd /work && tar xzf repo.tar.gz && cd repo && "
               "pip install -q -e . 2>&1 | tail -1; " + " && ".join(commands))
        proc = sb.exec("bash", "-lc", cmd, env=job_env(sb.object_id, repo))
        rc = stream_until(proc, sb, deadline_s=timeout + 300)
        print(f"\nexit {rc}")
        # Ledger FIRST, before harvest or recovery. The row records that a job ran
        # and how it ended; nothing downstream should be able to suppress it. A
        # recovery failure once lost the row for a run that had succeeded.
        if ledger:
            provenance.append_ledger(
                ledger, row | {"outcome": "ok" if rc == 0 else f"failed ({rc})"})
            ledger_written = True
        try:
            harvest(sb, out_dir)
        except Exception as e:  # noqa: BLE001 - a dead sandbox cannot be tarred
            print(f"sandbox harvest unavailable ({type(e).__name__}); "
                  "records on the volume are the durable copy")
    finally:
        if ledger and not ledger_written:
            # the job never reached an exit status -- still record the attempt
            provenance.append_ledger(ledger, row | {"outcome": f"failed ({rc})"})
        if detach:
            print(f"sandbox {sb.object_id} left running — terminate it when done")
        else:
            try:
                sb.terminate()
                print("sandbox terminated")
            except Exception as e:  # noqa: BLE001 - already gone is the common case
                print(f"sandbox already gone ({type(e).__name__})")
    return rc


def point_latest(run_dir: Path) -> None:
    """`<arm>/latest` -> this run. Convenience only, and gitignored as such."""
    link = run_dir.parent / "latest"
    try:
        link.unlink(missing_ok=True)
        link.symlink_to(run_dir.name)
    except OSError:
        pass    # a symlink is a nicety; never fail a run over one


POLL_S = 60


def stream_until(proc, sb, deadline_s: float) -> int:
    """Print the job's output, stopping when the sandbox dies or the deadline passes.

    Iterating proc.stdout blocks, and a sandbox that dies mid-job does not
    necessarily close the stream: one run hung here for eighty minutes after its
    sandbox had already expired, so a dead job looked like a running one. The
    reader therefore runs on a daemon thread while this loop watches the sandbox
    itself -- a deadline alone is not enough, since it would still wait hours
    after an early death.
    """
    def _pump():
        try:
            for line in proc.stdout:
                print(line, end="")
                sys.stdout.flush()
        except Exception as e:  # noqa: BLE001 - a dead stream must not raise here
            print(f"[stream ended: {type(e).__name__}]", flush=True)

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    waited = 0.0
    while t.is_alive() and waited < deadline_s:
        t.join(timeout=POLL_S)
        waited += POLL_S
        if not t.is_alive():
            break
        try:
            if sb.poll() is not None:      # sandbox exited; the stream will not
                print(f"\n[sandbox exited with {sb.poll()} while streaming; "
                      "stopping the reader]", flush=True)
                return sb.poll()
        except Exception as e:  # noqa: BLE001 - polling must never end the run
            # Report and keep reading: a transient poll failure is not evidence
            # the sandbox died, and treating it as such would abandon a live job.
            print(f"[sandbox poll failed: {type(e).__name__}]", flush=True)
    if t.is_alive():
        print(f"\n[no completion after {waited:.0f}s; giving up on the stream]",
              flush=True)
        return -1
    try:
        proc.wait()
        return proc.returncode
    except Exception as e:  # noqa: BLE001
        print(f"[could not read exit status: {type(e).__name__}]", flush=True)
        return -1


def fetch_volume_records(tags: list[str], out_dir: Path,
                         volume_name: str | None = None,
                         prefix: str = "records") -> dict:
    """Pull each tag's outputs off the weights volume.

    `prefix` selects what to recover: "records" for training records, or
    "generations" for generated sequences.

    The primary path, not a fallback: records written to the volume outlive the
    sandbox exactly as the adapter does, so this works even when the container is
    long dead -- which is the case the tar harvest cannot cover.
    """
    import modal
    from modal.volume import FileEntryType

    vol = modal.Volume.from_name(volume_name or modal_env.VOLUME)
    got = {}
    for tag in tags:
        base = f"{prefix}/{tag}"
        try:
            entries = list(vol.iterdir(base, recursive=True))
        except Exception as e:  # noqa: BLE001 - absence is a reportable outcome
            print(f"  {tag}: no record on the volume ({type(e).__name__})")
            continue
        n = 0
        for e in entries:
            rel = e.path[len(base):].lstrip("/")
            # Skip directory entries EXPLICITLY. Opening one for writing creates a
            # regular file where a directory belongs, and every file beneath it
            # then fails to land -- which is how a run whose records were all
            # safely on the volume still came home empty.
            if not rel or e.type == FileEntryType.DIRECTORY:
                continue
            dest = out_dir / tag / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(dest, "wb") as fh:
                    fh.writelines(vol.read_file(e.path))
                n += 1
            except Exception as exc:  # noqa: BLE001 - report, never abort recovery
                print(f"    {e.path}: {type(exc).__name__}: {str(exc)[:80]}")
        got[tag] = n
        print(f"  {tag}: {n} record files recovered from the volume")
    return got


def harvest(sb, out_dir: Path) -> bool:
    """Copy the remote out/ tree back. Runs even after a failure, on purpose.

    An adapter was lost to an expired sandbox once, and `error.txt` is tracked
    deliberately (docs/RUN_TRACKING.md) -- a failed run is a result about the
    code, so it has to come home too.

    `out_dir` must be unique per run (see provenance.run_dir): the archive is
    extracted in place, so a shared directory means each run destroys the record
    of the one before it -- which is how three earlier B2 attempts lost their
    tracebacks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # tar to stdout, read as bytes. `ls out` first so an absent out/ is reported
    # as "nothing ran" rather than as a corrupt archive.
    proc = sb.exec("bash", "-lc",
                   "cd /work/repo && ls out >/dev/null 2>&1 && tar czf - out",
                   text=False)
    blob = proc.stdout.read()
    proc.wait()
    if proc.returncode != 0 or not blob:
        print("nothing to harvest: out/ was never written on the sandbox")
        return False
    (out_dir / "out.tgz").write_bytes(blob)
    # --strip-components=1 drops the remote "out/" wrapper: the run directory is
    # already this run's, so the useful level below it is the per-tag one.
    subprocess.run(["tar", "xzf", "out.tgz", "--strip-components=1"],
                   cwd=out_dir, check=False)
    print(f"record harvested to {out_dir} ({len(blob)/1e6:.1f} MB)")
    return True
