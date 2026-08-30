"""Pre-flight check before launching a run. ~5 seconds, no GPU, no cost.

Run from the repository root:

    python scripts/check_modal_setup.py                            # training inputs
    python scripts/check_modal_setup.py --configs configs/b1_sparse_clade.yaml
    python scripts/check_modal_setup.py --part B --arms b2_balanced  # generation too
    python scripts/check_modal_setup.py --transport                  # CPU sandbox, ~30 s

It answers three questions, cheapest first: are the local inputs present, does
the pinned image and volume exist, and are the cached checkpoints (and any
adapter a generation run would need) actually on the volume.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from src import modal_env

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default=None,
                help="comma-separated; their `data:` corpora must exist locally. "
                     "Defaults to configs/all_fullcds.yaml, or to nothing when "
                     "--part is given without it -- generation needs no corpus.")
ap.add_argument("--part", choices=["A", "B"], default=None,
                help="also check what that paper part's generation needs")
ap.add_argument("--arms", default=None,
                help="part B adapter tags to look for on the volume")
ap.add_argument("--transport", action="store_true",
                help="round-trip a file through a CPU-only sandbox, exercising "
                     "the same upload/harvest code a real run uses. Costs "
                     "seconds of CPU, no GPU. Worth it after any Modal SDK "
                     "upgrade: the filesystem API this used to rely on was "
                     "retired server-side, and the failure looks like a healthy "
                     "launch right up until it isn't.")
args = ap.parse_args()

# Inputs a run needs that are NOT in git. data/*.jsonl is gitignored (corpora are
# rebuilt from NCBI, not versioned), so a fresh clone does not have them and a
# run would die after the GPU was already billed. Corpus paths come from the
# configs rather than a hardcoded list, so a new arm cannot skip the check.
# Training needs a corpus; generation does not. Checking for data/b2.jsonl
# before a Part A run would fail a run that never reads it.
configs = args.configs if args.configs is not None else (
    "" if (args.part or args.transport) else "configs/all_fullcds.yaml")
required = {}
for c in [x.strip() for x in configs.split(",") if x.strip()]:
    if not Path(c).exists():
        print(f"MISSING CONFIG: {c}")
        sys.exit(1)
    data = yaml.safe_load(Path(c).read_text())["data"]
    stem = Path(c).stem                      # b2_balanced -> ALL_fullcds
    required[data] = (f"python -m src.data.build_dataset "
                      f"--arm {stem[0].upper() + stem[1:]} --out {data}")
if args.part:
    required["data/prompts_corpus.csv"] = "(tracked in git — if missing, git pull)"
if args.part == "B":
    required["data/part_a_generated_corpus.csv"] = "(tracked in git — if missing, git pull)"

missing = [(p, how) for p, how in required.items() if not Path(p).exists()]
if missing:
    print("MISSING INPUTS — the run would fail after the GPU was billed:\n")
    for p, how in missing:
        print(f"  {p}\n     build with: {how}\n")
    sys.exit(1)
print("required inputs present:", ", ".join(required) or "(none needed)")

print("\nimage:", modal_env.IMAGE_ID)
img_handle = modal.Image.from_id(modal_env.IMAGE_ID)
# NOTE: from_id is lazy -- it constructs a handle without contacting Modal, so
# reaching this line proves nothing about the image existing. The volume
# listing below is the first call that actually hits the API.
print("  handle constructed (not yet validated — see volume listing)")

v = modal.Volume.from_name(modal_env.VOLUME)
print("volume:", modal_env.VOLUME)
for e in v.iterdir("/"):
    print("  ", e.path)

# the weights the run needs
print("\nchecking for cached checkpoints:")
try:
    for e in v.iterdir("/hub"):
        print("  ", e.path)
except Exception as exc:
    print("   /hub not found:", type(exc).__name__)

print("\nadapters on the volume:")
present = []
try:
    for e in v.iterdir("/adapters"):
        print("  ", e.path)
        present.append(Path(e.path).name)
except Exception:
    print("   none yet (expected — the first adapter was lost)")

# A part B generation run needs its adapters BEFORE it is worth launching.
for tag in [x.strip() for x in (args.arms or "").split(",") if x.strip() and x != "base"]:
    want = modal_env.adapter_path(tag).name
    ok = want in present
    print(f"   arm {tag}: {'found' if ok else 'MISSING'} ({want})")
    if not ok:
        print(f"     train it with: python scripts/run_on_modal.py "
              f"--configs configs/{tag}.yaml")


if args.transport:
    import tempfile

    from src import modal_job

    print("\ntransport check — CPU sandbox, no GPU:")
    app = modal.App.lookup(modal_env.APP_NAME, create_if_missing=True)
    sb = modal.Sandbox.create("bash", "-lc", "sleep 300", image=img_handle,
                              app=app, timeout=300, workdir="/work")
    print("  sandbox", sb.object_id)
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            payload = td / "probe.bin"
            # 5 MB, deliberately larger than modal's 2 MiB stdin buffer and
            # comparable to the real repo tarball: a probe that fits in one
            # write never exercises the chunking, which is where this broke.
            payload.write_bytes(b"rbcL" * 1_250_000)
            modal_job.upload(sb, payload, "/work/probe.bin")

            env = modal_job.job_env(sb.object_id)
            proc = sb.exec("bash", "-lc",
                           "mkdir -p /work/repo/out && "
                           "cp /work/probe.bin /work/repo/out/probe.bin && "
                           'echo "$CS_JOB_ID" > /work/repo/out/job_id.txt && '
                           "python -c \"import json,os;"
                           "print(json.loads(os.environ['CS_GIT_STATE'])['commit_short'])\" "
                           "> /work/repo/out/commit.txt && wc -c < /work/probe.bin",
                           env=env)
            uploaded = "".join(proc.stdout).strip()
            proc.wait()
            print(f"  upload: {uploaded} bytes remote vs "
                  f"{payload.stat().st_size} local", 
                  "OK" if uploaded == str(payload.stat().st_size) else "MISMATCH")

            ok = modal_job.harvest(sb, td / "harvested")
            got = td / "harvested" / "out" / "probe.bin"
            print("  harvest:", "OK" if ok and got.read_bytes() == payload.read_bytes()
                  else "FAILED")
            print("  CS_JOB_ID reached the job:",
                  (td / "harvested" / "out" / "job_id.txt").read_text().strip()
                  == sb.object_id)
            print("  CS_GIT_STATE reached the job:",
                  (td / "harvested" / "out" / "commit.txt").read_text().strip())
    finally:
        sb.terminate()
        print("  sandbox terminated")
