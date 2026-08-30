#!/usr/bin/env python3
"""TRAINING LAUNCHER -- finetunes on Modal, stores the adapter, and stops.

This script no longer generates. It exists to produce WEIGHTS, which are then
reused by as many generation runs as you like:

    python scripts/run_on_modal.py                       # train b2_balanced
    python scripts/run_on_modal.py --configs configs/b1_sparse_clade.yaml,configs/all_fullcds.yaml
    python scripts/run_on_modal.py --seed 1              # a training replicate
    python scripts/run_on_modal.py --detach              # keep the sandbox alive after

Roughly 30 min of H100 time per config. What it produces, per config:

  * /weights/adapters/<tag>_adapter_best.pt on the persistent volume, which
    survives sandbox expiry -- the first adapter this project trained did not
  * the full training record (run_summary.json, split_manifest.json,
    adapter_config.json, history.json, provenance.json) harvested to
    results/<tag>/<UTC>_<commit>/<tag>/, one directory per run
  * a row in RUNS.md carrying the sandbox id, which is what joins this repo's
    record to Modal's own expiring log
  * the adapter's sha256, PRINTED in scripts/generate_ab.py's --expect-sha form.
    That handoff is the whole point: retyping a sha256 by eye is exactly where
    the wrong weights get blessed.

Then generate, as often as the protocol changes, against those exact weights:

    python scripts/run_generate_on_modal.py --part A
    python scripts/run_generate_on_modal.py --part B --arms base,b2_balanced \\
        --expect-sha b2_balanced=<sha>

Why training and generation are separate: coupling them charged ~30 min of
retraining to every generation experiment, so changing a prompt protocol meant
retraining an identical model -- and nothing verified that a later generation
used the same weights. Now the sha256 does.

Why launched from here rather than through the agent's compute layer: job
submission there wedged (submits hang without raising and never surface an
approval card). This is the same work, against your own Modal account.

For a free five-second check that the image, cached weights and inputs are
present, run scripts/check_modal_setup.py first. THAT one is the test.

Needs: `pip install modal` and `modal token new` (once).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import modal_job, provenance

REPO = modal_job.REPO
# Calibrated on a 3,846-record corpus that trained in 1,820 s for 3 epochs.
# The budget MUST scale with corpus size: a flat per-config figure budgeted 2.4 h
# for three corpora needing 2.8 h, and a sandbox killed at its timeout cannot be
# harvested -- an undersized budget does not truncate the run, it loses all of it.
SECONDS_PER_RECORD_EPOCH = 0.473 / 3
STARTUP_S = 900             # image pull, pip install, checkpoint load


def budget(jobs: list[dict], repo: Path | None = None) -> int:
    """Sandbox timeout for these corpora, from their actual record counts.

    Generous on purpose: overshooting costs nothing, since the sandbox is
    terminated as soon as the work finishes, while undershooting loses the run.
    """
    repo = repo or REPO
    total = 0.0
    for j in jobs:
        path = repo / j["data"]
        if path.exists():
            with open(path) as fh:
                n = sum(1 for _ in fh)
        else:
            n = 4000        # a corpus not yet built; budget for a typical arm
        cfg = yaml.safe_load((repo / j["config"]).read_text())
        total += n * SECONDS_PER_RECORD_EPOCH * int(cfg.get("epochs", 3))
    # 2.5x, not 1.5x. The sandbox is terminated as soon as the work finishes, so
    # an over-large timeout costs nothing, while an under-large one killed a run
    # that had already trained all three adapters and lost every record.
    return int(total * 2.5 + STARTUP_S)


def plan(configs: list[str], seed: int | None) -> list[dict]:
    """Resolve each config to a tag and its corpus, failing before any GPU.

    data/*.jsonl is gitignored, so a fresh clone lacks the training corpus and
    the run would otherwise die ~1 min into a billed H100 with "no such file".
    The corpus path is read from the config rather than hardcoded, so a new arm
    cannot skip the check.
    """
    missing = [c for c in configs if not (REPO / c).exists()]
    if missing:
        sys.exit("missing configs: " + ", ".join(missing))
    jobs = []
    for c in configs:
        cfg = yaml.safe_load((REPO / c).read_text())
        stem = Path(c).stem
        jobs.append({"config": c, "data": cfg["data"],
                     # _seed<N> so a replicate cannot overwrite -- or be mistaken
                     # for -- the original adapter on the volume.
                     "tag": f"{stem}_seed{seed}" if seed is not None else stem})
    absent = sorted({j["data"] for j in jobs if not (REPO / j["data"]).exists()})
    if absent:
        sys.exit(
            "missing training corpora: " + ", ".join(absent)
            + "\nbuild them first, e.g.:\n"
              "  python -m src.data.build_dataset --arm ALL_fullcds --out data/all_fullcds.jsonl")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="configs/all_fullcds.yaml",
                    help="comma-separated. Several are trained in sequence in one "
                         "sandbox, so the arms of a corpus-composition comparison "
                         "share an image, a checkpoint and a machine.")
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if an adapter with this tag is on the volume")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--timeout", type=int, default=None,
                    help="seconds; default scales with corpus size + 15 min startup")
    ap.add_argument("--detach", action="store_true",
                    help="leave the sandbox running once the job finishes, to "
                         "poke at it. The launcher still streams and blocks; "
                         "terminate it yourself or it bills until timeout.")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not configs:
        sys.exit("--configs is empty")
    jobs = plan(configs, args.seed)
    for j in jobs:
        print(f"  train {j['config']} -> tag {j['tag']} (corpus {j['data']})")

    # One --out per tag: with a shared out/ the second config's run_summary would
    # overwrite the first, and the record is the deliverable of a training job.
    commands = []
    for j in jobs:
        c = (f"python scripts/finetune.py --config {j['config']} "
             f"--tag {j['tag']} --out out/{j['tag']}")
        if args.seed is not None:
            c += f" --seed {args.seed}"
        if args.force:
            c += " --force"
        commands.append(c)

    # results/<arm>/<UTC>_<sha>/ -- never a fixed path. A shared directory means
    # a rerun destroys the previous run's record, including its failure.
    arm = "+".join(j["tag"] for j in jobs)
    out = REPO / provenance.run_dir("results", arm, repo=REPO)
    rc = modal_job.launch(
        commands, arm=arm, out_dir=out, gpu=args.gpu,
        timeout=args.timeout or budget(jobs),
        detach=args.detach, note=f"train {len(jobs)} config(s)")

    # Records off the VOLUME, not just the sandbox tar. A sandbox that expires
    # takes its local out/ with it -- that is how three adapters once arrived
    # with no loss curves, no split manifests and no provenance.
    print("recovering training records from the weights volume:")
    modal_job.fetch_volume_records([j["tag"] for j in jobs], out)
    report_shas(out, jobs)
    return rc


def report_shas(out: Path, jobs: list[dict]) -> None:
    """Print the handoff to generation: the fingerprints, ready to paste."""
    pairs = []
    for j in jobs:
        summary = out / j["tag"] / "run_summary.json"
        if not summary.exists():
            print(f"  {j['tag']}: no run_summary.json recovered -- the adapter may "
                  "still be on the volume, but its training is undocumented")
            continue
        rec = json.loads(summary.read_text())
        sha = (rec.get("adapter") or {}).get("sha256")
        if sha:
            pairs.append(f"{j['tag']}={sha}")
        print(f"  {j['tag']}: sha256 {sha}  train {rec.get('train_s')}s  "
              f"source {rec.get('adapter_source')}")
    if pairs:
        arms = ",".join(j["tag"] for j in jobs)
        print("\ngenerate against exactly these weights:\n"
              f"  python scripts/run_generate_on_modal.py --part B --arms base,{arms} \\\n"
              f"      --expect-sha {','.join(pairs)}")


if __name__ == "__main__":
    sys.exit(main())
