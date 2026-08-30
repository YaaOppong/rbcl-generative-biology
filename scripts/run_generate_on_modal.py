#!/usr/bin/env python3
"""GENERATION LAUNCHER -- runs one paper part's generation on Modal. Never trains.

The counterpart to run_on_modal.py, which produces weights and stops. This
dispatches the per-part generators, which need the sandbox for the checkpoint
cache (Part A and B) and the adapter volume (Part B):

    python scripts/run_generate_on_modal.py --part A
    python scripts/run_generate_on_modal.py --part A --levels L1_donor_90 --replicates 0
    python scripts/run_generate_on_modal.py --part B --arms base,b2_balanced \\
        --expect-sha b2_balanced=<sha256>
    python scripts/run_generate_on_modal.py --part B \\
        --arms base,b1_sparse_clade,b2_balanced \\
        --expect-sha b1_sparse_clade=<sha1>,b2_balanced=<sha2>

Both parts run scripts/generate_ab.py -- the base arm is the same work either
way, so the part is a matter of which arms you ask for. --part A runs base Evo 2
alone; --part B adds named adapters and refuses any whose sha256 was not stated.

Everything else -- packing the repo, exporting CS_JOB_ID and the commit,
harvesting out/ even after a failure, one row in RUNS.md -- is
src/modal_job.launch, shared with the training launcher.

Roughly 25 min of H100 time per arm per 120 prompts.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import modal_job, provenance
from src.generate import runner

REPO = modal_job.REPO
# Calibrated from the B2 runs: ~12.5 s per generated sequence on an H100 at
# 1800 nt. The budget MUST scale with the prompt count -- a flat per-arm figure
# budgeted 45 min for a full Part A titration, which needs six hours, and a
# sandbox killed at its timeout cannot be harvested at all.
SECONDS_PER_SEQUENCE = 12.5
STARTUP_S = 900          # image pull, pip install, checkpoint load


def budget(n_seqs: int) -> int:
    """Sandbox timeout for a run of this size, with headroom.

    Generous on purpose: overshooting costs nothing (the sandbox is terminated
    when the work finishes), while undershooting kills the job and loses every
    sequence it had already produced, because harvest needs a live sandbox.
    """
    return int(n_seqs * SECONDS_PER_SEQUENCE * 1.4 + STARTUP_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B"],
                    help="A: base model only. B: base vs named adapters.")
    ap.add_argument("--arms", default=None,
                    help="part B only; comma-separated 'base' and adapter tags")
    ap.add_argument("--levels", default=None,
                    help="default: all for part A, L1_donor_90 for part B")
    ap.add_argument("--replicates", default=None,
                    help="default: all for part A, 0 for part B")
    ap.add_argument("--total-len", type=int, default=None)
    ap.add_argument("--expect-sha", default=None, help="part B only; <tag>=<sha256>")
    ap.add_argument("--allow-unverified-adapter", action="store_true",
                    help="part B only; generate against weights nobody checked")
    ap.add_argument("--split-by-level", action="store_true",
                    help="one command per prompt level in the same sandbox, so a "
                         "failure part-way still harvests the levels that finished")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--detach", action="store_true",
                    help="leave the sandbox running once the job finishes; the "
                         "launcher still streams and blocks")
    args = ap.parse_args()

    if args.part == "A" and args.expect_sha:
        sys.exit("part A runs the base model alone, so there is no adapter to "
                 "verify. Use --part B to compare weights.")
    if args.part == "A" and args.arms and args.arms.strip() != "base":
        sys.exit(f"part A is the base model by definition; got --arms {args.arms}. "
                 "Use --part B for adapter arms.")
    if args.part == "B" and not args.arms:
        sys.exit("--part B needs --arms, e.g. --arms base,b2_balanced")

    # generate_ab.py refuses an unverified adapter anyway -- but it does so on the
    # sandbox, a minute and a machine later. The same refusal is free here.
    adapter_arms = [a.strip() for a in (args.arms or "").split(",")
                    if a.strip() and a.strip() != "base"]
    if adapter_arms and not (args.expect_sha or args.allow_unverified_adapter):
        sys.exit(
            f"arms {adapter_arms} are adapters, so --expect-sha is required.\n"
            "The shas are printed at the end of the training run, and are in\n"
            "results/<tag>/<UTC>_<commit>/<tag>/run_summary.json -> "
            "adapter.sha256 (or results/<tag>/latest/).\n"
            "Or pass --allow-unverified-adapter to generate against weights "
            "nobody checked.")

    # The prompt corpus is tracked, but check anyway: a fresh clone that has not
    # pulled would otherwise fail after the GPU was billed.
    prompts = REPO / "data" / "prompts_corpus.csv"
    if not prompts.exists():
        sys.exit(f"missing {prompts} — it is tracked in git, so: git pull")

    arms = [a.strip() for a in (args.arms or "base").split(",") if a.strip()]
    levels = args.levels or ("all" if args.part == "A" else "L1_donor_90")
    replicates = args.replicates or ("all" if args.part == "A" else "0")
    total_len = args.total_len or runner.DEFAULT_TOTAL_LEN

    # Resolve the prompts HERE, on the laptop, to size the budget and to fail on
    # an empty selection before a GPU is allocated.
    prompts = runner.load_prompts(REPO / "data" / "prompts_corpus.csv",
                                  levels, replicates, total_len)
    per_level = sorted({p["level_name"] for p in prompts})
    n_seqs = len(prompts) * len(arms)

    # One command per level when there are several: they are independent, they
    # write to separate directories, and `&&` stops at the first failure -- so a
    # run that dies in level 4 still brings levels 1-3 home. One sandbox, so the
    # checkpoint loads once.
    run_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    groups = per_level if (args.split_by_level and len(per_level) > 1) else [levels]
    commands = []
    for g in groups:
        tag = g if g != levels else f"part_{args.part.lower()}"
        cmd = [f"python scripts/generate_ab.py --out out/{tag}",
               f"--run-tag {run_stamp}_{tag}",
               f"--arms {','.join(arms)}", f"--levels {g}",
               f"--replicates {replicates}", f"--total-len {total_len}"]
        if args.expect_sha:
            cmd.append(f"--expect-sha {args.expect_sha}")
        if args.allow_unverified_adapter:
            cmd.append("--allow-unverified-adapter")
        commands.append(" ".join(cmd))
    groups_tags = [g if g != levels else f"part_{args.part.lower()}" for g in groups]
    for c in commands:
        print(f"  {c}")
    print(f"  {len(prompts)} prompts x {len(arms)} arm(s) = {n_seqs} sequences, "
          f"budget {budget(n_seqs)/3600:.1f} h")

    arm = f"generate_{args.part.lower()}"
    out_dir = REPO / provenance.run_dir("results", arm, repo=REPO)
    rc = modal_job.launch(
        commands, arm=arm, out_dir=out_dir, gpu=args.gpu,
        timeout=args.timeout or budget(n_seqs), detach=args.detach,
        note=f"part {args.part}: {','.join(arms)}, {n_seqs} sequences")

    # Sequences off the VOLUME. The sandbox tar needs a live container and one
    # died before it ran, taking a level's generations with it.
    print("recovering generated sequences from the weights volume:")
    modal_job.fetch_volume_records([f"{run_stamp}_{t}" for t in groups_tags],
                                   out_dir, prefix="generations")
    return rc


if __name__ == "__main__":
    sys.exit(main())
