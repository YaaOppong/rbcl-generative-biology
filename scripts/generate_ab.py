"""Generate rbcL sequences from Part A's prompt corpus, one arm per model.

Covers both paper parts, because they are the same decoding procedure over the
same prompts and differ only in which models are run:

    Part A -- generation without finetuning.  --arms base
    Part B -- does finetuning add anything?   --arms base,all_fullcds

The base arm is identical work in both cases, so it is one script: a separate
Part A entry point would regenerate the same rows from the same seeds and give
the protocol two places to drift. `part` is recorded per run and derived from
the arms -- no adapters means Part A.

ARMS ARE NAMED ADAPTERS, not a base/finetuned pair, because Part B asks several
questions at once and they are only comparable when run together:

  1. does finetuning help at all?      base vs all_fullcds
  2. does gating the corpus on an      all_fullcds vs all_fullcds_atg
     intact start codon matter?
  3. is clade coverage or clade        all_fullcds vs b1_sparse_clade
     breadth doing the work?

Run them as arms of one job and every comparison is paired on prompt, seed and
batch position; run them on different days and none of them are.

`finetuned` is accepted as an alias for --config's stem, so the pre-split
command line still means what it used to.

Inputs
    --arms          'base' and/or adapter tags on the weights volume
    --levels        prompt levels, comma-separated or 'all'
    --replicates    comma-separated or 'all'
    --total-len     uniform total (prefix + generated); 1800 by default
    --expect-sha    <tag>=<sha256>, REQUIRED for every adapter arm
    --config        supplies the model name and the default adapter tag
    --prompts       Part A's prompt corpus
    --out           where the record lands

Outputs, in --out
    generated_<arm>.csv   one row per prompt: the sequence, its CDS truncated at
                          the first in-frame stop, and whether it read through
    run_summary.json      arms, adapter shas and whether each was verified,
                          levels, per-arm timings, model, GPU
    provenance.json       commit, corpus digest, and the sampling regime
    error.txt             on failure, the traceback

Defaults are Part B's powered design: L1 at one replicate, 120 donors per arm.
L1 is where base Evo 2 fails (48.3% full-length), so it is the level with room
to measure an effect, and exact McNemar over the paired design gives power 0.86
against a 60% alternative. Part A's full titration is `--levels all
--replicates all` -- five levels x three replicates.

--expect-sha IS REQUIRED for every adapter arm, mirroring provenance.py's
dirty-tree rule: the unsafe thing stays possible but must be asked for by name
(--allow-unverified-adapter). Without the check, "the finetuned arm" is a claim
about weights nobody verified. Take the value from the finetune run's
run_summary.json -> adapter.sha256; run_on_modal.py prints them in this script's
tag=sha form.

Usage (from the repository root, or via scripts/run_generate_on_modal.py):
    python scripts/generate_ab.py --arms base --levels all --replicates all
    python scripts/generate_ab.py --arms base,all_fullcds \
        --expect-sha all_fullcds=<sha256>
    python scripts/generate_ab.py --levels all \
        --arms base,all_fullcds,all_fullcds_atg,b1_sparse_clade \
        --expect-sha all_fullcds=<sha1>,all_fullcds_atg=<sha2>,b1_sparse_clade=<sha3>
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src import modal_env, provenance

modal_env.apply_hf_env()    # cache on the volume; must precede torch and evo2

import torch, yaml

from src.generate import runner

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/all_fullcds.yaml")
ap.add_argument("--arms", default="base",
                help="comma-separated: 'base', or an adapter tag on the weights "
                     "volume. 'finetuned' aliases --config's stem. base alone is "
                     "Part A; adding adapters makes it Part B.")
ap.add_argument("--levels", default="L1_donor_90",
                help="comma-separated, or 'all'. L1 is where base Evo 2 fails "
                     "(48.3% full-length), so it is the level with room to "
                     "measure an effect.")
ap.add_argument("--replicates", default="0",
                help="comma-separated, or 'all'. One replicate (120 donors) gives "
                     "exact-McNemar power 0.86 to detect 60%% against the 48.3%% "
                     "base rate and ~1.00 for >=70%%, so 'all' buys little.")
ap.add_argument("--total-len", type=int, default=runner.DEFAULT_TOTAL_LEN)
ap.add_argument("--expect-sha", default=None,
                help="'<tag>=<sha256>[,<tag>=<sha256>...]', or a bare sha256 when "
                     "there is exactly one adapter arm. Refuses on a mismatch.")
ap.add_argument("--allow-unverified-adapter", action="store_true",
                help="generate without checking the adapter sha256. Records "
                     "adapter_verified: false -- the result is not citable as a "
                     "comparison against a known set of weights.")
ap.add_argument("--prompts", default="data/prompts_corpus.csv")
ap.add_argument("--out", default="out")
ap.add_argument("--run-tag", default=None,
                help="name for this run's mirror on the weights volume. Sequences "
                     "are copied there as each arm finishes, so a sandbox that "
                     "dies before the harvest does not take them with it.")
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open(args.config))
TOTAL_LEN = args.total_len
DEFAULT_TAG = Path(args.config).stem

# ---- arms, and the sha gate. Both live in the runner so they are unit-tested
# ---- (tests/test_generation_protocol.py); everything here runs BEFORE any
# ---- checkpoint loads, so a mismatch costs a second rather than a GPU-hour.
ARMS, adapters = runner.resolve_arms(
    args.arms, config=args.config, expect_sha=args.expect_sha,
    allow_unverified=args.allow_unverified_adapter)
ADAPTER_ARMS = [(a, t) for a, t in ARMS if t]
# Part A is defined by the ABSENCE of finetuning, so the part is read off the
# arms rather than declared -- a run cannot be mislabelled as the part it isn't.
PART = "B" if ADAPTER_ARMS else "A"

# ---- per-arm LoRA config: an adapter must be rebuilt with the shape it was
# ---- trained at, and b1/b2 are separate configs that may diverge.
lora_cfgs = {}
if ADAPTER_ARMS:
    from src.train.lora_finetune import load_config
    for arm, tag in ADAPTER_ARMS:
        cand = Path("configs") / f"{tag}.yaml"
        src_cfg = str(cand) if cand.exists() else args.config
        lora_cfgs[arm] = load_config(src_cfg).lora
        adapters[arm]["lora_config_from"] = src_cfg

prompts = runner.load_prompts(args.prompts, args.levels, args.replicates, TOTAL_LEN)

prov = provenance.write(OUT, config=cfg, inputs=[args.prompts],
                        extra={"stage": "generate_ab", "part": PART,
                               "entrypoint": "scripts/generate_ab.py",
                               "arms": [a for a, _ in ARMS], "adapters": adapters,
                               "levels": args.levels, "replicates": args.replicates,
                               **runner.sampling_provenance(TOTAL_LEN)},
                        allow_dirty=True)
RUN_TAG = args.run_tag or f"{PART.lower()}_{args.levels.replace(',', '-')}_r{args.replicates}"
log = {"stage": "generate_ab", "part": PART, "run_tag": RUN_TAG, "t_start": time.time(),
       "torch": torch.__version__, "commit": prov["git"]["commit"],
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "arms": [a for a, _ in ARMS], "adapters": adapters,
       # None, not True: a run with no adapter arms has nothing to verify, and
       # "verified" on an empty set would read as a checked comparison.
       "adapter_verified": (all(v["verified"] for v in adapters.values())
                            if adapters else None),
       "total_len": TOTAL_LEN,
       "levels_run": sorted({pr["level_name"] for pr in prompts}),
       "replicates_run": sorted({pr["replicate"] for pr in prompts}),
       "n_prompts": len(prompts)}
runner.save_summary(OUT, log)

checkpoint = runner.resolve_checkpoint(cfg["model"], log, out_dir=OUT)
runner.save_summary(OUT, log)

# Arms run sequentially against one prompt list, so the design stays paired:
# same donors, same per-batch seeds, same batch positions in every arm. The
# summary is rewritten after each arm, so an arm that dies leaves the arms that
# finished intact and says which one failed.
for arm, tag in ARMS:
    try:
        runner.generate_arm(arm=arm, checkpoint=checkpoint, prompts=prompts,
                            out_csv=OUT / f"generated_{arm}.csv",
                            adapter=adapters[arm]["path"] if tag else None,
                            lora_cfg=lora_cfgs.get(arm), log=log)
    except Exception:
        import traceback
        runner.fail(OUT, log, traceback.format_exc(), stage=f"generate_{arm}")
        raise
    runner.save_summary(OUT, log)

log["t_total_s"] = round(time.time() - log["t_start"], 1)
runner.save_summary(OUT, log)
print(json.dumps(log, indent=2)[:2500])
