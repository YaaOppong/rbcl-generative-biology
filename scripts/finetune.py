"""LoRA-finetune Evo 2 7B on an rbcL corpus, save the adapter, and stop.

Reads a config (model, corpus, LoRA rank/alpha, optimizer, seed, split
fractions), runs src.train.lora_finetune as a subprocess, and persists both the
weights and the record of how they were made. It does not generate.

Inputs
    --config    an arm's YAML; `data:` names the corpus, gitignored and built by
                src.data.build_dataset
    --tag       the adapter's name on the weights volume (default: config stem)
    --seed      overrides the config seed; pair it with a distinct --tag
    --force     retrain even when an adapter with this tag already exists
    --out       where the record lands (default: out/)

Outputs
    <volume>/adapters/<tag>_adapter_best.pt   the adapter, on storage that
                                              outlives the sandbox
    out/adapter_best.pt                       a second copy, harvested back to
                                              whoever launched the job
    out/adapter_manifest.json                 every tensor name, shape and dtype
    out/run_summary.json                      model, GPU, seed, wall time, the
                                              per-epoch losses, and
                                              adapter.sha256
    out/config.used.yaml, provenance.json     the config as resolved, and the
                                              commit, packages and input digests
    out/error.txt                             on failure, the traceback

Three copies of the adapter because the first one this project trained was lost
when its sandbox expired. The sha256 in run_summary.json is what generate_ab.py
checks with --expect-sha, which is how a generation run proves it loaded the
weights it claims.

Idempotent: an adapter already present under this tag means the run exits 0
without touching the GPU. Use --force to retrain, or a new --tag for a
replicate -- reusing a tag would overwrite weights that results already cite.

Usage (from the repository root, or via scripts/run_on_modal.py):
    python scripts/finetune.py --config configs/all_fullcds.yaml
    python scripts/finetune.py --config configs/all_fullcds.yaml \
        --tag b2_balanced_seed1 --seed 1        # a training replicate
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src import modal_env

modal_env.apply_hf_env()    # cache on the volume; must precede torch and evo2

import torch, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/all_fullcds.yaml")
ap.add_argument("--tag", default=None,
                help="adapter name on the volume; defaults to the config stem")
ap.add_argument("--seed", type=int, default=None, help="override the config seed")
ap.add_argument("--force", action="store_true",
                help="retrain even if an adapter with this tag already exists")
ap.add_argument("--out", default="out")
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)


def persist_record():
    """Copy this run's record to the weights volume, beside the adapter.

    The adapter has been written to persistent storage since an early one was
    lost to an expired sandbox. The RECORD was not -- and a later run proved what
    that asymmetry costs: three adapters survived a sandbox timeout while their
    loss curves, split manifests and provenance died with the container. A model
    whose training cannot be documented is not a result.

    Called on the failure paths too. A run that failed is a result about the
    code, and its traceback is the part worth keeping.
    """
    dest_root = modal_env.WEIGHTS_MOUNT / "records" / TAG
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.suffix in {".json", ".yaml", ".txt"}:
                dest = dest_root / f.relative_to(OUT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                n += 1
        print(f"record persisted to {dest_root} ({n} files)", flush=True)
    except OSError as e:
        # Never fail a completed run over a copy; the tar harvest is the fallback.
        print(f"WARNING: could not persist record to {dest_root}: {e}", flush=True)
cfg = yaml.safe_load(open(args.config))
if args.seed is not None:
    cfg["seed"] = args.seed
TAG = args.tag or Path(args.config).stem
cfg["out_dir"] = str(OUT / "train")
CFG_USED = str(OUT / "config.used.yaml")
yaml.safe_dump(cfg, open(CFG_USED, "w"))

from src import provenance

prov = provenance.write(OUT, config=cfg, inputs=[cfg["data"]],
                        extra={"stage": "finetune", "tag": TAG,
                               "entrypoint": "scripts/finetune.py",
                               "seed": cfg.get("seed")},
                        allow_dirty=True)
log = {"stage": "finetune", "tag": TAG, "t_start": time.time(),
       "torch": torch.__version__, "seed": cfg.get("seed"),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "commit": prov["git"]["commit"], "config": args.config}

# modal_env owns the filename: finetune.py writes it and generate_ab.py looks it
# up, so a disagreement would surface as "no adapter found" after a paid train.
VOL_ADAPTER = modal_env.adapter_path(TAG)
if VOL_ADAPTER.exists() and not args.force:
    print(f"adapter already on the volume: {VOL_ADAPTER}")
    print("nothing to do (pass --force to retrain). Exiting 0.")
    log["adapter_source"] = "already present; not retrained"
    log["adapter_path"] = str(VOL_ADAPTER)
else:
    from src.evo2_loader import load_evo2

    chosen = None
    for cand in [cfg["model"], "evo2_7b"]:
        try:
            m, info = load_evo2(cand)
            del m; torch.cuda.empty_cache()
            chosen, log["load_info"] = cand, info
            break
        except Exception as e:
            log.setdefault("load_failures", {})[cand] = \
                f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    if chosen is None:
        json.dump(log, open(OUT / "run_summary.json", "w"), indent=2)
        provenance.record_failure(OUT, json.dumps(log.get("load_failures"), indent=2),
                                  stage="checkpoint_load")
        persist_record()
        raise SystemExit("no checkpoint loaded")
    log["model"] = chosen

    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "src.train.lora_finetune",
                        "--config", CFG_USED], capture_output=True, text=True,
                       check=False)
    log["train_s"] = round(time.time() - t0, 1)
    log["train_exit"] = r.returncode
    log["train_tail"] = (r.stdout + r.stderr)[-4000:]
    if r.returncode != 0:
        json.dump(log, open(OUT / "run_summary.json", "w"), indent=2)
        provenance.record_failure(OUT, log["train_tail"], stage="train")
        persist_record()
        raise SystemExit("training failed")

    log["history"] = json.loads((Path(cfg["out_dir"]) / "history.json").read_text())
    trained = Path(cfg["out_dir"]) / "adapter_best.pt"
    VOL_ADAPTER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained, VOL_ADAPTER)          # survives sandbox expiry
    shutil.copy2(trained, OUT / "adapter_best.pt")   # harvested to the caller
    log["adapter_source"] = "trained this run"
    log["adapter_path"] = str(VOL_ADAPTER)

# Fingerprint the weights so generation can prove which adapter it used.
# provenance.file_digest, not a local hashlib call: generation checks the same
# adapter through the same function, so the two cannot compute it differently.
_ad = torch.load(str(VOL_ADAPTER), map_location="cpu")
log["adapter"] = {
    **provenance.file_digest(VOL_ADAPTER),   # sha256, size, mtime
    "n_tensors": len(_ad),
    "n_params": int(sum(v.numel() for v in _ad.values())),
    "dtypes": sorted({str(v.dtype) for v in _ad.values()}),
    "tag": TAG,
}
json.dump({k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in _ad.items()},
          open(OUT / "adapter_manifest.json", "w"), indent=2)
del _ad

log["t_total_s"] = round(time.time() - log["t_start"], 1)
json.dump(log, open(OUT / "run_summary.json", "w"), indent=2)

persist_record()

print(json.dumps({k: v for k, v in log.items()
                  if k not in ("train_tail", "history")}, indent=2)[:2000])
