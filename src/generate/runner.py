"""The decoding procedure, written once and shared by every part.

Parts A, B and C differ in their ARMS, not in how a sequence is produced. Part A
varies the prompt against the base model; Part B holds Part A's prompt protocol
and varies the weights. The batching, the
per-batch seeding, the stop-codon truncation and the CSV schema are common to all
three, and a difference in any of them would be indistinguishable from a real
effect -- a changed batch size perturbs logits (padding, reduction order, FFT
prefill), and sampling amplifies that. So the procedure lives here as one
function and each entry point declares only what it varies.

The invariants below are Part A's, and they are constants rather than flags on
purpose: they are the reason arms are comparable at all.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from src import modal_env, provenance

# Where generated sequences are mirrored so they outlive the container. The
# adapter and the training record are both written here for the same reason; a
# generation run that lost 76 minutes of H100 to a sandbox dying before its
# harvest is why sequences are too.
GENERATION_DIR = modal_env.WEIGHTS_MOUNT / "generations"
from src.eval.metrics import STOPS

BATCH_SIZE = 4      # Part A invariant: uniform across levels, replicates, arms
TEMPERATURE = 0.7   # Part A: explicit
TOP_K = 4           # Arc's evo2 default, which Part A inherited by not passing it
TOP_P = 1.0         # Arc's default

# Part A used a 1500 nt total. That is only 33 nt (2%) above the LONGEST natural
# rbcL CDS (1467 nt), so a generation running slightly long is CENSORED rather
# than failed: 39/1800 of Part A's sequences hit the budget with no stop codon
# and were scored full_length=False, indistinguishable from frame breakage. At L1
# that was 25 sequences = 13.4% of all L1 failures. 1800 is 23% above the natural
# maximum, so a read-through now means the model genuinely failed to terminate.
DEFAULT_TOTAL_LEN = 1800

# 'prompt' is deliberately absent: the prompt is recoverable from the donor
# accession and prefix_nt, and seq_nt already contains it as its own prefix.
FIELDS = ["prompt_id", "donor_acc", "organism", "tax_group", "cluster", "split",
          "level_name", "prefix_nt", "n_tokens", "replicate", "seed",
          "seq_nt", "seq_len", "cds_nt", "cds_len", "censored"]

_COMPUTED = {"seq_nt", "seq_len", "cds_nt", "cds_len", "censored"}


def resolve_arms(arms: str, *, config: str, expect_sha: str | None = None,
                 allow_unverified: bool = False,
                 adapter_dir=None) -> tuple[list[tuple[str, str | None]], dict]:
    """Parse Part B's arms and check every adapter BEFORE anything expensive.

    Returns `([(arm, tag or None)], {arm: {tag, path, sha256, verified}})`.

    Arms are named adapters rather than a base/finetuned pair, because Part B's
    second question -- whether the already-represented clades must be in the
    corpus to keep the model flat -- needs b1 and b2 as arms of ONE paired run.
    `finetuned` stays valid as an alias for the config's own stem, so the
    pre-split command line still means what it used to.

    Every gate here raises SystemExit rather than warning. A mismatch caught now
    costs a second; caught after the checkpoint loads it costs a GPU-hour, and
    not caught at all it costs the claim -- "the finetuned arm" would be a
    statement about weights nobody verified.
    """
    default_tag = Path(config).stem
    parsed: list[tuple[str, str | None]] = []
    for raw in [x.strip() for x in arms.split(",") if x.strip()]:
        if raw == "base":
            parsed.append((raw, None))
        elif raw == "finetuned":
            parsed.append((raw, default_tag))
        else:
            parsed.append((raw, raw))
    if not parsed:
        raise SystemExit("no arms given")
    names = [a for a, _ in parsed]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arms: {names}. Each arm writes "
                         "generated_<arm>.csv, so a repeat would overwrite itself.")

    adapter_arms = [(a, t) for a, t in parsed if t]
    expected: dict[str, str] = {}
    for item in [x.strip() for x in (expect_sha or "").split(",") if x.strip()]:
        if "=" in item:
            tag, sha = item.split("=", 1)
            expected[tag.strip()] = sha.strip().lower()
        elif len(adapter_arms) == 1:
            expected[adapter_arms[0][1]] = item.lower()
        else:
            raise SystemExit(
                "a bare --expect-sha is only unambiguous with one adapter arm; "
                f"got {len(adapter_arms)}. Use <tag>=<sha256> pairs.")
    unknown = set(expected) - {t for _, t in adapter_arms}
    if unknown:
        raise SystemExit(f"--expect-sha names tags that are not arms: {sorted(unknown)}")

    adapters: dict[str, dict] = {}
    for arm, tag in adapter_arms:
        path = modal_env.adapter_path(tag, adapter_dir)
        if not path.exists():
            raise SystemExit(
                f"no adapter at {path}\n"
                f"train one first:  python scripts/finetune.py --config configs/{tag}.yaml\n"
                f"or on Modal:      python scripts/run_on_modal.py --configs configs/{tag}.yaml")
        # provenance.file_digest, the same function finetune.py recorded the
        # fingerprint with, so the two sides cannot hash differently.
        sha = provenance.file_digest(path)["sha256"]
        if tag in expected:
            if sha != expected[tag]:
                raise SystemExit(
                    f"adapter sha256 mismatch for '{tag}' -- refusing to generate.\n"
                    f"  on volume: {sha}\n  expected:  {expected[tag]}\n"
                    "The weights are not the ones this experiment was designed around.")
        elif not allow_unverified:
            raise SystemExit(
                f"arm '{arm}' has no --expect-sha. Adapter on the volume is\n"
                f"  {tag}={sha}\n"
                "Pass that (from the finetune run's run_summary.json -> "
                "adapter.sha256) or --allow-unverified-adapter to proceed unverified.")
        adapters[arm] = {"tag": tag, "path": str(path), "sha256": sha,
                         "verified": tag in expected}
    return parsed, adapters


def _is_lora(name: str) -> bool:
    """LoRA factors, as src.train.lora names them: <site>.a.weight / .b.weight."""
    return name.endswith((".a.weight", ".b.weight"))


def cds_of(seq: str) -> str | None:
    """Truncate at the first in-frame stop, inclusive. None => read-through.

    Matches src.analysis_l1.cds_of and Part A's NaN cds_len for read-through.
    """
    s = str(seq).upper()
    for i in range(0, len(s) - 2, 3):
        if s[i:i + 3] in STOPS:
            return s[:i + 3]
    return None


def load_prompts(path: str | Path, levels: str = "all", replicates: str = "all",
                 total_len: int = DEFAULT_TOTAL_LEN) -> list[dict]:
    """Read Part A's prompt corpus, filter, and re-derive the token budget.

    n_tokens is RECOMPUTED rather than trusted: the shipped file was built for a
    1500 nt total, so its stored value would silently generate short.
    """
    prompts = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            row["prefix_nt"] = int(row["prefix_nt"])
            row["replicate"] = int(row["replicate"])
            row["seed"] = int(row["seed"])
            prompts.append(row)
    if levels != "all":
        keep = {x.strip() for x in levels.split(",") if x.strip()}
        prompts = [pr for pr in prompts if pr["level_name"] in keep]
    if replicates != "all":
        reps = {int(x) for x in replicates.split(",") if x.strip()}
        prompts = [pr for pr in prompts if pr["replicate"] in reps]
    for pr in prompts:
        pr["n_tokens"] = total_len - pr["prefix_nt"]
    if not prompts:
        raise SystemExit(
            f"no prompts matched levels={levels} replicates={replicates} in {path}")
    short = [pr["prompt_id"] for pr in prompts if pr["n_tokens"] <= 0]
    if short:
        raise SystemExit(
            f"total_len {total_len} is not longer than the prefix for "
            f"{len(short)} prompts (e.g. {short[0]}); nothing would be generated")
    return prompts


def batches(prompts: list[dict]) -> list[list[dict]]:
    """Split into fixed-size batches that are HOMOGENEOUS in token budget.

    Two properties the pairing depends on, which is why this is a function with
    tests rather than a loop inside the decoder:

    * one token budget per batch -- mixing budgets changes the padding, and
      batched autoregressive decoding is not bit-identical across padding
    * a deterministic order from the prompt list alone, so a donor lands in the
      same batch, at the same position, in every arm
    """
    groups: dict[int, list[dict]] = {}
    for pr in prompts:
        groups.setdefault(pr["n_tokens"], []).append(pr)
    out = []
    for n_tokens in sorted(groups, reverse=True):
        grp = groups[n_tokens]
        out += [grp[i:i + BATCH_SIZE] for i in range(0, len(grp), BATCH_SIZE)]
    return out


def sampling_provenance(total_len: int) -> dict:
    """The sampling regime, recorded explicitly rather than left to the library."""
    return {
        "total_len": total_len, "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P,
        "censoring_note": (
            f"total_len {total_len} vs longest natural rbcL CDS 1467 nt; "
            "Part A's 1500 censored 2.2% of its generations"),
        "top_k_note": ("inherited from ArcInstitute/evo2's Evo2.generate default, "
                       "not tuned here; held constant across every arm"),
    }


def fail(out_dir, log: dict, exc_text: str, stage: str) -> None:
    """Persist the record and the traceback before dying.

    A run that failed is a result about the code -- three failed runs in this
    project produced four real fixes, and those tracebacks were nearly lost
    (docs/RUN_TRACKING.md). Raising without writing loses the evidence, and the
    evidence is the only thing a failed GPU hour buys.
    """
    save_summary(out_dir, log)
    provenance.record_failure(out_dir, exc_text, stage=stage)


def resolve_checkpoint(preferred: str, log: dict, fallback: str = "evo2_7b",
                       out_dir=None) -> str:
    """Load and immediately release each candidate, cheapest failure first.

    A checkpoint that cannot load must cost seconds, not a whole generation run.
    7b is the documented fallback if the 1b fp8 patch misbehaves (docs/DESIGN.md).
    """
    import torch

    from src.evo2_loader import load_evo2

    for cand in dict.fromkeys([preferred, fallback]):
        try:
            m, info = load_evo2(cand)
            del m
            torch.cuda.empty_cache()
            log["load_info"] = info
            log["model"] = cand
            log["fell_back"] = cand != preferred
            return cand
        except Exception as e:  # noqa: BLE001 - a checkpoint can fail to load
            # in many version-dependent ways; each candidate must be tried and
            # the failure recorded, not raised.
            log.setdefault("load_failures", {})[cand] = \
                f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    if out_dir is not None:
        fail(out_dir, log, json.dumps(log.get("load_failures"), indent=2),
             stage="checkpoint_load")
    raise SystemExit("no checkpoint loaded")


def generate_arm(*, arm: str, checkpoint: str, prompts: list[dict], out_csv: Path,
                 adapter: str | Path | None = None, lora_cfg=None,
                 log: dict | None = None) -> Path:
    """Generate one arm. `adapter=None` is the base arm -- no LoRA is applied.

    Rows are written and flushed as they are produced: a run that dies at
    sequence 200 of 240 should leave 200 usable sequences, not an empty file.
    """
    import torch

    from src.evo2_loader import load_evo2

    log = log if log is not None else {}
    mdl, _ = load_evo2(checkpoint)
    if adapter is not None:
        if lora_cfg is None:
            raise ValueError("an adapter arm needs the LoRA config it was trained with")
        from src.evo2_loader import make_trainable
        from src.train.lora import apply_lora
        make_trainable(mdl.model)
        apply_lora(mdl.model, lora_cfg)
        sd = torch.load(str(adapter), map_location="cpu")
        res = mdl.model.load_state_dict(sd, strict=False)
        # BOTH directions, because strict=False is silent in both:
        #   unexpected -- the adapter carries tensors this model has no slot for,
        #     so it does not fit and nothing can be concluded from the arm;
        #   missing    -- the model has LoRA sites the adapter does not cover.
        #     Those keep their initialisation, and LoRA B is zero-init, so each
        #     uncovered site is the IDENTITY. The arm would run partly (or
        #     entirely) as base while being labelled finetuned, and would still
        #     produce a clean csv and a plausible pass rate.
        if res.unexpected_keys:
            raise SystemExit(
                f"adapter does not fit the model: {len(res.unexpected_keys)} "
                f"unexpected tensors, e.g. {res.unexpected_keys[:3]}")
        sites = [n for n, _ in mdl.model.named_parameters() if _is_lora(n)]
        uncovered = sorted(set(sites) - set(sd))
        if uncovered:
            raise SystemExit(
                f"adapter covers {len(sd)} of {len(sites)} LoRA sites; "
                f"{len(uncovered)} would run as base, e.g. {uncovered[:3]}. "
                "Refusing to label this arm finetuned.")
        log.setdefault("adapter_load", {})[arm] = {
            "tensors": len(sd), "lora_sites_in_model": len(sites),
            "fully_covered": True, "path": str(adapter)}
    mdl.model.eval()

    t0 = time.time()
    n_done = 0
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for batch in batches(prompts):
            # Part A: the seed comes from the batch's FIRST prompt, so the same
            # donor lands in the same batch position in every arm.
            torch.manual_seed(batch[0]["seed"])
            torch.cuda.manual_seed_all(batch[0]["seed"])
            with torch.no_grad():
                out = mdl.generate(prompt_seqs=[b["prompt"] for b in batch],
                                   n_tokens=batch[0]["n_tokens"],
                                   temperature=TEMPERATURE, top_k=TOP_K,
                                   top_p=TOP_P, cached_generation=True)
            for row, seq in zip(batch, out.sequences):
                full = row["prompt"] + seq
                cds = cds_of(full)
                w.writerow({k: row.get(k, "") for k in FIELDS if k not in _COMPUTED}
                           | {"seq_nt": full, "seq_len": len(full),
                              "cds_nt": cds or "",
                              "cds_len": len(cds) if cds else "",
                              "censored": cds is None})
            n_done += len(batch)
            if n_done % (BATCH_SIZE * 20) == 0:
                print(f"{arm}: {n_done}/{len(prompts)}", flush=True)
                fh.flush()
    log[f"generate_{arm}_s"] = round(time.time() - t0, 1)
    log[f"{arm}_rows"] = n_done
    mirror_to_volume(out_csv.parent, log.get("run_tag"))
    del mdl
    torch.cuda.empty_cache()
    return out_csv


def save_summary(out_dir: str | Path, log: dict) -> None:
    """Rewritten after every arm, so a crash still leaves the arms that finished."""
    with open(Path(out_dir) / "run_summary.json", "w") as fh:
        json.dump(log, fh, indent=2)
    mirror_to_volume(out_dir, log.get("run_tag"))


def mirror_to_volume(out_dir: str | Path, run_tag: str | None) -> None:
    """Copy this run's outputs to the weights volume.

    The sandbox tar harvest needs a LIVE sandbox, and three sandboxes in this
    project have exited before it ran. Training survived because finetune.py
    mirrors its record to the volume as each adapter completes; generation did
    not, and one level's sequences were lost outright. The volume outlives the
    container, so this is the durable copy and the tar is the convenience.

    Best effort by design: a copy failure must never end a run that is producing
    the sequences it exists to produce.
    """
    if not run_tag:
        return
    dest = GENERATION_DIR / run_tag
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(out_dir).rglob("*")):
            if f.is_file() and f.suffix in {".csv", ".json"}:
                target = dest / f.relative_to(out_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    except OSError as e:
        print(f"WARNING: could not mirror to {dest}: {e}", flush=True)
