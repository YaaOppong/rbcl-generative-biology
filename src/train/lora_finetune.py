"""LoRA finetuning of Evo 2 on rbcL coding sequences (causal-LM objective).

Reference LoRA implementation, CPU-testable without weights. NOT the production
training path -- released adapters are trained with NVIDIA BioNeMo's Evo2LoRA
(`recipes/evo2_megatron`); see docs/DESIGN.md. Kept because it is what the test
suite exercises without a GPU, and it caught a real defect (every target name in
the original configs was absent from the real model).

Adapter configuration follows the NVIDIA BioNeMo recipe for Evo 2
(rank 16, alpha 32, adapters on attention/MLP projections and the Hyena mixer,
~1.4% of parameters trainable).

STATUS: the recipe this is derived from demonstrates a *discriminative* task
(a classification head on pooled hidden states). This module targets a
*generative* objective instead. The adapter machinery is expected to transfer;
that expectation is unverified. See docs/DESIGN.md "Known risks".

Usage:
    python -m src.train.lora_finetune --config configs/demo_small.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from src.train.lora import (
    apply_lora,
    causal_lm_loss,
    evaluate,
    model_device,
    save_adapters,
)


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["attn_qkv", "attn_out", "mlp_in", "mlp_out", "hyena_mixer"]
    )


@dataclass
class TrainConfig:
    model: str = "evo2_1b_base"
    data: str = "data/b1.jsonl"
    max_length: int = 1600
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 1e-4
    epochs: int = 3
    warmup_ratio: float = 0.03
    seed: int = 0
    val_fraction: float = 0.2
    novel_share: float = 0.5
    early_stopping_patience: int = 2
    out_dir: str = "results/run"
    lora: LoraConfig = field(default_factory=LoraConfig)


def load_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    lora = LoraConfig(**raw.pop("lora", {}))
    return TrainConfig(lora=lora, **raw)


def species_key(record: dict) -> str:
    """Group key for leakage control: taxid when present, else binomial.

    taxid is authoritative; the organism string is a fallback for records built
    before taxid was carried through, and takes the first two whitespace tokens
    (genus + species), which is what a GenBank organism field leads with.
    """
    taxid = record.get("taxid")
    if taxid not in (None, "", 0):
        return f"taxid:{taxid}"
    return "name:" + " ".join(str(record.get("organism", "NA")).split()[:2]).lower()


def three_way_split(
    records: list[dict],
    val_fraction: float = 0.2,
    novel_share: float = 0.5,
    seed: int = 0,
):
    """Split into train / val_seen / val_novel, clade-stratified.

    Two validation sets, because "held out" is ambiguous for a barcode locus and
    the two readings measure different things:

    * ``val_novel`` -- whole species held out. No species appears in training,
      so loss here measures generalisation to a species never seen. This is the
      early-stopping signal.
    * ``val_seen`` -- single records pulled from species that REMAIN in
      training. Leaky by construction, and kept deliberately: it measures fit on
      familiar species, and the gap against ``val_novel`` quantifies how much of
      the model's competence is species-specific memorisation.

    Why both, measured on B1 after exact-duplicate removal: conspecific records
    are median 99.26% identical (56.6% of pairs >=99%) against 88.68% for
    between-species same-clade pairs. So a record-level split does leak. But
    1,516 of 2,079 species are singletons -- 42% of records -- and a pure
    species-disjoint split additionally removes ~20% of species from the
    gradient entirely, which is real taxonomic diversity lost for a hypothesis
    that is *about* taxonomic structure. Reporting both costs nothing and
    resolves the tension rather than picking a side.

    Every species that can be in training is in training: all singletons, and
    every multi-record species except those assigned to ``val_novel``.

    ``val_fraction`` is the TOTAL held back; ``novel_share`` is the portion of
    that going to ``val_novel`` (0.5 => 10% + 10% at val_fraction=0.2).

    ``val_seen`` is capped by donor availability: it can only take one record
    per multi-record species that stayed in training, so a clade whose records
    are mostly singletons (or concentrated in one species) yields fewer than
    ``target_seen``. The realised total is therefore <= ``val_fraction``, never
    more. On B1 the shortfall is negligible (20.1% realised against 20%); on a
    singleton-dominated clade it can be large, and that is preferable to the
    alternatives -- inventing leakage by splitting a species' only record, or
    silently converting the shortfall into extra ``val_novel`` and diluting the
    contrast the two sets exist to measure.
    """
    import random

    rng = random.Random(seed)
    by_clade: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by_clade.setdefault(r.get("clade", "NA"), {}).setdefault(
            species_key(r), []
        ).append(r)

    train, val_seen, val_novel = [], [], []
    for _clade, species in sorted(by_clade.items()):
        keys = sorted(species)
        rng.shuffle(keys)
        n_clade = sum(len(species[k]) for k in keys)
        target_novel = round(n_clade * val_fraction * novel_share)
        target_seen = round(n_clade * val_fraction * (1 - novel_share))

        # 1. Whole species -> val_novel, until the budget is met. Species are
        #    visited in shuffled order and taken only if they fit the remaining
        #    budget, so a dominant species cannot invert the split. Note the
        #    fit rule means singleton species are taken disproportionately
        #    often (they always fit): on B1, 170 of the 229 held-out species
        #    are singletons. That is sound -- holding out a whole species is
        #    what makes it novel, regardless of its record count -- but it does
        #    mean val_novel skews toward less-sequenced species, which are
        #    plausibly the harder cases. Read it as a conservative estimate.
        n_novel, held = 0, set()
        for k in keys:
            rows = species[k]
            over = (n_novel + len(rows)) - target_novel
            fits = n_novel < target_novel and over <= target_novel - n_novel
            # Never hold out a clade's only species: it would leave the clade
            # with no training signal while still driving the loss.
            if fits and len(keys) > 1:
                val_novel.extend(rows)
                n_novel += len(rows)
                held.add(k)

        # 2. One record from multi-record species that stayed in training.
        remaining = [k for k in keys if k not in held]
        donors = [k for k in remaining if len(species[k]) > 1]
        rng.shuffle(donors)
        n_seen = 0
        for k in donors:
            if n_seen >= target_seen:
                break
            rows = species[k]
            val_seen.append(rows[-1])       # >=1 record stays in training
            train.extend(rows[:-1])
            # Not an enumerate index (SIM113): n_seen is a budget counter read
            # by the break above, and the loop skips donors, so it does not
            # track iteration count.
            n_seen += 1  # noqa: SIM113
            remaining.remove(k)

        # 3. Everything else trains.
        for k in remaining:
            train.extend(species[k])

    rng.shuffle(train)
    return train, val_seen, val_novel


def clade_stratified_split(records: list[dict], val_fraction: float, seed: int):
    """Clade-stratified, SPECIES-DISJOINT train/validation split.

    Stratified by clade so early stopping is not driven by whichever clade
    dominates the arm, and split on whole species so no species appears on both
    sides.

    Why species-disjoint is load-bearing here: rbcL is a barcode locus, so the
    corpus carries many records per species (B1: mean 2.4, max 87) and 25% of
    records were byte-identical duplicates before the build-time dedup gate.
    Under a record-level split, 71% of B1 validation records had their species
    in the training set. Validation loss then measures partial memorisation of
    near-identical congeneric sequence, and "val_loss improved" supports no
    claim about generalisation -- which is the only claim early stopping and
    the epoch -1 baseline are there to make.

    Species are assigned whole, largest-first, to whichever side is furthest
    from its target -- greedy rather than random, because a species holding 87
    of a clade's records cannot be placed by coin flip without wrecking the
    intended validation fraction.
    """
    import random

    rng = random.Random(seed)
    by_clade: dict[str, dict[str, list[dict]]] = {}
    for r in records:
        by_clade.setdefault(r.get("clade", "NA"), {}).setdefault(
            species_key(r), []
        ).append(r)

    train, val = [], []
    for _clade, species in sorted(by_clade.items()):
        keys = sorted(species)
        rng.shuffle(keys)  # seed decides WHICH species are held out
        n_clade = sum(len(species[k]) for k in keys)
        target_val = round(n_clade * val_fraction)
        n_val = 0
        for k in keys:
            rows = species[k]
            # A species is taken for validation only if it fits: adding it must
            # not overshoot the target by more than it undershoots. Without this,
            # one dominant species (B1's Pinnularia borealis holds 87 records)
            # lands in validation first and inverts the split -- an 80/100-record
            # species produced 80% validation in testing.
            over = (n_val + len(rows)) - target_val
            fits = n_val < target_val and over <= target_val - n_val
            # Never send a clade's only species to validation: it would leave the
            # clade with no training signal while still driving the loss we
            # early-stop on.
            if fits and len(keys) > 1:
                val.extend(rows)
                n_val += len(rows)
            else:
                train.extend(rows)
    rng.shuffle(train)
    return train, val


def main(config_path: str) -> None:
    import random

    import torch
    from torch.utils.data import DataLoader, Dataset

    cfg = load_config(config_path)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Seed CPU *and* CUDA generators. Since adapters are constructed on the base
    # layer's device (see LoRALinear), kaiming_uniform_ draws from the CUDA RNG
    # on a GPU run -- which torch.manual_seed alone does not cover, so adapter
    # initialisation was not reproducible across runs at the same seed.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    # A seeded generator for the training DataLoader: shuffle=True otherwise
    # draws from global state that anything else touching the RNG can perturb.
    _dl_gen = torch.Generator()
    _dl_gen.manual_seed(cfg.seed)
    with open(cfg.data) as fh:
        records = [json.loads(line) for line in fh]
    train_rows, seen_rows, novel_rows = three_way_split(
        records, cfg.val_fraction, cfg.novel_share, cfg.seed
    )
    print(
        f"train={len(train_rows)} val_seen={len(seen_rows)} "
        f"val_novel={len(novel_rows)} clades={len({r['clade'] for r in records})}"
    )

    # Persist the split, not just print it. Training and generation are separate
    # jobs now, so the training record has to stand alone: without this, "which
    # species were held out" is only recoverable by re-deriving the split, which
    # silently changes if the split code or the corpus ever does.
    split_manifest = {
        "seed": cfg.seed,
        "val_fraction": cfg.val_fraction,
        "novel_share": cfg.novel_share,
        "counts": {"train": len(train_rows), "val_seen": len(seen_rows),
                   "val_novel": len(novel_rows), "total": len(records)},
        "species": {
            "train": sorted({species_key(r) for r in train_rows}),
            "val_seen": sorted({species_key(r) for r in seen_rows}),
            "val_novel": sorted({species_key(r) for r in novel_rows}),
        },
        "leaked_species_into_val_novel": len(
            {species_key(r) for r in novel_rows} & {species_key(r) for r in train_rows}
        ),
        "per_clade": {
            c: {
                "train": sum(1 for r in train_rows if r["clade"] == c),
                "val_seen": sum(1 for r in seen_rows if r["clade"] == c),
                "val_novel": sum(1 for r in novel_rows if r["clade"] == c),
            }
            for c in sorted({r["clade"] for r in records})
        },
    }
    # Recorded, NOT asserted. The split is species-disjoint per clade, so a
    # species assigned to two clades in the source metadata can legitimately be
    # held out in one and trained on in the other. That is a data-quality
    # question about the corpus, not a bug here -- and it must not kill a
    # half-hour GPU run. It goes in the record where analysis can see it.
    if split_manifest["leaked_species_into_val_novel"]:
        print(f"WARNING: {split_manifest['leaked_species_into_val_novel']} species "
              "appear in both train and val_novel — check for cross-clade taxid "
              "assignments in the corpus; see split_manifest.json")
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    from src.evo2_loader import load_evo2

    # Routed through load_evo2, not Evo2() directly: 1B needs the fp8 config
    # patch (see src/evo2_loader.py). info records which numerical path the
    # weights took, so it belongs in the run's provenance.
    model, load_info = load_evo2(cfg.model)
    base = model.model

    # vortex loads weights under torch.inference_mode(), which marks every
    # parameter an inference tensor -- unusable in autograd. Must run before
    # apply_lora so adapters wrap normal layers.
    from src.evo2_loader import make_trainable

    load_info["inference_tensors_converted"] = make_trainable(base)
    (out / "load_info.json").write_text(json.dumps(load_info, indent=2))
    tokenizer = model.tokenizer

    apply_lora(base, cfg.lora)
    dev = model_device(base)
    trainable = [p for p in base.parameters() if p.requires_grad]
    total = sum(p.numel() for p in base.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable {n_train:,} / {total:,} params ({100 * n_train / total:.2f}%)")

    # Record what was actually adapted. "LoRA r=16 on these target modules" is a
    # config claim; this is what the config produced against this checkpoint.
    (out / "adapter_config.json").write_text(json.dumps({
        "trainable_params": n_train,
        "total_params": total,
        "trainable_frac": round(n_train / total, 6),
        "n_injection_sites": sum(1 for n, _ in base.named_modules()
                                 if n.endswith((".a", ".b"))) // 2,
        "lora": asdict(cfg.lora) if hasattr(cfg.lora, "__dataclass_fields__")
        else dict(cfg.lora),
        # Field is max_length, not max_len -- a getattr default would have
        # silently recorded None and the record would understate the run.
        "optimizer": {"lr": cfg.lr, "batch_size": cfg.batch_size,
                      "grad_accum": cfg.grad_accum, "epochs": cfg.epochs,
                      "max_length": cfg.max_length,
                      "warmup_ratio": cfg.warmup_ratio,
                      "early_stopping_patience": cfg.early_stopping_patience},
    }, indent=2))

    class SeqDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            ids = tokenizer.tokenize(self.rows[i]["sequence"])[: cfg.max_length]
            return torch.tensor(ids, dtype=torch.long)

    def collate(batch):
        n = max(len(b) for b in batch)
        x = torch.zeros(len(batch), n, dtype=torch.long)
        mask = torch.zeros(len(batch), n, dtype=torch.bool)
        for i, b in enumerate(batch):
            x[i, : len(b)] = b
            mask[i, : len(b)] = True
        return x, mask

    dl = DataLoader(SeqDataset(train_rows), batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate, generator=_dl_gen)
    vl_seen = DataLoader(SeqDataset(seen_rows), batch_size=cfg.batch_size, collate_fn=collate)
    vl_novel = DataLoader(SeqDataset(novel_rows), batch_size=cfg.batch_size, collate_fn=collate)

    opt = torch.optim.AdamW(trainable, lr=cfg.lr)
    steps = (len(dl) // cfg.grad_accum) * cfg.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(1, steps), pct_start=cfg.warmup_ratio
    )

    # Validation loss BEFORE any weight update. Without this there is no
    # baseline to claim improvement against: "val_loss 0.71 at epoch 2" is
    # uninterpretable unless the epoch -1 value is on the record. Adapters are
    # zero-initialised on the B branch, so this is the base model's loss on
    # exactly the same held-out records and tokenisation.
    base_seen, base_novel = evaluate(base, vl_seen), evaluate(base, vl_novel)
    print(
        f"epoch -1 val_seen {base_seen:.4f} val_novel {base_novel:.4f} "
        "(base model, no adapters trained)"
    )

    # Early stopping follows val_novel: it is the only one of the two that
    # measures generalisation to an unseen species. Stopping on val_seen would
    # reward memorising the species already in training.
    history = [{"epoch": -1, "val_seen": base_seen, "val_novel": base_novel}]
    best, bad = float("inf"), 0
    for epoch in range(cfg.epochs):
        base.train()
        for i, (x, mask) in enumerate(dl):
            loss = causal_lm_loss(base, x.to(dev), mask.to(dev))
            (loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        v_seen, v_novel = evaluate(base, vl_seen), evaluate(base, vl_novel)
        history.append({"epoch": epoch, "val_seen": v_seen, "val_novel": v_novel,
                        "memorisation_gap": v_seen - v_novel})
        print(f"epoch {epoch} val_seen {v_seen:.4f} val_novel {v_novel:.4f} "
              f"gap {v_seen - v_novel:+.4f}")
        vloss = v_novel
        if vloss < best:
            best, bad = vloss, 0
            save_adapters(base, out / "adapter_best.pt")
        else:
            bad += 1
            if bad >= cfg.early_stopping_patience:
                print(f"early stop at epoch {epoch}")
                break
    (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
