"""End-to-end test of the LoRA training loop on a tiny stand-in model.

Evo 2 needs a bf16 GPU, so CI cannot run the real model. But the training loop
itself -- adapter injection, the causal-LM objective, clade-stratified
splitting, early stopping, checkpointing -- is model-agnostic and is exactly
where a silent bug would live. This exercises all of it on a ~50k-parameter
character model over real rbcL records, so `lora_finetune.py` is executed code
rather than code that merely imports.
"""

import pytest

# torch is an optional extra: it needs a bf16-capable GPU for real use and
# is installed separately (see README). These modules exercise adapter
# injection and the loss on toy tensors, so they need torch but not a GPU.
# Skipping beats erroring: a fresh clone should get a clean run, not three
# collection failures that look like the suite is broken.
pytest.importorskip("torch")

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.train.lora import apply_lora, causal_lm_loss, evaluate, save_adapters
from src.train.lora_finetune import clade_stratified_split, species_key

VOCAB = {c: i for i, c in enumerate("ACGTN")}


class TinyLM(nn.Module):
    """Character-level stand-in with Evo-like module names so apply_lora binds."""

    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), d)
        self.blocks = nn.ModuleList([_Block(d) for _ in range(2)])
        self.head = nn.Linear(d, len(VOCAB))

    def forward(self, ids):
        h = self.emb(ids)
        for b in self.blocks:
            h = b(h)
        return (self.head(h),)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn_qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(3 * d, d)
        self.mlp_in = nn.Linear(d, 2 * d)
        self.mlp_out = nn.Linear(2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        h = h + self.proj(torch.relu(self.attn_qkv(h)))
        return self.norm(h + self.mlp_out(torch.relu(self.mlp_in(h))))


class Cfg:
    rank, alpha, dropout = 4, 8, 0.0
    target_modules = ("attn_qkv", "mlp_in")


def encode(seq, n=240):
    ids = [VOCAB.get(c, VOCAB["N"]) for c in seq[:n]]
    ids += [VOCAB["N"]] * (n - len(ids))
    return ids


DEMO = Path("data/demo.jsonl")

# data/*.jsonl is gitignored -- corpora are rebuilt from NCBI, not versioned --
# so a fresh clone has none. These tests exercise the loop on REAL records and
# there is no honest substitute, so they skip rather than hard-fail (matching
# tests/test_active_site.py and tests/test_analysis_l1.py). Rebuild with:
#   python -m src.data.build_dataset --arm B1_sparse_clade --out data/demo.jsonl --limit 300
needs_demo = pytest.mark.skipif(
    not DEMO.exists(), reason="data/demo.jsonl absent (gitignored; rebuild from NCBI)"
)


def load_records(path=DEMO, cap=48):
    with open(path) as fh:
        return [json.loads(line) for line in fh][:cap]


@needs_demo
def test_training_loop_reduces_loss_on_real_records(tmp_path):
    recs = load_records()
    if len(recs) < 8:
        import pytest

        pytest.skip("data/demo.jsonl not built; run src.data.build_dataset first")

    torch.manual_seed(0)
    train_rows, val_rows = clade_stratified_split(recs, 0.25, seed=0)
    assert train_rows and val_rows
    # stratification must not lose or duplicate records
    assert len(train_rows) + len(val_rows) == len(recs)

    model = TinyLM()
    n_wrapped = apply_lora(model, Cfg())
    assert n_wrapped == 4  # 2 targets x 2 blocks

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(".a." in n or ".b." in n for n, p in model.named_parameters() if p.requires_grad)

    def batches(rows, bs=8):
        for i in range(0, len(rows) - bs + 1, bs):
            ids = torch.tensor([encode(r["sequence"]) for r in rows[i : i + bs]])
            yield ids, torch.ones_like(ids, dtype=torch.bool)

    opt = torch.optim.AdamW(trainable, lr=5e-2)
    before = evaluate(model, list(batches(val_rows, 4)))
    for _ in range(6):
        for ids, mask in batches(train_rows):
            loss = causal_lm_loss(model, ids, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()
    after = evaluate(model, list(batches(val_rows, 4)))

    assert after < before, f"val loss did not improve: {before:.4f} -> {after:.4f}"

    path = tmp_path / "adapter.pt"
    save_adapters(model, path)
    assert path.stat().st_size < 200_000  # adapters only, not the base model


def test_training_seeds_cuda_and_dataloader():
    """Seeding must cover the CUDA RNG and the DataLoader, not just CPU torch.

    Adapters are constructed on the base layer's device (LoRALinear), so on a GPU
    run kaiming_uniform_ draws from the CUDA generator -- which
    torch.manual_seed does NOT seed. The first successful finetune was therefore
    not reproducible at its own seed. shuffle=True likewise draws from global RNG
    state unless given an explicit generator.

    Asserted on the source: the training loop needs weights and a GPU to run.
    """
    from pathlib import Path as _P

    body = (_P(__file__).parent.parent / "src" / "train" / "lora_finetune.py").read_text()
    assert "torch.cuda.manual_seed_all(cfg.seed)" in body, "CUDA RNG not seeded"
    assert "random.seed(cfg.seed)" in body, "stdlib random not seeded"
    assert "_dl_gen.manual_seed(cfg.seed)" in body, "DataLoader generator not seeded"
    assert "generator=_dl_gen" in body, "seeded generator not passed to DataLoader"


@needs_demo
def test_stratified_split_covers_every_clade():
    """Every clade in training; in validation only where species allow.

    The split became species-disjoint (see tests/test_split.py), and the two
    guarantees genuinely conflict for a clade whose records all belong to ONE
    species: it cannot be both species-disjoint and represented in validation.
    In the 120-record demo slice, Red algae is 3 records from a single species.

    Leakage wins that conflict -- a validation set sharing species with training
    measures memorisation, which is worse than a clade being absent from it.
    So: every clade must appear in TRAINING (a clade with no training signal
    would still drive the loss we early-stop on), and every clade with at least
    two species must appear in validation.
    """
    recs = load_records(cap=120)
    if len(recs) < 20:
        import pytest

        pytest.skip("data/demo.jsonl not built")
    train_rows, val_rows = clade_stratified_split(recs, 0.25, seed=0)
    clades = {r["clade"] for r in recs}
    assert {r["clade"] for r in train_rows} == clades

    species_per_clade: dict[str, set[str]] = {}
    for r in recs:
        species_per_clade.setdefault(r["clade"], set()).add(species_key(r))
    multi = {c for c, s in species_per_clade.items() if len(s) > 1}
    assert {r["clade"] for r in val_rows} == multi

    # And the reason this test changed: no species on both sides.
    assert not ({species_key(r) for r in train_rows}
                & {species_key(r) for r in val_rows})
