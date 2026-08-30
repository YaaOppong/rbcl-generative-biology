"""Minimal LoRA adapter injection, loss, and checkpointing.

Kept dependency-light and explicit rather than delegating to a library, so the
adapter placement is auditable against the recipe it claims to follow.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        # Match the wrapped layer's device AND dtype. Vortex places the model on
        # CUDA (and in bf16) as it loads, so adapters built with nn.Linear's
        # defaults land on CPU in fp32 and the first forward pass dies with
        # "Expected all tensors to be on the same device". The CPU test suite
        # exercises this path too, where base.weight is already CPU/fp32.
        self.a = nn.Linear(base.in_features, rank, bias=False,
                           device=base.weight.device, dtype=base.weight.dtype)
        self.b = nn.Linear(rank, base.out_features, bias=False,
                           device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)  # update starts at exactly zero
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.b(self.a(self.drop(x))) * self.scale


def apply_lora(model: nn.Module, cfg) -> int:
    """Replace matching nn.Linear modules in-place. Returns count wrapped."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped = 0
    skipped_non_linear: set[str] = set()
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            matched = any(t in full for t in cfg.target_modules)
            if not isinstance(child, nn.Linear):
                # A requested target that is not an nn.Linear is an error, not a
                # skip: silently ignoring it yields a finetune with fewer
                # adapters than the config claims, which is unfalsifiable later.
                if matched:
                    skipped_non_linear.add(f"{full} ({type(child).__name__})")
                continue
            if matched:
                setattr(module, child_name, LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout))
                wrapped += 1
    if wrapped == 0:
        raise RuntimeError(
            f"no modules matched target_modules={cfg.target_modules!r}. "
            "Module names differ between Evo 2 releases -- print "
            "[n for n,_ in model.named_modules()] and update the config. "
            "Verified names for StripedHyena (vortex): inner_mha_cls.Wqkv, "
            "inner_mha_cls.out_proj, mlp.l1, mlp.l2, mlp.l3, out_filter_dense."
        )
    if skipped_non_linear:
        raise RuntimeError(
            "target_modules matched non-nn.Linear modules, which this "
            f"implementation cannot wrap: {sorted(skipped_non_linear)}. "
            "In StripedHyena, blocks.N.projections is a TELinear -- it "
            "subclasses nn.Module, not nn.Linear, and returns (out, bias) "
            "rather than a tensor, so a LoRALinear around it would add a "
            "tensor to a tuple. Remove it from target_modules, or extend "
            "LoRALinear to handle the tuple convention explicitly."
        )
    return wrapped


def causal_lm_loss(model, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy over unpadded positions only."""
    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    logits = logits[:, :-1].reshape(-1, logits.size(-1))
    targets = input_ids[:, 1:].reshape(-1)
    keep = mask[:, 1:].reshape(-1)
    return nn.functional.cross_entropy(logits[keep], targets[keep])


def model_device(model: nn.Module) -> torch.device:
    """Device the model's parameters live on.

    Used instead of a hardcoded .cuda(): the loop must run on CPU so CI can
    execute it against a small stand-in model. Hardcoding cuda makes the training
    code untestable without a GPU, which is how it stays unrun and unverified.
    """
    return next(model.parameters()).device


@torch.no_grad()
def evaluate(model, loader) -> float:
    """Token-weighted mean validation loss.

    The weight must be the number of positions the loss actually averaged
    over, which is mask[:, 1:] -- next-token prediction has one fewer target
    than tokens. Weighting by mask.sum() instead over-weights short sequences
    (their off-by-one is proportionally larger), which biases early stopping
    toward whichever batch composition happens to hold the short records.
    """
    model.eval()
    dev = model_device(model)
    total, n = 0.0, 0
    for x, mask in loader:
        loss = causal_lm_loss(model, x.to(dev), mask.to(dev))
        ntok = int(mask[:, 1:].sum().item())
        total += loss.item() * ntok
        n += ntok
    return total / max(1, n)


def save_adapters(model: nn.Module, path: Path) -> None:
    """Save only trainable adapter weights (a few MB, not the full model)."""
    state = {k: v.cpu() for k, v in model.state_dict().items() if ".a." in k or ".b." in k}
    torch.save(state, path)
