"""Tests for LoRA adapter injection.

These run on a toy module tree, not Evo 2 -- they verify the injection logic
and the zero-initialisation property without needing a GPU or model weights.
"""

import pytest

# torch is an optional extra: it needs a bf16-capable GPU for real use and
# is installed separately (see README). These modules exercise adapter
# injection and the loss on toy tensors, so they need torch but not a GPU.
# Skipping beats erroring: a fresh clone should get a clean run, not three
# collection failures that look like the suite is broken.
pytest.importorskip("torch")

import re
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

from src.train.lora import LoRALinear, apply_lora, causal_lm_loss, save_adapters


class Cfg:
    rank, alpha, dropout = 8, 16, 0.0
    target_modules: ClassVar[list[str]] = ["attn_qkv", "mlp_in"]


def toy_model():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_qkv = nn.Linear(16, 48)
            self.attn_out = nn.Linear(16, 16)
            self.mlp_in = nn.Linear(16, 32)

    m = nn.Module()
    m.blocks = nn.ModuleList([Block() for _ in range(3)])
    return m


def test_wraps_only_targeted_modules():
    m = toy_model()
    assert apply_lora(m, Cfg()) == 6  # 2 targets x 3 blocks
    assert isinstance(m.blocks[0].attn_qkv, LoRALinear)
    assert isinstance(m.blocks[0].mlp_in, LoRALinear)
    assert isinstance(m.blocks[0].attn_out, nn.Linear)  # untargeted, untouched


def test_raises_when_no_module_matches():
    class BadCfg(Cfg):
        target_modules: ClassVar[list[str]] = ["does_not_exist"]

    with pytest.raises(RuntimeError, match="no modules matched"):
        apply_lora(toy_model(), BadCfg())


def test_update_starts_at_exactly_zero():
    """B is zero-initialised, so a freshly adapted model must be numerically
    identical to the base model. If this fails, training starts from a
    perturbed model and the finetune is not an adaptation of the baseline."""
    m = toy_model()
    x = torch.randn(4, 16)
    before = m.blocks[0].attn_qkv(x).clone()
    apply_lora(m, Cfg())
    torch.testing.assert_close(m.blocks[0].attn_qkv(x), before)


def test_only_adapter_params_are_trainable():
    m = toy_model()
    apply_lora(m, Cfg())
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable
    assert all(".a." in n or ".b." in n for n in trainable)
    # Exact expected count rather than a fraction: on a 16-dim toy model a rank-8
    # adapter is legitimately a large share of parameters. The share only becomes
    # small at real model width (~1.4% at Evo2-1B).
    r = Cfg.rank
    per_block = (r * 16 + 48 * r) + (r * 16 + 32 * r)  # attn_qkv + mlp_in
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_train == 3 * per_block


def test_adapter_checkpoint_excludes_base_weights(tmp_path):
    m = toy_model()
    apply_lora(m, Cfg())
    path = tmp_path / "adapter.pt"
    save_adapters(m, path)
    keys = torch.load(path).keys()
    assert keys
    assert all(".a." in k or ".b." in k for k in keys)
    assert not any("base" in k for k in keys)


def test_causal_lm_loss_ignores_padding():
    """Padded positions must not contribute. Two batches differing only in
    pad content must give the same loss."""
    torch.manual_seed(0)
    vocab, n = 8, 6

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, 12)
            self.head = nn.Linear(12, vocab)

        def forward(self, ids):
            return (self.head(self.emb(ids)),)

    lm = LM()
    ids = torch.randint(1, vocab, (2, n))
    mask = torch.ones(2, n, dtype=torch.bool)
    mask[1, 4:] = False
    a = ids.clone(); a[1, 4:] = 0
    b = ids.clone(); b[1, 4:] = 5
    torch.testing.assert_close(causal_lm_loss(lm, a, mask), causal_lm_loss(lm, b, mask))

class _TELinearAlike(nn.Module):
    """Mirrors vortex TELinear: nn.Module subclass returning (out, bias)."""

    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i))

    def forward(self, x):
        return x @ self.weight.T, None


class _GatedMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, 2 * d, bias=False)
        self.l2 = nn.Linear(d, 2 * d, bias=False)
        self.l3 = nn.Linear(2 * d, d, bias=False)


class _MHA(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)


class _HyenaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.projections = _TELinearAlike(d, 3 * d)
        self.out_filter_dense = nn.Linear(d, d, bias=False)
        self.mlp = _GatedMLP(d)


class _AttnBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inner_mha_cls = _MHA(d)
        self.mlp = _GatedMLP(d)


class _StripedHyenaAlike(nn.Module):
    """Block layout matching Evo 2: hyena blocks with periodic attention."""

    def __init__(self, d=16, n=8, attn_every=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_AttnBlock(d) if (i + 1) % attn_every == 0 else _HyenaBlock(d) for i in range(n)]
        )


SHIPPED_TARGETS = [
    "inner_mha_cls.Wqkv",
    "inner_mha_cls.out_proj",
    "mlp.l1",
    "mlp.l2",
    "mlp.l3",
    "out_filter_dense",
]


def _cfg(targets, rank=4, alpha=8):
    class C(Cfg):
        target_modules: ClassVar[list[str]] = list(targets)
    C.rank, C.alpha, C.dropout = rank, alpha, 0.0
    return C()


def test_shipped_targets_match_striped_hyena_layout():
    """The configs' target names must resolve on the real block layout.

    Guards the defect found in the 0b check: the original config named
    attn_qkv/attn_out/mlp_in/mlp_out/hyena_mixer, none of which exist in
    vortex, so apply_lora raised on the real model.
    """
    m = _StripedHyenaAlike(d=16, n=8, attn_every=4)
    n = apply_lora(m, _cfg(SHIPPED_TARGETS))
    # 8 blocks: 6 hyena (out_filter_dense + 3 mlp = 4 each) + 2 attn (2 + 3 = 5 each)
    assert n == 6 * 4 + 2 * 5, n
    wrapped = [k for k, v in m.named_modules() if isinstance(v, LoRALinear)]
    assert any("inner_mha_cls.Wqkv" in w for w in wrapped)
    assert any("out_filter_dense" in w for w in wrapped)
    assert any("mlp.l3" in w for w in wrapped)


def test_original_guessed_names_would_have_raised():
    m = _StripedHyenaAlike()
    with pytest.raises(RuntimeError, match="no modules matched"):
        apply_lora(m, _cfg(["attn_qkv", "attn_out", "mlp_in", "mlp_out"]))


def test_telinear_target_is_rejected_not_skipped():
    """projections is TELinear -> must raise, not silently under-adapt."""
    m = _StripedHyenaAlike()
    with pytest.raises(RuntimeError, match="non-nn.Linear"):
        apply_lora(m, _cfg(["projections", "mlp.l1"]))


def test_shipped_configs_use_verified_targets():
    """Every arm config must carry the verified names, not guesses."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    for cfg_path in sorted((root / "configs").glob("*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        got = cfg["lora"]["target_modules"]
        assert got == SHIPPED_TARGETS, f"{cfg_path.name}: {got}"

# Layer counts and attention indices read from ArcInstitute/evo2
# evo2/configs/{evo2-1b-8k.yml, evo2-7b-1m.yml}. The two checkpoints differ only
# in width, depth and layer indices -- both are StripedHyena2 built from the same
# two block classes (AttentionBlock / ParallelGatedConvBlock) by get_block().
# Targets must therefore be index-agnostic so one config drives either scale.
EVO2_LAYOUTS = {
    "evo2_1b_base": (25, (3, 10, 17, 24)),
    "evo2_7b": (32, (3, 10, 17, 24, 31)),
}


def _module_names(n_layers, attn_idxs):
    """Real StripedHyena2 module names for a given depth."""
    names = []
    for i in range(n_layers):
        if i in attn_idxs:
            names += [f"blocks.{i}.inner_mha_cls.Wqkv", f"blocks.{i}.inner_mha_cls.out_proj"]
        else:
            names += [f"blocks.{i}.filter.out_filter_dense", f"blocks.{i}.projections"]
        names += [f"blocks.{i}.mlp.l1", f"blocks.{i}.mlp.l2", f"blocks.{i}.mlp.l3"]
    return names


def test_make_trainable_clears_inference_tensors():
    """Weights loaded under inference_mode must become autograd-capable.

    vortex.model.utils.load_checkpoint loads inside torch.inference_mode(), which
    marks every parameter an inference tensor. Training then dies with "Inference
    tensors cannot be saved for backward" inside vortex's own RMSNorm -- before
    any adapter code runs, so no LoRA-side change can fix it. A real B2 run failed
    exactly this way. Reproducible on CPU.
    """
    from src.evo2_loader import make_trainable

    with torch.inference_mode():
        model = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Linear(4, 2))
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    assert any(p.is_inference() for p in model.parameters()), "fixture precondition"

    n = make_trainable(model)
    assert n >= 4  # 2 weights + 2 biases
    assert not any(p.is_inference() for p in model.parameters())

    # and a backward pass now actually works
    loss = model(torch.randn(3, 6)).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_adapters_inherit_device_and_dtype_of_wrapped_layer():
    """Adapters must be built on the base layer's device and dtype.

    Vortex loads Evo 2 onto CUDA in bf16, so adapters created with nn.Linear's
    defaults (CPU, fp32) make the first forward pass raise "Expected all tensors
    to be on the same device". A real B2 run failed exactly this way. Testable on
    CPU by wrapping a non-default dtype.
    """
    base = torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16)
    wrapped = LoRALinear(base, rank=2, alpha=4)
    assert wrapped.a.weight.dtype == base.weight.dtype
    assert wrapped.b.weight.dtype == base.weight.dtype
    assert wrapped.a.weight.device == base.weight.device
    # and it must actually run without a dtype promotion error
    out = wrapped(torch.randn(3, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert out.shape == (3, 4)


@pytest.mark.parametrize("model_key", sorted(EVO2_LAYOUTS))
def test_targets_are_index_agnostic_across_scales(model_key):
    """The shipped target names must resolve at both 1B and 7B depth.

    Guards the model-scale decision: if a target name were coupled to a layer
    index, switching checkpoints would silently under-adapt.
    """
    n_layers, attn_idxs = EVO2_LAYOUTS[model_key]
    names = _module_names(n_layers, attn_idxs)
    hit = [n for n in names if any(t in n for t in SHIPPED_TARGETS)]
    # Every attention block contributes 2, every block contributes 3 MLP
    # projections, every hyena block contributes 1 output filter.
    n_attn = len(attn_idxs)
    n_hyena = n_layers - n_attn
    assert len(hit) == 2 * n_attn + 3 * n_layers + n_hyena
    # No layer is skipped.
    assert {int(n.split(".")[1]) for n in hit} == set(range(n_layers))


@pytest.mark.parametrize("model_key", sorted(EVO2_LAYOUTS))
def test_telinear_never_matched_at_either_scale(model_key):
    """blocks.N.projections returns (out, bias) -- it must never be targeted."""
    n_layers, attn_idxs = EVO2_LAYOUTS[model_key]
    names = _module_names(n_layers, attn_idxs)
    leaked = [n for n in names if n.endswith("projections")
              and any(t in n for t in SHIPPED_TARGETS)]
    assert leaked == []


def test_configs_do_not_pin_a_layer_index():
    """A digit-bearing target would break on a checkpoint switch.

    'mlp.l1' etc. are literal module names, not indices -- so the assertion is
    specifically that no target contains a 'blocks.<N>' path segment.
    """
    for t in SHIPPED_TARGETS:
        assert not re.search(r"blocks\.\d", t), t
