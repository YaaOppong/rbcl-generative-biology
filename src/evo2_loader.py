"""Load an Evo 2 checkpoint at either scale, working around the fp8 gate.

Part A generated with ``evo2_7b`` (hardcoded in ``generate_corpus.py``). Part B
needs a *matched* control at the same scale as the finetune, and 1B is the
cheaper path -- but 1B will not load without Transformer Engine, for a reason
that is a packaging accident rather than a real numerical requirement:

  * Every shipped config sets ``use_fp8_input_projections: true``, which needs
    Transformer Engine (TE) for the FP8 input projections.
  * evo2 0.5.5 knows how to fall back to bf16 projections without TE, but the
    fallback is gated on a literal ``"7b"`` substring in the model name or the
    config path (``is_7b_model`` in ``evo2/models.py``). Any other checkpoint
    raises ``ImportError`` instead of falling back.
  * The GPU image used here deliberately omits TE: its import-time CUDA init
    hangs during Modal's image-save phase, and TE 1.13 caps flash-attn at
    <=2.6.3.

So loading 1B means handing ``Evo2`` a config copy with the flag set to false.
That is the *same* numerical path the maintainers' own 7B fallback takes -- it
is not a novel or unsupported configuration -- but it is applied by us rather
than by their gate.

``import evo2`` requires a live CUDA driver (Triton raises "0 active drivers"
on CPU), so nothing in this module can be exercised without a GPU. The config
rewriting is factored out into :func:`patched_config` precisely so that part
*is* CPU-testable; see ``tests/test_evo2_loader.py``.
"""

from __future__ import annotations

import os
import pkgutil
import tempfile
from pathlib import Path

import yaml

#: Checkpoints this project may load, and whether evo2's own gate handles fp8.
#: The gate matches a literal "7b" substring, so only 7B variants self-heal.
SELF_HEALING = ("evo2_7b", "evo2_7b_base", "evo2_7b_262k", "evo2_7b_microviridae")


def needs_fp8_patch(model_name: str, config: dict) -> bool:
    """Would ``Evo2(model_name)`` raise ImportError without Transformer Engine?

    True when the config asks for FP8 projections *and* the model name falls
    outside evo2's ``is_7b_model`` fallback gate.
    """
    if not config.get("use_fp8_input_projections", False):
        return False
    return "7b" not in model_name


def patched_config(model_name: str, config_rel: str) -> tuple[dict, bool]:
    """Return (config dict, was_patched) for ``model_name``.

    Reads the config shipped inside the installed ``evo2`` package -- not a
    vendored copy -- so it tracks the installed version rather than drifting
    from it.
    """
    raw = pkgutil.get_data("evo2", config_rel)
    if raw is None:
        raise FileNotFoundError(f"evo2 package has no config at {config_rel!r}")
    config = yaml.safe_load(raw)
    if needs_fp8_patch(model_name, config):
        config["use_fp8_input_projections"] = False
        return config, True
    return config, False


def load_evo2(model_name: str = "evo2_7b", *, use_kernels: bool = False):
    """Load ``model_name``, applying the fp8 workaround only when required.

    Returns ``(model, info)``. ``info`` records whether the patch was applied
    and which checkpoint file was used, so a run's provenance block can state
    the numerical path the weights actually took.
    """
    from evo2.models import Evo2
    from evo2.utils import CONFIG_MAP
    from huggingface_hub import snapshot_download

    if model_name not in CONFIG_MAP:
        raise ValueError(
            f"unknown checkpoint {model_name!r}; known: {sorted(CONFIG_MAP)}"
        )

    config, was_patched = patched_config(model_name, CONFIG_MAP[model_name])
    info = {
        "model_name": model_name,
        "fp8_patch_applied": was_patched,
        "self_healing_gate": model_name in SELF_HEALING,
    }

    if not was_patched:
        # Let evo2 take its own documented path (download + gate + fallback).
        return Evo2(model_name), info

    # Patched path: Evo2 only honours a custom config alongside an explicit
    # local checkpoint, so resolve the weights ourselves. HF_HUB_OFFLINE=1 in
    # the image means this reads the hydrated Volume rather than the network.
    snap = snapshot_download(f"arcinstitute/{model_name}")
    ckpt = os.path.join(snap, f"{model_name}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found at {ckpt}")

    # The config path MUST be inside the installed evo2 package directory.
    # load_evo2_model reads it with pkgutil.get_data(__name__, config_path),
    # which resolves relative to site-packages/evo2/ -- so an absolute /tmp path
    # is looked up as site-packages/evo2/tmp/... and raises FileNotFoundError.
    import evo2 as _evo2pkg

    pkg_dir = Path(_evo2pkg.__file__).parent
    cfg_dir = pkg_dir / "_patched_configs"
    cfg_dir.mkdir(exist_ok=True)
    fd, abs_path = tempfile.mkstemp(prefix=f"{model_name}-nofp8-", suffix=".yml",
                                    dir=str(cfg_dir))
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(config, fh)
    # pass the package-RELATIVE path, which is what pkgutil.get_data expects
    cfg_path = f"_patched_configs/{Path(abs_path).name}"

    info["checkpoint"] = ckpt
    info["config_path"] = cfg_path

    # `Evo2.__init__` takes only (model_name, local_path, use_kernels) -- there is
    # NO config_path parameter, though the lower-level `load_evo2_model` has one.
    # Passing it to the constructor raises TypeError. So bypass __init__ and drive
    # load_evo2_model directly, mirroring exactly what __init__ does:
    #   - with local_path set, __init__ passes model_name=None
    #   - load_evo2_model returns the MODEL ONLY, not a (model, tokenizer) tuple
    #   - the tokenizer is constructed separately as CharLevelTokenizer(512)
    from vortex.model.tokenizer import CharLevelTokenizer

    model = Evo2.__new__(Evo2)
    # NOTE: the INSTALLED evo2 0.5.5 signature is
    #   load_evo2_model(model_name, config_path, local_path, remove_shards)
    # with NO use_kernels parameter -- that argument exists on GitHub main but
    # not in the released package, so passing it raises TypeError. Positional
    # call against the installed signature; `use_kernels` is accepted by this
    # function only for API compatibility and is applied via config instead.
    if use_kernels:
        config["use_hcs_kernel"] = True
        config["use_hcm_kernel"] = True
        config["use_hcl_kernel"] = True
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(config, fh)
    model.model = model.load_evo2_model(None, cfg_path, ckpt)
    model.tokenizer = CharLevelTokenizer(512)
    return model, info


def make_trainable(model) -> int:
    """Undo vortex's inference-tensor marking so autograd can run.

    `vortex.model.utils.load_checkpoint` loads weights inside
    `torch.inference_mode()`, which marks every parameter as an *inference
    tensor*. Those can never take part in autograd -- a training forward pass
    dies with "Inference tensors cannot be saved for backward", inside vortex's
    own RMSNorm, before any adapter code is reached. Nothing on the LoRA side can
    fix it: the base parameters themselves carry the flag.

    Cloning each parameter out of inference mode produces normal tensors. Returns
    the number of parameters converted. Call BEFORE apply_lora so the adapters
    wrap already-normal layers.
    """
    import torch

    n = 0
    with torch.no_grad():
        for module in model.modules():
            for name, param in list(module.named_parameters(recurse=False)):
                if param.is_inference():
                    setattr(module, name, torch.nn.Parameter(
                        param.clone(), requires_grad=param.requires_grad))
                    n += 1
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf is not None and buf.is_inference():
                    module.register_buffer(name, buf.clone())
                    n += 1
    return n
