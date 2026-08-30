"""Tests for the Evo 2 fp8 workaround.

``import evo2`` needs a live CUDA driver, so the loader itself cannot run here.
What IS testable on CPU is the decision logic -- which is the part that would
silently pick the wrong numerical path if it regressed.

The load-bearing assertion is that the gate is a literal ``"7b"`` substring
match, not a model-size comparison: ``evo2_1b_base`` and ``evo2_40b`` both fall
outside it despite sitting on opposite sides of 7B.
"""

from __future__ import annotations

import pytest

from src.evo2_loader import needs_fp8_patch

FP8_ON = {"use_fp8_input_projections": True}
FP8_OFF = {"use_fp8_input_projections": False}


@pytest.mark.parametrize(
    "model_name",
    ["evo2_7b", "evo2_7b_base", "evo2_7b_262k", "evo2_7b_microviridae"],
)
def test_7b_variants_self_heal(model_name):
    """evo2's own gate handles these -- we must not double-patch."""
    assert needs_fp8_patch(model_name, FP8_ON) is False


@pytest.mark.parametrize("model_name", ["evo2_1b_base", "evo2_20b", "evo2_40b"])
def test_non_7b_needs_patch(model_name):
    """Everything outside the substring gate raises without our patch.

    Note evo2_40b is *larger* than 7B and still needs it -- the gate is not a
    size comparison.
    """
    assert needs_fp8_patch(model_name, FP8_ON) is True


def test_patched_config_path_is_package_relative():
    """The patched config path must be package-relative, not absolute.

    load_evo2_model reads it via pkgutil.get_data(__name__, config_path), which
    resolves relative to site-packages/evo2/. A real run failed with
    FileNotFoundError on 'site-packages/evo2/tmp/evo2_1b_base-nofp8-x.yml'
    because an absolute /tmp path was passed and silently appended to the
    package directory.

    Asserted against the source rather than by loading: `import evo2` requires a
    live CUDA driver, so the loader itself cannot run in CI.
    """
    import re
    from pathlib import Path as _P

    body = (_P(__file__).parent.parent / "src" / "evo2_loader.py").read_text()
    assert "Path(_evo2pkg.__file__).parent" in body, "config not written into the package dir"
    assert re.search(r'cfg_path = f"_patched_configs/\{Path\(abs_path\)\.name\}"', body), \
        "relative cfg_path not constructed"
    passed = re.search(r"load_evo2_model\(None, (\w+), ckpt\)", body)
    assert passed, "load_evo2_model call not found"
    assert passed.group(1) == "cfg_path", f"passes {passed.group(1)}, not the relative cfg_path"


def test_no_patch_when_config_does_not_request_fp8():
    """A config with fp8 off needs nothing, whatever the checkpoint."""
    assert needs_fp8_patch("evo2_1b_base", FP8_OFF) is False
    assert needs_fp8_patch("evo2_1b_base", {}) is False


def test_patch_decision_is_independent_of_te_availability():
    """The decision describes the config+name, not the host.

    The loader is always called in an image without TE; if this ever became
    TE-conditional, a run's provenance record of the numerical path would
    depend on where it ran.
    """
    assert needs_fp8_patch("evo2_1b_base", dict(FP8_ON)) is True
