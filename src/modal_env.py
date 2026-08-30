"""The pinned remote environment and the layout of the weights volume.

One place, because these constants were previously restated in every script that
touched Modal -- and a truncated copy of IMAGE_ID is a bug this project already
had (commit daee32b). The adapter filename matters for the same reason: training
writes it and generation looks it up, so if the two ever disagree the failure is
"no adapter found" after a paid half-hour of training.
"""
from __future__ import annotations

import os
from pathlib import Path

# The image and volume already exist in the account; neither is built here.
IMAGE_ID = "im-ObsG9qtru3aZlO314k92Mv"
VOLUME = "claude-science-evo2-weights"
APP_NAME = "rbcl-finetune"

# Where the volume is mounted inside a sandbox. The HuggingFace cache lives on it
# too, so checkpoints are downloaded once per account rather than once per run.
WEIGHTS_MOUNT = Path("/weights")
ADAPTER_DIR = WEIGHTS_MOUNT / "adapters"


def adapter_path(tag: str, adapter_dir: str | Path | None = None) -> Path:
    """`<volume>/adapters/<tag>_adapter_best.pt` -- written by finetune.py,
    read by generate_ab.py. The tag is the config stem, plus `_seed<N>` for a
    training replicate so a replicate cannot overwrite the original."""
    return Path(adapter_dir or ADAPTER_DIR) / f"{tag}_adapter_best.pt"


def hf_env() -> dict:
    """Environment for a remote job: cache on the volume, no network reachback.

    HF_HUB_OFFLINE is deliberate -- a run that silently re-downloads a
    checkpoint is a run whose weights are not the cached ones.
    """
    return {"HF_HOME": str(WEIGHTS_MOUNT), "HF_HUB_OFFLINE": "1"}


def apply_hf_env() -> None:
    """Set the cache variables unless the caller already chose otherwise."""
    for k, v in hf_env().items():
        os.environ.setdefault(k, v)
