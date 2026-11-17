"""Checkpoint identity and trust: what may be unpickled, and which weights are kept.

A PTv3 checkpoint is a pickle, so reading it is only as safe as the certainty that it is the file
the configuration pinned. This module owns that boundary, in the order it must happen:

1. :func:`verify_checkpoint_sha256` proves the file is exactly the pinned bytes, before anything is
   unpickled;
2. :func:`read_backbone_state` reads the checkpoint on the CPU, optionally with
   ``weights_only=True``;
3. :func:`extract_backbone_state` keeps the ``backbone.*`` weights and discards the classification
   head on purpose, because this stage emits no classification.

The module imports no model library: ``torch`` is passed to :func:`read_backbone_state` by the
caller, which is the only place that imports it, so importing this module loads nothing heavy.
"""

from __future__ import annotations

import hashlib
import pickle
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextmap.point_representation.backends.ptv3.adapter import PTv3RuntimeUnavailableError

_BACKBONE_PREFIX = "backbone."
_DISTRIBUTED_PREFIX = "module."
_HASH_CHUNK_BYTES = 1 << 20
# Falhas de leitura de torch.load; qualquer outra exceção é um defeito e deve aparecer como tal.
_UNREADABLE_CHECKPOINT = (
    pickle.UnpicklingError,
    zipfile.BadZipFile,
    EOFError,
    OSError,
    RuntimeError,
)


def verify_checkpoint_sha256(path: Path, expected: str) -> None:
    """Require that a checkpoint file is exactly the bytes the configuration pinned.

    Args:
        path: Checkpoint file.
        expected: ``"sha256:<64 hex>"`` from :attr:`PTv3Config.checkpoint_hash`.

    Raises:
        PTv3RuntimeUnavailableError: If the file is missing or its SHA-256 differs.
    """
    if not path.is_file():
        raise PTv3RuntimeUnavailableError(f"checkpoint file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    actual = f"sha256:{digest.hexdigest()}"
    if actual != expected:
        raise PTv3RuntimeUnavailableError(
            f"the checkpoint {path} has hash {actual}, which does not match the configured "
            f"checkpoint_hash {expected}"
        )


def extract_backbone_state(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the backbone weights of a Pointcept segmentation checkpoint.

    Pointcept saves a ``DefaultSegmentor`` under the distributed-training prefix, so the backbone
    lives under ``module.backbone.`` next to the classification head ``module.seg_head.``; the head
    is discarded on purpose, because this stage emits no classification.

    Args:
        state_dict: The ``state_dict`` entry of the checkpoint.

    Returns:
        The backbone weights keyed as ``PointTransformerV3`` names them.

    Raises:
        PTv3RuntimeUnavailableError: If no ``backbone.*`` weight is present.
    """
    backbone: dict[str, Any] = {}
    for key, value in state_dict.items():
        name = key.removeprefix(_DISTRIBUTED_PREFIX)
        if name.startswith(_BACKBONE_PREFIX):
            backbone[name.removeprefix(_BACKBONE_PREFIX)] = value
    if not backbone:
        raise PTv3RuntimeUnavailableError(
            "the checkpoint has no 'backbone.*' weights, so it is not a Pointcept PTv3 model"
        )
    return backbone


def read_backbone_state(torch: Any, path: Path, *, weights_only: bool) -> dict[str, Any]:
    """Read the checkpoint on the CPU and keep the backbone weights.

    The caller must have verified the file with :func:`verify_checkpoint_sha256` first, so that
    what is unpickled here is exactly the pinned bytes.

    Args:
        torch: The imported ``torch`` module; passed in so this module imports no model library.
        path: Checkpoint file.
        weights_only: Forwarded to ``torch.load``; ``False`` is an explicit decision to unpickle
            arbitrary objects, made only for a checkpoint of verified provenance.

    Returns:
        The backbone weights keyed as ``PointTransformerV3`` names them.

    Raises:
        PTv3RuntimeUnavailableError: If the file cannot be read, has no ``state_dict`` entry or no
            ``backbone.*`` weight.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=weights_only)
    except _UNREADABLE_CHECKPOINT as error:
        raise PTv3RuntimeUnavailableError(
            f"the checkpoint cannot be read with weights_only={weights_only}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise PTv3RuntimeUnavailableError("the checkpoint has no 'state_dict' entry")
    return extract_backbone_state(checkpoint["state_dict"])
