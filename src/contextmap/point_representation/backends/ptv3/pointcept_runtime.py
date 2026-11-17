"""Real PTv3 runtime: the Pointcept backbone behind the :class:`PTv3Runtime` boundary.

:class:`PointceptPTv3Runtime` gives :class:`PTv3PointEncoder` a real forward pass. It is a
*backbone-only* runtime: it builds the Point Transformer V3 encoder-decoder from a Pointcept
checkout, loads only the ``backbone.*`` weights of a supervised segmentation checkpoint and never
builds or applies the classification head. The per-point features it returns are a learned 3D
descriptor of the local structure, not a label; a segmentation checkpoint is used because it is what
the PTv3 authors publish, not because segmentation is wanted here. This is inference with a generic
PTv3, not Sonata- or Vernata-style representation learning.

The module imports no model library at import time (no torch, spconv, torch-scatter, timm or
Pointcept, and not even NumPy): everything is imported when the first support is encoded, so the
capability and the rest of the pipeline work without them and disabling this backend changes no
canonical contract. Pointcept is not pip-installable; it must be a clone of the official repository
at a pinned commit, passed as ``pointcept_root``. Importing its backbone module installs module
stubs in ``sys.modules``; that side effect is owned and documented by :mod:`.pointcept_module`.

Failure is explicit and there is no fallback. Anything that prevents a forward pass from being
trustworthy raises :class:`PTv3RuntimeUnavailableError` before a weight is used: a missing
dependency or Pointcept checkout, a checkpoint whose SHA-256 differs from the configured
``checkpoint_hash``, a checkpoint that does not fit the declared backbone, a device without CUDA, a
backbone whose weights do not fit in device memory. Running out of device memory on one support
raises :class:`PTv3OutOfMemoryError`, wherever that support allocates: staging its inputs, the
forward pass or copying the result back.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmap.point_representation.backends.ptv3.adapter import (
    PTv3Config,
    PTv3Inference,
    PTv3OutOfMemoryError,
    PTv3RuntimeUnavailableError,
)
from contextmap.point_representation.backends.ptv3.checkpoint import (
    read_backbone_state,
    verify_checkpoint_sha256,
)
from contextmap.point_representation.backends.ptv3.pointcept_backbone import (
    PointceptBackbone,
    check_runtime_compatibility,
)
from contextmap.point_representation.backends.ptv3.pointcept_module import (
    import_pointcept_ptv3,
    require_dependency,
)
from contextmap.point_representation.backends.ptv3.voxelization import (
    Voxelization,
    pool_voxel_features,
    voxelize,
)
from contextmap.shared import Vector3

if TYPE_CHECKING:
    from numpy.typing import NDArray

POINTCEPT_RUNTIME_VERSION = "1"
"""Version of this runtime's mapping from a support to the backbone input and back."""


def _release_device_memory_after_out_of_memory(torch: Any, error: BaseException) -> None:
    """Return to the driver the device memory a failed allocation left behind.

    The traceback of ``error`` keeps every finished frame alive, and with them the tensors held in
    their locals (staged inputs, activations, weights already moved). ``empty_cache`` only returns
    memory nobody references, so the frames are cleared first; the traceback itself stays readable.
    """
    traceback.clear_frames(error.__traceback__)
    torch.cuda.empty_cache()


@dataclass(frozen=True, eq=False)
class _LoadedBackbone:
    """A backbone resident on the device, with the identity it was loaded under."""

    torch: Any
    model: Any
    device: Any
    device_name: str
    checkpoint_hash: str


class PointceptPTv3Runtime:
    """Runs the Pointcept PTv3 backbone on one support and pools its per-point features.

    The model is built lazily, on the first :meth:`infer`, so constructing the runtime imports and
    loads nothing. A support is voxelized, the centroids (plus zero padding channels) go through the
    backbone, the per-voxel features are mapped back to the support's points and pooled as
    ``config.pooling`` says.

    ``peak_memory_bytes`` in the result is ``torch.cuda.max_memory_allocated`` for the call: the
    peak of tensor memory the process allocated on the device, including the resident weights, and
    excluding the CUDA context and allocator cache that ``nvidia-smi`` also counts.
    """

    def __init__(
        self,
        *,
        pointcept_root: Path,
        checkpoint_path: Path,
        backbone: PointceptBackbone,
        weights_only: bool = True,
    ) -> None:
        """Remember where the code and weights are; nothing is imported or loaded yet.

        Args:
            pointcept_root: Clone of the official Pointcept repository at a pinned commit.
            checkpoint_path: Local checkpoint file whose SHA-256 must equal
                ``PTv3Config.checkpoint_hash``.
            backbone: The architecture the checkpoint was trained with.
            weights_only: Load with ``torch.load(weights_only=True)``. Pointcept training
                checkpoints also pickle optimizer/scheduler state and NumPy scalars, which that
                mode rejects. Passing ``False`` is an explicit decision to unpickle the file, made
                only for a checkpoint of verified provenance; the SHA-256 is always verified first,
                so what is unpickled is exactly the pinned bytes.
        """
        self._pointcept_root = pointcept_root
        self._checkpoint_path = checkpoint_path
        self._backbone = backbone
        self._weights_only = weights_only
        self._loaded: _LoadedBackbone | None = None

    def infer(
        self, *, coordinates_m: Sequence[Vector3], center_index: int, config: PTv3Config
    ) -> PTv3Inference:
        """Encode one support.

        Args:
            coordinates_m: The prepared local coordinates of the support.
            center_index: Position of the center element in ``coordinates_m``.
            config: The effective configuration.

        Returns:
            The pooled vector and the peak device memory of the call.

        Raises:
            ValueError: If the coordinates are empty, not ``(N, 3)`` or not finite, or the center
                is outside the support.
            PTv3OutOfMemoryError: If this support exhausts device memory, at any allocation it
                causes on the device.
            PTv3RuntimeUnavailableError: If the runtime cannot run at all, including a backbone
                whose weights do not fit in device memory.
        """
        voxels = voxelize(coordinates_m, config.grid_size_m)
        if not 0 <= center_index < len(coordinates_m):
            raise ValueError(
                f"center_index {center_index} is outside a support of {len(coordinates_m)} points"
            )
        loaded = self._ensure_loaded(config)
        features, peak_memory_bytes = self._forward(loaded, voxels, config)
        pooled = pool_voxel_features(
            features, voxels, center_index=center_index, pooling=config.pooling
        )
        return PTv3Inference(
            vector=tuple(float(value) for value in pooled), peak_memory_bytes=peak_memory_bytes
        )

    def _forward(
        self, loaded: _LoadedBackbone, voxels: Voxelization, config: PTv3Config
    ) -> tuple[NDArray[Any], int]:
        """Run the backbone on the voxels and return their features and the memory peak.

        Every allocation this support causes on the device happens under the out-of-memory guard,
        from staging the inputs to copying the result back, so running out of memory anywhere
        becomes the same explicit :class:`PTv3OutOfMemoryError` for this support alone.
        """
        torch = loaded.torch
        try:
            voxel_features = self._run_backbone(loaded, voxels, config)
        except torch.cuda.OutOfMemoryError as error:
            _release_device_memory_after_out_of_memory(torch, error)
            raise PTv3OutOfMemoryError(str(error)) from error
        return voxel_features, int(torch.cuda.max_memory_allocated(loaded.device))

    def _run_backbone(
        self, loaded: _LoadedBackbone, voxels: Voxelization, config: PTv3Config
    ) -> NDArray[Any]:
        """Stage the voxels on the device, run the backbone and copy the features back."""
        import numpy as np

        torch = loaded.torch
        voxel_count = len(voxels.counts)
        padding = np.zeros((voxel_count, config.padding_channels), dtype=np.float64)
        features = np.concatenate([voxels.centroid_m, padding], axis=1).astype(np.float32)
        data = {
            "coord": torch.from_numpy(voxels.centroid_m.astype(np.float32)).to(loaded.device),
            "grid_coord": torch.from_numpy(voxels.grid_coord).to(loaded.device),
            "feat": torch.from_numpy(features).to(loaded.device),
            "offset": torch.tensor([voxel_count], dtype=torch.long, device=loaded.device),
        }
        torch.cuda.reset_peak_memory_stats(loaded.device)
        with torch.no_grad(), self._autocast(torch, config):
            output = loaded.model(data)
        voxel_features: NDArray[Any] = output.feat.float().cpu().numpy()
        return voxel_features

    @staticmethod
    def _autocast(torch: Any, config: PTv3Config) -> Any:
        """Mixed-precision context for float16/bfloat16; float32 runs without autocast."""
        if config.precision == "float32":
            return torch.autocast(device_type="cuda", enabled=False)
        dtype = torch.float16 if config.precision == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _ensure_loaded(self, config: PTv3Config) -> _LoadedBackbone:
        """Build the backbone on first use, trusting no weight until its hash is verified."""
        check_runtime_compatibility(config, self._backbone)
        if self._loaded is not None:
            if (self._loaded.checkpoint_hash, self._loaded.device_name) != (
                config.checkpoint_hash,
                config.device,
            ):
                raise PTv3RuntimeUnavailableError(
                    "this runtime already loaded another checkpoint or device; build one runtime "
                    "per checkpoint and device"
                )
            return self._loaded
        if not (self._pointcept_root / "pointcept" / "models").is_dir():
            raise PTv3RuntimeUnavailableError(
                f"no Pointcept checkout found at {self._pointcept_root}; clone the official "
                "repository at a pinned commit"
            )
        verify_checkpoint_sha256(self._checkpoint_path, config.checkpoint_hash)
        torch = require_dependency("torch")
        for dependency in ("spconv.pytorch", "torch_scatter", "timm", "addict"):
            require_dependency(dependency)
        if not torch.cuda.is_available():
            raise PTv3RuntimeUnavailableError("CUDA is not available to torch")
        try:
            device = torch.device(config.device)
            torch.zeros(1, device=device)
        except (RuntimeError, AssertionError) as error:
            raise PTv3RuntimeUnavailableError(
                f"device {config.device!r} cannot be used: {error}"
            ) from error
        ptv3 = import_pointcept_ptv3(self._pointcept_root)
        model = ptv3.PointTransformerV3(**self._backbone.constructor_kwargs())
        state = read_backbone_state(torch, self._checkpoint_path, weights_only=self._weights_only)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise PTv3RuntimeUnavailableError(
                f"the checkpoint does not fit the declared backbone: {error}"
            ) from error
        try:
            model.to(device).eval()
        except torch.cuda.OutOfMemoryError as error:
            # A cópia dos pesos é interrompida no meio: parte deles já está no dispositivo.
            # Devolve-os à CPU e libera a memória antes de falhar, sem deixar modelo meio carregado.
            model.to("cpu")
            _release_device_memory_after_out_of_memory(torch, error)
            raise PTv3RuntimeUnavailableError(
                f"the PTv3 backbone of checkpoint {config.checkpoint_hash} does not fit on device "
                f"{config.device!r}: {error}"
            ) from error
        self._loaded = _LoadedBackbone(
            torch=torch,
            model=model,
            device=device,
            device_name=config.device,
            checkpoint_hash=config.checkpoint_hash,
        )
        return self._loaded
