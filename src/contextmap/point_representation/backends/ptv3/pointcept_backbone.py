"""Architecture of one published Pointcept PT-v3m1 backbone and what it can honour.

This module owns three things that change together, and only when a new published checkpoint is
adopted: the architecture a checkpoint was trained with (:class:`PointceptBackbone`), the
constructor arguments that turn it into a deterministic inference network
(:meth:`PointceptBackbone.constructor_kwargs`), and the refusal of an adapter configuration that
the backbone cannot serve (:func:`check_runtime_compatibility`).

It is pure data and pure checks: it imports no NumPy, no torch and no Pointcept, so a backbone can
be described, compared and validated without any model library installed. Building the network
and loading its weights belong to :mod:`.pointcept_runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextmap.point_representation.backends.ptv3.adapter import (
    PTv3Config,
    PTv3RuntimeUnavailableError,
)
from contextmap.point_representation.backends.ptv3.voxelization import XYZ_CHANNELS


@dataclass(frozen=True, kw_only=True)
class PointceptBackbone:
    """Architecture of one Pointcept PTv3 (``PT-v3m1``) backbone: the checkpoint's own settings.

    The values must be the ones the checkpoint was trained with (its ``config.py``); loading is
    strict, so a mismatch fails instead of producing a different network.

    Attributes:
        in_channels: Input feature channels the stem convolution expects.
        order: Serialization curves the blocks cycle through.
        stride: Downsampling stride between encoder stages.
        enc_depths: Blocks per encoder stage.
        enc_channels: Channels per encoder stage.
        enc_num_head: Attention heads per encoder stage.
        enc_patch_size: Attention patch size per encoder stage.
        dec_depths: Blocks per decoder stage.
        dec_channels: Channels per decoder stage; the first one is the output feature size.
        dec_num_head: Attention heads per decoder stage.
        dec_patch_size: Attention patch size per decoder stage.
    """

    in_channels: int
    order: tuple[str, ...]
    stride: tuple[int, ...]
    enc_depths: tuple[int, ...]
    enc_channels: tuple[int, ...]
    enc_num_head: tuple[int, ...]
    enc_patch_size: tuple[int, ...]
    dec_depths: tuple[int, ...]
    dec_channels: tuple[int, ...]
    dec_num_head: tuple[int, ...]
    dec_patch_size: tuple[int, ...]

    @property
    def output_channels(self) -> int:
        """Size of the per-point feature the decoder returns."""
        return self.dec_channels[0]

    def constructor_kwargs(self) -> dict[str, Any]:
        """Arguments for ``PointTransformerV3``, configured for deterministic inference.

        Three settings differ from training on purpose. ``shuffle_orders`` is off because Pointcept
        draws a random permutation of the serialization curves on every forward pass, which would
        make the same support encode differently twice; the weights do not depend on which curve a
        block uses, so a fixed assignment is a valid inference configuration. ``enable_flash`` is
        off because FlashAttention is not installed, and the plain patch attention has the same
        parameters. ``drop_path`` is zero because there is no stochastic depth at inference.
        """
        return {
            "in_channels": self.in_channels,
            "order": self.order,
            "stride": self.stride,
            "enc_depths": self.enc_depths,
            "enc_channels": self.enc_channels,
            "enc_num_head": self.enc_num_head,
            "enc_patch_size": self.enc_patch_size,
            "dec_depths": self.dec_depths,
            "dec_channels": self.dec_channels,
            "dec_num_head": self.dec_num_head,
            "dec_patch_size": self.dec_patch_size,
            # Demais valores: os do config.py do checkpoint nuScenes (PT-v3m1 base).
            "mlp_ratio": 4,
            "qkv_bias": True,
            "qk_scale": None,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "pre_norm": True,
            "enable_rpe": False,
            "upcast_attention": False,
            "upcast_softmax": False,
            # Três divergências deliberadas do treino, todas para inferência determinística:
            # sem estocasticidade de profundidade (drop_path), sem permutação aleatória das
            # curvas de serialização a cada passada (shuffle_orders) e com a atenção por
            # blocos comum no lugar da FlashAttention, que não está instalada e tem os mesmos
            # parâmetros (enable_flash). Os pesos não dependem de nenhuma das três.
            "drop_path": 0.0,
            "shuffle_orders": False,
            "enable_flash": False,
        }


NUSCENES_SEMSEG_PTV3M1_BASE = PointceptBackbone(
    in_channels=4,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
)
"""Backbone of ``Pointcept/PointTransformerV3`` ``nuscenes-semseg-pt-v3m1-0-base`` (MIT).

Its four input channels are XYZ plus the LiDAR return intensity, so the runtime is configured with
``padding_channels=1``: the intensity channel is filled with zeros and never measured.
"""


def check_runtime_compatibility(config: PTv3Config, backbone: PointceptBackbone) -> None:
    """Refuse a configuration that this backbone cannot honour.

    Args:
        config: The effective adapter configuration.
        backbone: The architecture the checkpoint was trained with.

    Raises:
        PTv3RuntimeUnavailableError: If the device is not CUDA, the output dimension is not the
            backbone's feature size, or the declared zero padding does not complete the input.
    """
    if not config.device.startswith("cuda"):
        raise PTv3RuntimeUnavailableError(
            f"the PTv3 backbone needs a CUDA device for sparse convolution, got {config.device!r}"
        )
    if config.output_dimension != backbone.output_channels:
        raise PTv3RuntimeUnavailableError(
            f"output_dimension is {config.output_dimension} but the checkpoint backbone produces "
            f"{backbone.output_channels} features"
        )
    if config.padding_channels + XYZ_CHANNELS != backbone.in_channels:
        raise PTv3RuntimeUnavailableError(
            f"padding_channels is {config.padding_channels} but the checkpoint backbone expects "
            f"{backbone.in_channels} input channels ({XYZ_CHANNELS} for XYZ)"
        )
