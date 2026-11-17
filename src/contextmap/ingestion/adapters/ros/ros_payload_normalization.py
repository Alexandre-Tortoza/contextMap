"""Normalização de layouts binários usados pelos payloads ROS suportados.

As funções removem padding, ajustam byte order de escalares conhecidos e enriquecem provenance
com as transformações realmente aplicadas. A normalização é lossless dentro dos formatos
suportados e não altera o significado físico da observação.
"""

from __future__ import annotations

from dataclasses import replace

from contextmap.ingestion.canonical_observations import SourceProvenance


def _remove_row_padding(
    data: bytes,
    *,
    height: int,
    row_step: int,
    row_payload_size: int,
    payload_name: str,
) -> bytes:
    """Validate a ROS row layout and return a tightly packed payload.

    A payload without row padding is already tightly packed and is returned as
    the same object, never copied.
    """
    if height <= 0 or row_payload_size <= 0:
        raise ValueError(f"{payload_name} dimensions must be positive")
    if row_step < row_payload_size:
        raise ValueError(
            f"{payload_name} row step {row_step} is smaller than payload {row_payload_size}"
        )
    expected_size = height * row_step
    if len(data) != expected_size:
        raise ValueError(
            f"{payload_name} data size {len(data)} does not match height*row_step={expected_size}"
        )
    if row_step == row_payload_size:
        return data
    return b"".join(
        data[row_index * row_step : row_index * row_step + row_payload_size]
        for row_index in range(height)
    )


def _swap_fixed_width_values(data: bytes, *, value_size: int) -> bytes:
    """Reverse every fixed-width scalar in a tightly packed byte buffer."""
    return b"".join(
        data[offset : offset + value_size][::-1] for offset in range(0, len(data), value_size)
    )


def _with_raw_metadata(provenance: SourceProvenance, **metadata: object) -> SourceProvenance:
    """Return provenance enriched with primitive normalization metadata."""
    return replace(provenance, raw_metadata={**provenance.raw_metadata, **metadata})
