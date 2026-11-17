"""Resíduos contra correspondências geométricas de referência confiáveis.

Distingue referências não avaliadas de projeções inválidas e mede erro em pixels brutos.
Não cria correspondências nem estima a referência a partir da própria associação.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from contextmap.sensor_association.projection_chain.frame_projection import FrameProjection
from contextmap.sensor_association.quality import ReprojectionStatistics

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True, eq=False)
class TrustedCorrespondences:
    """Reference pairs of map geometry and the pixel where it was really observed.

    They must come from a trusted source (surveyed landmarks, a calibration target); they are
    what makes a reprojection residual meaningful, and nothing here creates them.

    Attributes:
        reference_id: Identity of the trusted correspondence set.
        geometry_indices: ``(K,)`` global indices of the map geometry elements, integer
            dtype, the same identity :func:`~contextmap.geometric_mapping.geometry_id_for`
            uses.
        observed_pixels: ``(K, 2)`` raw-image pixels where each element was observed.
    """

    reference_id: str
    geometry_indices: NDArray[Any]
    observed_pixels: NDArray[Any]

    def __post_init__(self) -> None:
        """Validate the reference.

        Raises:
            ValueError: If the reference is unnamed or empty, the arrays disagree in shape, a
                pixel is not finite, or the indices are not an integer array. A float index
                names no map element, and because the lookup matches by equality it would be
                counted as geometry the candidate policy did not evaluate: a malformed
                reference would read as an effect of the candidate policy, in the one metric
                that measures that policy. ``NaN`` is worse, since it also defeats the bounds
                check in :func:`reprojection_statistics`.
        """
        import numpy as np

        if not self.reference_id:
            raise ValueError("reference_id must not be empty")
        if not np.issubdtype(self.geometry_indices.dtype, np.integer):
            raise ValueError(
                f"geometry_indices must be an integer array, got dtype "
                f"{self.geometry_indices.dtype}: a trusted reference names map geometry by its "
                "global index, never by a value that no element can have"
            )
        count = self.geometry_indices.shape[0] if self.geometry_indices.ndim == 1 else -1
        if count != self.observed_pixels.shape[0] or self.observed_pixels.shape[1:] != (2,):
            raise ValueError(
                f"geometry_indices must have shape (K,) and observed_pixels shape (K, 2), got "
                f"{self.geometry_indices.shape} and {self.observed_pixels.shape}"
            )
        if count < 1:
            raise ValueError("a trusted reference needs at least one correspondence")
        if not np.isfinite(self.observed_pixels).all():
            raise ValueError("observed_pixels must be finite")


def evaluated_correspondences(
    frame: FrameProjection, correspondences: TrustedCorrespondences
) -> int:
    """Count the reference correspondences whose geometry this frame evaluated.

    A frame only evaluates the geometry its candidate policy selected, so a reference may name
    map elements the frame never looked at. This is the one definition of that count; the
    residual statistics and the diagnostics both read it rather than each deriving their own.
    """
    _, found = frame.rows_for(correspondences.geometry_indices)
    return int(found.sum())


def reprojection_statistics(
    frame: FrameProjection, correspondences: TrustedCorrespondences
) -> ReprojectionStatistics | None:
    """Measure the reprojection residual of a projected frame against a trusted reference.

    The residual of a correspondence is the distance, in raw-image pixels, between where the
    frame projects the geometry and where it was observed. A correspondence the camera model
    cannot project is counted as invalid and contributes no residual. The 95th percentile uses
    linear interpolation between order statistics.

    A correspondence names geometry by its **global** map index, so it is looked up in the
    frame's candidate rows rather than used as one: geometry the frame's candidate policy
    did not select is reported as unevaluated and contributes no residual.

    Args:
        frame: The projection of one camera frame's candidate geometry.
        correspondences: The trusted reference.

    Returns:
        The statistics, or ``None`` when no correspondence could be projected.

    Raises:
        ValueError: If a correspondence names geometry outside the map.
    """
    import numpy as np

    indices = correspondences.geometry_indices
    if indices.min() < 0 or indices.max() >= frame.map_point_count:
        raise ValueError(
            f"geometry_indices must lie within the map's {frame.map_point_count} elements"
        )
    count = int(indices.shape[0])
    rows, evaluated = frame.rows_for(indices)
    valid = np.zeros(count, dtype=bool)
    valid[evaluated] = frame.projectable[rows[evaluated]]
    valid_count = int(valid.sum())
    if valid_count == 0:
        return None
    projected = frame.raw_pixels[rows[valid]]
    residuals = np.hypot(*(projected - correspondences.observed_pixels[valid]).T)
    return ReprojectionStatistics(
        reference_id=correspondences.reference_id,
        correspondence_count=count,
        invalid_count=int((evaluated & ~valid).sum()),
        unevaluated_count=count - int(evaluated.sum()),
        mean_px=float(residuals.mean()),
        median_px=float(np.median(residuals)),
        p95_px=float(np.percentile(residuals, 95)),
        max_px=float(residuals.max()),
    )
