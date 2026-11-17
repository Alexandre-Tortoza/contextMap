"""Region evidence: discovered, normalized, and validated candidate regions.

Uma região é um candidato espacial em uma imagem que representa estrutura visual coesa.
Uma `Region2D` é a evidência canônica de uma região aceita após descoberta e normalização.

Regiões são descobertas por backends (SAM2, SAM3, Florence2) e refinadas opcionalmente.
Uma `RegionCandidate` é a saída bruta de descoberta; uma `Region2D` é o contrato validado.

Este contrato é o mesmo de antes da decomposição, restaurado deliberadamente. A reescrita
havia trocado `region_kind` (dica de categoria do backend, texto livre) por um enum `kind` de
*origem*, acrescentado `confidence: float` validado em `[0, 1]`, duplicado a geometria em
`bounds` e `bounding_box` e perdido `coordinate_convention` e a validação de caixa dentro da
imagem. A origem de uma região não é campo: ela está na identidade, derivada do request que a
produziu (`grounding_region_id_for`, `refined_region_id_for`). E `confidence` aqui violava
AGENTS.md §5 — score de detector, similaridade CLIP e qualidade geométrica não são
probabilidades intercambiáveis, e `test_grounding_confidence_guard` existe para afirmar que
nenhum contrato de grounding ou refinamento carrega um.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, NewType

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception.region_geometry import CoordinateConvention

if TYPE_CHECKING:
    from contextmap.region_perception.region_geometry import (
        BoundingBox,
        BoundingBox2D,
        InlineMask,
        RegionProvenance,
    )
    from contextmap.shared import BackendProvenance


RegionId = NewType("RegionId", str)
"""Identity of one region candidate within a discovery pass."""


@dataclass(frozen=True, kw_only=True)
class Region2D:
    """A validated 2D region candidate in image space.

    A region represents spatial extent and provenance of a candidate area in one frame.
    It does not assert semantic identity; that comes from later evidence (claims, features).

    There is no confidence field. A backend's native score has backend-specific semantics
    and is kept where it was measured (the discovery audit, the grounding and refinement
    diagnostics), never promoted to a number on the region that a consumer could read as a
    probability (AGENTS.md §5).

    Nor is there a field for the region's origin. A discovered, a grounded and a refined
    region are told apart by their identity, which is derived from the request that produced
    them, so they never collide and the origin is not a second source of truth.

    Attributes:
        region_id: Producer-qualified identity in the published regional artifact.
            Normalization initially uses a frame-local identity before publication.
        bounding_box: Image-space bounding geometry.
        provenance: Backend that produced this region.
        mask_reference: Reference to a mask payload (e.g. a debug/output file path) — never
            the raw mask array inline. ``None`` for a box-only region.
        region_kind: Free-form category hint from the producing backend, e.g. ``"object"``,
            ``"surface"``. ``None`` when unclassified. It is a hint, not a label.
        is_accepted: Whether this region survived normalization/merge and belongs to the
            frozen canonical set. No production discovery backend emits
            ``is_accepted=False``: the canonical path records each rejected candidate as a
            ``RejectedRegionCandidate`` in the frame's ``RegionDiscoveryAudit``, persisted in
            the perception run.
        rejection_reason: Human-readable reason; set only when ``is_accepted`` is ``False``.
        source_observation_id: Physical frame this region was found in, or ``None`` when
            unknown. It is never invented, so a consumer can tell "unknown" apart from
            "belongs to another observation".
        image_width: Image width in pixels, for geometry validation and context.
        image_height: Image height in pixels, for geometry validation and context.
        area_pixels: Computed pixel area of the region (key discovery evidence).
        contributor_candidate_ids: Candidate IDs that merged into this region (provenance).
        discovery_provenance: Full provenance of each contributor candidate.
        mask: Full-image binary mask, if available (important discovery evidence).
        coordinate_convention: Pixel coordinate convention the geometry is expressed in.
    """

    region_id: RegionId
    bounding_box: BoundingBox2D
    provenance: BackendProvenance
    mask_reference: str | None = None
    region_kind: str | None = None
    is_accepted: bool = True
    rejection_reason: str | None = None
    source_observation_id: SourceObservationId | None = None
    image_width: int | None = None
    image_height: int | None = None
    area_pixels: float | None = None
    contributor_candidate_ids: tuple[str, ...] = ()
    discovery_provenance: tuple[RegionProvenance, ...] = ()
    mask: InlineMask | None = None
    coordinate_convention: CoordinateConvention = CoordinateConvention.PIXEL_XY_TOP_LEFT

    def __post_init__(self) -> None:
        """Validate identity, rejection metadata, image space, and geometry.

        Raises:
            ValueError: If ``region_id`` is empty, ``rejection_reason`` is set while
                ``is_accepted`` is ``True``, the image dimensions are partial or not
                positive, ``bounding_box`` extends past the image when its dimensions are
                known, or the area, contributor ids, or inline mask are invalid.
        """
        if not str(self.region_id).strip():
            raise ValueError("region_id must not be empty")
        # NOTE: a ausência da observação de origem permanece None. Fabricar um id aqui
        # transformaria "não sabemos de onde veio" em "veio de outra observação", e os
        # consumidores que confrontam a região com o frame — evidence_views, por exemplo —
        # rejeitam um id divergente, mas aceitam None.
        if self.source_observation_id is not None and not str(self.source_observation_id).strip():
            raise ValueError("source_observation_id must not be empty when provided")
        if self.is_accepted and self.rejection_reason is not None:
            raise ValueError("rejection_reason must be None when is_accepted is True")
        if not self.is_accepted and self.rejection_reason is None:
            raise ValueError("rejection_reason must be set when is_accepted is False")
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError("image_width and image_height must be provided together")
        if (
            self.image_width is not None
            and self.image_height is not None
            and (self.image_width <= 0 or self.image_height <= 0)
        ):
            raise ValueError("region image dimensions must be positive")
        # Mesma regra de RegionCandidate: a caixa é semiaberta, então tocar a borda é válido.
        if (
            self.image_width is not None
            and self.image_height is not None
            and (
                self.bounding_box.x_max > self.image_width
                or self.bounding_box.y_max > self.image_height
            )
        ):
            raise ValueError(
                "bounding box extends outside the region image: "
                f"x_max={self.bounding_box.x_max}, y_max={self.bounding_box.y_max} for a "
                f"{self.image_width}x{self.image_height} image"
            )
        if self.area_pixels is not None and (
            not isfinite(self.area_pixels) or self.area_pixels <= 0
        ):
            raise ValueError("area_pixels must be positive and finite")
        if any(not candidate_id for candidate_id in self.contributor_candidate_ids):
            raise ValueError("contributor candidate ids must not be empty")
        if len(set(self.contributor_candidate_ids)) != len(self.contributor_candidate_ids):
            raise ValueError("contributor candidate ids must be unique")
        if (
            self.mask is not None
            and self.image_width is not None
            and (self.mask.width != self.image_width or self.mask.height != self.image_height)
        ):
            raise ValueError("inline mask dimensions must match the region image dimensions")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible canonical region representation.

        Unlike :func:`~contextmap.region_perception.encode_region`, this inlines the mask
        when the region carries one: it is the shape a diagnostic or a golden records, not
        the shape the run artifact persists (#378).
        """
        mask = None if self.mask is None else self.mask.to_dict()
        return {
            "region_id": str(self.region_id),
            "bounding_box": {
                "x": self.bounding_box.x,
                "y": self.bounding_box.y,
                "width": self.bounding_box.width,
                "height": self.bounding_box.height,
            },
            "provenance": {
                "backend_id": self.provenance.backend_id,
                "capability": self.provenance.capability,
                "provider": self.provenance.provider,
                "model": self.provenance.model,
                "version": self.provenance.version,
                "configuration_fingerprint": self.provenance.configuration_fingerprint,
            },
            "mask_reference": self.mask_reference,
            "region_kind": self.region_kind,
            "is_accepted": self.is_accepted,
            "rejection_reason": self.rejection_reason,
            "source_observation_id": (
                None if self.source_observation_id is None else str(self.source_observation_id)
            ),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "area_pixels": self.area_pixels,
            "contributor_candidate_ids": list(self.contributor_candidate_ids),
            "discovery_provenance": [item.to_dict() for item in self.discovery_provenance],
            "mask": mask,
            "coordinate_convention": self.coordinate_convention.value,
        }


@dataclass(frozen=True, kw_only=True)
class RegionCandidate:
    """A region candidate from automatic discovery (before normalization/acceptance).

    A candidate is the raw output of a discovery backend. It has not yet been
    validated, merged, or filtered.

    Note:
        This is the reduced definition. The one the discovery pipeline actually consumes is
        :class:`contextmap.region_perception.region_geometry.RegionCandidate`, which carries
        the observation, the image dimensions and the mask. See
        ``region_perception/docs/architecture.md`` for the declared duplication.
    """

    candidate_id: str
    bounds: BoundingBox
    confidence: float
    backend_id: str
    region_kind: str | None = None
    mask_id: str | None = None
