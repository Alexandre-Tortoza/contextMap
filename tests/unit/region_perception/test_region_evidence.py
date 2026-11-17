"""Invariantes de `Region2D`, o contrato de evidência de região.

Esta suíte cobria o contrato que a decomposição escreveu, não o que ela devia mover. Foi
reescrita junto da restauração: o que mudou de verdade, e por quê, está no docstring de
`region_evidence.py` e em `docs/perception-dag-decomposition.md`.

Os dois invariantes que estes testes protegem e que a reescrita havia perdido:

- **região não carrega confidence.** Score de detector, similaridade CLIP e qualidade
  geométrica não são probabilidades intercambiáveis (AGENTS.md §5). O score nativo fica onde
  foi medido — auditoria de descoberta, diagnósticos de grounding e refinamento.
- **a caixa não escapa da imagem.** Com as dimensões conhecidas, uma caixa que passa da borda
  é erro de geometria, não um dado a propagar.
"""

from __future__ import annotations

import dataclasses

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox,
    BoundingBox2D,
    CoordinateConvention,
    InlineMask,
)
from contextmap.region_perception.region_evidence import Region2D, RegionCandidate, RegionId
from contextmap.shared import BackendProvenance

_PROVENANCE = BackendProvenance(
    backend_id="test_backend",
    capability="region_discovery",
    provider="test",
    model="test-v1",
    version="1",
    configuration_fingerprint="sha256:test",
)


def _mask(width: int, height: int) -> InlineMask:
    """Build an all-true mask of the given size."""
    import numpy as np

    return InlineMask(np.ones((height, width), dtype=np.bool_))


def _region(**overrides: object) -> Region2D:
    """Build an accepted region in a 640x480 image, varying only what a test varies."""
    fields: dict[str, object] = {
        "region_id": RegionId("region-0001"),
        "bounding_box": BoundingBox2D(x=10.0, y=20.0, width=30.0, height=40.0),
        "provenance": _PROVENANCE,
        "source_observation_id": SourceObservationId("frame-0001"),
        "image_width": 640,
        "image_height": 480,
    }
    fields.update(overrides)
    return Region2D(**fields)  # type: ignore[arg-type]


class TestIdentity:
    """A region without identity cannot be referenced by any downstream evidence."""

    @pytest.mark.parametrize("blank", ["", " ", "\t"])
    def test_an_empty_region_id_is_refused(self, blank: str) -> None:
        with pytest.raises(ValueError, match="region_id must not be empty"):
            _region(region_id=RegionId(blank))

    def test_an_absent_origin_stays_absent(self) -> None:
        """Omission means "the origin frame is unknown", never another observation."""
        assert _region(source_observation_id=None).source_observation_id is None

    def test_a_blank_origin_is_refused(self) -> None:
        """Absent is a valid state; blank is a malformed id."""
        with pytest.raises(ValueError, match="source_observation_id must not be empty"):
            _region(source_observation_id=SourceObservationId(" "))


class TestNoConfidence:
    """The contract has no confidence field, and that is the point (AGENTS.md §5)."""

    def test_a_region_has_no_confidence_field(self) -> None:
        names = {field.name for field in dataclasses.fields(Region2D)}
        assert "confidence" not in names

    def test_a_confidence_cannot_be_attached(self) -> None:
        with pytest.raises(TypeError):
            _region(confidence=0.95)


class TestGeometry:
    """Geometry is validated against the image whenever the image is known."""

    def test_the_bounding_box_must_stay_inside_the_image(self) -> None:
        with pytest.raises(ValueError, match="bounding box extends outside the region image"):
            _region(bounding_box=BoundingBox2D(x=600.0, y=0.0, width=100.0, height=10.0))

    def test_touching_the_edge_is_valid(self) -> None:
        """The box is half-open, so x_max == image_width is inside."""
        region = _region(bounding_box=BoundingBox2D(x=600.0, y=0.0, width=40.0, height=10.0))
        assert region.bounding_box.x_max == 640

    def test_image_dimensions_come_in_pairs(self) -> None:
        with pytest.raises(ValueError, match="must be provided together"):
            _region(image_width=640, image_height=None)

    def test_image_dimensions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be positive"):
            _region(image_width=0, image_height=0)

    def test_a_region_without_a_known_image_skips_the_box_rule(self) -> None:
        """A region may exist before its frame's dimensions are known."""
        region = _region(image_width=None, image_height=None)
        assert (region.image_width, region.image_height) == (None, None)

    def test_the_coordinate_convention_is_part_of_the_contract(self) -> None:
        """Pixel geometry without its convention is not interpretable."""
        assert _region().coordinate_convention is CoordinateConvention.PIXEL_XY_TOP_LEFT


class TestArea:
    """Area is discovery evidence; a non-positive or non-finite one is malformed."""

    @pytest.mark.parametrize("area", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_non_positive_or_non_finite_area_is_refused(self, area: float) -> None:
        with pytest.raises(ValueError, match="area_pixels must be positive and finite"):
            _region(area_pixels=area)

    def test_an_absent_area_is_accepted(self) -> None:
        assert _region().area_pixels is None


class TestContributors:
    """Merge provenance must name each contributor exactly once."""

    def test_an_empty_contributor_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="contributor candidate ids must not be empty"):
            _region(contributor_candidate_ids=("cand-1", ""))

    def test_a_repeated_contributor_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="contributor candidate ids must be unique"):
            _region(contributor_candidate_ids=("cand-1", "cand-1"))


class TestAcceptance:
    """Rejection is explicit in both directions."""

    def test_an_accepted_region_carries_no_reason(self) -> None:
        with pytest.raises(ValueError, match="rejection_reason must be None"):
            _region(is_accepted=True, rejection_reason="merged away")

    def test_a_rejected_region_must_carry_a_reason(self) -> None:
        with pytest.raises(ValueError, match="rejection_reason must be set"):
            _region(is_accepted=False, rejection_reason=None)

    def test_a_rejected_region_with_a_reason_is_valid(self) -> None:
        region = _region(is_accepted=False, rejection_reason="below the area threshold")
        assert (region.is_accepted, region.rejection_reason) == (
            False,
            "below the area threshold",
        )


class TestInlineMask:
    """An inline mask is full-image evidence, so it must match the image it came from."""

    def test_a_mask_of_another_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="inline mask dimensions must match"):
            _region(mask=_mask(32, 32))

    def test_a_matching_mask_is_accepted(self) -> None:
        mask = _mask(640, 480)
        assert _region(mask=mask).mask is mask


class TestSerialization:
    """``to_dict`` is the canonical record a diagnostic or a golden keeps."""

    def test_the_record_carries_the_hint_and_the_convention(self) -> None:
        record = _region(region_kind="surface").to_dict()
        assert record["region_kind"] == "surface"
        assert record["coordinate_convention"] == CoordinateConvention.PIXEL_XY_TOP_LEFT.value

    def test_an_unknown_origin_travels_as_null(self) -> None:
        """So a decoder gives it back absent instead of inventing one."""
        assert _region(source_observation_id=None).to_dict()["source_observation_id"] is None

    def test_a_known_origin_travels_as_text(self) -> None:
        assert _region().to_dict()["source_observation_id"] == "frame-0001"

    def test_the_record_has_no_confidence_key(self) -> None:
        assert "confidence" not in _region().to_dict()


class TestRegionCandidate:
    """The reduced candidate: raw backend output, before normalization."""

    def test_a_candidate_carries_its_backend_score_where_it_was_measured(self) -> None:
        candidate = RegionCandidate(
            candidate_id="cand-0001",
            bounds=BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0),
            confidence=0.95,
            backend_id="sam2",
        )
        assert (candidate.candidate_id, candidate.confidence) == ("cand-0001", 0.95)
        assert (candidate.region_kind, candidate.mask_id) == (None, None)
