"""Tests for versioned semantic evidence-view policies and exact view materialization (#524)."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import replace

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import Region2D, RegionId
from contextmap.semantic_interpretation import (
    SEMANTIC_VIEW_POLICY_VERSION,
    SemanticContextBoundary,
)
from contextmap.semantic_interpretation.evidence_views import (
    SemanticMaskOutline,
    SemanticViewPolicy,
    check_view_policy_supported,
    materialize_region_views,
    materialize_scene_view,
)
from contextmap.semantic_interpretation.interpretation_request import (
    SemanticInterpretationMode,
    SemanticInterpreterCapabilities,
    VisualViewKind,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.models import BoundingBox2D
from contextmap.visual_perception.region_models import InlineMask

SOURCE = SourceObservationId("frame-0007")
SOURCE_SHA = "a" * 64
FULL = VisualViewKind.FULL_FRAME
MASKED = VisualViewKind.MASKED_SUBJECT
TIGHT = VisualViewKind.TIGHT_CROP
CONTEXT = VisualViewKind.CONTEXTUAL_CROP
OUTLINE_CROP = VisualViewKind.MASK_OUTLINE_CROP
OUTLINE_FULL = VisualViewKind.MASK_OUTLINE_FULL_FRAME
OUTLINE = SemanticMaskOutline(rgb=(255, 0, 255), width_px=1)


def _image(width: int = 8, height: int = 6) -> np.ndarray:
    """Every pixel is unique, so a wrong crop or fill can never look right by accident."""
    ys, xs = np.mgrid[0:height, 0:width]
    return np.stack([xs * 10, ys * 10, xs + ys], axis=-1).astype(np.uint8)


def _decode_png(payload: bytes) -> np.ndarray:
    """Decode the RGB8, filter-0 PNGs this policy version writes (test helper, not a codec)."""
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, {}
    idat = b""
    while offset < len(payload):
        (length,) = struct.unpack(">I", payload[offset : offset + 4])
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", payload[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(kind + data)
        if kind == b"IDAT":
            idat += data
        else:
            chunks[kind] = data
        offset += 12 + length
    width, height, depth, color, _, _, _ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, color) == (8, 2)
    raw = zlib.decompress(idat)
    stride = 1 + 3 * width
    rows = [raw[row * stride : (row + 1) * stride] for row in range(height)]
    assert all(row[0] == 0 for row in rows)
    return np.frombuffer(b"".join(row[1:] for row in rows), dtype=np.uint8).reshape(
        height, width, 3
    )


def _region(
    box: BoundingBox2D,
    *,
    mask: InlineMask | None = None,
    width: int | None = 8,
    height: int | None = 6,
) -> Region2D:
    return Region2D(
        region_id=RegionId("run-0001--frame-0007--region-0000"),
        bounding_box=box,
        provenance=BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="fake",
            model="m",
            version="1",
        ),
        source_observation_id=SOURCE,
        image_width=width,
        image_height=height,
        mask=mask,
    )


def _mask(pixels: set[tuple[int, int]], width: int = 8, height: int = 6) -> InlineMask:
    return InlineMask(
        np.array([[(x, y) in pixels for x in range(width)] for y in range(height)], dtype=bool)
    )


def _policy(*views: VisualViewKind, **overrides: object) -> SemanticViewPolicy:
    values: dict[str, object] = {"region_views": views}
    if MASKED in views:
        values["mask_fill_rgb"] = (1, 2, 3)
    if CONTEXT in views:
        values["context_margin_ratio"] = 0.5
    if OUTLINE_CROP in views or OUTLINE_FULL in views:
        values["mask_outline"] = OUTLINE
    values.update(overrides)
    return SemanticViewPolicy(**values)  # type: ignore[arg-type]


def _materialize(
    policy: SemanticViewPolicy, region: Region2D, image: np.ndarray | None = None
) -> tuple:
    return materialize_region_views(
        _image() if image is None else image,
        source_observation_id=SOURCE,
        source_image_sha256=SOURCE_SHA,
        region=region,
        policy=policy,
    )


BOX = BoundingBox2D(x=2, y=1, width=3, height=2)
MASK = _mask({(2, 1), (3, 1), (3, 2)})


class TestMaterialization:
    def test_identical_inputs_and_policy_give_byte_identical_content_addressed_views(
        self,
    ) -> None:
        policy = _policy(MASKED, TIGHT, CONTEXT)

        first = _materialize(policy, _region(BOX, mask=MASK))
        second = _materialize(policy, _region(BOX, mask=MASK))

        assert first == second
        for item in first:
            assert item.view.sha256 == hashlib.sha256(item.payload).hexdigest()

    def test_tight_crop_rounds_outward_and_clamps_at_the_image_edge(self) -> None:
        fractional = BoundingBox2D(x=1.5, y=0.25, width=2.2, height=1.1)
        at_edge = BoundingBox2D(x=6.5, y=4, width=5, height=5)

        (inner,) = _materialize(_policy(TIGHT), _region(fractional))
        # Uma Region2D com dimensões de imagem não passa da borda (#619); só uma região sem
        # elas (só caixa) chega ao corte com a caixa além da imagem.
        (edge,) = _materialize(_policy(TIGHT), _region(at_edge, width=None, height=None))

        assert inner.view.construction.pixel_bounds == (1, 0, 4, 2)
        assert np.array_equal(_decode_png(inner.payload), _image()[0:2, 1:4])
        assert edge.view.construction.pixel_bounds == (6, 4, 8, 6)
        assert np.array_equal(_decode_png(edge.payload), _image()[4:6, 6:8])

    def test_masked_subject_keeps_the_mask_and_fills_the_rest_with_the_policy_color(
        self,
    ) -> None:
        (masked,) = _materialize(_policy(MASKED), _region(BOX, mask=MASK))

        pixels = _decode_png(masked.payload)
        assert masked.view.construction.pixel_bounds == (2, 1, 5, 3)
        expected = np.full((2, 3, 3), (1, 2, 3), dtype=np.uint8)
        for x, y in ((2, 1), (3, 1), (3, 2)):
            expected[y - 1, x - 2] = _image()[y, x]
        assert np.array_equal(pixels, expected)

    def test_contextual_crop_expands_by_the_ratio_and_is_clamped_not_padded(self) -> None:
        (inner,) = _materialize(_policy(CONTEXT), _region(BOX))
        (edge,) = _materialize(
            _policy(CONTEXT), _region(BoundingBox2D(x=0, y=0, width=2, height=2))
        )

        # 0.5 x (3, 2) de margem por lado: x em [0.5, 6.5) -> [0, 7), y em [0, 4).
        assert inner.view.construction.pixel_bounds == (0, 0, 7, 4)
        assert np.array_equal(_decode_png(inner.payload), _image()[0:4, 0:7])
        assert edge.view.construction.pixel_bounds == (0, 0, 3, 3)

    def test_the_boundary_outlines_the_tight_box_around_the_subject_only(self) -> None:
        policy = _policy(
            CONTEXT,
            context_boundary=SemanticContextBoundary(rgb=(255, 0, 255), width_px=1),
        )

        (view,) = _materialize(policy, _region(BOX))

        pixels = _decode_png(view.payload)
        crop = _image()[0:4, 0:7]
        outline = {(x, 0) for x in range(1, 6)} | {(x, 3) for x in range(1, 6)}
        outline |= {(1, y) for y in range(4)} | {(5, y) for y in range(4)}
        for y in range(4):
            for x in range(7):
                expected = (255, 0, 255) if (x, y) in outline else tuple(crop[y, x])
                assert tuple(pixels[y, x]) == expected, (x, y)

    @pytest.mark.parametrize(
        "views",
        [
            (MASKED,),
            (TIGHT,),
            (CONTEXT,),
            (FULL, TIGHT),
            (MASKED, TIGHT),
            (MASKED, CONTEXT),
            (TIGHT, CONTEXT),
            (MASKED, TIGHT, CONTEXT),
            (FULL, MASKED, TIGHT, CONTEXT),
        ],
    )
    def test_each_variant_yields_exactly_its_declared_views_in_order(
        self, views: tuple[VisualViewKind, ...]
    ) -> None:
        materialized = _materialize(_policy(*views), _region(BOX, mask=MASK))

        assert tuple(item.view.kind for item in materialized) == views
        assert len({item.view.view_id for item in materialized}) == len(views)

    def test_every_view_keeps_its_observation_region_and_construction_lineage(self) -> None:
        policy = _policy(FULL, MASKED, TIGHT, CONTEXT)

        materialized = _materialize(policy, _region(BOX, mask=MASK))

        for item in materialized:
            view = item.view
            assert view.source_observation_id == SOURCE
            assert view.construction is not None
            assert view.construction.policy_fingerprint == policy.fingerprint()
            assert view.construction.source_image_sha256 == SOURCE_SHA
            assert view.payload_reference.startswith("outputs/semantic-views/")
            if view.kind is FULL:
                assert view.region_id is None
                assert view.construction.pixel_bounds == (0, 0, 8, 6)
            else:
                assert view.region_id == RegionId("run-0001--frame-0007--region-0000")

    def test_the_full_frame_of_a_region_request_is_the_scene_view(self) -> None:
        policy = _policy(FULL, TIGHT)

        scene = materialize_scene_view(
            _image(), source_observation_id=SOURCE, source_image_sha256=SOURCE_SHA, policy=policy
        )
        full, _ = _materialize(policy, _region(BOX))

        assert full == scene
        assert np.array_equal(_decode_png(scene.payload), _image())

    def test_a_masked_subject_needs_the_region_inline_mask(self) -> None:
        with pytest.raises(ValueError, match="inline mask"):
            _materialize(_policy(MASKED), _region(BOX))

    def test_a_region_outside_the_image_or_of_another_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            _materialize(_policy(TIGHT), _region(BoundingBox2D(x=9, y=1, width=1, height=1)))
        with pytest.raises(ValueError, match="dimensions"):
            _materialize(_policy(TIGHT), _region(BOX), image=_image(width=9))

    def test_a_region_of_another_observation_is_refused(self) -> None:
        other = replace(_region(BOX), source_observation_id=SourceObservationId("frame-0008"))

        with pytest.raises(ValueError, match="frame-0008"):
            _materialize(_policy(TIGHT), other)

    def test_only_an_rgb8_image_is_accepted(self) -> None:
        with pytest.raises(ValueError, match="RGB"):
            _materialize(_policy(TIGHT), _region(BOX), image=_image()[:, :, 0])


class TestPolicy:
    def test_the_fingerprint_is_deterministic_and_follows_every_parameter(self) -> None:
        base = _policy(MASKED, CONTEXT)

        assert base.fingerprint() == _policy(MASKED, CONTEXT).fingerprint()
        variants = {
            _policy(CONTEXT, MASKED).fingerprint(),
            _policy(MASKED, CONTEXT, mask_fill_rgb=(0, 0, 0)).fingerprint(),
            _policy(MASKED, CONTEXT, context_margin_ratio=0.25).fingerprint(),
            _policy(
                MASKED,
                CONTEXT,
                context_boundary=SemanticContextBoundary(rgb=(255, 0, 0), width_px=2),
            ).fingerprint(),
        }
        assert base.fingerprint() not in variants
        assert len(variants) == 4

    def test_the_document_records_the_fixed_rules_of_its_version(self) -> None:
        document = _policy(TIGHT).to_document()

        assert document["version"] == SEMANTIC_VIEW_POLICY_VERSION
        assert document["region_views"] == ["tight_crop"]
        assert {"crop_rounding", "resize", "encoding"} <= set(document)
        assert _policy(TIGHT).fingerprint().startswith("sha256:")

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"region_views": ()}, "at least one"),
            ({"region_views": (TIGHT, TIGHT)}, "unique"),
            ({"region_views": (FULL,)}, "region-bound"),
            ({"region_views": (MASKED,)}, "mask_fill_rgb"),
            ({"region_views": (CONTEXT,)}, "context_margin_ratio"),
            ({"region_views": (TIGHT,), "mask_fill_rgb": (0, 0, 0)}, "mask_fill_rgb"),
            ({"region_views": (TIGHT,), "context_margin_ratio": 0.5}, "context_margin_ratio"),
            (
                {
                    "region_views": (TIGHT,),
                    "context_boundary": SemanticContextBoundary(rgb=(0, 0, 0), width_px=1),
                },
                "context_boundary",
            ),
            ({"region_views": (MASKED,), "mask_fill_rgb": (0, 0, 256)}, "0..255"),
            ({"region_views": (CONTEXT,), "context_margin_ratio": 0.0}, "positive"),
            ({"region_views": (CONTEXT,), "context_margin_ratio": float("inf")}, "finite"),
        ],
    )
    def test_an_ambiguous_or_dead_parameter_is_refused(
        self, values: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SemanticViewPolicy(**values)  # type: ignore[arg-type]

    def test_a_boundary_has_a_color_and_a_positive_width(self) -> None:
        with pytest.raises(ValueError, match="width_px"):
            SemanticContextBoundary(rgb=(0, 0, 0), width_px=0)


class TestPreflight:
    @staticmethod
    def _single_region_filling_view() -> SemanticInterpreterCapabilities:
        """What Florence-2's region tasks declare."""
        return SemanticInterpreterCapabilities(
            supported_modes=frozenset({SemanticInterpretationMode.REGION}),
            supported_view_kinds=frozenset({TIGHT, MASKED}),
            accepts_visual_features=False,
            accepts_scene_context=False,
            max_visual_views=1,
        )

    def test_a_single_region_filling_view_is_accepted(self) -> None:
        check_view_policy_supported(_policy(TIGHT), self._single_region_filling_view())
        check_view_policy_supported(_policy(MASKED), self._single_region_filling_view())

    @pytest.mark.parametrize(
        ("views", "message"),
        [((MASKED, TIGHT), "at most 1"), ((CONTEXT,), "contextual_crop"), ((FULL, TIGHT), "full")],
    )
    def test_a_combination_the_interpreter_cannot_consume_fails_before_inference(
        self, views: tuple[VisualViewKind, ...], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            check_view_policy_supported(_policy(*views), self._single_region_filling_view())

    def test_region_views_are_not_checked_for_an_interpreter_without_region_mode(self) -> None:
        scene_only = replace(
            self._single_region_filling_view(),
            supported_modes=frozenset({SemanticInterpretationMode.SCENE}),
            supported_view_kinds=frozenset({FULL}),
        )

        check_view_policy_supported(_policy(MASKED, TIGHT, CONTEXT), scene_only)


def _ring(
    mask_pixels: set[tuple[int, int]], width_px: int, width: int, height: int
) -> set[tuple[int, int]]:
    """Oráculo por força bruta: pixels fora da máscara a distância de Chebyshev <= width_px."""
    return {
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in mask_pixels
        and any(max(abs(x - mx), abs(y - my)) <= width_px for mx, my in mask_pixels)
    }


# Uma máscara em L no interior de uma imagem 10x8 e outra encostada na borda esquerda: uma
# dilatação que desse a volta na imagem (np.roll) pintaria a borda direita.
WIDE = (10, 8)
L_MASK = {(4, 3), (5, 3), (5, 4)}
EDGE_MASK = {(0, 0), (0, 1)}


class TestMaskOutlineViews:
    """C6 arms B and D: the mask is shown by an outline drawn only outside it (D5)."""

    @pytest.mark.parametrize("mask_pixels", [L_MASK, EDGE_MASK])
    @pytest.mark.parametrize("width_px", [1, 2])
    def test_the_full_frame_outline_rings_the_mask_from_outside_at_the_declared_width(
        self, mask_pixels: set[tuple[int, int]], width_px: int
    ) -> None:
        image = _image(*WIDE)
        outline = SemanticMaskOutline(rgb=(255, 0, 255), width_px=width_px)
        region = _region(BOX, mask=_mask(mask_pixels, *WIDE), width=WIDE[0], height=WIDE[1])

        (view,) = _materialize(_policy(OUTLINE_FULL, mask_outline=outline), region, image)

        pixels = _decode_png(view.payload)
        ring = _ring(mask_pixels, width_px, *WIDE)
        assert ring
        assert pixels.shape == image.shape
        assert view.view.construction.pixel_bounds == (0, 0, *WIDE)
        for y in range(WIDE[1]):
            for x in range(WIDE[0]):
                expected = (255, 0, 255) if (x, y) in ring else tuple(image[y, x])
                assert tuple(pixels[y, x]) == expected, (x, y)

    def test_the_outline_never_paints_a_mask_pixel(self) -> None:
        # A cor do contorno é a cor do próprio sujeito: se o contorno invadisse a máscara, os
        # pixels do sujeito mudariam de cor e contaminariam o atributo "color" (D5).
        image = _image(*WIDE)
        subject_rgb = tuple(int(value) for value in image[3, 4])
        outline = SemanticMaskOutline(rgb=subject_rgb, width_px=3)  # type: ignore[arg-type]
        region = _region(BOX, mask=_mask(L_MASK, *WIDE), width=WIDE[0], height=WIDE[1])

        views = _materialize(
            _policy(OUTLINE_CROP, OUTLINE_FULL, mask_outline=outline), region, image
        )

        full = _decode_png(views[1].payload)
        for x, y in L_MASK:
            assert tuple(full[y, x]) == tuple(image[y, x])
        changed = {(x, y) for y, x in zip(*np.nonzero((full != image).any(axis=-1)), strict=True)}
        assert changed <= _ring(L_MASK, 3, *WIDE)

    def test_the_crop_outline_is_the_tight_box_with_the_ring_clipped_to_it(self) -> None:
        image = _image(*WIDE)
        box = BoundingBox2D(x=3, y=2, width=4, height=3)  # janela (3, 2, 7, 5): 4 x 3 px.
        region = _region(box, mask=_mask(L_MASK, *WIDE), width=WIDE[0], height=WIDE[1])

        (view,) = _materialize(_policy(OUTLINE_CROP), region, image)

        pixels = _decode_png(view.payload)
        assert view.view.construction.pixel_bounds == (3, 2, 7, 5)
        assert pixels.shape == (3, 4, 3)
        ring = _ring(L_MASK, 1, *WIDE)
        for y in range(2, 5):
            for x in range(3, 7):
                expected = (255, 0, 255) if (x, y) in ring else tuple(image[y, x])
                assert tuple(pixels[y - 2, x - 3]) == expected, (x, y)

    def test_identical_inputs_give_byte_identical_outline_views(self) -> None:
        region = _region(BOX, mask=_mask(L_MASK, *WIDE), width=WIDE[0], height=WIDE[1])
        policy = _policy(OUTLINE_CROP, OUTLINE_FULL)

        first = _materialize(policy, region, _image(*WIDE))
        second = _materialize(policy, region, _image(*WIDE))

        assert first == second
        assert [item.view.kind for item in first] == [OUTLINE_CROP, OUTLINE_FULL]
        for item in first:
            assert item.view.sha256 == hashlib.sha256(item.payload).hexdigest()
            assert item.view.region_id == region.region_id
            assert item.view.construction.policy_fingerprint == policy.fingerprint()
        assert len({item.view.view_id for item in first}) == 2
        assert len({item.view.payload_reference for item in first}) == 2

    def test_the_outline_views_join_the_other_views_in_declared_order(self) -> None:
        views = (TIGHT, OUTLINE_FULL, MASKED, OUTLINE_CROP, FULL)

        materialized = _materialize(_policy(*views), _region(BOX, mask=MASK))

        assert tuple(item.view.kind for item in materialized) == views

    @pytest.mark.parametrize("kind", [OUTLINE_CROP, OUTLINE_FULL])
    def test_an_outline_view_needs_the_region_inline_mask(self, kind: VisualViewKind) -> None:
        # Uma caixa não tem contorno de máscara; inventá-lo da caixa fabricaria evidência.
        with pytest.raises(ValueError, match=f"{kind.value} needs the region's inline mask"):
            _materialize(_policy(kind), _region(BOX))

    @pytest.mark.parametrize(
        ("views", "required"),
        [
            ((TIGHT,), False),
            ((CONTEXT,), False),
            ((FULL, TIGHT), False),
            ((MASKED,), True),
            ((OUTLINE_CROP,), True),
            ((OUTLINE_FULL,), True),
            ((TIGHT, OUTLINE_FULL), True),
        ],
    )
    def test_a_policy_with_an_outline_view_requires_the_region_mask(
        self, views: tuple[VisualViewKind, ...], required: bool
    ) -> None:
        assert _policy(*views).requires_region_mask is required

    def test_a_request_of_outline_views_round_trips_through_its_audit_record(self) -> None:
        from contextmap.semantic_interpretation import (
            SemanticInterpretationRunId,
            decode_semantic_request,
            encode_semantic_request,
        )
        from contextmap.semantic_interpretation.interpretation_request import (
            SemanticInterpretationRequest,
            SemanticRequestId,
        )

        region = _region(BOX, mask=MASK)
        views = _materialize(_policy(OUTLINE_CROP, OUTLINE_FULL), region)
        request = SemanticInterpretationRequest(
            request_id=SemanticRequestId("request-0001"),
            source_observation_id=SOURCE,
            semantic_interpretation_run_id=SemanticInterpretationRunId("run-0001--frame-0007"),
            mode=SemanticInterpretationMode.REGION,
            region_id=region.region_id,
            visual_views=tuple(item.view for item in views),
            prompt_template_id="region/v1",
            requested_output_schema="semantic-response/1",
            configuration_fingerprint="sha256:config",
        )

        encoded = encode_semantic_request(request)

        assert [view["kind"] for view in encoded["visual_views"]] == [
            "mask_outline_crop",
            "mask_outline_full_frame",
        ]
        assert decode_semantic_request(encoded) == request


class TestMaskOutlinePolicy:
    def test_an_outline_full_frame_alone_is_a_region_bound_policy(self) -> None:
        # Arm D é uma view só: o frame inteiro identifica a região pelo contorno.
        policy = SemanticViewPolicy(region_views=(OUTLINE_FULL,), mask_outline=OUTLINE)

        assert policy.region_views == (OUTLINE_FULL,)

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"region_views": (OUTLINE_CROP,)}, "mask_outline"),
            ({"region_views": (OUTLINE_FULL,)}, "mask_outline"),
            ({"region_views": (TIGHT, OUTLINE_FULL)}, "mask_outline"),
            ({"region_views": (TIGHT,), "mask_outline": OUTLINE}, "mask_outline"),
            (
                {"region_views": (MASKED,), "mask_fill_rgb": (0, 0, 0), "mask_outline": OUTLINE},
                "mask_outline",
            ),
        ],
    )
    def test_a_missing_or_dead_outline_is_refused(
        self, values: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SemanticViewPolicy(**values)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            ({"rgb": (0, 0, 256), "width_px": 1}, "0..255"),
            ({"rgb": (0, 0), "width_px": 1}, "0..255"),
            ({"rgb": (0, 0, 0), "width_px": 0}, "width_px"),
        ],
    )
    def test_an_outline_has_a_color_and_a_positive_width(
        self, values: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SemanticMaskOutline(**values)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("policy", "fingerprint"),
        [
            (
                _policy(TIGHT),
                "sha256:e9ed1c324a30c359211ef8e51bcb8fd9e99526fcc582744256145ad204f77233",
            ),
            (
                _policy(
                    MASKED,
                    TIGHT,
                    CONTEXT,
                    context_boundary=SemanticContextBoundary(rgb=(255, 0, 255), width_px=1),
                ),
                "sha256:4a6c40fecf884d1a1c6763ba4b8c4561dbca8d2d2829c7dd3dc4d4c9d64cae14",
            ),
            (
                _policy(FULL, MASKED, TIGHT, CONTEXT),
                "sha256:72718cca719db01b19472951b08b9eb5928a9d7731ddd26ee97033739a480fbf",
            ),
        ],
    )
    def test_a_policy_without_outline_keeps_its_identity_from_before_the_outline_views(
        self, policy: SemanticViewPolicy, fingerprint: str
    ) -> None:
        """The outline is additive: runs made before it keep their stage identity (C6)."""
        assert "mask_outline" not in policy.to_document()
        assert policy.fingerprint() == fingerprint

    def test_views_without_outline_keep_their_bytes_from_before_the_outline_views(self) -> None:
        """Arm A (``tight_crop``) must stay byte-identical to the E0 baseline (C6)."""
        materialized = _materialize(_policy(FULL, MASKED, TIGHT, CONTEXT), _region(BOX, mask=MASK))

        assert {item.view.kind.value: item.view.sha256 for item in materialized} == {
            "full_frame": "8483d3c6da00c70ac8cff83b3b67a5c7eb60dfedcfbae1e873b3c729cb161538",
            "masked_subject": "3b26d26161b047ac67ff9d096354109e6e48814e7706ab90369e4ec8738795b1",
            "tight_crop": "edcfb687ac91115599c7506cb7bfe9d0532e7b3e20cf7a604157f7b90088308b",
            "contextual_crop": "f05bc4b1c47e37da1e1cf55d04eba0e05949d3181062dbe9b995fb90a76ba43d",
        }

    def test_the_outline_is_recorded_and_follows_the_fingerprint(self) -> None:
        base = _policy(OUTLINE_CROP)

        assert base.to_document()["mask_outline"] == {"rgb": [255, 0, 255], "width_px": 1}
        variants = {
            _policy(OUTLINE_FULL).fingerprint(),
            _policy(
                OUTLINE_CROP, mask_outline=SemanticMaskOutline(rgb=(0, 255, 0), width_px=1)
            ).fingerprint(),
            _policy(
                OUTLINE_CROP, mask_outline=SemanticMaskOutline(rgb=(255, 0, 255), width_px=2)
            ).fingerprint(),
            _policy(TIGHT).fingerprint(),
        }
        assert base.fingerprint() == _policy(OUTLINE_CROP).fingerprint()
        assert base.fingerprint() not in variants
        assert len(variants) == 4

    @pytest.mark.parametrize("kind", [OUTLINE_CROP, OUTLINE_FULL])
    def test_an_interpreter_that_does_not_declare_an_outline_view_refuses_it_before_inference(
        self, kind: VisualViewKind
    ) -> None:
        with pytest.raises(ValueError, match=kind.value):
            check_view_policy_supported(_policy(kind), TestPreflight._single_region_filling_view())


class _CountingResolver:
    """Resolves one reference to one mask and counts how often pixels were asked for."""

    def __init__(self, reference: str, mask: InlineMask) -> None:
        self._reference, self._mask = reference, mask
        self.loads = 0

    def resolve_mask(self, reference: str) -> InlineMask:
        assert reference == self._reference
        self.loads += 1
        return self._mask


class TestMasksResolvedFromAReference:
    """F02b: a region externalized by its producer carries only ``mask_reference``."""

    REFERENCE = "outputs/masks/frame-0007/region-0000.npy"

    def _referenced(self) -> tuple[Region2D, _CountingResolver]:
        region = replace(_region(BOX, mask=MASK), mask=None, mask_reference=self.REFERENCE)
        return region, _CountingResolver(self.REFERENCE, MASK)

    @pytest.mark.parametrize(
        "views",
        [(MASKED,), (OUTLINE_FULL,), (OUTLINE_CROP,), (MASKED, OUTLINE_FULL, TIGHT)],
    )
    def test_the_views_equal_the_ones_built_from_the_inline_mask(
        self, views: tuple[VisualViewKind, ...]
    ) -> None:
        policy = _policy(*views)
        region, resolver = self._referenced()

        from_reference = materialize_region_views(
            _image(),
            source_observation_id=SOURCE,
            source_image_sha256=SOURCE_SHA,
            region=region,
            policy=policy,
            mask_resolver=resolver,
        )

        assert from_reference == _materialize(policy, _region(BOX, mask=MASK))

    def test_the_pixels_are_loaded_once_per_region_however_many_views_need_them(self) -> None:
        region, resolver = self._referenced()

        materialize_region_views(
            _image(),
            source_observation_id=SOURCE,
            source_image_sha256=SOURCE_SHA,
            region=region,
            policy=_policy(MASKED, OUTLINE_FULL),
            mask_resolver=resolver,
        )

        assert resolver.loads == 1

    def test_a_policy_without_mask_views_never_loads_the_pixels(self) -> None:
        region, resolver = self._referenced()

        materialize_region_views(
            _image(),
            source_observation_id=SOURCE,
            source_image_sha256=SOURCE_SHA,
            region=region,
            policy=_policy(TIGHT),
            mask_resolver=resolver,
        )

        assert resolver.loads == 0

    def test_a_box_only_region_still_refuses_a_masked_subject(self) -> None:
        _, resolver = self._referenced()

        with pytest.raises(ValueError, match="needs the region's inline mask"):
            materialize_region_views(
                _image(),
                source_observation_id=SOURCE,
                source_image_sha256=SOURCE_SHA,
                region=_region(BOX),
                policy=_policy(MASKED),
                mask_resolver=resolver,
            )
