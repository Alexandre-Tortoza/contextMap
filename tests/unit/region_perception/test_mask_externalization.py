"""The run writer stages a region's mask at the producer boundary (finding F02b).

``externalize_region_mask`` persists an ``InlineMask`` into the run's staging mask store and
hands back the same region carrying ``mask_reference`` instead of pixels; the reference is the
exact one ``add_result`` has always produced, so the written artifact does not change.
"""

import gc
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    InlineMask,
    Region2D,
    RegionId,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
    region_mask_pixels,
)
from contextmap.region_perception import (
    RegionPerceptionRunError as RunArtifactError,
)
from contextmap.region_perception.artifacts.mask_store import (
    MaskPayloadIntegrityError,
    MaskStoreError,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage

from .test_artifacts import _manifest


@dataclass(frozen=True)
class _MaskedFrame:
    source_observation_id: SourceObservationId
    regions: tuple[Region2D, ...]


def _full_frame_mask(*, width: int, height: int) -> InlineMask:
    return InlineMask(np.ones((height, width), dtype=bool))


def _masked_result(frame: str, mask: InlineMask) -> _MaskedFrame:
    obs = SourceObservationId(frame)
    return _MaskedFrame(
        obs,
        (
            Region2D(
                region_id=RegionId(f"{frame}--region-0001"),
                source_observation_id=obs,
                mask=mask,
                bounding_box=BoundingBox2D(x=0, y=0, width=mask.width, height=mask.height),
                provenance=BackendProvenance(
                    backend_id="fake",
                    capability="region_discovery",
                    provider="fake",
                    model="fake",
                    version="1",
                    configuration_fingerprint="test",
                ),
            ),
        ),
    )


def _run_dir(root: Path) -> Path:
    return root / "run"


def _staging_dir(root: Path) -> Path:
    return next(root.glob(".tmp-run-masks-*"))


def _write_run(root: Path) -> RegionPerceptionRunWriter:
    writer = RegionPerceptionRunWriter(_run_dir(root))
    writer.write_manifest(_manifest(source_observation_ids=()))
    return writer


def _add_frame(writer: RegionPerceptionRunWriter, frame: _MaskedFrame) -> None:
    assert writer.manifest is not None
    writer.write_manifest(
        replace(
            writer.manifest,
            source_observation_ids=(
                *writer.manifest.source_observation_ids,
                str(frame.source_observation_id),
            ),
        )
    )
    width = frame.regions[0].bounding_box.width if frame.regions else 4
    height = frame.regions[0].bounding_box.height if frame.regions else 3
    writer.write_frame(
        PreparedImage(
            source_observation_id=frame.source_observation_id,
            payload_reference=f"source-observation:{frame.source_observation_id}",
            width=int(width),
            height=int(height),
        )
    )
    for region in frame.regions:
        writer.write_region(writer.externalize_region_mask(frame.source_observation_id, region))


FRAME = SourceObservationId("frame-0001")


def _live_inline_masks() -> int:
    gc.collect()
    return sum(1 for item in gc.get_objects() if type(item) is InlineMask)


def _mask(width: int = 4, height: int = 3) -> InlineMask:
    return _full_frame_mask(width=width, height=height)


def _masked_region(frame: str = "frame-0001", mask: InlineMask | None = None) -> Region2D:
    return _masked_result(frame, mask or _mask()).regions[0]


def test_externalizing_returns_the_region_with_a_reference_and_no_pixels(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    region = _masked_region()

    externalized = writer.externalize_region_mask(FRAME, region)

    assert externalized.mask is None
    assert externalized.mask_reference == "outputs/masks/frame-0001/frame-0001--region-0001.npy"
    assert replace(externalized, mask=region.mask, mask_reference=None) == replace(
        region, mask_reference=None
    )
    assert (_staging_dir(tmp_path) / externalized.mask_reference).is_file()


def test_the_pixels_die_once_the_caller_drops_the_original_region(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    before = _live_inline_masks()
    region = _masked_region()
    assert _live_inline_masks() == before + 1

    externalized = writer.externalize_region_mask(FRAME, region)
    del region

    assert _live_inline_masks() == before
    assert externalized.mask is None


def test_a_box_only_region_is_returned_untouched(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    box_only = replace(_masked_region(), mask=None)

    assert writer.externalize_region_mask(FRAME, box_only) is box_only
    assert list(tmp_path.iterdir()) == []


def test_the_staged_mask_is_resolvable_by_reference_during_the_run(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    mask = _mask()
    externalized = writer.externalize_region_mask(FRAME, _masked_region(mask=mask))

    assert externalized.mask_reference is not None
    assert writer.resolve_mask(externalized.mask_reference) == mask
    assert region_mask_pixels(externalized, writer) == mask


def test_region_mask_pixels_prefers_the_inline_mask_and_knows_a_box_only_region() -> None:
    region = _masked_region()

    assert region_mask_pixels(region, None) is region.mask
    assert region_mask_pixels(replace(region, mask=None), None) is None


def test_a_reference_without_a_resolver_is_an_explicit_error() -> None:
    referenced = replace(_masked_region(), mask=None, mask_reference="outputs/masks/x/y.npy")

    with pytest.raises(MaskStoreError, match="no resolver"):
        region_mask_pixels(referenced, None)


@pytest.mark.parametrize(
    "reference",
    [
        "outputs/masks/frame-9999/region-0001.npy",
        "elsewhere/frame-0001/frame-0001--region-0001.npy",
    ],
)
def test_resolving_a_reference_this_run_did_not_stage_fails_explicitly(
    tmp_path: Path, reference: str
) -> None:
    writer = _write_run(tmp_path)
    writer.externalize_region_mask(FRAME, _masked_region())

    with pytest.raises(MaskStoreError, match=r"not staged|not a mask of this run"):
        writer.resolve_mask(reference)


def test_resolving_before_any_mask_was_staged_fails_explicitly(tmp_path: Path) -> None:
    with pytest.raises(MaskStoreError, match="not a mask of this run"):
        _write_run(tmp_path).resolve_mask("outputs/masks/frame-0001/frame-0001--region-0001.npy")


def test_a_corrupted_staged_mask_fails_the_hash_check(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    externalized = writer.externalize_region_mask(FRAME, _masked_region())
    assert externalized.mask_reference is not None
    (_staging_dir(tmp_path) / externalized.mask_reference).write_bytes(b"corrupted")

    with pytest.raises(MaskPayloadIntegrityError, match="content hash mismatch"):
        writer.resolve_mask(externalized.mask_reference)


def test_add_result_keeps_an_externalized_reference_without_persisting_twice(
    tmp_path: Path,
) -> None:
    writer = _write_run(tmp_path)
    result = _masked_result("frame-0001", _mask())
    externalized = writer.externalize_region_mask(FRAME, result.regions[0])

    _add_frame(writer, replace(result, regions=(externalized,)))
    writer.finalize()

    reader = RegionPerceptionRunReader(_run_dir(tmp_path))
    assert reader.mask_store().region_keys() == ((FRAME, externalized.region_id),)
    (stored,) = reader.read_regions()
    assert stored.mask_reference == externalized.mask_reference
    assert reader.verify_integrity() == []


def test_the_externalized_run_equals_the_run_that_persisted_in_add_result(
    tmp_path: Path,
) -> None:
    inline_dir, external_dir = tmp_path / "inline", tmp_path / "external"
    inline_dir.mkdir()
    external_dir.mkdir()
    results = [_masked_result(f"frame-000{i}", _mask(4, 3 + i)) for i in range(3)]

    inline = _write_run(inline_dir)
    for result in results:
        _add_frame(inline, result)
    inline.finalize()

    external = _write_run(external_dir)
    for result in results:
        region = external.externalize_region_mask(result.source_observation_id, result.regions[0])
        _add_frame(external, replace(result, regions=(region,)))
    external.finalize()

    for relative in (
        "outputs/regions.jsonl",
        "outputs/masks/mask-index.jsonl",
        *(f"outputs/masks/frame-000{i}/frame-000{i}--region-0001.npy" for i in range(3)),
    ):
        assert (_run_dir(external_dir) / relative).read_bytes() == (
            _run_dir(inline_dir) / relative
        ).read_bytes(), relative


def test_the_mask_index_follows_the_result_order_not_the_staging_order(tmp_path: Path) -> None:
    """A stage-major run stages every frame's masks before any result is added."""
    writer = _write_run(tmp_path)
    results = [_masked_result(f"frame-000{i}", _mask()) for i in range(3)]
    staged = {
        str(r.source_observation_id): writer.externalize_region_mask(
            r.source_observation_id, r.regions[0]
        )
        for r in reversed(results)
    }
    for result in results:
        _add_frame(writer, replace(result, regions=(staged[str(result.source_observation_id)],)))
    writer.finalize()

    keys = RegionPerceptionRunReader(_run_dir(tmp_path)).mask_store().region_keys()
    assert [str(observation) for observation, _ in keys] == [
        "frame-0000",
        "frame-0001",
        "frame-0002",
    ]


def test_add_result_rejects_a_staged_looking_reference_this_run_never_staged(
    tmp_path: Path,
) -> None:
    writer = _write_run(tmp_path)
    result = _masked_result("frame-0001", _mask())
    foreign = replace(
        result.regions[0],
        mask=None,
        mask_reference="outputs/masks/frame-0001/frame-0001--region-0001.npy",
    )

    with pytest.raises(RunArtifactError, match="mask_reference"):
        _add_frame(writer, replace(result, regions=(foreign,)))


def test_finalize_rejects_a_staged_mask_no_result_references(tmp_path: Path) -> None:
    writer = _write_run(tmp_path)
    result = _masked_result("frame-0001", _mask())
    writer.externalize_region_mask(FRAME, result.regions[0])
    _add_frame(writer, replace(result, regions=()))

    with pytest.raises(RunArtifactError, match="no published region references"):
        writer.finalize()


def test_a_failed_mask_write_discards_the_staging_and_poisons_the_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _write_run(tmp_path)
    first = writer.externalize_region_mask(FRAME, _masked_region())
    assert first.mask_reference is not None

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(np, "packbits", boom)
    with pytest.raises(OSError, match="disk full"):
        writer.externalize_region_mask(
            SourceObservationId("frame-0002"), _masked_region("frame-0002")
        )

    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    # Um writer descartado nao recria o staging nem aceita mais mascaras: a run esta perdida.
    with pytest.raises(RunArtifactError, match="discarded"):
        writer.externalize_region_mask(
            SourceObservationId("frame-0003"), _masked_region("frame-0003")
        )
    assert list(tmp_path.iterdir()) == []


def test_an_interrupted_run_leaves_no_staged_payload_behind(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="interrupted"), _write_run(tmp_path) as writer:
        writer.externalize_region_mask(FRAME, _masked_region())
        raise RuntimeError("interrupted between frames")

    assert list(tmp_path.iterdir()) == []


def test_a_staged_mask_cannot_be_relabelled_as_another_region(tmp_path: Path) -> None:
    """Mask provenance preserves its producer's physical observation and regional identity."""
    writer = _write_run(tmp_path)
    original = writer.externalize_region_mask(FRAME, _masked_region())
    with pytest.raises(RunArtifactError, match="mask_reference"):
        writer.write_region(replace(original, region_id=RegionId("other-region")))
