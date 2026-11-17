"""Regional evidence is published once and consumed by the independent visual producer.

The regional artifact retains masks and normalization audits. Visual evidence cites the
regional producer and its regions, without copying the upstream payloads or rerunning discovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.ingestion import (
    FrameId,
    ImageEncoding,
    ImageObservation,
    SensorId,
    SequenceArtifactId,
    SequenceArtifactWriter,
    SourceObservationId,
    SourceProvenance,
)
from contextmap.region_perception import (
    AuditedRegions,
    BoundingBox2D,
    InlineMask,
    MaskStoreError,
    Region2D,
    RegionDiscoveryAudit,
    RegionDiscoveryIdentity,
    RegionId,
    RegionPerceptionRunError,
    RegionPerceptionRunReader,
)
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.composition import RegionDiscoveryPlan
from contextmap.runtime.executors import (
    ExecutorError,
    RegionPerceptionExecutor,
    VisualPerceptionExecutor,
    inventory_digest,
)
from contextmap.shared import BackendProvenance, SourceTimestamp
from contextmap.visual_perception import PreparedImage, VisualEvidenceRunReader
from contextmap.visual_perception.visual_features import FeatureId, FeatureScope, VisualFeature

WIDTH, HEIGHT = 8, 6
FRAMES = 2

_DISCOVERY_BACKEND = BackendProvenance(
    backend_id="fake_region_discovery",
    capability="region_discovery",
    provider="fake",
    model="fake-discovery",
    version="0.1",
    configuration_fingerprint="sha256:fake-discovery-config",
)


def _mask(row: int) -> InlineMask:
    """A full-image mask whose single true row makes each region's pixels distinguishable."""
    pixels = np.zeros((HEIGHT, WIDTH), dtype=bool)
    pixels[row, :] = True
    return InlineMask(pixels)


class _FakeDiscovery:
    """Two mask-backed regions per frame, with the audit a real backend reports."""

    def __init__(self) -> None:
        self.frames: list[str] = []

    def backend_provenance(self) -> BackendProvenance:
        return _DISCOVERY_BACKEND

    def discovery_identity(self) -> RegionDiscoveryIdentity:
        return RegionDiscoveryIdentity.effective()

    def discover(self, image: PreparedImage) -> tuple[Region2D, ...]:
        return self.discover_audited(image).regions

    def discover_audited(self, image: PreparedImage) -> AuditedRegions:
        self.frames.append(str(image.source_observation_id))
        regions = tuple(
            Region2D(
                region_id=RegionId(f"region-{index:04d}"),
                bounding_box=BoundingBox2D(x=0.0, y=float(index), width=float(WIDTH), height=1.0),
                provenance=_DISCOVERY_BACKEND,
                source_observation_id=image.source_observation_id,
                image_width=WIDTH,
                image_height=HEIGHT,
                area_pixels=float(WIDTH),
                mask=_mask(index),
            )
            for index in range(2)
        )
        return AuditedRegions(
            regions=regions,
            audit=RegionDiscoveryAudit(
                source_observation_id=image.source_observation_id,
                backend=_DISCOVERY_BACKEND,
                passes=(),
                diagnostics=(),
                pass_rejections=(),
                discovery_identity=self.discovery_identity(),
                normalization_rejections=(),
                merge_decisions=(),
            ),
        )


class _FakeFeatures:
    """One feature per scope, so the perception graph's feature stages succeed."""

    def __init__(self, scope: FeatureScope) -> None:
        self._scope = scope

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id=f"fake_{self._scope.value}_features",
            capability="feature_extractor",
            provider="fake",
            model="fake",
            version="0.1",
        )

    def required_scope(self) -> FeatureScope:
        return self._scope

    def extract(
        self, image: PreparedImage, regions: tuple[Region2D, ...] = ()
    ) -> list[VisualFeature]:
        if self._scope is FeatureScope.DENSE:
            return [
                VisualFeature(
                    feature_id=FeatureId(f"{image.source_observation_id}--feature-dense-0000"),
                    source_observation_id=image.source_observation_id,
                    scope=FeatureScope.DENSE,
                    embedding_space_id="fake-dense-space",
                    shape=(2,),
                    dtype="float32",
                    payload_reference=f"{image.source_observation_id}-dense.npy",
                    provenance=self.backend_provenance(),
                )
            ]
        return [
            VisualFeature(
                feature_id=FeatureId(f"feature-{region.region_id}"),
                source_observation_id=image.source_observation_id,
                scope=FeatureScope.REGION,
                embedding_space_id="fake-region-space",
                shape=(2,),
                dtype="float32",
                payload_reference=f"{image.source_observation_id}-{region.region_id}.npy",
                provenance=self.backend_provenance(),
                region_id=region.region_id,
            )
            for region in regions
        ]


def _sequence(workspace: Path) -> ArtifactRef:
    """Publish a real sequence artifact of ``FRAMES`` decodable images."""
    output = workspace / "S1" / "run-0001" / "ingestion"
    with SequenceArtifactWriter(
        output_dir=output, sequence_name="S1", artifact_id=SequenceArtifactId("sequence-0001")
    ) as writer:
        for index in range(FRAMES):
            writer.add_observation(
                ImageObservation(
                    observation_id=SourceObservationId(f"frame-{index:04d}"),
                    sensor_id=SensorId("camera_1"),
                    frame_id=FrameId("camera_1_optical"),
                    timestamp=SourceTimestamp(seconds=index, nanoseconds=0, clock_id="fixture"),
                    provenance=SourceProvenance(source_type="fixture", source_path="images"),
                    width=WIDTH,
                    height=HEIGHT,
                    encoding=ImageEncoding.RGB8,
                    data=bytes((index * 7 + offset) % 256 for offset in range(WIDTH * HEIGHT * 3)),
                )
            )
        manifest = writer.finalize()
    return ArtifactRef(
        stage_id="ingestion",
        contract="SequenceArtifact",
        artifact_id="sequence-0001",
        content_hash=inventory_digest(manifest.file_inventory),
        location="S1/run-0001/ingestion",
    )


def _region_executor(discovery: _FakeDiscovery) -> RegionPerceptionExecutor:
    return RegionPerceptionExecutor(
        region_discovery=RegionDiscoveryPlan(
            factory=lambda _root: discovery,
            backend=discovery.backend_provenance(),
            discovery_identity=discovery.discovery_identity(),
        ),
        code_version="test-code",
    )


def _perception_executor() -> VisualPerceptionExecutor:
    return VisualPerceptionExecutor(
        dense_features=lambda _scope: _FakeFeatures(FeatureScope.DENSE),  # type: ignore[arg-type,return-value]
        region_features=lambda _scope: _FakeFeatures(FeatureScope.REGION),  # type: ignore[arg-type,return-value]
    )


def _run_region_stage(
    workspace: Path, sequence: ArtifactRef, *, selection: dict[str, Any] | None = None
) -> tuple[ArtifactRef, _FakeDiscovery]:
    discovery = _FakeDiscovery()
    reference = _region_executor(discovery).execute(
        StageRequest(
            stage_id="region_perception",
            inputs={"sequence": (sequence,)},
            components={},
            config_digest="sha256:region-stage",
            observation_selection=selection,
            output_dir=workspace / "S1" / "run-0001" / "region_perception",
            workspace=workspace,
        )
    )
    return reference, discovery


def _run_perception_stage(
    workspace: Path, sequence: ArtifactRef, regions: ArtifactRef
) -> ArtifactRef:
    return _perception_executor().execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence,), "regions": (regions,)},
            components={},
            config_digest="sha256:perception-stage",
            output_dir=workspace / "S1" / "run-0001" / "visual_perception",
            workspace=workspace,
        )
    )


@pytest.mark.usefixtures("fake_pillow")
class TestTheRegionStage:
    """The stage runs discovery and publishes what a consumer needs to replay it."""

    def test_it_publishes_the_regions_masks_and_audit_of_its_selection(
        self, tmp_path: Path
    ) -> None:
        """Regions alone are not enough: a replay needs the pixels and the evidence too."""
        sequence = _sequence(tmp_path)

        reference, discovery = _run_region_stage(tmp_path, sequence)

        assert reference.contract == "RegionPerceptionRunArtifact"
        assert discovery.frames == ["frame-0000", "frame-0001"]
        reader = RegionPerceptionRunReader(tmp_path / "S1" / "run-0001" / "region_perception")
        assert reader.verify_integrity() == []
        manifest = reader.read_manifest()
        assert list(manifest.source_observation_ids) == ["frame-0000", "frame-0001"]
        assert manifest.region_count == 2 * FRAMES
        assert manifest.discovery_backend_id == "fake_region_discovery"
        assert manifest.code_version == "test-code"
        # A descoberta é o único componente deste stage: nada de grounding nem refinamento.
        assert manifest.grounding_backend_id is None
        assert manifest.refinement_backend_id is None

        regions = list(reader.read_regions())
        assert len({region.region_id for region in regions}) == 2 * FRAMES
        assert all(str(manifest.run_id) in str(region.region_id) for region in regions)
        # O registro publicado nunca inlina pixels; ele carrega a referência do store do run.
        assert all(region.mask is None for region in regions)
        resolver = reader.mask_resolver()
        for region in regions:
            assert region.mask_reference is not None
            assert region.mask_reference.startswith("outputs/masks/")
            row = int(str(region.region_id).rsplit("--region-", 1)[1])
            assert np.array_equal(
                resolver.resolve_mask(region.mask_reference).as_array(), _mask(row).as_array()
            )

        audits = list(reader.read_region_discovery_audits())
        assert [str(audit.source_observation_id) for audit in audits] == [
            "frame-0000",
            "frame-0001",
        ]
        assert {audit.backend for audit in audits} == {_DISCOVERY_BACKEND}

    def test_a_run_that_publishes_no_audit_cannot_be_read_as_one_without_evidence(
        self, tmp_path: Path
    ) -> None:
        """A missing audit stream is a named error, never an empty list (AGENTS.md §26)."""
        sequence = _sequence(tmp_path)
        _run_region_stage(tmp_path, sequence)
        run_dir = tmp_path / "S1" / "run-0001" / "region_perception"
        (run_dir / "outputs" / "region-discovery-audit.jsonl").unlink()

        with pytest.raises(RegionPerceptionRunError, match="publishes no outputs/region-discovery"):
            RegionPerceptionRunReader(run_dir).read_region_discovery_audits()

    def test_a_mask_a_region_claims_but_the_store_lacks_is_an_error(self, tmp_path: Path) -> None:
        """A lost mask must not read back as a box-only region."""
        sequence = _sequence(tmp_path)
        _run_region_stage(tmp_path, sequence)
        run_dir = tmp_path / "S1" / "run-0001" / "region_perception"
        resolver = RegionPerceptionRunReader(run_dir).mask_resolver()

        with pytest.raises(MaskStoreError, match="names no mask of this region perception run"):
            resolver.resolve_mask("outputs/masks/frame-0000/region-9999.npy")


@pytest.mark.usefixtures("fake_pillow")
class TestPerceptionReplaysTheRegionRun:
    """Perception reads the published regions instead of running discovery again."""

    def test_it_publishes_features_linked_to_regions_without_copying_masks(
        self, tmp_path: Path
    ) -> None:
        """The independent feature producer preserves the upstream regional identities."""
        sequence = _sequence(tmp_path)
        regions_ref, discovery = _run_region_stage(tmp_path, sequence)
        discovered = list(
            RegionPerceptionRunReader(
                tmp_path / "S1" / "run-0001" / "region_perception"
            ).read_regions()
        )

        _run_perception_stage(tmp_path, sequence, regions_ref)

        # A descoberta rodou uma única vez, no stage dela: a percepção não a chamou de novo.
        assert discovery.frames == ["frame-0000", "frame-0001"]
        perception = VisualEvidenceRunReader(tmp_path / "S1/run-0001/visual_perception")
        assert perception.verify_integrity() == []
        assert perception.manifest.region_artifact_id == regions_ref.artifact_id
        assert perception.manifest.source_observation_ids == ("frame-0000", "frame-0001")
        featured_regions = [
            feature.region_id
            for result in perception.iter_results()
            for feature in result.features
            if feature.region_id is not None
        ]
        assert featured_regions == [region.region_id for region in discovered]
        assert not (tmp_path / "S1/run-0001/visual_perception/outputs/masks").exists()

    def test_a_region_run_that_misses_a_selected_frame_is_refused(self, tmp_path: Path) -> None:
        """Perception never discovers a frame its pinned region run does not cover."""
        sequence = _sequence(tmp_path)
        regions_ref, discovery = _run_region_stage(
            tmp_path,
            sequence,
            selection={"kind": "frame_range", "start_frame_index": 0, "end_frame_index": 1},
        )
        with pytest.raises(
            ExecutorError, match="does not cover the requested physical observations"
        ):
            _run_perception_stage(tmp_path, sequence, regions_ref)
        assert discovery.frames == ["frame-0000"]
