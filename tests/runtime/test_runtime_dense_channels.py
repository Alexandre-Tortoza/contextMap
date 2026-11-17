"""The perception run's dense feature maps reach Sensor Association as declared channels (#651).

A runtime perception run persists the sampling geometry of every dense feature it extracts
(``metrics/feature-extraction.jsonl``), its reader rebuilds each ``DenseFeatureMap`` from it, and
``SensorAssociationExecutor``, when the optional ``sensor_association.dense_sampling`` component is
selected, samples one declared channel per dense feature of that run. The real DINOv2 and SigLIP2
adapters run here with a fixed native output instead of a model, so the geometry they compute,
the sink that carries it and the executors that persist and consume it are the production ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from runtime_documents import effective_from, selected_document
from runtime_foundation import foundation_refs
from runtime_perception import publish_region_run, stub_perception_executor
from runtime_pillow import fake_pillow  # noqa: F401

from contextmap.region_perception import RegionPerceptionRunReader
from contextmap.runtime import ArtifactRef, StageRequest
from contextmap.runtime.artifacts import inventory_digest
from contextmap.runtime.composition import FeatureBuildScope, compose, compose_executors
from contextmap.runtime.executors import ExecutorError, SensorAssociationExecutor
from contextmap.sensor_association import (
    AssociationInputError,
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    InterpolationPolicy,
    OcclusionPolicy,
    SensorAssociationRunReader,
)
from contextmap.shared import BackendProvenance
from contextmap.state_estimation import LookupPolicy
from contextmap.visual_perception import PreparedImage, VisualEvidenceRunReader
from contextmap.visual_perception.visual_features import (
    DenseFeatureMap,
    FeatureScope,
    VisualFeature,
)
from contextmap.visual_perception.visual_features.adapters.dinov2 import (
    DinoV2Config,
    DinoV2DenseFeatureBackend,
    DinoV2NativeOutput,
)
from contextmap.visual_perception.visual_features.adapters.siglip2 import (
    Siglip2Config,
    Siglip2FeatureBackend,
    Siglip2NativeOutput,
)

_REVISION = "e" * 40
_DENSE_STAGE = "dense_feature_extraction"
_DENSE_OUTPUTS = ("outputs/dense-feature-associations.jsonl", "outputs/dense-feature-cells.bin")


class _NativeRuntime:
    """Serves one fixed native encoder output: the model is what CI must not run."""

    def __init__(self, output: Any) -> None:
        self._output = output

    def infer(self, image: PreparedImage) -> Any:
        return self._output


def _grid(height: int, width: int) -> Any:
    """A float32 grid whose cell ``(row, column)`` holds ``(row + 1, column + 1)``."""
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return (np.stack((rows, columns), axis=-1) + 1).astype(np.float32)


def _dinov2(scope: FeatureBuildScope) -> DinoV2DenseFeatureBackend:
    """DINOv2 over a 4x4 patch grid: 4 x 3 px cells on the 16 x 12 synthetic images."""
    return DinoV2DenseFeatureBackend(
        config=DinoV2Config(
            checkpoint="facebook/dinov2-small", revision=_REVISION, input_width=56, input_height=56
        ),
        run_id=scope.run_id,
        feature_stage_id=scope.feature_stage_id,
        source_artifact_id=scope.source_artifact_id,
        payload_sink=scope.payload_sink,
        runtime=_NativeRuntime(
            DinoV2NativeOutput(
                array=_grid(4, 4),
                model_input_width=56,
                model_input_height=56,
                patch_width=14,
                patch_height=14,
            )
        ),
    )


def _siglip2(scope: FeatureBuildScope) -> Siglip2FeatureBackend:
    """SigLIP2 over a 2x2 patch grid: 8 x 6 px cells, another stride and support than DINOv2."""
    return Siglip2FeatureBackend(
        config=Siglip2Config(
            checkpoint="google/siglip2-base-patch16-32",
            revision=_REVISION,
            scope=FeatureScope.DENSE,
            input_size=32,
        ),
        run_id=scope.run_id,
        feature_stage_id=scope.feature_stage_id,
        source_artifact_id=scope.source_artifact_id,
        payload_sink=scope.payload_sink,
        runtime=_NativeRuntime(
            Siglip2NativeOutput(
                array=_grid(2, 2),
                model_input_width=32,
                model_input_height=32,
                patch_width=16,
                patch_height=16,
            )
        ),
    )


class _Recording:
    """The composed dense adapter, keeping the map it computed for each image in memory.

    ``extract`` is the adapter's own: one ``extract_dense`` whose feature goes to the graph.
    """

    def __init__(self, backend: Any, maps: dict[str, DenseFeatureMap]) -> None:
        self._backend = backend
        self._maps = maps

    def backend_provenance(self) -> Any:
        return self._backend.backend_provenance()

    def required_scope(self) -> FeatureScope:
        return FeatureScope.DENSE

    def extract(self, image: PreparedImage, regions: Any = ()) -> tuple[VisualFeature, ...]:
        extraction = self._backend.extract_dense(image)
        self._maps[str(image.source_observation_id)] = extraction.dense_map
        return (extraction.dense_map.feature,)


class _FailingOn:
    """A dense adapter whose extraction fails on one frame, as a model error would."""

    def __init__(self, backend: Any, observation_id: str) -> None:
        self._backend = backend
        self._observation_id = observation_id

    def backend_provenance(self) -> Any:
        return self._backend.backend_provenance()

    def required_scope(self) -> FeatureScope:
        return FeatureScope.DENSE

    def extract(self, image: PreparedImage, regions: Any = ()) -> Any:
        if str(image.source_observation_id) == self._observation_id:
            raise RuntimeError("dense extraction failed on this frame")
        return self._backend.extract(image, regions)


def _perceive(workspace: Path, sequence: ArtifactRef, dense_features: Any = None) -> ArtifactRef:
    output = workspace / "S1" / "run-0001" / "visual_perception"
    regions = publish_region_run(workspace, sequence)
    return stub_perception_executor(dense_features=dense_features).execute(
        StageRequest(
            stage_id="visual_perception",
            inputs={"sequence": (sequence,), "regions": (regions,)},
            components={},
            config_digest="sha256:perception",
            output_dir=output,
            workspace=workspace,
        )
    )


def _run_dir(workspace: Path, ref: ArtifactRef) -> Path:
    assert ref.location is not None
    return workspace / ref.location


def _executor(**options: Any) -> SensorAssociationExecutor:
    return SensorAssociationExecutor(
        candidates=CandidateGeometryPolicy(max_range_m=None),
        occlusion=OcclusionPolicy(
            cell_size_px=4, neighborhood_radius_cells=0, depth_margin_m=0.1, depth_margin_ratio=0.02
        ),
        tolerances=DiagnosticTolerances(
            max_pose_time_delta_ns=1_000_000,
            max_map_window_offset_ns=None,
            max_reprojection_p95_px=None,
            max_reprojection_invalid_rate=None,
        ),
        pose_policy=LookupPolicy.exact(),
        **options,
    )


def _associate(
    workspace: Path,
    refs: dict[str, ArtifactRef],
    perception: ArtifactRef,
    executor: Any,
    *,
    run: str,
) -> Path:
    output = workspace / "S1" / run / "sensor_association"
    assert refs["sequence"].location is not None
    regional_dir = (workspace / refs["sequence"].location).parent / "region_perception"
    regional = RegionPerceptionRunReader(regional_dir)
    executor.execute(
        StageRequest(
            stage_id="sensor_association",
            inputs={
                "sequence": (refs["sequence"],),
                "regions": (
                    ArtifactRef(
                        artifact_id=VisualEvidenceRunReader(
                            _run_dir(workspace, perception)
                        ).manifest.region_artifact_id,
                        stage_id="region_perception",
                        contract="RegionPerceptionRunArtifact",
                        content_hash=inventory_digest(regional.manifest.file_inventory),
                        location=regional_dir.relative_to(workspace).as_posix(),
                    ),
                ),
                "visual_evidence": (perception,),
                "trajectory": (refs["state_estimation"],),
                "geometry": (refs["geometry"],),
            },
            components={},
            config_digest=f"sha256:{run}",
            output_dir=output,
            workspace=workspace,
        )
    )
    return output


def _dense_features(reader: VisualEvidenceRunReader) -> list[tuple[str, VisualFeature]]:
    return [
        (str(result.source_observation_id), feature)
        for result in reader.iter_results()
        for feature in result.features
        if feature.scope is FeatureScope.DENSE
    ]


@pytest.mark.usefixtures("fake_pillow")
class TestPerceptionRunDenseGeometry:
    def test_the_run_records_the_geometry_of_each_dense_feature_and_the_reader_rebuilds_it(
        self, tmp_path: Path
    ) -> None:
        refs = foundation_refs(tmp_path)
        maps: dict[str, DenseFeatureMap] = {}
        perception = _perceive(
            tmp_path, refs["sequence"], lambda scope: _Recording(_dinov2(scope), maps)
        )
        run_dir = _run_dir(tmp_path, perception)
        metrics = run_dir / "metrics" / "feature-extraction.jsonl"

        assert metrics.is_file()
        records = [json.loads(line) for line in metrics.read_text("utf-8").splitlines()]
        reader = VisualEvidenceRunReader(run_dir)
        dense = _dense_features(reader)
        assert len(dense) == 3
        assert {record["feature_id"] for record in records if record["dense"]} == {
            str(feature.feature_id) for _, feature in dense
        }
        assert "metrics/feature-extraction.jsonl" in {
            entry.path for entry in reader.manifest.file_inventory
        }
        for observation_id, feature in dense:
            recorded = reader.dense_feature_map(observation_id, feature)  # type: ignore[arg-type]
            assert recorded.stage_id == _DENSE_STAGE
            assert recorded.dense_map == maps[observation_id]


@pytest.mark.usefixtures("fake_pillow")
class TestDenseChannelsInAssociation:
    def test_the_selected_component_samples_the_dense_channel_of_the_perception_run(
        self, tmp_path: Path
    ) -> None:
        refs = foundation_refs(tmp_path)
        perception = _perceive(tmp_path, refs["sequence"], _dinov2)
        (_, feature), *_ = _dense_features(VisualEvidenceRunReader(_run_dir(tmp_path, perception)))

        output = _associate(
            tmp_path,
            refs,
            perception,
            _executor(dense_sampling=InterpolationPolicy.BILINEAR),
            run="run-0002",
        )

        for relative in _DENSE_OUTPUTS:
            assert (output / relative).is_file()
        reader = SensorAssociationRunReader(output)
        (channel,) = reader.manifest.dense_channels
        # A identidade do canal vem do estágio e do espaço de embedding, nunca do backend.
        assert channel["channel_id"] == f"{_DENSE_STAGE}@{feature.embedding_space_id}"
        assert "dinov2" not in channel["channel_id"]
        assert channel["interpolation"] == "bilinear"
        (source,) = channel["feature_sources"]
        assert source["source_artifact_id"] == perception.artifact_id
        assert source["embedding_space_id"] == feature.embedding_space_id
        assert source["extractor"]["backend_id"] == "dinov2_huggingface"
        assert source["frame_count"] == 3
        association = reader.dense_association("frame-0000", channel["channel_id"])
        assert association.terms == 4
        assert any(association.sampled)

    def test_the_configuration_fingerprint_names_the_channels_and_their_interpolation(
        self, tmp_path: Path
    ) -> None:
        refs = foundation_refs(tmp_path)
        perception = _perceive(tmp_path, refs["sequence"], _dinov2)

        fingerprints = [
            SensorAssociationRunReader(
                _associate(tmp_path, refs, perception, executor, run=run)
            ).manifest.configuration_fingerprint
            for run, executor in (
                ("run-0002", _executor()),
                ("run-0003", _executor(dense_sampling=InterpolationPolicy.NEAREST)),
                ("run-0004", _executor(dense_sampling=InterpolationPolicy.BILINEAR)),
            )
        ]

        assert len(set(fingerprints)) == 3

    @pytest.mark.parametrize("backend", [_dinov2, _siglip2], ids=["dinov2", "siglip2"])
    def test_grids_of_different_stride_and_support_take_the_same_path(
        self, tmp_path: Path, backend: Any
    ) -> None:
        refs = foundation_refs(tmp_path)
        maps: dict[str, DenseFeatureMap] = {}
        perception = _perceive(
            tmp_path, refs["sequence"], lambda scope: _Recording(backend(scope), maps)
        )

        output = _associate(
            tmp_path,
            refs,
            perception,
            _executor(dense_sampling=InterpolationPolicy.NEAREST),
            run="run-0002",
        )

        reader = SensorAssociationRunReader(output)
        (channel,) = reader.manifest.dense_channels
        (source,) = channel["feature_sources"]
        sampling = maps["frame-0000"].sampling
        assert source["coordinate_transform_id"] == sampling.coordinate_transform_id
        association = reader.dense_association("frame-0000", channel["channel_id"])
        assert association.terms == 1
        assert any(association.sampled)

    def test_without_the_component_the_association_stays_geometry_only(
        self, tmp_path: Path
    ) -> None:
        """Control: a run whose perception has dense geometry, but the component is unselected.

        The composed executor of a configuration that selects no dense sampling writes exactly
        what the executor built without the option writes: no channel and no dense output.
        """
        refs = foundation_refs(tmp_path)
        perception = _perceive(tmp_path, refs["sequence"], _dinov2)
        effective = effective_from(tmp_path, selected_document())
        composed = compose(
            effective, stages=["sensor_association"], module_available=lambda _: True, environ={}
        )
        assert composed.association_dense_sampling is None
        assert composed.association_candidates is not None
        assert composed.occlusion_policy is not None
        assert composed.association_tolerances is not None
        assert composed.association_pose_policy is not None
        executor = compose_executors(effective, module_available=lambda _: True, environ={})[
            "sensor_association"
        ]
        today = SensorAssociationExecutor(
            candidates=composed.association_candidates,
            occlusion=composed.occlusion_policy,
            tolerances=composed.association_tolerances,
            pose_policy=composed.association_pose_policy,
        )

        composed_output = _associate(tmp_path, refs, perception, executor, run="run-0002")
        today_output = _associate(tmp_path, refs, perception, today, run="run-0003")

        composed_reader = SensorAssociationRunReader(composed_output)
        today_reader = SensorAssociationRunReader(today_output)
        assert composed_reader.manifest.dense_channels == ()
        assert (
            composed_reader.manifest.configuration_fingerprint
            == today_reader.manifest.configuration_fingerprint
        )
        for relative in _DENSE_OUTPUTS:
            assert not (composed_output / relative).exists()
        outputs = sorted(path.name for path in (composed_output / "outputs").iterdir())
        assert outputs == sorted(path.name for path in (today_output / "outputs").iterdir())
        for name in outputs:
            assert (composed_output / "outputs" / name).read_bytes() == (
                today_output / "outputs" / name
            ).read_bytes()


@pytest.mark.usefixtures("fake_pillow")
class TestDenseChannelsFailExplicitly:
    def test_a_perception_run_without_dense_geometry_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        # O fake padrão produz uma feature densa sem geometria de amostragem registrada.
        refs = foundation_refs(tmp_path)
        perception = _perceive(tmp_path, refs["sequence"])

        with pytest.raises(ExecutorError, match="dense geometry"):
            _associate(
                tmp_path,
                refs,
                perception,
                _executor(dense_sampling=InterpolationPolicy.NEAREST),
                run="run-0002",
            )

        assert not (tmp_path / "S1" / "run-0002").exists()

    def test_a_perception_run_without_any_dense_feature_is_refused(self, tmp_path: Path) -> None:
        refs = foundation_refs(tmp_path)

        class _NoFeatures:
            def backend_provenance(self) -> Any:
                return _fake_provenance()

            def required_scope(self) -> FeatureScope:
                return FeatureScope.DENSE

            def extract(self, image: PreparedImage, regions: Any = ()) -> tuple[()]:
                return ()

        perception = _perceive(tmp_path, refs["sequence"], lambda _scope: _NoFeatures())

        with pytest.raises(ExecutorError, match="no dense feature"):
            _associate(
                tmp_path,
                refs,
                perception,
                _executor(dense_sampling=InterpolationPolicy.NEAREST),
                run="run-0002",
            )

        assert not (tmp_path / "S1" / "run-0002").exists()

    def test_a_frame_without_the_map_of_a_declared_channel_fails_the_association(
        self, tmp_path: Path
    ) -> None:
        refs = foundation_refs(tmp_path)
        perception = _perceive(
            tmp_path, refs["sequence"], lambda scope: _FailingOn(_dinov2(scope), "frame-0001")
        )

        with pytest.raises(AssociationInputError, match="no dense map for the declared channel"):
            _associate(
                tmp_path,
                refs,
                perception,
                _executor(dense_sampling=InterpolationPolicy.NEAREST),
                run="run-0002",
            )

        assert not (tmp_path / "S1" / "run-0002" / "sensor_association").exists()


def _fake_provenance() -> BackendProvenance:
    return BackendProvenance(
        backend_id="fake_dense_feature_extractor",
        capability="feature_extractor",
        provider="fake",
        model="fake",
        version="0.1",
    )
