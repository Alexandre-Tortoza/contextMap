"""Point Representation as an opt-in derived stage of the runtime (issue #663).

The composition root composes the stage's executor from configuration alone, only when the stage
is enabled and with the code identity its run records; a learned encoder without its model
runtime is left out with that cause, never replaced. The executor's identity follows the encoder
and the center rule, not only the parameters the configuration declares, and the executor refuses
an association run built over another map than the one it represents, even one that holds no
observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pointrep_ptv3_fakes import FakePTv3Runtime
from runtime_documents import SUPPORT_POLICY, effective_from, selected_document
from runtime_foundation import DATASET, build_map, foundation_refs

from contextmap.geometric_mapping import GeometricMapArtifactReader
from contextmap.ingestion import SequenceArtifactReader
from contextmap.point_representation import (
    ObservationSupportAnchorPolicy,
    PointRepresentationRunReader,
    SupportPolicy,
)
from contextmap.point_representation.backends.geometric_descriptor import (
    GeometricDescriptorEncoder,
)
from contextmap.runtime import ArtifactRef, ConfigurationError, StageRequest
from contextmap.runtime.coercion import build_config
from contextmap.runtime.components.contracts import ASSOCIATION
from contextmap.runtime.composition import compose_executors
from contextmap.runtime.errors import (
    BackendRuntimeMissingError,
    CompositionError,
    ExecutorIdentityMissingError,
)
from contextmap.runtime.executors import (
    ExecutorError,
    PointRepresentationExecutor,
    inventory_digest,
)
from contextmap.sensor_association import (
    CandidateGeometryPolicy,
    DiagnosticTolerances,
    OcclusionPolicy,
    SensorAssociationRequest,
    SensorAssociationRunId,
    SensorAssociationRunReader,
    SensorAssociationRunWriter,
    SensorAssociationService,
)
from contextmap.state_estimation import LookupPolicy, StateEstimationRunReader, TrajectoryLookup

PTV3 = {
    "variant": "ptv3-base",
    "checkpoint": "ptv3-x",
    "checkpoint_hash": "sha256:" + "0" * 64,
    "precision": "float32",
    "grid_size_m": 0.02,
    "output_dimension": 8,
    "pooling": "mean",
    "normalization": "none",
    "min_support_points": 1,
    "support_policy": SUPPORT_POLICY,
}


def _compose(
    tmp_path: Path, document: dict[str, Any], **options: Any
) -> tuple[dict[str, Any], dict[str, CompositionError | ConfigurationError]]:
    failures: dict[str, CompositionError | ConfigurationError] = {}
    options.setdefault("code_version", "code-663")
    executors = compose_executors(
        effective_from(tmp_path, document),
        module_available=lambda _name: True,
        environ={},
        on_composition_failure=failures.__setitem__,
        **options,
    )
    return dict(executors), failures


def _with_ptv3() -> dict[str, Any]:
    document = selected_document()
    document["components"]["point_representation"]["encoder"] = {"backend": "ptv3", "ptv3": PTV3}
    return document


def test_the_deterministic_descriptor_composes_the_stage_with_no_provider(tmp_path: Path) -> None:
    executors, failures = _compose(tmp_path, selected_document())

    assert isinstance(executors["point_representation"], PointRepresentationExecutor)
    assert "point_representation" not in failures


def test_ptv3_composes_with_its_runtime_from_a_provider(tmp_path: Path) -> None:
    runtimes: list[Any] = []

    def provide(config: Any, _secrets: Any) -> FakePTv3Runtime:
        runtimes.append(config)
        return FakePTv3Runtime()

    executors, failures = _compose(
        tmp_path, _with_ptv3(), providers={"point_representation.encoder": provide}
    )

    assert isinstance(executors["point_representation"], PointRepresentationExecutor)
    assert "point_representation" not in failures
    # O provider recebe a configuração já validada da capability, com o dispositivo resolvido.
    assert [(config.variant, config.device) for config in runtimes] == [("ptv3-base", "cpu")]


def test_ptv3_without_its_runtime_is_left_out_never_replaced_by_the_descriptor(
    tmp_path: Path,
) -> None:
    executors, failures = _compose(tmp_path, _with_ptv3())

    assert "point_representation" not in executors
    failure = failures["point_representation"]
    assert isinstance(failure, BackendRuntimeMissingError)
    assert failure.component_id == "point_representation.encoder"
    # Os demais estágios não pagam pela falta do runtime.
    assert {"sensor_association", "semantic_fusion", "context_map"} <= set(executors)


def test_without_a_code_identity_the_stage_is_left_out_naming_it(tmp_path: Path) -> None:
    executors, failures = _compose(tmp_path, selected_document(), code_version=None)

    assert "point_representation" not in executors
    failure = failures["point_representation"]
    assert isinstance(failure, ExecutorIdentityMissingError)
    assert "code_version" in str(failure)


def test_a_disabled_stage_is_neither_composed_nor_reported(tmp_path: Path) -> None:
    document = selected_document()
    document["pipeline"]["stages"]["point_representation"] = False

    executors, failures = _compose(tmp_path, document)

    assert "point_representation" not in executors
    assert "point_representation" not in failures


def test_a_stage_without_its_center_selection_is_left_out_not_given_a_default(
    tmp_path: Path,
) -> None:
    document = selected_document()
    del document["components"]["point_representation"]["center_selection"]

    executors, failures = _compose(tmp_path, document)

    assert "point_representation" not in executors
    assert "point_representation.center_selection" in str(failures["point_representation"])


def _descriptor_executor(radius_m: float = 0.5) -> PointRepresentationExecutor:
    """The stage's executor over the deterministic descriptor, which needs no model runtime."""
    policy = build_config(SupportPolicy, {**SUPPORT_POLICY, "radius_m": radius_m})
    return PointRepresentationExecutor(
        center_selection=ObservationSupportAnchorPolicy(),
        encoder=GeometricDescriptorEncoder(policy),
        code_version="code-663",
    )


def test_the_configuration_identity_follows_the_encoder_and_its_support_policy() -> None:
    identity = _descriptor_executor(0.5).configuration_identity

    assert _descriptor_executor(0.5).configuration_identity == identity
    assert _descriptor_executor(0.25).configuration_identity != identity


def _directory(workspace: Path, ref: ArtifactRef) -> Path:
    assert ref.location is not None
    return workspace / ref.location


def _empty_association(
    workspace: Path, foundation: dict[str, ArtifactRef], geometry: ArtifactRef, *, run: str
) -> ArtifactRef:
    """Publish an association run over ``geometry`` without any frame, in ``run``.

    Without frames the run holds no spatial observation, so its manifest is the only record of
    the map it was built over.
    """
    sequence = SequenceArtifactReader(_directory(workspace, foundation["sequence"]))
    trajectory = StateEstimationRunReader(_directory(workspace, foundation["state_estimation"]))
    calibration = sequence.read_calibration()
    assert calibration is not None, "the synthetic sequence carries its calibration"
    output = workspace / DATASET / run / "sensor_association"
    writer = SensorAssociationRunWriter(
        output_dir=output,
        sequence_name=sequence.manifest.sequence_name,
        run_id=SensorAssociationRunId(f"association-{run}"),
        run_index=1,
    )
    with (
        GeometricMapArtifactReader(_directory(workspace, geometry)) as source,
        writer.transaction() as sink,
    ):
        outcome = SensorAssociationService().run(
            SensorAssociationRequest(
                association_run_id=str(writer.run_id),
                sequence_artifact_id=sequence.manifest.artifact_id,
                selection_id="full-sequence",
                geometry=source.geometry(),
                trajectory=TrajectoryLookup(trajectory.trajectory()),
                pose_policy=LookupPolicy.exact(),
                calibration=calibration,
                candidate_policy=CandidateGeometryPolicy(max_range_m=None),
                occlusion_policy=OcclusionPolicy(
                    cell_size_px=4,
                    neighborhood_radius_cells=0,
                    depth_margin_m=0.1,
                    depth_margin_ratio=0.02,
                ),
                tolerances=DiagnosticTolerances(
                    max_pose_time_delta_ns=1_000_000,
                    max_map_window_offset_ns=None,
                    max_reprojection_p95_px=None,
                    max_reprojection_invalid_rate=None,
                ),
                frames=(),
                state_estimation_run_id=trajectory.manifest.run_id,
                code_version="test",
            ),
            sink=sink,
        )
        manifest = sink.finalize(outcome)
    return ArtifactRef(
        stage_id="sensor_association",
        contract=ASSOCIATION,
        artifact_id=str(manifest.run_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location=output.relative_to(workspace).as_posix(),
    )


def _representation_request(
    workspace: Path, *, association: ArtifactRef, geometry: ArtifactRef
) -> StageRequest:
    return StageRequest(
        stage_id="point_representation",
        inputs={"association": (association,), "geometry": (geometry,)},
        components={},
        config_digest="pr-1",
        output_dir=workspace / DATASET / "run-0003" / "point_representation",
        workspace=workspace,
    )


def _map_id(workspace: Path, geometry: ArtifactRef) -> str:
    with GeometricMapArtifactReader(_directory(workspace, geometry)) as reader:
        return str(reader.manifest.map_id)


def test_an_association_over_another_map_is_refused_before_anything_is_published(
    tmp_path: Path,
) -> None:
    foundation = foundation_refs(tmp_path)
    sequence, trajectory = foundation["sequence"], foundation["state_estimation"]
    other_map = build_map(tmp_path, sequence, trajectory, run="run-0002", digest="gm-2")
    # Sem observações, a checagem por observação da política de centros não vê nada: só o
    # manifest da associação diz sobre qual mapa ela foi construída.
    association = _empty_association(tmp_path, foundation, other_map, run="run-0002")
    request = _representation_request(
        tmp_path, association=association, geometry=foundation["geometry"]
    )
    represented = _map_id(tmp_path, foundation["geometry"])
    associated = _map_id(tmp_path, other_map)
    assert represented != associated
    association_run = SensorAssociationRunReader(_directory(tmp_path, association)).manifest.run_id

    with pytest.raises(ExecutorError) as refused:
        _descriptor_executor().execute(request)

    message = str(refused.value)
    assert str(association_run) in message
    assert represented in message and associated in message
    assert request.output_dir is not None and not request.output_dir.exists()


def test_an_empty_association_over_the_same_map_publishes_a_run_without_representations(
    tmp_path: Path,
) -> None:
    foundation = foundation_refs(tmp_path)
    association = _empty_association(tmp_path, foundation, foundation["geometry"], run="run-0001")

    published = _descriptor_executor().execute(
        _representation_request(tmp_path, association=association, geometry=foundation["geometry"])
    )

    reader = PointRepresentationRunReader(_directory(tmp_path, published))
    assert reader.verify_integrity() == []
    assert str(reader.manifest.geometric_map_id) == _map_id(tmp_path, foundation["geometry"])
    assert reader.manifest.association_context_id == association.artifact_id
    assert reader.manifest.representation_count == 0
    assert list(reader.iter_representations()) == []
