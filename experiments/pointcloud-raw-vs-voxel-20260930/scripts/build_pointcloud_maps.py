r"""Build two geometry-only ContextMaps — raw and voxel — from one shared geometry plan.

This script serves a testing need (control vs. test): produce two independent, complete
``map_id``s that the viewer catalogs side by side, so the effect of voxel aggregation on the
point cloud can be seen directly. Both maps are built from a *single* ``GeometryInputPlan``
(same sequence, same trajectory, same pose-lookup and motion-correction policies), so the only
variable that differs between them is the scan aggregation:

* the raw map is written with ``aggregation=None`` (every measured point);
* the voxel map is written with ``aggregation=ScanVoxelPolicy(cell_m=...)`` (intra-scan
  voxel-centroid aggregation).

Each ``GeometricMapArtifact`` is then wrapped in a geometry-only ``ContextMap`` (zero entities,
zero relations), which is the only artifact the viewer catalogs and the smallest one that still
renders geometry. No semantic stage runs. Nothing in ``src/`` or in the viewer changes; this is
pure orchestration over the public APIs (``contextmap.geometric_mapping``,
``contextmap.artifact`` and ``contextmap.runtime``), the same category of work
``experiments/geometric-aggregation-raw-vs-voxel-20260926/scripts/arm.py`` already does.

The pose-lookup and motion-correction policies are taken from the very same effective
configuration the runtime would use (``compose(effective, stages=["geometric_mapping"])``), so
the maps this script builds match what ``contextmap run --stage geometric_mapping`` would build
for the raw arm — the voxel arm only adds the aggregation the CLI cannot pass today.

Usage:
    python build_pointcloud_maps.py \\
        --sequence DIR --trajectory DIR \\
        --output-dir DIR --map-name NAME --cell-m 0.10 \\
        [--profile canonical/2] [-c exp.toml ...] [--set path=value ...] \\
        [--code-version <commit>]

``--sequence`` and ``--trajectory`` are the ingestion ``SequenceArtifact`` and the
``state_estimation`` run directories (both produced by, e.g.,
``contextmap run --stage geometric_mapping`` or ``--stage state_estimation``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from contextmap.artifact import (
    CONTEXT_MAP_ASSEMBLY_POLICY_ID,
    ArtifactKind,
    ContextMapId,
    ContextMapMetadata,
    DeclaredCapabilities,
    MapCapability,
    MapCreation,
    ObservationWindow,
    PolicyRef,
    SourceSequence,
    UpstreamArtifact,
    ValidationLevel,
    ValidationStatus,
    artifact_digest,
    assemble_context_map_with_metrics,
    estimator_local_map_frame,
    validate_context_map_artifact,
    write_context_map_with_metrics,
)
from contextmap.geometric_mapping import (
    GeometricMapArtifactManifest,
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    ScanVoxelPolicy,
    build_geometry_mapping_plan_from_artifacts,
)
from contextmap.ingestion import (
    FullSequenceSelection,
    SequenceArtifactReader,
    selection_identity,
)
from contextmap.runtime import compose, resolve_effective_config
from contextmap.shared import SourceTimestamp
from contextmap.state_estimation import StateEstimationRunReader

_ASSEMBLY_POLICY = PolicyRef(policy_id=CONTEXT_MAP_ASSEMBLY_POLICY_ID, version="2")

# Vetor "up" (contrário à gravidade) por eixo principal do frame do mapa. corridor-02 é Z-up
# (a trajetória GT varia ~1,8 m em z contra ~130 m em x/y), então +z é o padrão -- o mesmo
# AxisDirection.POSITIVE_Z que o preset canônico compõe em spatial_relations. Sem essa declaração
# o context_map sai com up_direction=None e o viewer cai no default Y-up: o botão "topo" passa a
# olhar pelo eixo errado e mostra a lateral do mapa (resolveUpAxis em navigation.ts do viewer).
_UP_AXIS_VECTORS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def _geometry_only_configuration_fingerprint(
    up_direction: tuple[float, float, float] | None,
) -> str:
    """Deterministic fingerprint of a geometry-only assembly, mirroring the runtime executor.

    ``ContextMapExecutor`` records an effective-configuration digest built from the assembly
    policy and the declared up direction (``_context_map_configuration_fingerprint`` in
    ``runtime/executors.py``). Computing it the same way, including the same ``sha256:`` prefix
    and the same ``up_direction`` payload, keeps the map's provenance consistent with a
    runtime-assembled one.

    Args:
        up_direction: The map frame's declared up axis, or ``None`` when none is declared.
    """
    payload = json.dumps(
        {
            "assembly_policy_id": _ASSEMBLY_POLICY.policy_id,
            "assembly_policy_version": _ASSEMBLY_POLICY.version,
            "up_direction": list(up_direction) if up_direction is not None else None,
        },
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _observation_window(manifest: GeometricMapArtifactManifest) -> ObservationWindow:
    """The map's time window, from the geometric map manifest's own clock and range."""

    def _timestamp(total_nanoseconds: int) -> SourceTimestamp:
        seconds, nanoseconds = divmod(total_nanoseconds, 1_000_000_000)
        return SourceTimestamp(seconds=seconds, nanoseconds=nanoseconds, clock_id=manifest.clock_id)

    return ObservationWindow(
        start=_timestamp(manifest.start_time_ns), end=_timestamp(manifest.end_time_ns)
    )


def _wrap_geometry_only(
    *,
    context_map_id: str,
    geometric_map_dir: Path,
    sequence_dir: Path,
    output_dir: Path,
    code_version: str | None,
    up_direction: tuple[float, float, float] | None,
) -> Path:
    """Assemble and write a geometry-only ContextMap over one geometric map, then verify it.

    Mirrors ``ContextMapExecutor.execute`` but with no entity-resolution and no spatial-relations
    inputs: the map declares only the geometry capability. Returns the context_map directory.

    Args:
        context_map_id: Stable identifier assigned to the assembled context map.
        geometric_map_dir: Directory containing the geometric map artifact to wrap.
        sequence_dir: Directory containing the source sequence artifact.
        output_dir: Directory in which the context map artifact is written.
        code_version: Code identity recorded in the context map metadata, when available.
        up_direction: Unit vector, in the map frame, that the viewer treats as "up" (see
            ``_UP_AXIS_VECTORS``). Declared here exactly as ``ContextMapExecutor`` declares it from
            Spatial Relations' ``FrameConventions.up_axis``; ``None`` leaves it undeclared.

    Raises:
        SystemExit: If the written artifact does not verify at ``ValidationLevel.FULL``.
    """
    with GeometricMapArtifactReader(geometric_map_dir) as reader:
        manifest = reader.manifest
        bounds = reader.geometry().geometric_map.bounds

    # Mesma guarda do executor: o sequence dado tem de ser aquele sobre o qual o mapa foi
    # construído, senão a lineage do SEQUENCE apontaria para outro artefato.
    sequence = SequenceArtifactReader(sequence_dir)
    if sequence.manifest.artifact_id != manifest.sequence_artifact_id:
        raise SystemExit(
            f"sequence {sequence.manifest.artifact_id!r} is not the one the geometric map "
            f"{manifest.map_id!r} was built over ({manifest.sequence_artifact_id!r})"
        )
    expected_selection_id = selection_identity(
        sequence.manifest.artifact_id, FullSequenceSelection()
    )
    if manifest.selection_id != expected_selection_id:
        raise SystemExit(
            f"selection {expected_selection_id!r} does not match the geometric map's own "
            f"selection {manifest.selection_id!r}"
        )

    metadata = ContextMapMetadata(
        creation=MapCreation(
            assembly_policy=_ASSEMBLY_POLICY,
            code_version=code_version,
            configuration_fingerprint=_geometry_only_configuration_fingerprint(up_direction),
        ),
        source_sequences=(
            SourceSequence(
                sequence_artifact_id=str(manifest.sequence_artifact_id),
                selection_id=manifest.selection_id,
            ),
        ),
        frame=estimator_local_map_frame(
            frame_id=str(manifest.map_frame), up_direction=up_direction
        ),
        bounds=bounds,
        time_bounds=_observation_window(manifest),
        capabilities=DeclaredCapabilities(
            content=(MapCapability.GEOMETRY,), relation_predicates=()
        ),
    )
    sequence_lineage = (
        UpstreamArtifact(
            artifact_id=str(manifest.sequence_artifact_id),
            kind=ArtifactKind.SEQUENCE,
            content_identity=artifact_digest(sequence_dir),
            configuration_fingerprint=None,
            code_version=None,
            model_identities=(),
        ),
    )
    result = assemble_context_map_with_metrics(
        context_map_id=ContextMapId(context_map_id),
        metadata=metadata,
        geometric_map_location=geometric_map_dir,
        entity_resolution_location=None,
        spatial_relations_location=None,
        additional_lineage=sequence_lineage,
    )
    context_map_dir = output_dir / "context_map"
    write_context_map_with_metrics(
        result.context_map,
        output_dir=context_map_dir,
        upstream_locations={str(manifest.map_id): geometric_map_dir},
    )
    report = validate_context_map_artifact(
        context_map_dir,
        level=ValidationLevel.FULL,
        dependency_paths={
            str(manifest.map_id): geometric_map_dir,
            str(manifest.sequence_artifact_id): sequence_dir,
        },
    )
    if report.status is not ValidationStatus.VERIFIED:
        findings = "; ".join(f"{item.code}: {item.message}" for item in report.findings[:5])
        raise SystemExit(
            f"the context map at {context_map_dir} does not verify "
            f"({report.status.value}): {findings}"
        )
    return context_map_dir


def _build_map(
    *,
    output_dir: Path,
    sequence: SequenceArtifactReader,
    sequence_dir: Path,
    plan_factory,
    run_id: str,
    aggregation: ScanVoxelPolicy | None,
    context_map_id: str,
    code_version: str | None,
    up_direction: tuple[float, float, float] | None,
) -> str:
    """Write one geometric map (raw or voxel) and its geometry-only ContextMap wrapper.

    ``plan_factory`` is called to obtain the shared ``GeometryInputPlan``; passing a factory
    (rather than the plan itself) keeps this function honest about the one plan both maps share.
    Returns the resulting ``map_id`` string.
    """
    geometry_dir = output_dir / "geometric_mapping"
    manifest = GeometricMapArtifactWriter(
        output_dir=geometry_dir,
        sequence_name=sequence.manifest.sequence_name,
        run_id=GeometricMapRunId(run_id),
        run_index=0,
    ).finalize(plan=plan_factory(), aggregation=aggregation, code_version=code_version)
    _wrap_geometry_only(
        context_map_id=context_map_id,
        geometric_map_dir=geometry_dir,
        sequence_dir=sequence_dir,
        output_dir=output_dir,
        code_version=code_version,
        up_direction=up_direction,
    )
    return str(manifest.map_id)


def main() -> int:
    """Build the raw and voxel geometry-only maps and print the viewer env to see them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="SequenceArtifact directory")
    parser.add_argument(
        "--trajectory", type=Path, required=True, help="state_estimation run directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--map-name", required=True, help="base run id; '-raw' and '-voxel' are appended"
    )
    parser.add_argument("--cell-m", type=float, required=True, help="voxel edge length, meters")
    parser.add_argument("--profile", default=None, help="base config profile (e.g. canonical/2)")
    parser.add_argument(
        "-c", "--config", action="append", default=[], help="config file (repeatable)"
    )
    parser.add_argument(
        "--set", action="append", default=[], help="'dotted.path=value' override (repeatable)"
    )
    parser.add_argument("--code-version", default=None, help="commit recorded in the lineage")
    parser.add_argument(
        "--up-axis",
        choices=sorted(_UP_AXIS_VECTORS),
        default="z",
        help="map-frame up axis declared to the viewer; 'none' would leave it undeclared "
        "(corridor-02 is z-up, the viewer default is y-up, so 'z' is required for 'top' view)",
    )
    options = parser.parse_args()
    up_direction = _UP_AXIS_VECTORS[options.up_axis]

    profile_kwargs = {"profile": options.profile} if options.profile else {}
    effective = resolve_effective_config(
        files=options.config, overrides=options.set, **profile_kwargs
    )
    composed = compose(effective, stages=["geometric_mapping"])
    if composed.geometric_mapping_pose_lookup is None or composed.motion_correction is None:
        raise SystemExit(
            "the effective configuration did not compose geometric_mapping policies; "
            "check --profile/-c so geometric_mapping is enabled"
        )
    pose_lookup = composed.geometric_mapping_pose_lookup
    motion_correction = composed.motion_correction

    sequence = SequenceArtifactReader(options.sequence)

    def plan_factory():
        """Build the shared geometry plan; identical for both maps, so only aggregation differs."""
        # Reconstruído a cada uso porque o writer consome o plano ao finalizar; os parâmetros
        # (sequence, trajectory, políticas) são idênticos, então a única variável entre raw e
        # voxel é a agregação.
        return build_geometry_mapping_plan_from_artifacts(
            sequence=sequence,
            selection=FullSequenceSelection(),
            run=StateEstimationRunReader(options.trajectory),
            pose_lookup=pose_lookup,
            motion_correction_policy=motion_correction,
        )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = options.output_dir / f"{options.map_name}-raw"
    voxel_dir = options.output_dir / f"{options.map_name}-voxel"

    raw_map_id = _build_map(
        output_dir=raw_dir,
        sequence=sequence,
        sequence_dir=options.sequence,
        plan_factory=plan_factory,
        run_id=f"{options.map_name}-raw",
        aggregation=None,
        context_map_id=f"{options.map_name}-raw--pointcloud",
        code_version=options.code_version,
        up_direction=up_direction,
    )
    voxel_map_id = _build_map(
        output_dir=voxel_dir,
        sequence=sequence,
        sequence_dir=options.sequence,
        plan_factory=plan_factory,
        run_id=f"{options.map_name}-voxel",
        aggregation=ScanVoxelPolicy(cell_m=options.cell_m),
        context_map_id=f"{options.map_name}-voxel--pointcloud",
        code_version=options.code_version,
        up_direction=up_direction,
    )

    print(
        json.dumps(
            {
                "raw": {"map_id": raw_map_id, "bundle": str(raw_dir)},
                "voxel": {
                    "map_id": voxel_map_id,
                    "cell_m": options.cell_m,
                    "bundle": str(voxel_dir),
                },
            },
            indent=1,
        )
    )
    print("\n# viewer .env (both cards are cataloged under one root):", file=sys.stderr)
    print(f"NUXT_CONTEXT_MAP_ARTIFACTS_ROOT={options.output_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
