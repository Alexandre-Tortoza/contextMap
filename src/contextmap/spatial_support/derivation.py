"""Seleção dos três artifacts a montante e a derivação suportada do suporte espacial.

Esta é a única camada da capability que abre artifact de outra capability. Ela recusa três
artifacts que não compartilham um mapa bruto, adapta as entidades do ContextMap para a
entrada da junção, monta a linhagem a partir dos três manifests e delega: a regra à
``service``, a persistência ao ``run_artifact``.

Ela não possui a regra de junção nem o layout em disco, e nada em ``run_artifact`` depende
dela -- é o que mantém o artifact independente de quem o alimenta.
"""

from __future__ import annotations

from pathlib import Path

from contextmap.artifact import ContextMapArtifactError, ContextMapArtifactReader, artifact_digest
from contextmap.geometric_mapping import (
    GeometricMapArtifactReader,
    MapArtifactError,
    VoxelAggregationArtifactReader,
)
from contextmap.spatial_support.models import EntityCellInput
from contextmap.spatial_support.run_artifact import (
    SpatialSupportArtifactError,
    SpatialSupportArtifactWriter,
    SpatialSupportManifest,
    SpatialSupportRunId,
)
from contextmap.spatial_support.service import (
    IMPLEMENTED_POLICY,
    SemanticSpatialSupportError,
    build_semantic_spatial_support,
)


def derive_semantic_spatial_support_artifact(
    *,
    context_map_dir: Path,
    voxel_aggregation_dir: Path,
    raw_geometry_dir: Path,
    output_dir: Path,
    run_id: SpatialSupportRunId,
    code_version: str | None,
) -> SpatialSupportManifest:
    """Derive the semantic spatial support of a ContextMap over a voxel aggregation.

    Args:
        context_map_dir: The finalized ContextMap artifact whose entities are placed.
        voxel_aggregation_dir: The voxel aggregation of the ContextMap's raw map.
        raw_geometry_dir: The raw GeometricMap artifact the aggregation was derived from.
        output_dir: Where the finished derivation will live.
        run_id: Identity of the derivation.
        code_version: Code revision that derives it, when known.

    Returns:
        The manifest of the finalized artifact.

    Raises:
        SpatialSupportArtifactError: If the three artifacts do not share one raw
            map, or the support cannot be built or written.
    """
    try:
        aggregation_reader = VoxelAggregationArtifactReader(voxel_aggregation_dir)
        with GeometricMapArtifactReader(raw_geometry_dir) as raw:
            raw_map_id = str(raw.manifest.map_id)
            with ContextMapArtifactReader.open(
                context_map_dir,
                dependency_paths={raw_map_id: raw_geometry_dir},
                verify_hashes=True,
            ) as context_map:
                context_map_id = str(context_map.geometry_link().map_id)
                map_ids = {
                    "context map": context_map_id,
                    "raw geometry": raw_map_id,
                    "voxel aggregation": str(aggregation_reader.manifest.source_map_id),
                }
                if len(set(map_ids.values())) != 1:
                    raise SpatialSupportArtifactError(
                        f"the three artifacts do not share one raw map: {map_ids}"
                    )
                problems = aggregation_reader.verify_integrity(source=raw)
                if problems:
                    raise SpatialSupportArtifactError(
                        "the voxel aggregation does not verify against the raw artifact: "
                        + "; ".join(problems)
                    )
                # A localização explícita faz o reader confrontar o digest registrado pelo
                # ContextMap com este artifact bruto, sem confiar no locator do manifest.
                context_map.dependency_location(context_map_id)
                aggregation = aggregation_reader.aggregation()
                entities = [
                    EntityCellInput(entity_id=entity.entity_id, geometry_refs=entity.geometry_refs)
                    for entity in context_map.entities()
                ]
                support = build_semantic_spatial_support(
                    entities=entities, aggregation=aggregation, raw_geometry=raw.geometry()
                )
                lineage = {
                    "source": {
                        "context_map": {
                            "context_map_id": context_map.manifest.context_map_id,
                            "schema_version": context_map.manifest.schema_version,
                            "content_identity": context_map.manifest.content_identity,
                        },
                        "geometric_map": {
                            "run_id": str(raw.manifest.run_id),
                            "map_id": raw_map_id,
                            "configuration_fingerprint": raw.manifest.configuration_fingerprint,
                            "content_identity": artifact_digest(raw_geometry_dir),
                        },
                        "voxel_aggregation": {
                            "run_id": str(aggregation_reader.manifest.run_id),
                            "policy_fingerprint": aggregation_reader.manifest.policy_fingerprint,
                            "aggregation_rule": aggregation_reader.manifest.aggregation_rule,
                            "content_identity": artifact_digest(voxel_aggregation_dir),
                        },
                    },
                    "code_version": code_version,
                }
                return SpatialSupportArtifactWriter(output_dir=output_dir, run_id=run_id).finalize(
                    support=support,
                    source_context_map_id=context_map.manifest.context_map_id,
                    policy=IMPLEMENTED_POLICY,
                    lineage=lineage,
                    code_version=code_version,
                )
    except (ContextMapArtifactError, MapArtifactError, SemanticSpatialSupportError) as error:
        raise SpatialSupportArtifactError(str(error)) from error
