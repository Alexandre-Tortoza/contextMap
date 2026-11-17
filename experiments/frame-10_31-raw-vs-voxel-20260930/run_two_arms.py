"""Duas runs completas do frame ~631s (10:31): mapa RAW vs mapa VOXEL, ambas com ego-mask.

Compartilha o que a imagem determina (state_estimation, visual_perception, semantic_scoring),
porque a ego-mask e a imagem são idênticas nos dois arms; o que difere é o MAPA geométrico
(raw = todos os pontos; voxel = ScanVoxelPolicy intra-scan) e tudo a jusante da associação.

Dirige executores direto (como os scripts canônicos), pois o CLI usa catálogo/workspace e não
passa a agregação voxel ao geometric_mapping.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    ComparisonChannels,
    ConservativeResolutionPolicy,
    GeometryComparisonPolicy,
    MatchChannel,
    MatchEvidenceBuilder,
)
from contextmap.geometric_mapping import ScanVoxelPolicy
from contextmap.ingestion import SequenceArtifactReader, SourceObservationId, encode_selection
from contextmap.ingestion.sequence_observation_selection import ExplicitIdsSelection
from contextmap.runtime import ArtifactRef, StageRequest, resolve_effective_config
from contextmap.runtime.composition import compose, compose_executors
from contextmap.runtime.executors import (
    EntityResolutionExecutor,
    GeometricMappingExecutor,
    SemanticMappingExecutor,
    SpatialRelationsExecutor,
)
from contextmap.semantic_mapping import (
    EntityMaterializationPolicy,
    GeometrySummaryPolicy,
    SemanticMapId,
)
from contextmap.spatial_relations import (
    AxisDirection,
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationPredicate,
    RelationsRunPolicies,
)


def ref(stage: str, contract: str, artifact_id: str, location: str) -> ArtifactRef:
    return ArtifactRef(
        stage_id=stage,
        contract=contract,
        artifact_id=artifact_id,
        content_hash="sha256:driven",
        location=location,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--sequence-id", required=True)
    p.add_argument("--pose-id", required=True)
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--cell-m", type=float, default=0.10)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--stage", default="all", help="all | foundation | perception | downstream")
    args = p.parse_args()

    ws = args.workspace
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    seq_loc = f"sequences/corridor-02/{args.sequence_id}"
    pose_loc = f"sequences/corridor-02-pose/{args.pose_id}"

    effective = resolve_effective_config(files=list(args.config))
    ex = compose_executors(
        effective, module_available=lambda _m: True, environ={}, code_version="frame-10_31"
    )
    # Abandona a VLM de cena/contexto: remove o estágio scene_interpretation do preset
    # canonical/1 (region_interpretation depende só de {image, regions}, não da cena; e
    # assemble_perception_result tolera a cena ausente via .get()). O Qwen só rotula regiões.
    from contextmap.visual_perception.execution.pipeline_resolution import (
        CANONICAL_PRESET_V1,
        PipelinePreset,
    )

    _scene_less = PipelinePreset(
        preset_id="canonical/1-labels-no-scene",
        stages=tuple(s for s in CANONICAL_PRESET_V1.stages if s.stage_id != "scene_interpretation"),
        description="canonical/1 sem scene_interpretation (apenas labels de região).",
    )
    ex["visual_perception"]._preset = lambda: _scene_less
    # As 3 etapas de cauda não têm params em nenhum config commitado (todo run completo as
    # constrói direto); uso os valores canônicos do e2e-real-canonical-run.
    sm_exec = SemanticMappingExecutor(
        policy=EntityMaterializationPolicy(
            geometry=GeometrySummaryPolicy(sparse_point_threshold=3, connectivity_radius_m=0.5)
        ),
        semantic_map_id=SemanticMapId("semantic-map-corridor-02-frame-10_31"),
        code_digest="sha256:" + "31" * 32,
        code_version="frame-10_31",
    )
    er_exec = EntityResolutionExecutor(
        retrieval=CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1),
        builder=MatchEvidenceBuilder(
            ComparisonChannels(
                geometry=GeometryComparisonPolicy(
                    min_shared_support_jaccard=0.5,
                    min_bounds_iou=0.5,
                    min_bounds_containment=0.9,
                    min_conflict_gap_m=0.5,
                    min_extent_ratio=0.3,
                )
            )
        ),
        resolution=ConservativeResolutionPolicy(
            use_channels=(MatchChannel.GEOMETRY,), min_supporting_channels=1
        ),
        code_version="frame-10_31",
    )
    sr_exec = SpatialRelationsExecutor(
        code_version="frame-10_31",
        policies=RelationsRunPolicies(
            frame_conventions=FrameConventions(
                map_frame="map",
                up_axis=AxisDirection.POSITIVE_Z,
                forward_axis=AxisDirection.POSITIVE_X,
            ),
            candidate=CandidatePolicy(
                predicates=(RelationPredicate.NEXT_TO, RelationPredicate.TOUCHING),
                proximity_radius_m=0.6,
                directional_radius_m=2.0,
            ),
            geometric=GeometricPredicatePolicy(
                boundary_tolerance_m=0.02,
                next_to_max_gap_m=0.5,
                adjacent_penetration_m=0.05,
                containment_slack_m=0.05,
                directional_overlap_fraction=0.5,
            ),
            contact=ContactPredicatePolicy(
                contact_distance_m=0.05,
                contact_tolerance_m=0.02,
                min_contact_points=3,
                support_height_tolerance_m=0.05,
                support_footprint_fraction=0.5,
                leaning_min_tilt_deg=10.0,
                leaning_max_tilt_deg=80.0,
                tilt_tolerance_deg=2.0,
                leaning_min_vertical_overlap_m=0.3,
            ),
            geometry_summary=GeometrySummaryPolicy(
                sparse_point_threshold=3, connectivity_radius_m=0.5
            ),
        ),
    )

    seq_ref = ref("ingestion", "SequenceArtifact", args.sequence_id, seq_loc)
    pose_ref = ref("pose_ingestion", "SequenceArtifact", args.pose_id, pose_loc)

    def run_state() -> ArtifactRef:
        out = pipe / "state_estimation"
        r = ex["state_estimation"].execute(
            StageRequest(
                stage_id="state_estimation",
                inputs={"sequence": (seq_ref,), "pose_sequence": (pose_ref,)},
                components={}, config_digest="s", output_dir=out, workspace=ws,
            )
        )
        print("STATE", r.artifact_id, flush=True)
        return ref("state_estimation", "StateEstimationRunArtifact", r.artifact_id,
                   "pipeline/state_estimation")

    def run_geometry(arm: str, aggregation) -> ArtifactRef:
        out = pipe / f"geometric_mapping_{arm}"
        state_ref = ref("state_estimation", "StateEstimationRunArtifact",
                        _read_id(pipe / "state_estimation", "state"), "pipeline/state_estimation")
        request = StageRequest(
            stage_id="geometric_mapping",
            inputs={"sequence": (seq_ref,), "trajectory": (state_ref,)},
            components={}, config_digest=f"geo-{arm}", output_dir=out, workspace=ws,
        )
        if aggregation is None:
            executor = ex["geometric_mapping"]
        else:
            g = compose(effective, stages=["geometric_mapping"])
            executor = GeometricMappingExecutor(
                pose_lookup=g.geometric_mapping_pose_lookup,
                motion_correction=g.motion_correction,
                aggregation=aggregation,
                code_version="frame-10_31",
            )
        r = executor.execute(request)
        print(f"GEOMETRY[{arm}]", r.artifact_id, flush=True)
        return ref("geometric_mapping", "GeometricMapArtifact", r.artifact_id,
                   f"pipeline/geometric_mapping_{arm}")

    def run_perception() -> ArtifactRef:
        out = pipe / "visual_perception"
        reader = SequenceArtifactReader(ws / seq_loc)
        image_ids = [str(e.observation_id) for e in reader.iter_index() if e.modality == "image"]
        chosen = image_ids[args.frame_index:args.frame_index + 1]
        print("IMAGE_IDS", image_ids, "CHOSEN", chosen, flush=True)
        r = ex["visual_perception"].execute(
            StageRequest(
                stage_id="visual_perception",
                inputs={"sequence": (seq_ref,)},
                components={}, config_digest="perc", output_dir=out, workspace=ws,
                observation_selection=encode_selection(
                    ExplicitIdsSelection(
                        observation_ids=frozenset(SourceObservationId(i) for i in chosen)
                    )
                ),
            )
        )
        print("PERCEPTION", r.artifact_id, flush=True)
        return ref("visual_perception", "PerceptionRunArtifact", r.artifact_id,
                   "pipeline/visual_perception")

    def run_scoring(perc_ref: ArtifactRef) -> ArtifactRef:
        out = pipe / "semantic_scoring"
        r = ex["semantic_scoring"].execute(
            StageRequest(stage_id="semantic_scoring", inputs={"perception": (perc_ref,)},
                         components={}, config_digest="score", output_dir=out, workspace=ws)
        )
        print("SCORING", r.artifact_id, flush=True)
        return ref("semantic_scoring", "SemanticScoringRunArtifact", r.artifact_id,
                   "pipeline/semantic_scoring")

    def run_downstream(arm: str, geo_ref: ArtifactRef, perc_ref: ArtifactRef,
                       score_ref: ArtifactRef) -> None:
        state_ref = ref("state_estimation", "StateEstimationRunArtifact",
                        _read_id(pipe / "state_estimation", "state"), "pipeline/state_estimation")
        a_out = pipe / f"{arm}_sensor_association"
        a = ex["sensor_association"].execute(StageRequest(
            stage_id="sensor_association",
            inputs={"sequence": (seq_ref,), "trajectory": (state_ref,), "geometry": (geo_ref,),
                    "perception": (perc_ref,)},
            components={}, config_digest=f"{arm}-assoc", output_dir=a_out, workspace=ws))
        a_ref = ref("sensor_association", "SensorAssociationRunArtifact", a.artifact_id,
                    f"pipeline/{arm}_sensor_association")
        print(f"ASSOC[{arm}]", a.artifact_id, flush=True)

        f_out = pipe / f"{arm}_semantic_fusion"
        f = ex["semantic_fusion"].execute(StageRequest(
            stage_id="semantic_fusion",
            inputs={"sequence": (seq_ref,), "association": (a_ref,), "perception": (perc_ref,),
                    "geometry": (geo_ref,), "semantic_scores": (score_ref,)},
            components={}, config_digest=f"{arm}-fusion", output_dir=f_out, workspace=ws))
        f_ref = ref("semantic_fusion", "SemanticFusionRunArtifact", f.artifact_id,
                    f"pipeline/{arm}_semantic_fusion")
        print(f"FUSION[{arm}]", f.artifact_id, flush=True)

        m_out = pipe / f"{arm}_semantic_mapping"
        m = sm_exec.execute(StageRequest(
            stage_id="semantic_mapping",
            inputs={"fusion": (f_ref,), "geometry": (geo_ref,)},
            components={}, config_digest=f"{arm}-map", output_dir=m_out, workspace=ws))
        m_ref = ref("semantic_mapping", "SemanticEntityArtifact", m.artifact_id,
                    f"pipeline/{arm}_semantic_mapping")
        print(f"MAPPING[{arm}]", m.artifact_id, flush=True)

        e_out = pipe / f"{arm}_entity_resolution"
        e = er_exec.execute(StageRequest(
            stage_id="entity_resolution",
            inputs={"entities": (m_ref,)},
            components={}, config_digest=f"{arm}-ent", output_dir=e_out, workspace=ws))
        e_ref = ref("entity_resolution", "EntityResolutionRunArtifact", e.artifact_id,
                    f"pipeline/{arm}_entity_resolution")
        print(f"ENTITY[{arm}]", e.artifact_id, flush=True)

        r_out = pipe / f"{arm}_spatial_relations"
        rr = sr_exec.execute(StageRequest(
            stage_id="spatial_relations",
            inputs={"entities": (e_ref,), "geometry": (geo_ref,)},
            components={}, config_digest=f"{arm}-rel", output_dir=r_out, workspace=ws))
        r_ref = ref("spatial_relations", "SpatialRelationsRunArtifact", rr.artifact_id,
                    f"pipeline/{arm}_spatial_relations")
        print(f"RELATIONS[{arm}]", rr.artifact_id, flush=True)

        c_out = ws / arm / "context_map"
        c_out.parent.mkdir(parents=True, exist_ok=True)
        c = ex["context_map"].execute(StageRequest(
            stage_id="context_map",
            inputs={"sequence": (seq_ref,), "geometry": (geo_ref,), "entities": (e_ref,),
                    "relations": (r_ref,)},
            components={}, config_digest=f"{arm}-ctx", output_dir=c_out, workspace=ws))
        print(f"CONTEXT_MAP[{arm}]", c.artifact_id, "at", c_out, flush=True)

    # --- orquestração ---
    if args.stage in ("all", "foundation"):
        run_state()
        run_geometry("raw", None)
        run_geometry("voxel", ScanVoxelPolicy(cell_m=args.cell_m))
    if args.stage in ("all", "perception"):
        run_perception()
        run_scoring(ref("visual_perception", "PerceptionRunArtifact",
                        _read_id(pipe / "visual_perception", "perc"), "pipeline/visual_perception"))
    if args.stage in ("all", "downstream"):
        perc_ref = ref("visual_perception", "PerceptionRunArtifact",
                       _read_id(pipe / "visual_perception", "perc"), "pipeline/visual_perception")
        score_ref = ref("semantic_scoring", "SemanticScoringRunArtifact",
                        _read_id(pipe / "semantic_scoring", "score"), "pipeline/semantic_scoring")
        for arm, sub in (("raw", "geometric_mapping_raw"), ("voxel", "geometric_mapping_voxel")):
            geo_ref = ref("geometric_mapping", "GeometricMapArtifact",
                          _read_id(pipe / sub, "geo"), f"pipeline/{sub}")
            run_downstream(arm, geo_ref, perc_ref, score_ref)
    return 0


def _read_id(run_dir: Path, kind: str) -> str:
    """Lê o run_id publicado de um estágio a partir do manifest do artefato."""
    import json
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for key in ("run_id", "context_map_id", "artifact_id"):
        if key in manifest:
            return str(manifest[key])
    raise KeyError(f"no run id in {run_dir}/manifest.json")


if __name__ == "__main__":
    raise SystemExit(main())
