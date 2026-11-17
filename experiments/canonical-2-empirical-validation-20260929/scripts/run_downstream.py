"""Downstream of one perception run: semantic_scoring -> sensor_association -> semantic_fusion.

On a canonical/2 (pooling-on) perception run this exercises the PR #688 round-1 fixes on real
data: the CLIP scorer ignores the pooled DINO region features (other embedding space) and still
scores the CLIP crops (#1); the raw scores reach fusion's semantic_scores channel and the fusion
lineage records the scoring run (#2/#3); Region2D -> SpatialObservation -> FusedEvidence survives.

Executors are composed from the same config; providers/checkpoints come from it, not from here.

Usage:
  run_downstream.py --config base.toml [--config arm.toml] --workspace DIR --sequence-id ID
      --perception-subdir pool_on_f0 --state-id ID --geo-id ID --out-prefix f0on
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contextmap.runtime import ArtifactRef, StageRequest, resolve_effective_config
from contextmap.runtime.composition import compose_executors
from contextmap.semantic_fusion import SemanticFusionRunReader
from contextmap.sensor_association import SensorAssociationRunReader
from contextmap.visual_perception import PerceptionRunReader, SemanticScoringRunReader


def _ref(stage: str, contract: str, artifact_id: str, location: str) -> ArtifactRef:
    return ArtifactRef(
        stage_id=stage,
        contract=contract,
        artifact_id=artifact_id,
        content_hash="sha256:placeholder",
        location=location,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--perception-subdir", required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--geo-id", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    ws, pipe = args.workspace, args.workspace / "pipeline"
    effective = resolve_effective_config(files=list(args.config))
    executors = compose_executors(
        effective,
        module_available=lambda _m: True,
        environ={},
        code_version="canonical-2-validation",
    )

    perception_dir = pipe / args.perception_subdir
    perception_id = str(PerceptionRunReader(perception_dir).manifest.run_id)
    perception_ref = _ref(
        "visual_perception",
        "PerceptionRunArtifact",
        perception_id,
        f"pipeline/{args.perception_subdir}",
    )
    sequence_ref = _ref(
        "ingestion",
        "SequenceArtifact",
        args.sequence_id,
        f"sequences/corridor-02/{args.sequence_id}",
    )
    state_ref = _ref(
        "state_estimation", "StateEstimationRunArtifact", args.state_id, "pipeline/state_estimation"
    )
    geo_ref = _ref(
        "geometric_mapping", "GeometricMapArtifact", args.geo_id, "pipeline/geometric_mapping"
    )

    scoring_out = pipe / f"{args.out_prefix}_semantic_scoring"
    scoring_ref = executors["semantic_scoring"].execute(
        StageRequest(
            stage_id="semantic_scoring",
            inputs={"perception": (perception_ref,)},
            components={},
            config_digest=f"{args.out_prefix}-scoring",
            output_dir=scoring_out,
            workspace=ws,
        )
    )
    sr = SemanticScoringRunReader(scoring_out)
    print(
        "SCORING_ID",
        scoring_ref.artifact_id,
        "SCORES",
        len(sr.list_scores()),
        "unscored_region_claims",
        sr.manifest.counts.unscored_region_claim_count,
        "verify",
        sr.verify_integrity(),
    )

    assoc_out = pipe / f"{args.out_prefix}_sensor_association"
    assoc_ref = executors["sensor_association"].execute(
        StageRequest(
            stage_id="sensor_association",
            inputs={
                "sequence": (sequence_ref,),
                "trajectory": (state_ref,),
                "geometry": (geo_ref,),
                "perception": (perception_ref,),
            },
            components={},
            config_digest=f"{args.out_prefix}-association",
            output_dir=assoc_out,
            workspace=ws,
        )
    )
    ar = SensorAssociationRunReader(assoc_out)
    print(
        "ASSOCIATION_ID",
        assoc_ref.artifact_id,
        "SPATIAL_OBSERVATIONS",
        len(list(ar.observations())),
        "verify",
        ar.verify_integrity(),
    )

    fusion_out = pipe / f"{args.out_prefix}_semantic_fusion"
    fusion_ref = executors["semantic_fusion"].execute(
        StageRequest(
            stage_id="semantic_fusion",
            inputs={
                "sequence": (sequence_ref,),
                "association": (
                    _ref(
                        "sensor_association",
                        "SensorAssociationRunArtifact",
                        assoc_ref.artifact_id,
                        f"pipeline/{args.out_prefix}_sensor_association",
                    ),
                ),
                "perception": (perception_ref,),
                "geometry": (geo_ref,),
                "semantic_scores": (
                    _ref(
                        "semantic_scoring",
                        "SemanticScoringRunArtifact",
                        scoring_ref.artifact_id,
                        f"pipeline/{args.out_prefix}_semantic_scoring",
                    ),
                ),
            },
            components={},
            config_digest=f"{args.out_prefix}-fusion",
            output_dir=fusion_out,
            workspace=ws,
        )
    )
    fr = SemanticFusionRunReader(fusion_out)
    m = fr.manifest
    print(
        "FUSION_ID",
        fusion_ref.artifact_id,
        "supports",
        m.support_count,
        "contributions",
        m.contribution_count,
        "verify",
        fr.verify_integrity(),
    )
    print("fusion semantic_scores identities", m.evidence_identities.get("semantic_scores"))
    print("fusion visual_features identities", m.evidence_identities.get("visual_features"))
    print("fusion lineage semantic_scoring_run_ids", m.lineage.semantic_scoring_run_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
