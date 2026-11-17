"""The v0.1.0 scope freeze and how the code may move past it.

``docs/release-v0.1.0.md`` freezes what v0.1.0 shipped, and ``V0_1_0_*`` below is that record: it
never changes. A frozen scope that only lives in prose drifts, so every public schema version the
code writes is checked against it. A version may leave the record only with its transition
(``<name> <old> → <new>``) under ``[Não lançado]`` in ``CHANGELOG.md``: a schema change is never
silent, and the released artifacts stay described as they were. The frozen demo in
``examples/v0.1.0`` stays readable (``test_release_examples.py``).

The canonical profile and the acceptance contract the release shipped remain pinned here. The
live scenario may be frozen again past it -- 1.0.6 re-expressed the profile's component ids
against the decomposed perception capabilities and promoted ``region_perception`` to a stage --
but the release contract stays 1.0.5, and the divergence has to be written in both release
documents rather than inferred.
"""

from __future__ import annotations

from pathlib import Path

from release_record import unrecorded_changes

from contextmap.artifact.versioning import CONTEXT_MAP_SCHEMA_VERSION
from contextmap.entity_resolution.run_artifact import SCHEMA_VERSION as ENTITY_RESOLUTION_RUN
from contextmap.evaluation import SCENARIO_ID, SCENARIO_VERSION
from contextmap.geometric_mapping.run_artifact import SCHEMA_VERSION as GEOMETRIC_MAPPING_RUN
from contextmap.ingestion.calibration import SCHEMA_VERSION as CALIBRATION
from contextmap.ingestion.sequence_artifact import SCHEMA_VERSION as SEQUENCE_ARTIFACT
from contextmap.ingestion.sequence_observation_diagnostics import (
    SCHEMA_VERSION as INGESTION_DIAGNOSTICS,
)
from contextmap.ingestion.sequence_provenance import SCHEMA_VERSION as SEQUENCE_PROVENANCE
from contextmap.point_representation.run_artifact import SCHEMA_VERSION as POINT_REPRESENTATION_RUN
from contextmap.region_perception import REGION_PERCEPTION_SCHEMA_VERSION
from contextmap.region_perception.artifacts.mask_store import MASK_INDEX_SCHEMA_VERSION
from contextmap.runtime.components import CANONICAL_PROFILE_ID
from contextmap.runtime.configuration import CONFIG_SCHEMA_VERSION
from contextmap.runtime.ingestion_service import INGESTION_REQUEST_SCHEMA_VERSION
from contextmap.runtime.lifecycle import RUN_SCHEMA_VERSION
from contextmap.runtime.pipeline import PLAN_SCHEMA_VERSION
from contextmap.runtime.reuse import REUSE_SCHEMA_VERSION
from contextmap.runtime.selection import CATALOG_SCHEMA_VERSION
from contextmap.semantic_fusion.run_artifact import SCHEMA_VERSION as SEMANTIC_FUSION_RUN
from contextmap.semantic_interpretation import SEMANTIC_INTERPRETATION_SCHEMA_VERSION
from contextmap.semantic_mapping.run_artifact import SCHEMA_VERSION as SEMANTIC_MAPPING_RUN
from contextmap.semantic_mapping.serialization import ENTITY_SCHEMA_VERSION
from contextmap.sensor_association.artifacts import SCHEMA_VERSION as SENSOR_ASSOCIATION_RUN
from contextmap.spatial_relations.run_artifact import SCHEMA_VERSION as SPATIAL_RELATIONS_RUN
from contextmap.state_estimation.run_artifact import SCHEMA_VERSION as STATE_ESTIMATION_RUN
from contextmap.visual_perception import VISUAL_EVIDENCE_SCHEMA_VERSION
from contextmap.visual_perception.execution.pipeline_resolution import PIPELINE_SCHEMA_VERSION
from contextmap.visual_perception.visual_features.feature_payload_store import (
    FEATURE_INDEX_SCHEMA_VERSION,
)
from contextmap.visual_perception.visual_features.visual_evidence import (
    EMBEDDING_SPACE_SCHEMA_VERSION as EMBEDDING_SPACE,
)

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "CHANGELOG.md"
RELEASE_SCOPE = ROOT / "docs/release-v0.1.0.md"
RELEASE_NOTES = ROOT / "docs/releases/v0.1.0.md"

# Contrato de aceitação que a v0.1.0 lançou. É registro, como as tabelas abaixo: a versão do
# cenário vivo (`SCENARIO_VERSION`) pode avançar, este par não. O id continua sendo verificado
# contra o código; a versão é verificada contra os documentos de release.
V0_1_0_SCENARIO = ("solution-1-canonical", "1.0.5")

# O que a v0.1.0 entregou (docs/release-v0.1.0.md). É registro: nunca muda depois do release.
V0_1_0_ARTIFACTS = {
    "SequenceArtifact": "0.2.0",
    "CalibrationSet": "0.1.0",
    "SequenceProvenance": "0.1.0",
    "ingestion diagnostics": "0.2.0",
    "PerceptionRunArtifact": "0.5.0",
    "perception pipeline": "0.2.0",
    "embedding space": "0.1.0",
    "feature index": "0.1.0",
    "mask index": "0.1.0",
    "StateEstimationRunArtifact": "0.2.0",
    "GeometricMapArtifact": "0.1.0",
    "SensorAssociationRunArtifact": "0.2.0",
    "PointRepresentationRunArtifact": "0.1.0",
    "SemanticFusionRunArtifact": "0.2.0",
    "SemanticEntityArtifact": "0.1.0",
    "entity serialization": "0.1.0",
    "EntityResolutionRunArtifact": "0.1.0",
    "SpatialRelationsRunArtifact": "0.1.0",
    "ContextMapArtifact": "0.1.0",
}
V0_1_0_RUNTIME = {
    "config": "0.1.0",
    "plan": "0.1.0",
    "run": "0.1.0",
    "catalog": "0.1.0",
    "reuse": "0.1.0",
    "ingestion_request": "0.1.0",
}


def _code_artifact_versions() -> dict[str, str]:
    """One entry per public artifact of the canonical pipeline, in topology order."""
    return {
        "SequenceArtifact": SEQUENCE_ARTIFACT,
        "CalibrationSet": CALIBRATION,
        "SequenceProvenance": SEQUENCE_PROVENANCE,
        "ingestion diagnostics": INGESTION_DIAGNOSTICS,
        "RegionPerceptionRunArtifact": REGION_PERCEPTION_SCHEMA_VERSION,
        "VisualEvidenceArtifact": VISUAL_EVIDENCE_SCHEMA_VERSION,
        "SemanticInterpretationArtifact": SEMANTIC_INTERPRETATION_SCHEMA_VERSION,
        "perception pipeline": PIPELINE_SCHEMA_VERSION,
        "embedding space": EMBEDDING_SPACE,
        "feature index": FEATURE_INDEX_SCHEMA_VERSION,
        "mask index": MASK_INDEX_SCHEMA_VERSION,
        "StateEstimationRunArtifact": STATE_ESTIMATION_RUN,
        "GeometricMapArtifact": GEOMETRIC_MAPPING_RUN,
        "SensorAssociationRunArtifact": SENSOR_ASSOCIATION_RUN,
        "PointRepresentationRunArtifact": POINT_REPRESENTATION_RUN,
        "SemanticFusionRunArtifact": SEMANTIC_FUSION_RUN,
        "SemanticEntityArtifact": SEMANTIC_MAPPING_RUN,
        "entity serialization": ENTITY_SCHEMA_VERSION,
        "EntityResolutionRunArtifact": ENTITY_RESOLUTION_RUN,
        "SpatialRelationsRunArtifact": SPATIAL_RELATIONS_RUN,
        "ContextMapArtifact": CONTEXT_MAP_SCHEMA_VERSION,
    }


def _code_runtime_versions() -> dict[str, str]:
    return {
        "config": CONFIG_SCHEMA_VERSION,
        "plan": PLAN_SCHEMA_VERSION,
        "run": RUN_SCHEMA_VERSION,
        "catalog": CATALOG_SCHEMA_VERSION,
        "reuse": REUSE_SCHEMA_VERSION,
        "ingestion_request": INGESTION_REQUEST_SCHEMA_VERSION,
    }


def test_every_artifact_schema_version_is_v0_1_0_or_a_recorded_transition() -> None:
    code = _code_artifact_versions()

    frozen_survivors = {name: version for name, version in V0_1_0_ARTIFACTS.items() if name in code}
    assert "PerceptionRunArtifact" not in code
    assert {
        "RegionPerceptionRunArtifact",
        "VisualEvidenceArtifact",
        "SemanticInterpretationArtifact",
    } <= code.keys()
    assert set(frozen_survivors) == set(code) - {
        "RegionPerceptionRunArtifact",
        "VisualEvidenceArtifact",
        "SemanticInterpretationArtifact",
    }
    unreleased = CHANGELOG.read_text("utf-8").split("## [Não lançado]", 1)[1].split("\n## [", 1)[0]
    assert "PerceptionRunArtifact 0.5.0 → retired" in unreleased
    assert all(
        f"{name} {code[name]}" in unreleased
        for name in (
            "RegionPerceptionRunArtifact",
            "VisualEvidenceArtifact",
            "SemanticInterpretationArtifact",
        )
    )
    current_survivors = {name: code[name] for name in frozen_survivors}
    assert not unrecorded_changes(
        current_survivors, frozen_survivors, CHANGELOG.read_text("utf-8")
    ), (
        "a public artifact schema changed after v0.1.0 without its transition in "
        "CHANGELOG.md [Não lançado] (write '<name> <old> → <new>'); the v0.1.0 record in "
        "docs/release-v0.1.0.md never changes"
    )


def test_every_runtime_schema_version_is_v0_1_0_or_a_recorded_transition() -> None:
    code = _code_runtime_versions()

    assert code.keys() == V0_1_0_RUNTIME.keys()
    assert not unrecorded_changes(code, V0_1_0_RUNTIME, CHANGELOG.read_text("utf-8"))


def test_the_frozen_canonical_profile_and_release_scenario_are_the_ones_documented() -> None:
    """The scope names one topology and one acceptance contract; both are pinned.

    The acceptance contract is pinned to what the release documents say it is, not to the live
    ``SCENARIO_VERSION``. The two are allowed to diverge -- a new scenario version is how a
    frozen decision changes after the release -- but only out loud: when they differ, both
    release documents must name the live version as well, so the contract never drifts with the
    code in silence.
    """
    scenario_id, contract_version = V0_1_0_SCENARIO
    scope, notes = RELEASE_SCOPE.read_text("utf-8"), RELEASE_NOTES.read_text("utf-8")

    assert CANONICAL_PROFILE_ID == "canonical/1"
    assert scenario_id == SCENARIO_ID
    assert all(f"`{scenario_id}` **{contract_version}**" in text for text in (scope, notes)), (
        "docs/release-v0.1.0.md and docs/releases/v0.1.0.md must both name the acceptance "
        f"contract the release shipped (`{scenario_id}` **{contract_version}**)"
    )
    if contract_version != SCENARIO_VERSION:
        assert all(SCENARIO_VERSION in text for text in (scope, notes)), (
            f"the live scenario is {SCENARIO_VERSION} and the v0.1.0 release contract is "
            f"{contract_version}; both release documents must say so instead of leaving the "
            "reader to assume the release was validated against the current scenario"
        )


def test_a_schema_change_since_v0_1_0_must_name_its_transition_in_the_changelog() -> None:
    frozen = {"SpatialRelationsRunArtifact": "0.1.0", "GeometricMapArtifact": "0.1.0"}
    code = {"SpatialRelationsRunArtifact": "0.2.0", "GeometricMapArtifact": "0.1.0"}

    assert unrecorded_changes(code, frozen, "## [Não lançado]\n\n## [0.1.0] - 2026-09-25\n") == [
        "SpatialRelationsRunArtifact 0.1.0 → 0.2.0"
    ]
    recorded = (
        "## [Não lançado]\n\n### Alterado\n\n"
        "- Schema do `SpatialRelationsRunArtifact` 0.1.0 → 0.2.0: exclusões limitadas.\n\n"
        "## [0.1.0] - 2026-09-25\n"
    )
    assert unrecorded_changes(code, frozen, recorded) == []
    # Uma transição registrada numa release passada não vale para o código ainda não lançado.
    released_only = (
        "## [Não lançado]\n\n## [0.1.0] - 2026-09-25\n\n"
        "- `SpatialRelationsRunArtifact` 0.1.0 → 0.2.0\n"
    )
    assert unrecorded_changes(code, frozen, released_only) == [
        "SpatialRelationsRunArtifact 0.1.0 → 0.2.0"
    ]
