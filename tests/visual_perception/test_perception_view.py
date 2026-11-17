from dataclasses import fields

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PerceptionResult


def test_a_perception_view_has_selected_producers_and_no_execution_identity() -> None:
    view = PerceptionResult(
        source_observation_id=SourceObservationId("frame"),
        sequence_artifact_id="sequence",
        region_artifact_id="regions",
    )

    assert view.regions == ()
    assert {field.name for field in fields(view)}.isdisjoint({"run_id", "result_id", "created_at"})


def test_the_view_rejects_a_claim_from_another_selected_semantic_producer() -> None:
    claim = SemanticClaim(
        claim_id=ClaimId("claim"),
        source_observation_id=SourceObservationId("frame"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("semantics-other"),
        hypothesis="room",
        role=HypothesisRole.PRIMARY,
        provenance=SemanticInferenceProvenance(
            backend=BackendProvenance(
                backend_id="fake",
                capability="semantic_interpreter",
                provider="fake",
                model="fake",
                version="1",
            ),
            task_identity="scene/1",
            prompt_template_id="scene/v1",
            output_schema_version="1",
        ),
    )

    with pytest.raises(ValueError, match=r"semantic.*producer"):
        PerceptionResult(
            source_observation_id=claim.source_observation_id,
            sequence_artifact_id="sequence",
            region_artifact_id="regions",
            semantic_interpretation_artifact_id="semantics-selected",
            claims=(claim,),
        )
