"""Producer-qualified evidence does not multiply physical observations."""

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_fusion import VisualFeatureRef
from contextmap.visual_perception import FeatureId, FeatureScope


def test_feature_reference_names_visual_artifact_and_physical_observation() -> None:
    reference = VisualFeatureRef(
        visual_artifact_id="visual",
        source_observation_id=SourceObservationId("obs"),
        feature_id=FeatureId("feature"),
        embedding_space_id="space",
        scope=FeatureScope.REGION,
        region_id=RegionId("region"),
    )
    assert reference.visual_artifact_id == "visual"


def test_shared_semantic_artifact_across_visual_variants_contributes_once() -> None:
    from dataclasses import replace

    from evidence_builders import ClaimSpec, View, build_scenario

    from contextmap.semantic_fusion import (
        BaselineAccumulationPolicy,
        EvidenceChannel,
        accumulate_baseline_evidence,
    )
    from contextmap.shared import BackendProvenance
    from contextmap.visual_perception import VisualFeature

    scenario = build_scenario([View("run-a", "frame-0120", (ClaimSpec("door"),))])
    key, (base,) = next(iter(scenario.results.items()))
    feature = VisualFeature(
        feature_id=FeatureId("local-feature"),
        source_observation_id=base.source_observation_id,
        scope=FeatureScope.REGION,
        region_id=RegionId("region-0001"),
        embedding_space_id="space",
        shape=(2,),
        dtype="float32",
        payload_reference="features/a.npy",
        provenance=BackendProvenance(
            backend_id="fake",
            capability="feature_extraction",
            provider="fake",
            model="fake",
            version="1",
        ),
    )
    views = tuple(
        replace(base, visual_evidence_artifact_id=producer, features=(feature,))
        for producer in ("visual-a", "visual-b", "visual-a")
    )
    fused = accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results={key: views},
        policy=BaselineAccumulationPolicy(
            channels=frozenset({EvidenceChannel.SEMANTIC_CLAIMS, EvidenceChannel.VISUAL_FEATURES})
        ),
    )
    assert len(fused.contributions) == 1
    assert len(fused.hypotheses[0].evidence) == 1
    assert fused.physical_observation_groups[0].inference_result_count == 1
    assert [
        (ref.visual_artifact_id, ref.feature_id)
        for ref in fused.contributions[0].visual_feature_refs
    ] == [("visual-a", "local-feature"), ("visual-b", "local-feature")]


def test_claims_require_an_explicit_semantic_producer() -> None:
    from dataclasses import replace

    import pytest
    from fusion_builders import make_contribution

    with pytest.raises(ValueError, match="claims require a semantic producer"):
        replace(make_contribution(), semantic_interpretation_artifact_id=None)


def test_visual_reference_must_describe_the_contributions_physical_observation() -> None:
    from dataclasses import replace

    import pytest
    from fusion_builders import make_contribution, make_region_feature

    foreign = replace(make_region_feature(), source_observation_id=SourceObservationId("other"))
    with pytest.raises(ValueError, match="feature source observation"):
        replace(make_contribution(), visual_feature_refs=(foreign,))


def test_empty_semantic_envelope_keeps_coverage_without_inventing_a_hypothesis() -> None:
    from evidence_builders import View, build_scenario

    from contextmap.semantic_fusion import accumulate_baseline_evidence

    scenario = build_scenario([View("run-a", "frame-0120", ())])
    fused = accumulate_baseline_evidence(
        scenario.support,
        observations=scenario.observations,
        grouping=scenario.grouping,
        perception_results=scenario.results,
    )
    assert len(fused.contributions) == 1
    assert fused.hypotheses == ()
    assert len(fused.physical_observation_groups) == 1
    assert fused.physical_observation_groups[0].inference_result_count == 1
    assert fused.physical_observation_groups[0].semantic_interpretation_artifact_ids == (
        "run-a--frame-0120",
    )
