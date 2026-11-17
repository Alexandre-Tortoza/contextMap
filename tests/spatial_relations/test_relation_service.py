"""Deriving relations as one operation of the capability, not a sequence assembled outside it.

The stage used to be composed in the runtime: resolve geometry, derive the interaction reach,
generate candidates, evaluate each declared channel, decide. That sequence is the science of this
capability, so it has to produce exactly the same relations when the capability owns it.
"""

from __future__ import annotations

import dataclasses

import pytest
from relation_resolution_fixture import er_scene_source, write_er_run
from relation_scene import CONNECTED_POLICY

from contextmap.entity_resolution import EntityResolutionRunReader, ResolvedEntitySet
from contextmap.geometric_mapping import GeometrySource
from contextmap.spatial_relations import (
    CandidatePolicy,
    ContactPredicatePolicy,
    FrameConventions,
    GeometricPredicatePolicy,
    RelationEvidence,
    RelationEvidenceChannel,
    RelationPredicate,
    SemanticPredicatePolicy,
)
from contextmap.spatial_relations.candidates import (
    generate_relation_candidates,
    resolve_interaction_reach,
)
from contextmap.spatial_relations.contact_predicates import evaluate_contact_candidates
from contextmap.spatial_relations.decision import decide_relations
from contextmap.spatial_relations.geometric_predicates import evaluate_geometric_candidates
from contextmap.spatial_relations.policies import RelationsRunPolicies
from contextmap.spatial_relations.resolved_geometry import resolved_entity_geometries
from contextmap.spatial_relations.semantic_predicates import (
    evaluate_interaction_candidates,
    interaction_radii,
)
from contextmap.spatial_relations.service import DerivedRelations, derive_relations

P = RelationPredicate
CONVENTIONS = FrameConventions(map_frame="map")
GEOMETRIC = GeometricPredicatePolicy(
    boundary_tolerance_m=0.02,
    next_to_max_gap_m=0.6,
    adjacent_penetration_m=0.05,
    containment_slack_m=0.05,
    directional_overlap_fraction=0.5,
)
CONTACT = ContactPredicatePolicy(
    contact_distance_m=0.05,
    contact_tolerance_m=0.02,
    min_contact_points=3,
    support_height_tolerance_m=0.05,
    support_footprint_fraction=0.5,
    leaning_min_tilt_deg=10.0,
    leaning_max_tilt_deg=80.0,
    tilt_tolerance_deg=2.0,
    leaning_min_vertical_overlap_m=0.3,
)
CANDIDATES = CandidatePolicy(
    predicates=(P.NEXT_TO, P.TOUCHING), proximity_radius_m=0.6, directional_radius_m=1.0
)
POLICIES = RelationsRunPolicies(
    frame_conventions=CONVENTIONS,
    candidate=CANDIDATES,
    geometry_summary=CONNECTED_POLICY,
    geometric=GEOMETRIC,
    contact=CONTACT,
)


@pytest.fixture(scope="module")
def resolved(tmp_path_factory: pytest.TempPathFactory) -> ResolvedEntitySet:
    directory = tmp_path_factory.mktemp("resolution") / "run"
    write_er_run(directory)
    return EntityResolutionRunReader(directory).resolved_entities()


@pytest.fixture(scope="module")
def source() -> GeometrySource:
    return er_scene_source()


def _by_hand(
    resolved: ResolvedEntitySet, source: GeometrySource, policies: RelationsRunPolicies
) -> DerivedRelations:
    """The sequence the runtime used to assemble, kept here as the expected behavior."""
    radii = interaction_radii(resolved.entities)
    resolved_policies = dataclasses.replace(
        policies, candidate=resolve_interaction_reach(policies.candidate, radii)
    )
    entities = resolved_entity_geometries(
        resolved, source=source, policy=resolved_policies.geometry_summary
    )
    candidates = generate_relation_candidates(
        entities, policy=resolved_policies.candidate, conventions=policies.frame_conventions
    )
    evidence: list[RelationEvidence] = []
    if policies.geometric is not None:
        evidence.extend(
            evaluate_geometric_candidates(
                candidates,
                entities=entities,
                policy=policies.geometric,
                conventions=policies.frame_conventions,
            )
        )
    if policies.contact is not None:
        evidence.extend(
            evaluate_contact_candidates(
                candidates,
                entities=entities,
                geometry_source=source,
                policy=policies.contact,
                conventions=policies.frame_conventions,
            )
        )
    if policies.semantic is not None:
        evidence.extend(
            evaluate_interaction_candidates(
                candidates,
                entities=entities,
                radii=radii,
                policy=policies.semantic,
                conventions=policies.frame_conventions,
            )
        )
    return DerivedRelations(
        policies=resolved_policies,
        candidates=candidates,
        evidence=tuple(evidence),
        decisions=decide_relations(candidates, evidence),
    )


def test_the_operation_reproduces_the_sequence_the_runtime_assembled(
    resolved: ResolvedEntitySet, source: GeometrySource
) -> None:
    derived = derive_relations(resolved, source=source, policies=POLICIES)
    expected = _by_hand(resolved, source, POLICIES)
    assert derived.candidates == expected.candidates
    assert derived.evidence == expected.evidence
    assert derived.decisions == expected.decisions
    assert derived.policies == expected.policies


def test_the_relations_are_decided_and_traceable(
    resolved: ResolvedEntitySet, source: GeometrySource
) -> None:
    derived = derive_relations(resolved, source=source, policies=POLICIES)
    assert derived.candidates.candidates
    assert derived.decisions.relations
    # Toda decisão corresponde a uma relação, e nenhuma relação chega sem evidência que a decidiu.
    assert len(derived.decisions.decisions) == len(derived.decisions.relations)
    identities = {item.evidence_id for item in derived.evidence}
    for relation in derived.decisions.relations:
        assert set(relation.relation_evidence_refs) <= identities


def test_a_channel_the_policies_do_not_declare_produces_no_evidence(
    resolved: ResolvedEntitySet, source: GeometrySource
) -> None:
    geometry_only = dataclasses.replace(POLICIES, contact=None)
    derived = derive_relations(resolved, source=source, policies=geometry_only)
    channels = {item.channel for item in derived.evidence}
    assert channels == {RelationEvidenceChannel.GEOMETRY}


def test_the_operation_never_produces_observation_evidence(
    resolved: ResolvedEntitySet, source: GeometrySource
) -> None:
    # O canal OBSERVATION tem contratos e testes próprios, mas não é composto por esta operação:
    # nenhum artifact upstream da pipeline entrega statements. Ver observation_channel.
    derived = derive_relations(resolved, source=source, policies=POLICIES)
    assert all(item.channel is not RelationEvidenceChannel.OBSERVATION for item in derived.evidence)


def test_the_recorded_candidate_policy_is_the_one_the_reach_was_derived_into(
    resolved: ResolvedEntitySet, source: GeometrySource
) -> None:
    # Um alcance de interação derivado vira o maior raio das entidades resolvidas deste run, e é a
    # política resolvida — não a configurada — que gera os candidatos e que o run grava.
    derived_reach = dataclasses.replace(
        POLICIES,
        candidate=dataclasses.replace(
            CANDIDATES,
            predicates=(P.NEXT_TO, P.TOUCHING, P.WITHIN_INTERACTION_RANGE),
            interaction_radius_m=None,
            derive_interaction_radius=True,
        ),
        semantic=SemanticPredicatePolicy(boundary_tolerance_m=0.02),
    )
    radii = interaction_radii(resolved.entities)
    derived = derive_relations(resolved, source=source, policies=derived_reach)
    expected = resolve_interaction_reach(derived_reach.candidate, radii)
    assert derived.policies.candidate == expected
    assert derived.candidates.provenance.configuration_fingerprint == expected.fingerprint()
