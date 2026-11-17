"""Deriving the relations of one run: the capability's own sequence, from entities to decisions.

Resolving the geometry of the entities, deriving the interaction reach, narrowing the pairs worth
measuring, measuring each declared channel and deciding the relations is one coherent scientific
transformation, and every step of it constrains the next: the candidate set carries the fingerprint
of the policy that generated it, and the evaluators refuse candidates generated under other
conventions. Assembling that sequence outside the capability would put its order, its optional
channels and the identity of the policy a run records in the hands of the caller.

So the sequence lives here, and the composition root keeps what is really its own: selecting the
upstream artifacts, supplying the policies, choosing where the run is written and persisting it.
This operation performs no I/O of its own -- it reads the geometric map through the read boundary
it is given -- and it never writes an artifact.

The ``OBSERVATION`` channel is deliberately not part of it. That channel has its own contracts,
rules and tests (see :mod:`contextmap.spatial_relations.observation_channel`), but it needs upstream
relational statements, and no artifact of the pipeline produces them today. Taking statements here
would suggest the canonical path supports a channel that every real execution leaves empty, and the
decision policy ignores observation evidence for the verdict anyway.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from contextmap.entity_resolution import ResolvedEntitySet
from contextmap.geometric_mapping import GeometrySource
from contextmap.spatial_relations.candidates import (
    RelationCandidateSet,
    generate_relation_candidates,
    resolve_interaction_reach,
)
from contextmap.spatial_relations.contact_predicates import evaluate_contact_candidates
from contextmap.spatial_relations.decision import RelationDecisionResult, decide_relations
from contextmap.spatial_relations.evidence import RelationEvidence
from contextmap.spatial_relations.geometric_predicates import evaluate_geometric_candidates
from contextmap.spatial_relations.policies import RelationsRunPolicies
from contextmap.spatial_relations.resolved_geometry import resolved_entity_geometries
from contextmap.spatial_relations.semantic_predicates import (
    evaluate_interaction_candidates,
    interaction_radii,
)


@dataclass(frozen=True, kw_only=True)
class DerivedRelations:
    """Everything one derivation produced, ready to be persisted as a run.

    Attributes:
        policies: The *effective* policies, which may differ from the ones supplied: a candidate
            policy that derives its interaction reach is resolved against the entities of this
            run. These are the policies that generated the candidates, so these are the ones a
            run must record.
        candidates: The pairs that were worth measuring, and why the others were not.
        evidence: What every declared channel measured, in channel order.
        decisions: The relations and the rule behind each one.
    """

    policies: RelationsRunPolicies
    candidates: RelationCandidateSet
    evidence: tuple[RelationEvidence, ...]
    decisions: RelationDecisionResult


def derive_relations(
    resolved: ResolvedEntitySet, *, source: GeometrySource, policies: RelationsRunPolicies
) -> DerivedRelations:
    """Relate the entities of one resolution run over the geometric map they were resolved on.

    Args:
        resolved: The resolved entities of one Entity Resolution run.
        source: The read boundary of the geometric map those entities belong to. The contact
            channel resolves points through it; the other channels only read bounds.
        policies: The policies of the run. The channels that are ``None`` produce no evidence, and
            a channel that is declared always runs.

    Returns:
        The effective policies, the candidate set, the evidence of every declared channel and the
        decided relations.

    Raises:
        ValueError: If the candidate policy derives its interaction reach and no resolved entity
            declares a radius while the interaction predicate is the only one requested, or if the
            geometry read from the map is not the geometry the resolution persisted.
        IncompatibleFrameError: If the entities are not in the frame the conventions declare, or
            they come from more than one geometric map.
        UndeclaredAxisError: If a requested predicate needs an axis the conventions do not declare.
    """
    radii = interaction_radii(resolved.entities)
    # Um alcance de interação derivado vira o maior raio das entidades resolvidas deste run; a
    # política resolvida é a que gera os candidatos e a que o run grava.
    effective = dataclasses.replace(
        policies, candidate=resolve_interaction_reach(policies.candidate, radii)
    )
    conventions = effective.frame_conventions
    entities = resolved_entity_geometries(
        resolved, source=source, policy=effective.geometry_summary
    )
    candidates = generate_relation_candidates(
        entities, policy=effective.candidate, conventions=conventions
    )
    evidence: list[RelationEvidence] = []
    if effective.geometric is not None:
        evidence.extend(
            evaluate_geometric_candidates(
                candidates,
                entities=entities,
                policy=effective.geometric,
                conventions=conventions,
            )
        )
    if effective.contact is not None:
        evidence.extend(
            evaluate_contact_candidates(
                candidates,
                entities=entities,
                geometry_source=source,
                policy=effective.contact,
                conventions=conventions,
            )
        )
    if effective.semantic is not None:
        evidence.extend(
            evaluate_interaction_candidates(
                candidates,
                entities=entities,
                radii=radii,
                policy=effective.semantic,
                conventions=conventions,
            )
        )
    return DerivedRelations(
        policies=effective,
        candidates=candidates,
        evidence=tuple(evidence),
        decisions=decide_relations(candidates, evidence),
    )
