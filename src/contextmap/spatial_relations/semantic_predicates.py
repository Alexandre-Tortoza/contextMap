"""The semantic interaction channel: does the object fall within the subject's interaction radius?

This channel is deliberately separate from the geometric and contact channels. Those decide a
relation from geometry alone -- no semantic label reaches their arithmetic. The interaction radius,
by contrast, is a *semantic* property of the subject, derived upstream from the interpreter's
interaction factor and the entity's own measured size (``semantic_mapping`` attaches it as the
``interaction_radius_m`` attribute). So it gets its own channel: the geometric invariant stays
intact, and the semantics enter only here.

The predicate ``WITHIN_INTERACTION_RANGE`` is directed and asymmetric. ``a WITHIN_INTERACTION_RANGE
b`` says the bounds of ``b`` fall within ``a``'s interaction radius; it says nothing about whether
``a`` falls within ``b``'s radius, because the radii differ per entity. The rule compares the
measured bounds gap between the two supports with the subject's radius, under a boundary tolerance,
exactly as the geometric evaluators compare a gap with a policy threshold -- the only difference is
that the threshold is a per-subject datum, not a configured constant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from contextmap.entity_resolution import ResolvedEntity, ResolvedEntityReference
from contextmap.semantic_mapping import EntityGeometry
from contextmap.spatial_relations._assessment import (
    Assessment,
    at_most,
    decide,
    measured_geometry,
    quantity,
)
from contextmap.spatial_relations._bounds import bounds_gap_m
from contextmap.spatial_relations._checks import require_finite
from contextmap.spatial_relations.candidates import RelationCandidateSet
from contextmap.spatial_relations.evidence import (
    RelationEvidence,
    RelationEvidenceChannel,
    RelationEvidenceProvenance,
    evidence_id_for,
)
from contextmap.spatial_relations.frame_conventions import FrameConventions
from contextmap.spatial_relations.taxonomy import (
    TAXONOMY_VERSION,
    RelationPredicate,
    predicate_spec,
)

SEMANTIC_POLICY_ID = "interaction-range-predicate-v1"
"""Versioned identity of the evaluation rule described in this module."""

SEMANTIC_PREDICATES = frozenset({RelationPredicate.WITHIN_INTERACTION_RANGE})
"""The predicates this channel evaluates directly."""

INTERACTION_RADIUS_ATTRIBUTE = "interaction_radius_m"
"""The entity attribute Semantic Mapping derives the interaction radius into, in meters."""

_RULE_ID = "interaction-radius-within-v1"


def interaction_radii(
    entities: Iterable[ResolvedEntity],
) -> dict[ResolvedEntityReference, float]:
    """The interaction radius, in meters, of every resolved entity that declares one.

    The radius is the ``interaction_radius_m`` attribute Semantic Mapping derived upstream from the
    interpreter's factor and the entity's measured size. An entity without the attribute, or whose
    value is not a finite positive number, is omitted: it simply declares no interaction zone, and
    the semantic channel produces no evidence with it as the subject.

    Args:
        entities: The resolved entities of one resolution artifact.

    Returns:
        The radius of each entity that carries a usable one, by reference.
    """
    radii: dict[ResolvedEntityReference, float] = {}
    for entity in entities:
        for attribute in entity.semantic_state.attributes:
            if attribute.name != INTERACTION_RADIUS_ATTRIBUTE:
                continue
            try:
                radius = float(attribute.value)
            except (TypeError, ValueError):
                break
            if isfinite(radius) and radius > 0.0:
                radii[entity.reference] = radius
            break
    return radii


@dataclass(frozen=True, kw_only=True)
class SemanticPredicatePolicy:
    """The tolerance of the interaction-range evaluator.

    The radius itself is a per-subject datum carried into the evaluator, not a policy threshold, so
    the only configuration here is the uncertainty band around comparing the bounds gap with it.

    Attributes:
        boundary_tolerance_m: Uncertainty of a bounds face, in meters. A comparison whose outcome
            depends on where inside this tolerance the true face lies is ambiguous.
    """

    boundary_tolerance_m: float

    def __post_init__(self) -> None:
        """Validate the tolerance.

        Raises:
            ValueError: If the tolerance is not finite and positive.
        """
        require_finite("boundary_tolerance_m", self.boundary_tolerance_m)
        if self.boundary_tolerance_m <= 0.0:
            raise ValueError(
                f"boundary_tolerance_m must be positive, got {self.boundary_tolerance_m!r}"
            )

    def fingerprint(self) -> str:
        """Hash the policy identity and tolerance, for provenance.

        Returns:
            ``sha256:`` followed by the digest of the canonical configuration.
        """
        canonical = json.dumps(
            {"policy_id": SEMANTIC_POLICY_ID, "boundary_tolerance_m": self.boundary_tolerance_m},
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def evaluate_interaction_candidate(
    *,
    subject_entity_ref: ResolvedEntityReference,
    subject_geometry: EntityGeometry,
    subject_interaction_radius_m: float,
    object_entity_ref: ResolvedEntityReference,
    object_geometry: EntityGeometry,
    policy: SemanticPredicatePolicy,
    conventions: FrameConventions,
) -> RelationEvidence:
    """Measure ``WITHIN_INTERACTION_RANGE`` for one directed pair.

    The subject's interaction radius is a semantic property derived upstream; this rule only
    compares it with the measured bounds gap, so the geometric channel's invariant -- no semantics
    in its arithmetic -- is untouched, and the semantics enter only here, in the semantic channel.

    Args:
        subject_entity_ref: The entity the statement is about; its radius decides the reach.
        subject_geometry: Its geometry.
        subject_interaction_radius_m: The subject's interaction radius, in meters, finite and > 0.
        object_entity_ref: The entity it may be related to.
        object_geometry: Its geometry.
        policy: The tolerance.
        conventions: The axes the run declared for the map frame.

    Returns:
        The evidence in the ``SEMANTIC`` channel: the status, the bounds gap and the radius it was
        compared with, the geometry the measurement was taken over and any caveat that limits it.

    Raises:
        ValueError: If the radius is not finite and positive, or the entities are not a relatable
            pair.
        IncompatibleFrameError: If a geometry is not in the map frame of ``conventions`` or the two
            belong to different geometric maps.
    """
    require_finite("subject_interaction_radius_m", subject_interaction_radius_m)
    if subject_interaction_radius_m <= 0.0:
        raise ValueError(
            f"subject_interaction_radius_m must be positive, got {subject_interaction_radius_m!r}"
        )
    predicate = RelationPredicate.WITHIN_INTERACTION_RANGE
    spec = predicate_spec(predicate)
    conventions.require_evaluable(spec.frame_requirement, subject_geometry, object_geometry)
    gap = bounds_gap_m(subject_geometry.bounds, object_geometry.bounds)
    assessment = Assessment(
        measurements=[quantity("bounds_gap", gap, "m")],
        thresholds=[
            quantity("interaction_radius", subject_interaction_radius_m, "m"),
            quantity("boundary_tolerance", policy.boundary_tolerance_m, "m"),
        ],
        checks=[
            at_most("bounds_gap", gap, subject_interaction_radius_m, policy.boundary_tolerance_m),
        ],
    )
    status, caveats = decide(assessment, subject_geometry, object_geometry)
    return RelationEvidence(
        evidence_id=evidence_id_for(
            channel=RelationEvidenceChannel.SEMANTIC,
            subject_entity_ref=subject_entity_ref,
            predicate=predicate,
            object_entity_ref=object_entity_ref,
        ),
        channel=RelationEvidenceChannel.SEMANTIC,
        subject_entity_ref=subject_entity_ref,
        predicate=predicate,
        object_entity_ref=object_entity_ref,
        status=status,
        measurements=tuple(sorted(assessment.measurements, key=lambda item: item.name)),
        thresholds=tuple(sorted(assessment.thresholds, key=lambda item: item.name)),
        geometry=(
            measured_geometry("object", object_geometry),
            measured_geometry("subject", subject_geometry),
        ),
        caveats=tuple(sorted(caveats, key=lambda item: (item.kind.value, item.detail))),
        provenance=RelationEvidenceProvenance(
            rule_id=_RULE_ID,
            configuration_fingerprint=policy.fingerprint(),
            taxonomy_version=TAXONOMY_VERSION,
            map_frame=conventions.map_frame,
            geometric_map_id=subject_geometry.geometric_map_id,
            frame_conventions_fingerprint=None,
        ),
    )


def evaluate_interaction_candidates(
    candidates: RelationCandidateSet,
    *,
    entities: Mapping[ResolvedEntityReference, EntityGeometry],
    radii: Mapping[ResolvedEntityReference, float],
    policy: SemanticPredicatePolicy,
    conventions: FrameConventions,
) -> tuple[RelationEvidence, ...]:
    """Evaluate the interaction candidates of a candidate set, in candidate order.

    Candidates of other channels are left to their own evaluators; nothing is dropped. A candidate
    whose subject declares no interaction radius is skipped: without a radius there is no zone to
    compare the gap with, so the pair simply produces no semantic evidence.

    Args:
        candidates: The output of candidate generation.
        entities: The geometry of every resolved entity the candidates name.
        radii: The interaction radius, in meters, of the entities that declare one, by reference.
        policy: The tolerance.
        conventions: The axes the run declared; they must be the ones the candidates were
            generated under.

    Returns:
        One evidence record per interaction candidate whose subject declares a radius.

    Raises:
        ValueError: If the candidates were generated under other frame conventions.
    """
    if candidates.provenance.frame_conventions_fingerprint != conventions.fingerprint():
        raise ValueError(
            "the candidates were generated under other frame conventions than the ones given"
        )
    return tuple(
        evaluate_interaction_candidate(
            subject_entity_ref=candidate.subject_entity_ref,
            subject_geometry=entities[candidate.subject_entity_ref],
            subject_interaction_radius_m=radii[candidate.subject_entity_ref],
            object_entity_ref=candidate.object_entity_ref,
            object_geometry=entities[candidate.object_entity_ref],
            policy=policy,
            conventions=conventions,
        )
        for candidate in candidates.candidates
        if candidate.predicate in SEMANTIC_PREDICATES and candidate.subject_entity_ref in radii
    )
