"""Entities and relations as the ContextMap composes them.

The map does not own what an entity or a relation *is*: Entity Resolution and Spatial Relations
do, and their identities, states and predicates are used here as they define them. The map owns
how they are composed into one readable whole. Each record therefore states three things: its
identity inside the map, the upstream record it maps to (a :class:`ResolvedEntityReference` for
an entity, the run and relation identities for a relation, never a silent rewrite of the
upstream identity), and the part of the upstream result a consumer needs without opening the
upstream artifact.

Geometry stays authoritative in the geometric-map artifact, so an entity holds
:class:`GeometryReference` values and never coordinates. Uncertainty is never resolved here: an
ambiguous or conflicting entity keeps every hypothesis, and an unresolved relation stays
unresolved. Scores are deliberately not copied; the evidence behind a result stays reachable
through the upstream artifact. What *is* copied, since schema 0.2.0 and only when upstream
derived it, is the belief Entity Resolution recorded (a support status, a primary label only when
supported, and per label a *count* of distinct physical observations) and attributes either stated
by claims or derived from traceable evidence. Both are optional: without them an entity is written
exactly as in 0.1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from contextmap.artifact._checks import require_canonical, require_present
from contextmap.artifact.provenance import EvidenceOrigin
from contextmap.artifact.references import (
    ContextEntityId,
    ContextEntityReference,
    ContextRelationId,
)
from contextmap.entity_resolution import ResolutionDecisionId, ResolvedEntityReference
from contextmap.geometric_mapping import GeometryReference
from contextmap.semantic_mapping import EntityReference
from contextmap.spatial_relations import (
    RelationId,
    RelationPredicate,
    RelationState,
    RelationUncertaintyKind,
    SpatialRelationsRunId,
)


class AmbiguityStatus(Enum):
    """How settled the semantic state of an entity is.

    The value must agree with the number of hypotheses, so it can be read without inspecting
    them, and a competing state can never be collapsed into one label by the map.

    Attributes:
        UNAMBIGUOUS: Exactly one hypothesis exists and nothing competes with it.
        AMBIGUOUS: Several hypotheses compete, without contradicting observations.
        CONFLICTING: Distinct observations back incompatible hypotheses.
        INSUFFICIENT_EVIDENCE: No hypothesis is supported; an abstention, not negative evidence.
    """

    UNAMBIGUOUS = "unambiguous"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


OMIT_WHEN_ABSENT_KEY = "omit_when_absent"
"""Field-metadata key of an optional addition the record leaves out while it holds its default."""

_OMITTED_WHEN_ABSENT = MappingProxyType({OMIT_WHEN_ABSENT_KEY: True})
"""Metadata of the 0.2.0 additions: absent means "not present", and a record without them stays
byte-identical to the 0.1.0 form (see ``src/contextmap/artifact/docs/versioning.md``)."""


class ContextSupportStatus(Enum):
    """What the independent support of an entity justifies about its label.

    It is Entity Resolution's belief (``resolved-belief-min-support-v1``) as the map carries it:
    a derived judgement over counts of distinct physical observations, never a confidence.

    Attributes:
        TENTATIVE: Hypotheses exist, but no label has the minimum support; all stay listed.
        SUPPORTED: Exactly one label has the minimum support; it is the primary label.
        CONTESTED: Two or more labels have the minimum support; there is no primary label.
        INSUFFICIENT_EVIDENCE: The entity has no hypothesis.
    """

    TENTATIVE = "tentative"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, kw_only=True)
class LabelHypothesis:
    """One thing an entity may be, as an open-vocabulary label.

    A label is a semantic hypothesis, not the identity of an object in the world. There is no
    score: the numbers that support it stay in the upstream artifact, which the origin cites.

    Attributes:
        label: The label text, verbatim.
        origin: How the hypothesis was produced and the evidence it rests on.
        supporting_physical_observation_count: How many distinct physical observations support
            the label (labels that differ only typographically share the count), exactly as the
            belief counted them; ``None`` when no belief was derived. A count, never a score.
    """

    label: str
    origin: EvidenceOrigin
    supporting_physical_observation_count: int | None = field(
        default=None, metadata=_OMITTED_WHEN_ABSENT
    )

    def __post_init__(self) -> None:
        """Require the label and a count of at least one when present.

        Raises:
            ValueError: If the label is blank or the count is not an integer of at least one.
        """
        require_present(self, "label")
        count = self.supporting_physical_observation_count
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise ValueError(
                f"supporting_physical_observation_count must be an integer of at least one, "
                f"got {count!r}"
            )


class ContextAttributeOrigin(Enum):
    """Whether an entity property was stated by claims or computed from evidence."""

    OBSERVED = "observed"
    DERIVED = "derived"


@dataclass(frozen=True, kw_only=True)
class ContextAttribute:
    """One observed or derived property of an entity, with its evidence.

    It is a semantic hypothesis, not a fact about the object: several values of one name are
    several attributes, and none is preferred.

    Attributes:
        name: The property (``category``, ``color``, ``material``...), as stated.
        value: Its value, as text, verbatim.
        origin: How the value was produced (a model inference) and the upstream records it rests
            on.
        attribute_origin: Whether claims stated the value or a rule derived it.
        derivation_id: The complete upstream derivation identity, including its effective
            configuration fingerprint when one exists.
    """

    name: str
    value: str
    origin: EvidenceOrigin
    attribute_origin: ContextAttributeOrigin = field(
        default=ContextAttributeOrigin.OBSERVED, metadata=_OMITTED_WHEN_ABSENT
    )
    derivation_id: str | None = field(default=None, metadata=_OMITTED_WHEN_ABSENT)

    def __post_init__(self) -> None:
        """Require the name and the value.

        Raises:
            ValueError: If the name or the value is blank.
        """
        require_present(self, "name", "value")
        if self.derivation_id is not None:
            require_present(self, "derivation_id")
        if self.attribute_origin is ContextAttributeOrigin.DERIVED and self.derivation_id is None:
            raise ValueError("a derived attribute must name its derivation_id")


def _context_attribute_key(item: ContextAttribute) -> tuple[str, str, str, str]:
    """Return the complete attribute identity shared with Entity Resolution."""
    return (
        item.name,
        item.value,
        item.attribute_origin.value,
        item.derivation_id or "",
    )


@dataclass(frozen=True, kw_only=True)
class ContextSemanticState:
    """What an entity may be, with every hypothesis kept.

    Attributes:
        status: How settled the state is.
        hypotheses: Every label hypothesis, sorted by label and unique. The order carries no
            ranking.
        support_status: The belief Entity Resolution derived from the independent support, or
            ``None`` when it derived none. Independent of ``status``: an ambiguous entity can
            have a supported label, and every alternative stays listed.
        primary_label: The one label with the minimum support, only when ``supported``.
        attributes: Observed and derived properties, sorted and unique by name, value,
            attribute origin and derivation identity.
    """

    status: AmbiguityStatus
    hypotheses: tuple[LabelHypothesis, ...]
    support_status: ContextSupportStatus | None = field(default=None, metadata=_OMITTED_WHEN_ABSENT)
    primary_label: str | None = field(default=None, metadata=_OMITTED_WHEN_ABSENT)
    attributes: tuple[ContextAttribute, ...] = field(default=(), metadata=_OMITTED_WHEN_ABSENT)

    def __post_init__(self) -> None:
        """Validate that the status, the belief and the attributes agree with the hypotheses.

        Raises:
            ValueError: If the hypotheses are not sorted and unique, an unambiguous state does
                not have exactly one, a competing state has fewer than two, a state with
                insufficient evidence has any, the belief disagrees with the hypotheses (see
                :meth:`_require_belief`), or the attributes are not sorted and unique.
        """
        require_canonical(
            "hypotheses", self.hypotheses, lambda item: (item.label,), detail="by label "
        )
        count = len(self.hypotheses)
        if self.status is AmbiguityStatus.UNAMBIGUOUS and count != 1:
            raise ValueError(f"an unambiguous state has exactly one hypothesis, got {count}")
        if self.status in (AmbiguityStatus.AMBIGUOUS, AmbiguityStatus.CONFLICTING) and count < 2:
            raise ValueError(
                f"a {self.status.value} state keeps at least two hypotheses, got {count}"
            )
        if self.status is AmbiguityStatus.INSUFFICIENT_EVIDENCE and count:
            raise ValueError(f"a state with insufficient evidence has no hypothesis, got {count}")
        require_canonical(
            "attributes",
            self.attributes,
            _context_attribute_key,
            detail="by name, value, attribute origin and derivation identity ",
        )
        self._require_belief()

    def _require_belief(self) -> None:
        """Require the belief fields to exist together and to agree with the hypotheses.

        Raises:
            ValueError: If counts exist without a support status or the reverse, the status does
                not fit the number of hypotheses, or the primary label is present outside a
                supported state, missing from one, not one of the labels or not the best counted.
        """
        counts = [item.supporting_physical_observation_count for item in self.hypotheses]
        status = self.support_status
        if status is None:
            if any(value is not None for value in counts):
                raise ValueError(
                    "supporting_physical_observation_count is only recorded with a support_status"
                )
            if self.primary_label is not None:
                raise ValueError("primary_label is only exposed with a supported support_status")
            return
        if any(value is None for value in counts):
            raise ValueError(
                "every hypothesis needs its supporting_physical_observation_count when a "
                "support_status is recorded"
            )
        minimum = {
            ContextSupportStatus.INSUFFICIENT_EVIDENCE: 0,
            ContextSupportStatus.TENTATIVE: 1,
            ContextSupportStatus.SUPPORTED: 1,
            ContextSupportStatus.CONTESTED: 2,
        }[status]
        if (status is ContextSupportStatus.INSUFFICIENT_EVIDENCE) != (not counts) or len(
            counts
        ) < minimum:
            raise ValueError(
                f"a {status.value} support_status does not fit {len(counts)} hypotheses"
            )
        if status is not ContextSupportStatus.SUPPORTED:
            if self.primary_label is not None:
                raise ValueError(
                    f"primary_label is only exposed when supported, not when {status.value}"
                )
            return
        by_label = {
            item.label: item.supporting_physical_observation_count for item in self.hypotheses
        }
        if self.primary_label not in by_label:
            raise ValueError(
                f"primary_label {self.primary_label!r} must be one of the labels {sorted(by_label)}"
            )
        if by_label[self.primary_label] != max(value for value in counts if value is not None):
            raise ValueError(
                f"primary_label {self.primary_label!r} must be a label with the largest count"
            )


@dataclass(frozen=True, kw_only=True)
class ContextEntity:
    """One resolved entity, as the map composes it.

    Attributes:
        entity_id: Identity of the entity, unique inside the map.
        source: The resolved entity this entity maps to; the identity mapping is explicit and
            two entities never map to the same resolved entity.
        member_entities: The source entities this entity was resolved from, sorted and unique;
            at least one. An entity that merged several keeps all of them.
        resolution_decisions: The resolution decisions that grouped the members, local to the
            resolution run of ``source``, sorted and unique; empty when no decision was needed.
        unresolved_neighbors: Source entities outside this one that a decision left
            ``UNRESOLVED`` against a member: neither merged nor declared distinct. Sorted and
            unique, and never a member.
        geometry_refs: The authoritative geometry support, by reference: at least one element,
            sorted and unique. Never coordinates.
        semantic_state: What the entity may be, with every hypothesis kept.
        origin: How the entity was produced and the evidence it rests on.
    """

    entity_id: ContextEntityId
    source: ResolvedEntityReference
    member_entities: tuple[EntityReference, ...]
    resolution_decisions: tuple[ResolutionDecisionId, ...]
    unresolved_neighbors: tuple[EntityReference, ...]
    geometry_refs: tuple[GeometryReference, ...]
    semantic_state: ContextSemanticState
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        """Validate the identity, the resolution lineage and the geometry support.

        Raises:
            ValueError: If the identity is blank, the entity has no member entity or no
                geometry support, a list is not sorted and unique, or an unresolved neighbor is
                also a member.
        """
        require_present(self, "entity_id")
        if not self.member_entities:
            raise ValueError("member_entities must list the source entities of the entity")
        for name, references in (
            ("member_entities", self.member_entities),
            ("unresolved_neighbors", self.unresolved_neighbors),
        ):
            require_canonical(
                name,
                references,
                lambda item: (str(item.semantic_map_id), str(item.entity_id)),
            )
        require_canonical("resolution_decisions", self.resolution_decisions, lambda item: (item,))
        shared = set(self.member_entities) & set(self.unresolved_neighbors)
        if shared:
            raise ValueError(
                f"an unresolved neighbor cannot also be a member: {sorted(map(repr, shared))}"
            )
        if not self.geometry_refs:
            raise ValueError("geometry_refs must list the geometry that supports the entity")
        require_canonical(
            "geometry_refs",
            self.geometry_refs,
            lambda item: (item.map_id, item.geometry_id),
        )


@dataclass(frozen=True, kw_only=True)
class ContextRelation:
    """One directed relation between two entities of the map.

    A candidate relation is never a confirmed one: an unresolved or rejected relation stays in the
    map with its state instead of being dropped or promoted, and a consumer that wants only the
    confirmed ones filters by ``RelationState.SUPPORTED``.

    Attributes:
        relation_id: Identity of the relation, unique inside the map.
        source_run_id: The Spatial Relations run that decided the relation.
        source_relation_id: The relation, local to that run. Together with ``source_run_id`` it
            names the upstream relation; two relations never map to the same one.
        subject: The entity the predicate is about.
        predicate: The canonical predicate of the Spatial Relations taxonomy.
        object: The entity the predicate points to. ``subject predicate object`` is directed:
            the relation does not hold in the other direction unless it is stated.
        state: The state decided by Spatial Relations: supported, rejected or unresolved.
        uncertainty_kinds: Why an unresolved relation could not be decided, sorted and unique;
            empty for a supported or rejected one.
        origin: How the relation was derived and the evidence it rests on, or ``None`` when it
            rests on no evidence this map can cite. Only an unresolved relation may have none: it
            may have been left undecided precisely because there was nothing to decide on, and
            saying so is not the same as citing something. ``source_run_id`` and
            ``source_relation_id`` already name the upstream record structurally, so the relation
            never cites its own record as its evidence.
    """

    relation_id: ContextRelationId
    source_run_id: SpatialRelationsRunId
    source_relation_id: RelationId
    subject: ContextEntityReference
    predicate: RelationPredicate
    object: ContextEntityReference
    state: RelationState
    uncertainty_kinds: tuple[RelationUncertaintyKind, ...]
    origin: EvidenceOrigin | None = field(default=None, metadata=_OMITTED_WHEN_ABSENT)

    def __post_init__(self) -> None:
        """Validate the identities, the endpoints and that the state agrees with the uncertainty.

        Raises:
            ValueError: If an identity is blank, the relation relates an entity to itself, the
                uncertainty kinds are not sorted and unique, an unresolved relation does not say
                why, a supported or rejected one carries uncertainty, or a decided one has no
                origin.
        """
        require_present(self, "relation_id", "source_run_id", "source_relation_id")
        if self.subject == self.object:
            raise ValueError(
                f"relation {self.relation_id!r} relates entity {self.subject.entity_id!r} to itself"
            )
        require_canonical(
            "uncertainty_kinds",
            self.uncertainty_kinds,
            lambda item: (item.value,),
            detail="by value ",
        )
        if self.state is RelationState.UNRESOLVED:
            if not self.uncertainty_kinds:
                raise ValueError("an unresolved relation says why: it needs uncertainty kinds")
        else:
            if self.uncertainty_kinds:
                raise ValueError(f"a {self.state.value} relation carries no uncertainty")
            # Decidir exige a evidência que decidiu: só uma relação não decidida pode não ter
            # nenhuma proveniência para citar.
            if self.origin is None:
                raise ValueError(
                    f"a {self.state.value} relation needs an origin: only an unresolved one may "
                    f"rest on no evidence"
                )
