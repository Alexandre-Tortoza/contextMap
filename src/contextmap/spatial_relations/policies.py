"""The effective policies of one Spatial Relations run, and the coherence rule between them.

Each policy of this capability is valid on its own: the frame conventions, the candidate reach, the
geometry summary and the three evidence channels each validate their own parameters. Only together
do they form the configuration of a run, and only together can they be *incoherent*: candidate
generation drops a pair before any evaluator measures it, so a reach shorter than an evaluator's
tolerance loses true relations silently. The run would still look successful, with exclusions whose
reasons are technically correct.

That rule has no other owner. It is not artifact layout, because it must hold before anything is
written, and it is not a single policy's invariant, because no policy can see the others. So the
aggregate and the rule live here, and both the operation that derives relations and the writer that
persists them read the same object.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextmap.semantic_mapping import GeometrySummaryPolicy
from contextmap.spatial_relations.candidates import CandidateExclusionReason, CandidatePolicy
from contextmap.spatial_relations.contact_predicates import ContactPredicatePolicy
from contextmap.spatial_relations.frame_conventions import FrameConventions
from contextmap.spatial_relations.geometric_predicates import GeometricPredicatePolicy
from contextmap.spatial_relations.semantic_predicates import SemanticPredicatePolicy


@dataclass(frozen=True, kw_only=True)
class RelationsRunPolicies:
    """The effective policies a run was produced under.

    Attributes:
        frame_conventions: The axes the run declared for the map frame.
        candidate: The candidate generation policy.
        geometry_summary: The policy that summarized each resolved entity's geometry before
            candidate generation; it measurably affects the geometric and contact predicate
            outputs (see :func:`~contextmap.spatial_relations.resolved_entity_geometries`), so
            it is part of the run's provenance exactly like every other effective policy.
        geometric: The geometric predicate policy, when geometric evidence was produced.
        contact: The contact predicate policy, when contact evidence was produced.
        semantic: The interaction-range predicate policy, when semantic evidence was produced. Its
            reach is the candidate policy's ``interaction_radius_m``; this policy only carries the
            tolerance the gap is compared with the per-entity radius under.
    """

    frame_conventions: FrameConventions
    candidate: CandidatePolicy
    geometry_summary: GeometrySummaryPolicy
    geometric: GeometricPredicatePolicy | None = None
    contact: ContactPredicatePolicy | None = None
    semantic: SemanticPredicatePolicy | None = None

    def __post_init__(self) -> None:
        """Require the candidate reach to cover the distance tolerances of the declared evaluators.

        Candidate generation drops a pair before any evaluator measures it, so a reach shorter
        than an evaluator's tolerance loses true relations silently: the run would still look
        successful, with exclusions whose reasons are technically correct. Each policy is valid
        on its own; only here do they meet. Only the evaluators present are checked.

        Raises:
            ValueError: If ``proximity_radius_m`` of the candidate policy is below
                ``next_to_max_gap_m`` or ``2 * containment_slack_m`` of the geometric policy, or
                below ``contact_distance_m + contact_tolerance_m`` of the contact policy. The
                message names every such parameter, its value and the relations that would be
                lost.
        """
        problems = _uncovered_tolerances(self.candidate, self.geometric, self.contact)
        if problems:
            raise ValueError(
                "the candidate reach does not cover the evaluators' tolerances: "
                + "; ".join(problems)
            )


def _uncovered_tolerances(
    candidate: CandidatePolicy,
    geometric: GeometricPredicatePolicy | None,
    contact: ContactPredicatePolicy | None,
) -> list[str]:
    """State every evaluator tolerance that the proximity reach of candidate generation misses.

    Each condition keeps a precondition of candidate generation from excluding a pair the
    evaluator would accept:

    * ``NEXT_TO`` accepts a bounds gap up to ``next_to_max_gap_m``, and a gap beyond
      ``proximity_radius_m`` is excluded;
    * ``INSIDE`` accepts a protrusion of ``containment_slack_m`` beyond *each* face, so the
      subject's extent may exceed the object's by ``2 * containment_slack_m`` on one axis, and
      an excess beyond ``proximity_radius_m`` is excluded;
    * the contact evaluators decide point pairs up to their search radius,
      ``contact_distance_m + contact_tolerance_m``, and the bounds gap never exceeds the distance
      between two points of the entities.

    Returns:
        One explanation per tolerance the reach does not cover; empty when it covers them all.
    """
    reach = candidate.proximity_radius_m
    beyond = CandidateExclusionReason.BEYOND_PROXIMITY_RADIUS.value
    problems: list[str] = []
    if geometric is not None:
        if geometric.next_to_max_gap_m > reach:
            problems.append(
                f"next_to_max_gap_m={geometric.next_to_max_gap_m!r} exceeds "
                f"proximity_radius_m={reach!r}: true NEXT_TO relations would be excluded as "
                f"{beyond} before evaluation"
            )
        # Multiplicar por dois é exato em ponto flutuante: a condição não depende de
        # arredondamento.
        excess = 2.0 * geometric.containment_slack_m
        if excess > reach:
            problems.append(
                f"2 * containment_slack_m = {excess!r} (a subject may protrude "
                f"containment_slack_m={geometric.containment_slack_m!r} beyond both faces of an "
                f"axis) exceeds proximity_radius_m={reach!r}: true INSIDE relations would be "
                f"excluded as {CandidateExclusionReason.CONTAINMENT_IMPOSSIBLE.value} before "
                f"evaluation"
            )
    # O raio de busca é a própria soma que o avaliador de contato usa, com o mesmo
    # arredondamento, e não uma cópia da regra.
    if contact is not None and contact.search_radius_m > reach:
        problems.append(
            f"contact_distance_m + contact_tolerance_m = {contact.search_radius_m!r} "
            f"({contact.contact_distance_m!r} + {contact.contact_tolerance_m!r}) exceeds "
            f"proximity_radius_m={reach!r}: true TOUCHING, ON_TOP_OF and LEANING_AGAINST "
            f"relations would be excluded as {beyond} before evaluation"
        )
    return problems
