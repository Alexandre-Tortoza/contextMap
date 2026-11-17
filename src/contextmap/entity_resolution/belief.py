"""Belief after resolution: which label, if any, the independent support of an entity justifies.

Existence is not belief. A resolved entity exists because its members were materialized and
matched; *what* it is stays the union of the member hypotheses, none of them preferred. The rule
here, ``resolved-belief-min-support-v1``, is a separate, versioned judgement derived from that
union, never a replacement for it:

* hypotheses are grouped by Semantic Fusion's typographic label key (NFKC, case-folded, whitespace
  collapsed), so ``Door`` in one member and ``door`` in another are one label, and nothing more is
  assumed (no synonym, no taxonomy);
* the support of a label is the set of **distinct physical observations** whose claims support it,
  across every member: repeated inference over one frame, or two members that saw the same frame,
  count once. No score is read, averaged or thresholded;
* ``tentative`` when no label reaches the declared minimum, ``supported`` when exactly one does
  (it is then the primary label), ``contested`` when two or more do (no primary), and
  ``insufficient_evidence`` when there is no hypothesis at all. Labels below the minimum stay
  listed with their counts: nothing is discarded;
* ``semantic_entropy_bits`` is the Shannon entropy of the per-label *counts* of observations,
  never of scores, so a spread of support is visible without inventing a probability.

The minimum is a declared hypothesis of the configuration (two observations is the working value
to be evaluated), never a default. Observations are "distinct" by identity only: no minimum
temporal or viewpoint separation is required yet, so two frames a few milliseconds apart count as
two (see ``src/contextmap/entity_resolution/docs/resolved-entities.md``).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from contextmap.entity_resolution._checks import require_canonical, require_present
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import EntityHypothesis

RESOLVED_BELIEF_RULE_ID = "resolved-belief-min-support-v1"
"""Versioned identity of the belief rule described in this module."""

_ENTROPY_TOLERANCE_BITS = 1e-12
"""Tolerance when checking a recorded entropy against its counts: only float rounding differs."""


class SupportStatus(Enum):
    """What the independent support of a resolved entity justifies about its label.

    Attributes:
        TENTATIVE: At least one hypothesis, but no label reaches the minimum; all stay visible.
        SUPPORTED: Exactly one label reaches the minimum; it is the primary label.
        CONTESTED: Two or more labels reach the minimum; there is no primary label.
        INSUFFICIENT_EVIDENCE: The entity has no hypothesis at all.
    """

    TENTATIVE = "tentative"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, kw_only=True)
class ResolvedBeliefPolicy:
    """Explicit configuration of ``resolved-belief-min-support-v1``.

    Attributes:
        min_supporting_physical_observations: Distinct physical observations a label needs before
            it can be the primary label. There is no default: the value is a scientific choice
            (two is the working hypothesis) to be evaluated against a reference set.
    """

    min_supporting_physical_observations: int

    def __post_init__(self) -> None:
        """Validate the minimum.

        Raises:
            ValueError: If the minimum is not an integer of at least one.
        """
        minimum = self.min_supporting_physical_observations
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError(
                f"min_supporting_physical_observations must be an integer of at least one, "
                f"got {minimum!r}"
            )


@dataclass(frozen=True, kw_only=True)
class LabelSupport:
    """The independent support of one label of a resolved entity.

    Attributes:
        label_key: The typographic key the label's hypotheses share.
        labels: Every verbatim label of those hypotheses, sorted and unique, never empty.
        supporting_physical_observations: The distinct physical observations whose claims support
            any of those hypotheses, across every member, sorted and unique, never empty.
    """

    label_key: str
    labels: tuple[str, ...]
    supporting_physical_observations: tuple[SourceObservationId, ...]

    def __post_init__(self) -> None:
        """Validate the key, the labels and the observations.

        Raises:
            ValueError: If the key is empty, a list is empty or not sorted and unique.
        """
        require_present(self, "label_key")
        for name in ("labels", "supporting_physical_observations"):
            items: tuple[str, ...] = getattr(self, name)
            if not items:
                raise ValueError(f"{name} must not be empty")
            require_canonical(name, items, lambda item: (item,))

    @property
    def observation_count(self) -> int:
        """How many distinct physical observations support the label."""
        return len(self.supporting_physical_observations)


def support_status_of(counts: tuple[int, ...], minimum: int) -> SupportStatus:
    """The status the per-label counts imply under a minimum.

    Args:
        counts: The count of distinct supporting physical observations of each label.
        minimum: The minimum a label needs.

    Returns:
        ``INSUFFICIENT_EVIDENCE`` without labels, ``TENTATIVE`` when none reaches the minimum,
        ``SUPPORTED`` when exactly one does and ``CONTESTED`` when several do.
    """
    if not counts:
        return SupportStatus.INSUFFICIENT_EVIDENCE
    reaching = sum(1 for count in counts if count >= minimum)
    if reaching == 0:
        return SupportStatus.TENTATIVE
    return SupportStatus.SUPPORTED if reaching == 1 else SupportStatus.CONTESTED


def entropy_bits(counts: tuple[int, ...]) -> float:
    """The Shannon entropy, in bits, of a distribution of counts.

    Args:
        counts: Positive counts; an empty tuple has zero entropy.

    Returns:
        ``-sum(p * log2(p))`` with ``p = count / total``; ``0.0`` for zero or one count.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    return abs(math.fsum(-(count / total) * math.log2(count / total) for count in counts))


@dataclass(frozen=True, kw_only=True)
class ResolvedSemanticBelief:
    """The belief a resolved entity's independent support justifies, derived by a versioned rule.

    Attributes:
        rule_id: The rule that derived it, ``resolved-belief-min-support-v1``.
        min_supporting_physical_observations: The minimum the rule was configured with.
        support_status: What the counts justify under that minimum.
        primary_label: The label of the one group that reaches the minimum, verbatim, only when
            ``supported``; ``None`` otherwise.
        label_support: The support of every label, sorted by key and unique; empty only for
            insufficient evidence.
        semantic_entropy_bits: Shannon entropy of the per-label observation counts, in bits.
    """

    rule_id: str
    min_supporting_physical_observations: int
    support_status: SupportStatus
    primary_label: str | None
    label_support: tuple[LabelSupport, ...]
    semantic_entropy_bits: float

    def __post_init__(self) -> None:
        """Validate that the status, the primary label and the entropy agree with the counts.

        Raises:
            ValueError: If the rule is empty, the minimum is not at least one, the groups are not
                sorted and unique, or the status, the primary label or the entropy is not the one
                the counts imply.
        """
        require_present(self, "rule_id")
        ResolvedBeliefPolicy(
            min_supporting_physical_observations=self.min_supporting_physical_observations
        )
        require_canonical("label_support", self.label_support, lambda item: (item.label_key,))
        counts = tuple(item.observation_count for item in self.label_support)
        expected = support_status_of(counts, self.min_supporting_physical_observations)
        if self.support_status is not expected:
            raise ValueError(
                f"support_status {self.support_status.value!r} does not match the counts "
                f"{list(counts)!r} under a minimum of {self.min_supporting_physical_observations}, "
                f"which imply {expected.value!r}"
            )
        self._require_primary_label()
        bits = entropy_bits(counts)
        if not math.isclose(
            self.semantic_entropy_bits, bits, rel_tol=0.0, abs_tol=_ENTROPY_TOLERANCE_BITS
        ):
            raise ValueError(
                f"semantic_entropy_bits {self.semantic_entropy_bits!r} is not the entropy of the "
                f"counts {list(counts)!r}, {bits!r}"
            )

    def _require_primary_label(self) -> None:
        """Require a primary label exactly when supported, taken from the group that reaches."""
        if self.support_status is not SupportStatus.SUPPORTED:
            if self.primary_label is not None:
                raise ValueError(
                    f"primary_label is only exposed when supported, not when "
                    f"{self.support_status.value}"
                )
            return
        (reaching,) = (
            item
            for item in self.label_support
            if item.observation_count >= self.min_supporting_physical_observations
        )
        if self.primary_label not in reaching.labels:
            raise ValueError(
                f"primary_label {self.primary_label!r} is not a label of the supported group "
                f"{list(reaching.labels)!r}"
            )


def derive_resolved_belief(
    hypotheses: tuple[EntityHypothesis, ...], policy: ResolvedBeliefPolicy
) -> ResolvedSemanticBelief:
    """Derive the belief the union of a resolved entity's hypotheses justifies.

    Args:
        hypotheses: The union of the member hypotheses, each listing its supporting physical
            observations.
        policy: The minimum support a label needs.

    Returns:
        The belief; the hypotheses themselves are not changed.

    Raises:
        ValueError: If a hypothesis does not record its supporting physical observations: its
            support cannot be counted, and guessing it would be a silent fallback.
    """
    frames: dict[str, set[SourceObservationId]] = {}
    spellings: dict[str, Counter[str]] = {}
    for hypothesis in hypotheses:
        if not hypothesis.supporting_physical_observations:
            raise ValueError(
                f"hypothesis {hypothesis.hypothesis_id!r} of {hypothesis.fused_evidence_id!r} does "
                f"not record its supporting physical observations, so its support cannot be "
                f"counted; rebuild the semantic map with a version that records them"
            )
        key = hypothesis.label_key
        frames.setdefault(key, set()).update(hypothesis.supporting_physical_observations)
        spellings.setdefault(key, Counter())[hypothesis.label] += 1
    groups = tuple(
        LabelSupport(
            label_key=key,
            labels=tuple(sorted(spellings[key])),
            supporting_physical_observations=tuple(sorted(frames[key])),
        )
        for key in sorted(frames)
    )
    minimum = policy.min_supporting_physical_observations
    counts = tuple(item.observation_count for item in groups)
    status = support_status_of(counts, minimum)
    primary = None
    if status is SupportStatus.SUPPORTED:
        (reaching,) = (item for item in groups if item.observation_count >= minimum)
        # A grafia exibida é a mais comum entre as hipóteses do grupo (empate: a menor), a mesma
        # regra que a fusão usa para escolher a grafia de uma hipótese.
        counted = spellings[reaching.label_key]
        primary = min(counted, key=lambda spelling: (-counted[spelling], spelling))
    return ResolvedSemanticBelief(
        rule_id=RESOLVED_BELIEF_RULE_ID,
        min_supporting_physical_observations=minimum,
        support_status=status,
        primary_label=primary,
        label_support=groups,
        semantic_entropy_bits=entropy_bits(counts),
    )
