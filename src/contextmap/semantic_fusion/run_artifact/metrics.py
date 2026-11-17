"""Checagem e agregação, em fluxo, do que o manifest e ``metrics/`` precisam.

:class:`RunTally` acompanha os outcomes à medida que são escritos e faz duas coisas que um
run precisa antes de publicar: **recusa** o que a linhagem e a política única do run não
admitem, e **agrega** as contagens e distribuições. Ela não guarda nenhuma linha das
tabelas de ``outputs/``: essas são escritas conforme cada outcome chega, então a memória
cresce com as identidades distintas do run, não com o seu tamanho.

As métricas são separadas por natureza: contagens e distribuições descrevem a evidência,
e números de runtime (segundos, memória) ficam em outro arquivo e só existem quando o
chamador os mediu. Nenhuma métrica aqui é medida de qualidade científica.
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from contextmap.semantic_fusion.evidence import SupportSignalKind
from contextmap.semantic_fusion.fused_evidence import (
    EvidenceStance,
    FusedEvidence,
    FusionOutcome,
    UncertaintyKind,
)
from contextmap.semantic_fusion.identity import FusionSupportId
from contextmap.semantic_fusion.run_artifact.layout import FusionRunArtifactError
from contextmap.semantic_fusion.run_artifact.manifest import FusionRunLineage


class RunTally:
    """Checks the outcomes as they stream and aggregates what the metrics and manifest need.

    It keeps no row of the ``outputs/`` tables: those are written as each outcome arrives.
    """

    def __init__(self, lineage: FusionRunLineage) -> None:
        """Start a tally bound to the run's declared upstream selection.

        Args:
            lineage: The explicit upstream selection every outcome is checked against.
        """
        self._lineage = lineage
        self.support_count = 0
        self.contribution_count = 0
        self.excluded_count = 0
        self.output_sizes: dict[str, int] = {}
        self.identities: dict[str, set[str]] = {}
        self.grouping_policy_id: str | None = None
        self.support_policy_id: str | None = None
        self.support_fingerprint: str | None = None
        self.fusion_policy_id: str | None = None
        self.fusion_fingerprint: str | None = None
        self._previous: FusionSupportId | None = None
        self._physical: set[str] = set()
        self._results: set[tuple[str, str]] = set()
        self._per_support: dict[str, list[int]] = {
            "physical_observations_per_support": [],
            "inference_results_per_support": [],
            "contributions_per_support": [],
            "hypotheses_per_support": [],
        }
        self._claims_total = 0
        self._claims_seen: dict[tuple[str, str], float | None] = {}
        self._abstaining: set[tuple[str, str]] = set()
        self._stances: Counter[str] = Counter()
        self._uncertainty: Counter[str] = Counter()
        self._hypotheses = 0
        self._scorer_signals = 0
        self._supports_with_uncertainty = 0

    def check(self, outcome: FusionOutcome) -> None:
        """Refuse an outcome the run cannot hold, before anything of it is counted.

        Args:
            outcome: The next outcome of the stream.

        Raises:
            FusionRunArtifactError: If the supports are not strictly sorted by identity, a
                support disagrees with the contributions of its evidence, the support is over
                another map than the lineage names, a contribution comes from a perception,
                scoring or representation run the lineage does not list, or the outcome was
                built or fused under a different policy or configuration than the rest.
        """
        support, evidence = outcome.support, outcome.evidence
        if self._previous is not None and support.fusion_support_id <= self._previous:
            raise FusionRunArtifactError(
                f"supports must be strictly sorted by identity: {support.fusion_support_id!r} "
                f"follows {self._previous!r}"
            )
        self._previous = support.fusion_support_id
        contributed = sorted({item.spatial_observation_id for item in evidence.contributions})
        if list(support.spatial_observation_ids) != contributed:
            raise FusionRunArtifactError(
                f"support {support.fusion_support_id!r} lists spatial observations that differ "
                f"from the contributions of its evidence"
            )
        if support.geometric_map_id != self._lineage.geometric_map_id:
            raise FusionRunArtifactError(
                f"support {support.fusion_support_id!r} is over map {support.geometric_map_id!r}, "
                f"but the run's lineage names {self._lineage.geometric_map_id!r}"
            )
        for item in evidence.contributions:
            if (
                item.semantic_interpretation_artifact_id is not None
                and item.semantic_interpretation_artifact_id
                not in self._lineage.semantic_interpretation_artifact_ids
            ):
                raise FusionRunArtifactError(
                    "contribution semantic producer is absent from lineage"
                )
            for feature in item.visual_feature_refs:
                if feature.visual_artifact_id not in self._lineage.visual_artifact_ids:
                    raise FusionRunArtifactError(
                        "contribution visual producer is absent from lineage"
                    )
            if item.region_artifact_id not in self._lineage.region_artifact_ids:
                raise FusionRunArtifactError(
                    f"contribution {item.contribution_id!r} comes from perception run "
                    f"{item.region_artifact_id!r}, which the run's lineage does not list"
                )
            for raw in item.raw_score_refs:
                if raw.semantic_scoring_run_id not in self._lineage.semantic_scoring_run_ids:
                    raise FusionRunArtifactError(
                        f"contribution {item.contribution_id!r} references semantic scoring run "
                        f"{raw.semantic_scoring_run_id!r}, which the run's lineage does not list"
                    )
        for ref in evidence.point_representation_refs:
            if ref.run_id not in self._lineage.point_representation_run_ids:
                raise FusionRunArtifactError(
                    f"support {support.fusion_support_id!r} references point representation run "
                    f"{ref.run_id!r}, which the run's lineage does not list"
                )
        self._single_policy(outcome)

    def _single_policy(self, outcome: FusionOutcome) -> None:
        support, evidence = outcome.support.provenance, outcome.evidence.provenance
        current = (
            evidence.grouping_policy_id,
            support.support_policy_id,
            support.configuration_fingerprint,
            evidence.fusion_policy_id,
            evidence.configuration_fingerprint,
        )
        first = (
            self.grouping_policy_id,
            self.support_policy_id,
            self.support_fingerprint,
            self.fusion_policy_id,
            self.fusion_fingerprint,
        )
        if self.support_count == 0:
            (
                self.grouping_policy_id,
                self.support_policy_id,
                self.support_fingerprint,
                self.fusion_policy_id,
                self.fusion_fingerprint,
            ) = current
        elif current != first:
            raise FusionRunArtifactError(
                f"support {outcome.support.fusion_support_id!r} was built or fused under a "
                f"different policy or configuration than the rest of the run: one run keeps one"
            )

    def add(self, outcome: FusionOutcome) -> None:
        """Aggregate one accepted outcome into the run's counts and distributions.

        Args:
            outcome: The outcome, already accepted by :meth:`check`.
        """
        evidence = outcome.evidence
        self.support_count += 1
        self.contribution_count += len(evidence.contributions)
        self._hypotheses += len(evidence.hypotheses)
        self._physical.update(
            str(g.physical_observation_id) for g in evidence.physical_observation_groups
        )
        # Um resultado de percepção tem várias regiões e cada uma pode cair em um suporte diferente:
        # somar `inference_result_count` por suporte contaria o mesmo resultado várias vezes e
        # deixaria de ser comparável com os frames físicos distintos acima.
        self._results.update(
            (str(result), str(g.physical_observation_id))
            for g in evidence.physical_observation_groups
            for result in g.semantic_interpretation_artifact_ids
        )
        counts = self._per_support
        counts["physical_observations_per_support"].append(evidence.physical_observation_count)
        counts["inference_results_per_support"].append(evidence.inference_result_count)
        counts["contributions_per_support"].append(len(evidence.contributions))
        counts["hypotheses_per_support"].append(len(evidence.hypotheses))
        self._claims_total += sum(len(item.claim_refs) for item in evidence.contributions)
        self._add_hypotheses(evidence)
        self._add_uncertainty(evidence)
        if evidence.uncertainty:
            self._supports_with_uncertainty += 1
        for channel in evidence.channels:
            self.identities.setdefault(channel.channel.value, set()).update(channel.identities)

    def _add_hypotheses(self, evidence: FusedEvidence) -> None:
        for hypothesis in evidence.hypotheses:
            for item in hypothesis.evidence:
                self._stances[item.stance.value] += 1
                key = (str(item.contribution_id), str(item.claim_id))
                confidence = next(
                    (s.value for s in item.signals if s.kind is SupportSignalKind.CLAIM_CONFIDENCE),
                    None,
                )
                self._claims_seen.setdefault(key, confidence)
                self._scorer_signals += sum(
                    1 for s in item.signals if s.kind is SupportSignalKind.SCORER_SUPPORT
                )
                if item.stance is EvidenceStance.ABSTAINING:
                    self._abstaining.add(key)

    def _add_uncertainty(self, evidence: FusedEvidence) -> None:
        for record in evidence.uncertainty:
            self._uncertainty[record.kind.value] += 1
            if record.kind is UncertaintyKind.INSUFFICIENT_EVIDENCE:
                self._abstaining.update(
                    (str(ref.contribution_id), str(ref.claim_id))
                    for ref in record.evidence
                    if ref.claim_id is not None
                )

    def counts(self, excluded: int, warnings: int) -> dict[str, Any]:
        """Render the run's counts for ``metrics/counts.json`` and the manifest.

        Args:
            excluded: Observations that took part in no support.
            warnings: How many warnings the run recorded.

        Returns:
            A JSON-ready record. Physical observations and inference results are counted
            distinct over the whole run, never summed per support.
        """
        scored = sum(1 for value in self._claims_seen.values() if value is not None)
        return {
            "supports": self.support_count,
            "fused_evidence": self.support_count,
            "contributions": self.contribution_count,
            "hypotheses": self._hypotheses,
            "physical_observations": len(self._physical),
            "inference_results": len(self._results),
            "claims": {
                "total": self._claims_total,
                "scored": scored,
                "unscored": len(self._claims_seen) - scored,
                "without_hypothesis": self._claims_total - len(self._claims_seen),
            },
            "scorer_signals": self._scorer_signals,
            "abstaining_claims": len(self._abstaining),
            "evidence_stances": {
                stance.value: self._stances.get(stance.value, 0) for stance in EvidenceStance
            },
            "uncertainty": {
                kind.value: self._uncertainty.get(kind.value, 0)
                for kind in (
                    UncertaintyKind.CONTRADICTION,
                    UncertaintyKind.AMBIGUITY,
                    UncertaintyKind.NEAR_TIE,
                    UncertaintyKind.INSUFFICIENT_EVIDENCE,
                )
            },
            "supports_with_uncertainty": self._supports_with_uncertainty,
            "excluded_observations": excluded,
            "warnings": warnings,
        }

    def distributions(self) -> dict[str, Any]:
        """Render the per-support distributions for ``metrics/distributions.json``.

        Returns:
            A JSON-ready record, one distribution per counted quantity.
        """
        return {name: distribution_of(values) for name, values in self._per_support.items()}


def distribution_of(values: list[int]) -> dict[str, Any]:
    """Summarise a per-support quantity without keeping its samples.

    Args:
        values: One value per support, in any order.

    Returns:
        Count, minimum, median and maximum; all ``None`` but the count for an empty run.
    """
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }
