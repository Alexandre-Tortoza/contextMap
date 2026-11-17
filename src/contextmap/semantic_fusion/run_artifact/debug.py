"""Evidência humana de diagnóstico, nunca contratual.

Tudo que este módulo escreve vive em ``__DEBUG__/`` e **não** entra no inventário do
manifest: remover o diretório não pode invalidar o run, e Semantic Mapping ou qualquer
outro estágio a jusante não pode depender dele. É evidência para uma pessoa olhar quando
um resultado surpreende, não dado de contrato.

Os níveis são explícitos: ``STANDARD`` cobre uma amostra de suportes, ``FULL`` cobre todos
e acrescenta o trace por contribuição e o resumo de geometria.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from contextmap.semantic_fusion.evidence import EvidenceContribution, SupportSignalKind
from contextmap.semantic_fusion.fused_evidence import EvidenceStance, FusionOutcome
from contextmap.semantic_fusion.run_artifact.layout import (
    DEBUG_SUPPORT_LIMIT,
    SemanticFusionDebugLevel,
    encode_document,
    encode_line,
)
from contextmap.semantic_fusion.serialization import encode_fused_evidence
from contextmap.shared import DEBUG_DIRECTORY, AtomicRunDirectory


def write_support_debug(
    run: AtomicRunDirectory,
    outcome: FusionOutcome,
    *,
    level: SemanticFusionDebugLevel,
    support_number: int,
) -> None:
    """Write the non-contractual debug evidence of one support, if the level asks for it.

    Args:
        run: The run directory being built; every file is written as non-contractual.
        outcome: The support and its fused evidence.
        level: How much debug evidence to persist.
        support_number: Ordinal of this support in the run, used by the ``standard`` sample.
    """
    if level is SemanticFusionDebugLevel.NONE:
        return
    full = level is SemanticFusionDebugLevel.FULL
    if not full and support_number > DEBUG_SUPPORT_LIMIT:
        return
    base = f"{DEBUG_DIRECTORY}/supports/{outcome.support.fusion_support_id}"
    evidence = outcome.evidence
    # NOTE: a evidência é codificada uma única vez; as três visões abaixo são recortes do
    # mesmo registro, e recodificar por recorte só repetiria o mesmo trabalho.
    encoded = encode_fused_evidence(evidence)
    run.write_text(
        f"{base}/summary.json", encode_document(debug_summary(outcome)), contractual=False
    )
    run.write_text(
        f"{base}/hypotheses.json",
        encode_document({"hypotheses": encoded["hypotheses"]}),
        contractual=False,
    )
    run.write_text(
        f"{base}/conflicts.json",
        encode_document({"uncertainty": encoded["uncertainty"]}),
        contractual=False,
    )
    run.write_text(
        f"{base}/physical-observations.json",
        encode_document({"groups": encoded["physical_observation_groups"]}),
        contractual=False,
    )
    if full:
        run.write_text(
            f"{base}/contribution-trace.jsonl",
            "".join(
                encode_line(contribution_trace(item)).decode() + "\n"
                for item in evidence.contributions
            ),
            contractual=False,
        )
        run.write_text(
            f"{base}/geometry-summary.json",
            encode_document(geometry_summary(outcome)),
            contractual=False,
        )


def debug_summary(outcome: FusionOutcome) -> dict[str, Any]:
    """Summarise one support for a person: hypotheses, their support and the uncertainty.

    Args:
        outcome: The support and its fused evidence.

    Returns:
        A JSON-ready record. Diagnostic only: never contractual and never an input downstream.
    """
    evidence = outcome.evidence
    contributions = {item.contribution_id: item for item in evidence.contributions}
    hypotheses = []
    for hypothesis in evidence.hypotheses:
        supporting = [
            contributions[item.contribution_id]
            for item in hypothesis.evidence
            if item.stance is EvidenceStance.SUPPORTING
        ]
        stances = Counter(item.stance.value for item in hypothesis.evidence)
        hypotheses.append(
            {
                "hypothesis_id": str(hypothesis.hypothesis_id),
                "label": hypothesis.label,
                "supporting_physical_observations": sorted(
                    {str(item.physical_observation_id) for item in supporting}
                ),
                "supporting_inference_results": sorted(
                    {
                        str(item.semantic_interpretation_artifact_id)
                        for item in supporting
                        if item.semantic_interpretation_artifact_id is not None
                    }
                ),
                "stances": dict(sorted(stances.items())),
                "unscored_claims": sum(
                    1
                    for item in hypothesis.evidence
                    if any(
                        s.kind is SupportSignalKind.CLAIM_CONFIDENCE and s.value is None
                        for s in item.signals
                    )
                ),
            }
        )
    return {
        "fusion_support_id": str(outcome.support.fusion_support_id),
        "fused_evidence_id": str(evidence.fused_evidence_id),
        "physical_observation_count": evidence.physical_observation_count,
        "inference_result_count": evidence.inference_result_count,
        "hypotheses": hypotheses,
        "active_channels": [item.channel.value for item in evidence.channels],
        "uncertainty": [
            {
                "kind": record.kind.value,
                "hypothesis_ids": [str(h) for h in record.hypothesis_ids],
                "evidence_count": len(record.evidence),
                "rule_id": record.rule_id,
            }
            for record in evidence.uncertainty
        ],
        "weighted": evidence.weighting is not None,
    }


def contribution_trace(item: EvidenceContribution) -> dict[str, Any]:
    """Trace one contribution back to the upstream evidence it references.

    Args:
        item: The contribution to describe.

    Returns:
        A JSON-ready record with counts instead of payloads. Diagnostic only.
    """
    return {
        "contribution_id": str(item.contribution_id),
        "physical_observation_id": str(item.physical_observation_id),
        "semantic_interpretation_artifact_id": item.semantic_interpretation_artifact_id,
        "region_artifact_id": str(item.region_artifact_id),
        "spatial_observation_id": str(item.spatial_observation_id),
        "region_id": str(item.region_id),
        "claim_ids": [str(ref.claim_id) for ref in item.claim_refs],
        "score_refs": len(item.score_refs),
        "visual_feature_refs": [ref.embedding_space_id for ref in item.visual_feature_refs],
        "observation_quality": None
        if item.observation_quality is None
        else item.observation_quality.definitions_version,
        "geometry_count": len(item.geometry_support),
    }


def geometry_summary(outcome: FusionOutcome) -> dict[str, Any]:
    """Describe the extent of a support and how much geometry each view of it saw.

    Args:
        outcome: The support and its fused evidence.

    Returns:
        A JSON-ready record. Diagnostic only.
    """
    support = outcome.support
    return {
        "geometric_map_id": str(support.geometric_map_id),
        "geometry_count": len(support.geometry_support),
        "bounds": {
            "frame_id": str(support.bounds.frame_id),
            "minimum_m": list(support.bounds.minimum_m),
            "maximum_m": list(support.bounds.maximum_m),
        },
        "centroid_m": list(support.centroid_m),
        "geometry_per_contribution": {
            str(item.contribution_id): len(item.geometry_support)
            for item in outcome.evidence.contributions
        },
    }
