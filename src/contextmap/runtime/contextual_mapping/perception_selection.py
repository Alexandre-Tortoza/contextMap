"""Se um run de percepção é a evidência de uma seleção de observações sobre uma sequência.

Uma ContextRun pode referenciar um run de percepção que selecionou em vez de executar, e
registra a seleção de observações do plano como a seleção que a evidência dela cobre. Este
módulo lê o manifest do próprio run selecionado, pelo leitor público do produtor, e o
compara com a identidade que essa seleção tem sobre a sequência da fundação
(``selection_identity``, a regra que a própria Visual Perception registra). Nada é inferido da
configuração que produziu o run.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextmap.ingestion import (
    FullSequenceSelection,
    SequenceArtifactId,
    decode_selection,
    selection_identity,
)
from contextmap.runtime.artifacts import ArtifactRef


def perception_selection_mismatch(
    ref: ArtifactRef,
    *,
    workspace: Path,
    sequence_id: str,
    observation_selection: Mapping[str, Any] | None,
) -> str | None:
    """Explain why a perception run does not cover a selection of a sequence.

    Args:
        ref: The perception run, located relative to ``workspace``.
        workspace: The workspace the run lives in.
        sequence_id: The sequence the run must have been built over.
        observation_selection: The encoded selection the run must have processed, or ``None``
            for the whole sequence.

    Returns:
        ``None`` when the run covers exactly that selection of that sequence, otherwise the
        reason it does not, including a run that has no location or cannot be read.
    """
    # O stage_id identifica o contrato do reader; cada artifact guarda sua própria seleção.
    from contextmap.region_perception import RegionPerceptionRunManifest, RegionPerceptionRunReader
    from contextmap.semantic_interpretation import (
        SemanticInterpretationRunManifest,
        SemanticInterpretationRunReader,
    )
    from contextmap.visual_perception import VisualEvidenceRunManifest, VisualEvidenceRunReader

    if ref.stage_id not in (
        "region_perception",
        "visual_perception",
        "semantic_interpretation",
    ):
        return f"{ref.stage_id!r} is not a selectable perception producer"
    if ref.location is None:
        return f"selected {ref.stage_id} run {ref.artifact_id!r} has no location to check it by"
    manifest: (
        RegionPerceptionRunManifest | VisualEvidenceRunManifest | SemanticInterpretationRunManifest
    )
    try:
        if ref.stage_id == "region_perception":
            manifest = RegionPerceptionRunReader(workspace / ref.location).manifest
        elif ref.stage_id == "visual_perception":
            manifest = VisualEvidenceRunReader(workspace / ref.location).manifest
        else:
            manifest = SemanticInterpretationRunReader(workspace / ref.location).manifest
    except (OSError, ValueError) as error:
        return f"cannot read selected {ref.stage_id} run {ref.artifact_id!r}: {error}"
    if str(manifest.sequence_artifact_id) != sequence_id:
        return (
            f"selected {ref.stage_id} run {ref.artifact_id!r} was built over sequence "
            f"{manifest.sequence_artifact_id!r}, not the foundation's {sequence_id!r}"
        )
    selection = (
        FullSequenceSelection()
        if observation_selection is None
        else decode_selection(dict(observation_selection))
    )
    expected = selection_identity(SequenceArtifactId(sequence_id), selection)
    if manifest.selection_id != expected:
        return (
            f"selected {ref.stage_id} run {ref.artifact_id!r} processed observation selection "
            f"{manifest.selection_id!r}, but this ContextRun records {observation_selection!r} "
            f"({expected!r})"
        )
    return None
