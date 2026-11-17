"""Os vocabulários de estágio do mapeamento contextual, e por que são dois e não um.

Dois conjuntos parecidos significam coisas diferentes e não podem ser fundidos:

- os **membros da fundação espacial** são exatamente os estágios que produzem a sequência, a
  trajetória e o mapa (``ingestion``, ``state_estimation``, ``geometric_mapping``), um por
  papel da :class:`SpatialFoundation`. ``pose_ingestion`` não é membro: é uma fonte auxiliar
  opcional de ``state_estimation``, cuja proveniência o ``StateEstimationRunArtifact`` registra;
- os estágios que **um build nunca recomputa** são tudo o que está a montante da
  materialização: a fundação, ``pose_ingestion`` e os estágios de contexto. Um build só
  materializa a evidência das suas ContextRuns sobre a fundação delas.

Os estágios de contexto são os que uma ContextRun executa; todo outro estágio é fundação ou
materialização.
"""

from __future__ import annotations

from collections.abc import Mapping

FOUNDATION_STAGE_OF_ROLE: Mapping[str, str] = {
    "sequence": "ingestion",
    "state_estimation": "state_estimation",
    "geometry": "geometric_mapping",
}
"""The stage that produces the artifact of each foundation role, in role order."""

CONTEXT_STAGES = (
    "region_perception",
    "visual_perception",
    "semantic_interpretation",
    "semantic_scoring",
    "sensor_association",
    "point_representation",
)
"""The stages a ContextRun may execute; every other stage is foundation or materialization."""

SELECTABLE_CONTEXT_STAGES = (
    "region_perception",
    "visual_perception",
    "semantic_interpretation",
)
"""The context stages a ContextRun may take from an existing run instead of executing."""

NEVER_RECOMPUTED_BY_A_BUILD = frozenset(
    {*FOUNDATION_STAGE_OF_ROLE.values(), "pose_ingestion", *CONTEXT_STAGES}
)
"""The stages a ContextBuild refuses to recompute: everything upstream of the materialization."""
