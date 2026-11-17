"""Mapeamento contextual incremental: o subdomínio do runtime que faz um mapa crescer.

A cadeia é fundação espacial → ContextRun → revisão de ContextBranch → ContextBuild →
ContextMap. A fundação fixa a sequência, a trajetória e o mapa geométrico; cada ContextRun
acrescenta evidência de contexto sobre ela; a branch acumula ContextRuns por acréscimo; um
build congela uma revisão como a entrada de uma materialização e o índice de builds registra
cada mapa publicado. Cada elo tem o próprio registro versionado e o próprio erro.

O subdomínio só *usa* o DAG do runtime (plano, escopo, journal); não o estende. Esta fachada
expõe o que cruza a fronteira do subpacote; os módulos internos dependem uns dos outros
diretamente, na direção ``build_index → context_build → context_branch → context_run →
foundation``.
"""

from contextmap.runtime.contextual_mapping.build_index import (
    ClaimedContextBuild,
    ContextBuildEntry,
    claim_context_build,
    effective_maps_root,
    read_context_builds,
    record_context_build,
    resolve_map_artifact,
)
from contextmap.runtime.contextual_mapping.context_branch import (
    ContextBranch,
    ContextBranchMember,
    append_to_branch,
    create_branch,
    open_branch,
)
from contextmap.runtime.contextual_mapping.context_build import (
    CONTEXT_BUILD_FILENAME,
    MATERIALIZATION_TARGET,
    ContextBuild,
    ContextBuildId,
    PlannedContextBuild,
    plan_context_build,
    publish_context_build,
    read_context_build,
)
from contextmap.runtime.contextual_mapping.context_run import (
    ContextArtifact,
    ContextRun,
    ContextRunId,
    check_context_selection,
    context_scope,
    publish_context_run,
    read_context_run,
)
from contextmap.runtime.contextual_mapping.errors import (
    ContextBranchError,
    ContextBuildError,
    ContextRunError,
    SpatialFoundationError,
)
from contextmap.runtime.contextual_mapping.foundation import (
    SpatialFoundation,
    SpatialFoundationId,
    foundation_of_run,
    resolve_spatial_foundation,
)
from contextmap.runtime.contextual_mapping.stages import CONTEXT_STAGES, SELECTABLE_CONTEXT_STAGES

__all__ = [
    "CONTEXT_BUILD_FILENAME",
    "CONTEXT_STAGES",
    "MATERIALIZATION_TARGET",
    "SELECTABLE_CONTEXT_STAGES",
    "ClaimedContextBuild",
    "ContextArtifact",
    "ContextBranch",
    "ContextBranchError",
    "ContextBranchMember",
    "ContextBuild",
    "ContextBuildEntry",
    "ContextBuildError",
    "ContextBuildId",
    "ContextRun",
    "ContextRunError",
    "ContextRunId",
    "PlannedContextBuild",
    "SpatialFoundation",
    "SpatialFoundationError",
    "SpatialFoundationId",
    "append_to_branch",
    "check_context_selection",
    "claim_context_build",
    "context_scope",
    "create_branch",
    "effective_maps_root",
    "foundation_of_run",
    "open_branch",
    "plan_context_build",
    "publish_context_build",
    "publish_context_run",
    "read_context_build",
    "read_context_builds",
    "read_context_run",
    "record_context_build",
    "resolve_map_artifact",
    "resolve_spatial_foundation",
]
