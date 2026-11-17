"""ContextBuild: a entrada congelada de uma materialização de ContextMap.

Uma branch continua crescendo, então materializar "a branch" faria um mapa depender de quando
foi construído. Um ContextBuild congela, antes de qualquer estágio de jusante rodar,
exatamente quais ContextRuns de qual revisão da branch alimentam a Semantic Fusion e tudo
depois dela, e sob qual configuração de jusante. O ContextMapId que o build produz continua
sendo a única identidade de snapshot; o build é a proveniência da materialização, não uma
segunda identidade.

O build entrega ao plano os artifacts exatos que as ContextRuns dele referenciam, nunca um
catálogo, um seletor ``latest`` ou uma listagem de diretório. A identidade dele vem do que a
materialização consome: a fundação, esses artifacts, a configuração de jusante e o código. As
ContextRuns ficam registradas como proveniência mas fora da identidade, porque uma ContextRun
também carrega evidência que o mapa não consome (um run de Point Representation), e o
pareamento de runs de percepção e de associação de que a fusão depende vive dentro dos
próprios artifacts de associação. Sucesso, falha e o mapa final vêm do journal do run. O
índice dos builds publicados de um mapa fica em
:mod:`~contextmap.runtime.contextual_mapping.build_index`. Ver
``docs/runtime-composition.md``, "Contexto incremental".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NewType

from contextmap.runtime._files import publish_text
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.contextual_mapping.context_branch import ContextBranch
from contextmap.runtime.contextual_mapping.context_run import ContextRunId, read_context_run
from contextmap.runtime.contextual_mapping.errors import ContextBuildError, ContextRunError
from contextmap.runtime.contextual_mapping.foundation import SpatialFoundation
from contextmap.runtime.contextual_mapping.stages import (
    CONTEXT_STAGES,
    NEVER_RECOMPUTED_BY_A_BUILD,
)
from contextmap.runtime.pipeline import ExecutionPlan, PipelinePlan

CONTEXT_BUILD_FILENAME = "context_build.json"
CONTEXT_BUILD_SCHEMA_VERSION = "0.1.0"
"""Version of the ``context_build.json`` document."""

MATERIALIZATION_TARGET = "context_map"
"""The stage a build produces; Semantic Fusion and every stage after it run on the way."""

ContextBuildId = NewType("ContextBuildId", str)
"""``"sha256:<hex>"`` over the foundation, the provided artifacts, downstream config and code."""


@dataclass(frozen=True, kw_only=True)
class ContextBuild:
    """The frozen input of one materialization.

    Attributes:
        identity: Derived from everything the materialization depends on and nothing else:
            the foundation, the provided artifacts, the stage configurations and the code.
        foundation: The spatial foundation of the branch.
        branch: The branch the runs come from; provenance, outside the identity.
        revision: The branch revision frozen; later appends never enter the build. Provenance,
            outside the identity.
        context_run_ids: The ContextRuns the provided artifacts come from, in canonical order.
            Provenance, outside the identity: two ContextRuns that differ only in evidence the
            map does not consume materialize the same map.
        provided: The exact artifacts supplied to the plan for each context stage, deduplicated:
            one artifact reached through two ContextRuns is one piece of evidence.
        stage_configs: The configuration digest of each stage the build runs.
        code_identity: Identity of the code materializing the map.
    """

    identity: ContextBuildId
    foundation: SpatialFoundation
    branch: str
    revision: int
    context_run_ids: tuple[ContextRunId, ...]
    provided: Mapping[str, tuple[ArtifactRef, ...]]
    stage_configs: Mapping[str, str]
    code_identity: str

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form, without the run that holds it."""
        return {
            "schema_version": CONTEXT_BUILD_SCHEMA_VERSION,
            "context_build_id": self.identity,
            "foundation": self.foundation.to_document(),
            "branch": self.branch,
            "revision": self.revision,
            "context_run_ids": list(self.context_run_ids),
            "provided": {
                stage_id: [ref.to_document() for ref in refs]
                for stage_id, refs in self.provided.items()
            },
            "stage_configs": dict(self.stage_configs),
            "code_identity": self.code_identity,
        }


@dataclass(frozen=True, kw_only=True)
class PlannedContextBuild:
    """A frozen build and the execution that materializes it.

    Attributes:
        build: What the materialization consumes; publish it before running.
        execution: The plan scoped to the materialization stages over the provided artifacts.
    """

    build: ContextBuild
    execution: ExecutionPlan


def _needed_context_stages(plan: PipelinePlan, *, available: set[str]) -> set[str]:
    """Return the context stages the materialization still needs, given what is available.

    The scope's own traversal walks the inputs of each target and stops at every stage whose
    artifact is supplied, so a stage upstream of a supplied one is never reached and supplying
    it is refused. This mirrors that walk, so the caller supplies exactly the frontier.

    Args:
        plan: The resolved topology.
        available: Context stages whose artifacts the caller holds.

    Returns:
        The subset of ``available`` the materialization target reaches.
    """
    by_id = {stage.stage_id: stage for stage in plan.stages}
    frontier: set[str] = set()
    seen: set[str] = set()
    stack = [MATERIALIZATION_TARGET]
    while stack:
        stage_id = stack.pop()
        if stage_id in available:
            frontier.add(stage_id)
            continue
        if stage_id in seen or stage_id not in by_id:
            continue
        seen.add(stage_id)
        stack.extend(item.source for item in by_id[stage_id].inputs)
    return frontier


def plan_context_build(
    workspace: Path,
    branch: ContextBranch,
    plan: PipelinePlan,
    *,
    code_identity: str,
    revision: int | None = None,
    context_run_ids: Sequence[ContextRunId] | None = None,
) -> PlannedContextBuild:
    """Freeze a branch revision, or an explicit subset of it, as the input of a build.

    Every selected ContextRun is reopened from its record and must be the member the branch
    names, over the branch's foundation. The foundation and the runs' artifacts of the context
    stages the materialization consumes become the plan's provided artifacts, so the
    materialization never recomputes them. A context artifact nothing downstream consumes (a
    Point Representation run) stays in its ContextRun and is neither provided, recorded
    nor part of the build's identity: the map does not depend on it. The selected ContextRun
    ids are recorded as provenance only.

    Args:
        workspace: The workspace root.
        branch: The branch to build.
        plan: The resolved topology, with the downstream configuration of this build.
        code_identity: Identity of the code materializing the map.
        revision: The revision to freeze; the branch's current one when omitted.
        context_run_ids: An explicit subset of that revision's members; all of them when
            omitted. Order does not matter.

    Returns:
        The frozen build and its execution.

    Raises:
        ContextBuildError: If the revision has no run, a selected run is not one of its members,
            or a member's record is missing, altered or built on another foundation.
    """
    frozen = branch.at(branch.revision if revision is None else revision)
    members = {member.context_run_id: member for member in frozen.members}
    selected = sorted(set(members) if context_run_ids is None else set(context_run_ids))
    if not selected:
        raise ContextBuildError(
            f"a build needs at least one ContextRun; branch {branch.name!r} has none at "
            f"revision {frozen.revision}"
        )
    outside = [run_id for run_id in selected if run_id not in members]
    if outside:
        raise ContextBuildError(
            f"{outside} are not members of branch {branch.name!r} at revision {frozen.revision}"
        )
    provided: dict[str, dict[tuple[str, str, str, str | None], ArtifactRef]] = {}
    for run_id in selected:
        member = members[run_id]
        try:
            context = read_context_run(workspace / member.run, workspace=workspace)
        except ContextRunError as error:
            raise ContextBuildError(
                f"member {run_id!r} of branch {branch.name!r} cannot be read from "
                f"{member.run!r}: {error}"
            ) from error
        if context.identity != run_id or context.foundation.identity != branch.foundation.identity:
            raise ContextBuildError(
                f"the record at {member.run!r} is not member {run_id!r} over the foundation of "
                f"branch {branch.name!r}"
            )
        for artifact in context.artifacts:
            ref = artifact.ref
            key = (ref.stage_id, ref.contract, ref.artifact_id, ref.content_hash)
            provided.setdefault(ref.stage_id, {})[key] = ref
    # Só os estágios de contexto de que o alvo depende: a entrada congelada é o que o mapa
    # consome, e um artifact fornecido sem uso seria recusado pelo escopo, nunca ignorado.
    # NOTE: a travessia para em cada estágio fornecido, como a do escopo. Desde que a percepção
    # ocupa dois estágios, region_perception está a montante de visual_perception: oferecer o
    # artifact deste satisfaz o alvo e torna aquele desnecessário, e fornecê-lo seria recusado.
    consumed = _needed_context_stages(plan, available=set(provided) & set(CONTEXT_STAGES))
    context_refs = {
        stage_id: tuple(refs[key] for key in sorted(refs, key=lambda key: key[2]))
        for stage_id, refs in sorted(provided.items())
        if stage_id in consumed
    }
    # A materialização consome a sequência e o mapa da fundação, nunca a trajetória.
    execution = plan.scope(
        targets=[MATERIALIZATION_TARGET],
        provided={
            "ingestion": branch.foundation.sequence,
            "geometric_mapping": branch.foundation.geometry,
            **context_refs,
        },
    )
    recomputed = [
        stage.stage_id
        for stage in execution.stages
        if stage.stage_id in NEVER_RECOMPUTED_BY_A_BUILD
    ]
    if recomputed:
        raise ContextBuildError(
            f"the build would recompute {recomputed}; it only materializes the evidence of its "
            "ContextRuns over their foundation"
        )
    stage_configs = {stage.stage_id: stage.config_digest for stage in execution.stages}
    build = ContextBuild(
        identity=_identity(branch.foundation, context_refs, stage_configs, code_identity),
        foundation=branch.foundation,
        branch=branch.name,
        revision=frozen.revision,
        context_run_ids=tuple(selected),
        provided=context_refs,
        stage_configs=stage_configs,
        code_identity=code_identity,
    )
    return PlannedContextBuild(build=build, execution=execution)


def publish_context_build(run_directory: Path, *, workspace: Path, build: ContextBuild) -> None:
    """Publish the frozen input of a build in its run, once, before the materialization runs.

    Raises:
        FileExistsError: If the run already has a build record.
    """
    document = {**build.to_document(), "run": run_directory.relative_to(workspace).as_posix()}
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
    publish_text(run_directory, CONTEXT_BUILD_FILENAME, text + "\n")


def read_context_build(run_directory: Path, *, workspace: Path) -> ContextBuild:
    """Read the build record of a run, checking its schema version, its run and its identity.

    Besides requiring a readable, well-formed record, the read checks three things: the record
    follows this runtime's schema version; it names this run, so a record copied into another
    run is refused; and its recorded ``context_build_id`` equals the identity recomputed, by the
    rule the build was frozen with, from the recorded foundation (its identity), consumed
    evidence (``provided``: the contract, id and content hash of each artifact, per stage),
    downstream stage configurations and code identity, so an edit of those that leaves the
    recorded identity as it was is refused. Nothing else in the record is covered by that
    check: not the provenance fields outside the identity (``branch``, ``revision`` and
    ``context_run_ids``), nor the other fields of the references, such as their locations. The
    read does not establish that the record as a whole is unaltered.

    Raises:
        ContextBuildError: If the record is unreadable, follows another schema version, names
            another run, or its recorded identity differs from the one its identified content
            gives.
    """
    path = run_directory / CONTEXT_BUILD_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContextBuildError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContextBuildError(f"{path} is not a build record")
    if document.get("schema_version") != CONTEXT_BUILD_SCHEMA_VERSION:
        raise ContextBuildError(
            f"{path} follows schema_version {document.get('schema_version')!r}, this runtime "
            f"reads {CONTEXT_BUILD_SCHEMA_VERSION!r}"
        )
    run = run_directory.relative_to(workspace).as_posix()
    if document.get("run") != run:
        raise ContextBuildError(f"{path} is the record of run {document.get('run')!r}, not {run!r}")
    try:
        foundation = SpatialFoundation.from_document(document["foundation"])
        run_ids = tuple(ContextRunId(str(run_id)) for run_id in document["context_run_ids"])
        provided = {
            str(stage_id): tuple(ArtifactRef.from_document(item) for item in refs)
            for stage_id, refs in document["provided"].items()
        }
        stage_configs = {str(key): str(value) for key, value in document["stage_configs"].items()}
        code_identity = str(document["code_identity"])
        branch, revision = str(document["branch"]), int(document["revision"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ContextBuildError(f"{path} is not a valid build record: {error}") from error
    identity = _identity(foundation, provided, stage_configs, code_identity)
    if document.get("context_build_id") != identity:
        raise ContextBuildError(
            f"{path} does not match its identity: it records "
            f"{document.get('context_build_id')!r} but its content gives {identity!r}"
        )
    return ContextBuild(
        identity=identity,
        foundation=foundation,
        branch=branch,
        revision=revision,
        context_run_ids=run_ids,
        provided=provided,
        stage_configs=stage_configs,
        code_identity=code_identity,
    )


def _identity(
    foundation: SpatialFoundation,
    provided: Mapping[str, tuple[ArtifactRef, ...]],
    stage_configs: Mapping[str, str],
    code_identity: str,
) -> ContextBuildId:
    """Identify a build by what its materialization depends on, and nothing else.

    The materialization consumes artifacts, not ContextRuns: each downstream stage's execution
    identity comes from its configuration and the content of its inputs, and the fusion pairs
    each association run with its perception run through the association artifact itself. So
    neither the branch name, the revision nor the ContextRun ids enter the identity: the same
    artifacts and downstream configuration frozen from two branches, from two revisions, or
    from two ContextRuns that differ only in evidence the map does not consume (a Point
    Representation run) are the same build.
    """
    document = {
        "foundation": foundation.identity,
        "provided": {
            stage_id: sorted([ref.contract, ref.artifact_id, ref.content_hash] for ref in refs)
            for stage_id, refs in provided.items()
        },
        "stage_configs": dict(stage_configs),
        "code_identity": code_identity,
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return ContextBuildId(f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}")
