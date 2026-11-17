"""ContextRun: uma execução que acrescenta evidência de contexto sobre uma fundação espacial.

O contexto incremental (``docs/runtime-composition.md``, "Contexto incremental (v0.1.1)") faz
um mapa contextual crescer executando os estágios de contexto (Visual Perception e Sensor
Association, mais Point Representation quando habilitado) sobre seleções explícitas de uma
sequência, sempre contra a mesma
:class:`~contextmap.runtime.contextual_mapping.foundation.SpatialFoundation`. Uma ContextRun é
o registro de uma dessas execuções: qual fundação e qual seleção de observações ela usou, e
quais artifacts de capability ela produziu ou reusou.

É proveniência de orquestração, não evidência: referencia os artifacts de capability e não
copia nada deles. É publicada uma única vez, ao lado do journal do run, só quando o run
termina; um run que falhou tem o journal como único registro.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NewType

from contextmap.runtime._files import publish_text
from contextmap.runtime.artifacts import ArtifactRef
from contextmap.runtime.contextual_mapping.errors import ContextRunError
from contextmap.runtime.contextual_mapping.foundation import SpatialFoundation
from contextmap.runtime.contextual_mapping.perception_selection import (
    perception_selection_mismatch,
)
from contextmap.runtime.contextual_mapping.stages import (
    CONTEXT_STAGES,
    FOUNDATION_STAGE_OF_ROLE,
    SELECTABLE_CONTEXT_STAGES,
)
from contextmap.runtime.pipeline import ExecutionPlan, ExecutionRecord, PipelinePlan

CONTEXT_RUN_FILENAME = "context_run.json"
CONTEXT_RUN_SCHEMA_VERSION = "0.1.0"
"""Version of the ``context_run.json`` document."""

ContextRunId = NewType("ContextRunId", str)
"""``"sha256:<hex>"`` over the foundation, the selection and the content of the artifacts."""

Disposition = Literal["produced", "reused"]


@dataclass(frozen=True, kw_only=True)
class ContextArtifact:
    """One capability artifact a ContextRun references.

    Attributes:
        ref: The artifact, with its content hash and location.
        disposition: ``"produced"`` by this run, or ``"reused"`` from an earlier one.
    """

    ref: ArtifactRef
    disposition: Disposition

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form."""
        return {**self.ref.to_document(), "disposition": self.disposition}


@dataclass(frozen=True, kw_only=True)
class ContextRun:
    """The record of one execution of the context stages over a spatial foundation.

    Attributes:
        identity: Derived from the foundation, the observation selection and the content of
            the referenced artifacts; never from a path, a run number or a clock. Two executions
            that reference the same evidence are the same ContextRun.
        foundation: The spatial foundation the context stages ran against.
        observation_selection: The encoded selection of the observations processed, or
            ``None`` for the whole sequence.
        artifacts: The context artifacts, one per executed context stage, ordered by stage.
        run: Where the runtime run that holds the record is, relative to the workspace
            (``<dataset>/run-NNNN``). A locator, never part of the identity.
        plan_digest: Digest of the resolved plan the run executed.
    """

    identity: ContextRunId
    foundation: SpatialFoundation
    observation_selection: Mapping[str, Any] | None
    artifacts: tuple[ContextArtifact, ...]
    run: str
    plan_digest: str

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible record persisted as ``context_run.json``."""
        return {
            "schema_version": CONTEXT_RUN_SCHEMA_VERSION,
            "context_run_id": self.identity,
            "foundation": self.foundation.to_document(),
            "observation_selection": self.observation_selection,
            "artifacts": [artifact.to_document() for artifact in self.artifacts],
            "run": self.run,
            "plan_digest": self.plan_digest,
        }


def context_scope(
    plan: PipelinePlan,
    foundation: SpatialFoundation,
    *,
    selected: Mapping[str, Sequence[ArtifactRef]] | None = None,
) -> ExecutionPlan:
    """Scope a plan to the context stages over a provided spatial foundation.

    The foundation stages are supplied, never recomputed: a ContextRun adds evidence to a fixed
    world model. The observation selection comes from the plan (``inputs.observation_selection``);
    the Visual Perception executor resolves it against the foundation's sequence before any work.

    A perception run may be selected instead of executed, so that branches over different maps
    of one sequence associate the same regions and labels: a VLM in the loop makes perception
    differ between executions of the same configuration. :func:`check_context_selection` checks
    it against the foundation and the observation selection before anything runs.

    Args:
        plan: The resolved topology.
        foundation: The spatial foundation to run against.
        selected: Existing runs of a stage in :data:`SELECTABLE_CONTEXT_STAGES`, used instead
            of executing that stage.

    Returns:
        The execution of enabled context stages, with selected stages explicitly reused.

    Raises:
        ContextRunError: If a stage outside :data:`SELECTABLE_CONTEXT_STAGES` is selected; the
            foundation stages always come from the foundation.
    """
    chosen = dict(selected or {})
    refused = sorted(set(chosen) - set(SELECTABLE_CONTEXT_STAGES))
    if refused:
        raise ContextRunError(
            f"cannot select {refused}: a ContextRun may select only "
            f"{SELECTABLE_CONTEXT_STAGES}; foundation stages come from the branch's foundation"
        )
    if {"visual_perception", "semantic_interpretation"} & set(chosen) and (
        "region_perception" not in chosen
    ):
        raise ContextRunError(
            "select region_perception explicitly with selected visual_perception or "
            "semantic_interpretation evidence"
        )
    provided: dict[str, ArtifactRef | Sequence[ArtifactRef]] = {
        stage_id: getattr(foundation, role) for role, stage_id in FOUNDATION_STAGE_OF_ROLE.items()
    }
    provided.update(chosen)
    # A seleção explícita continua sendo alvo: plan.scope a registra como reutilizada sem
    # executá-la, mesmo se nenhum estágio restante consumir o artifact diretamente. Assim
    # a ContextRun conserva os três produtores independentes que o usuário selecionou.
    return plan.scope(
        targets=[stage.stage_id for stage in plan.stages if stage.stage_id in CONTEXT_STAGES],
        provided=provided,
    )


def check_context_selection(
    execution: ExecutionPlan, foundation: SpatialFoundation, *, workspace: Path
) -> None:
    """Check that a selected perception run is evidence this ContextRun may reference.

    The ContextRun records the plan's observation selection as the selection its evidence
    covers, so a selected perception run must have processed exactly that selection, of the
    foundation's sequence. Nothing is inferred from the configuration that produced it: the
    run's own manifest is read.

    Args:
        execution: The scoped execution, from :func:`context_scope`.
        foundation: The spatial foundation the execution is scoped to.
        workspace: The workspace the selected runs are located in.

    Raises:
        ContextRunError: If a selected perception run has no location, cannot be read, or was
            built over another sequence or another observation selection.
    """
    for stage_id in SELECTABLE_CONTEXT_STAGES:
        for ref in execution.reused.get(stage_id, ()):
            mismatch = perception_selection_mismatch(
                ref,
                workspace=workspace,
                sequence_id=foundation.sequence.artifact_id,
                observation_selection=_plan_selection(execution),
            )
            if mismatch is not None:
                raise ContextRunError(mismatch)
    regions = execution.reused.get("region_perception", ())
    if regions:
        from contextmap.semantic_interpretation import SemanticInterpretationRunReader
        from contextmap.visual_perception import VisualEvidenceRunReader

        selected_region_ids = {ref.artifact_id for ref in regions}
        for stage_id, reader in (
            ("visual_perception", VisualEvidenceRunReader),
            ("semantic_interpretation", SemanticInterpretationRunReader),
        ):
            for ref in execution.reused.get(stage_id, ()):
                assert ref.location is not None
                regional_id = reader(workspace / ref.location).manifest.region_artifact_id
                if regional_id not in selected_region_ids:
                    raise ContextRunError(
                        f"selected {stage_id} artifact {ref.artifact_id!r} depends on "
                        f"region_perception artifact {regional_id!r}, outside the selected "
                        "regional evidence"
                    )


def publish_context_run(
    run_directory: Path,
    *,
    workspace: Path,
    foundation: SpatialFoundation,
    execution: ExecutionPlan,
    record: ExecutionRecord,
) -> ContextRun:
    """Publish the ContextRun record of a completed run, once.

    Args:
        run_directory: The directory of the runtime run that executed ``execution``.
        workspace: The workspace the run lives in.
        foundation: The spatial foundation the execution was scoped to.
        execution: The scoped execution, from :func:`context_scope`.
        record: Its execution record.

    Returns:
        The published record.

    Raises:
        ContextRunError: If the record holds a stage outside the context, an artifact without
            a content hash or location, evidence built from another foundation's artifacts, or
            a selected perception run :func:`check_context_selection` refuses.
        FileExistsError: If the run already has a ContextRun record.
    """
    check_context_selection(execution, foundation, workspace=workspace)
    foundation_refs = {
        stage_id: _signature(getattr(foundation, role))
        for role, stage_id in FOUNDATION_STAGE_OF_ROLE.items()
    }
    artifacts: list[ContextArtifact] = []
    for stage in record.stages:
        if stage.stage_id not in CONTEXT_STAGES:
            raise ContextRunError(
                f"stage {stage.stage_id!r} is not a context stage; a ContextRun holds only "
                f"{list(CONTEXT_STAGES)}"
            )
        for name, refs in stage.inputs.items():
            for ref in refs:
                expected = foundation_refs.get(ref.stage_id)
                if expected is not None and _signature(ref) != expected:
                    raise ContextRunError(
                        f"stage {stage.stage_id!r} consumed {ref.artifact_id!r} as {name!r}, "
                        f"not the foundation's {expected[2]!r}"
                    )
        if stage.output.content_hash is None or stage.output.location is None:
            raise ContextRunError(
                f"stage {stage.stage_id!r} produced {stage.output.artifact_id!r} without a "
                "content hash and a location; a ContextRun cannot reference it"
            )
        reused = stage.decision is not None and stage.decision.kind == "reused"
        artifacts.append(
            ContextArtifact(ref=stage.output, disposition="reused" if reused else "produced")
        )
    # Um estágio de contexto selecionado não executou, mas a evidência dele é parte do registro.
    for stage_id in SELECTABLE_CONTEXT_STAGES:
        for ref in execution.reused.get(stage_id, ()):
            if ref.content_hash is None or ref.location is None:
                raise ContextRunError(
                    f"selected {stage_id!r} run {ref.artifact_id!r} has no content hash and "
                    "location; a ContextRun cannot reference it"
                )
            artifacts.append(ContextArtifact(ref=ref, disposition="reused"))
    ordered = tuple(sorted(artifacts, key=lambda artifact: artifact.ref.stage_id))
    selection = _plan_selection(execution)
    context = ContextRun(
        identity=_identity(foundation, selection, ordered),
        foundation=foundation,
        observation_selection=selection,
        artifacts=ordered,
        run=run_directory.relative_to(workspace).as_posix(),
        plan_digest=record.plan_digest,
    )
    text = json.dumps(context.to_document(), indent=2, sort_keys=True, ensure_ascii=True)
    publish_text(run_directory, CONTEXT_RUN_FILENAME, text + "\n")
    return context


def read_context_run(run_directory: Path, *, workspace: Path) -> ContextRun:
    """Read the ContextRun record of a run and check it was not altered or moved.

    Args:
        run_directory: The directory of the runtime run that holds the record.
        workspace: The workspace the run lives in.

    Returns:
        The record.

    Raises:
        ContextRunError: If the record is unreadable, follows another schema version, names
            another run, or does not match its own identity.
    """
    path = run_directory / CONTEXT_RUN_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContextRunError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise ContextRunError(f"{path} is not a ContextRun record")
    if document.get("schema_version") != CONTEXT_RUN_SCHEMA_VERSION:
        raise ContextRunError(
            f"{path} follows schema_version {document.get('schema_version')!r}, this runtime "
            f"reads {CONTEXT_RUN_SCHEMA_VERSION!r}"
        )
    run = run_directory.relative_to(workspace).as_posix()
    if document.get("run") != run:
        raise ContextRunError(f"{path} is the record of run {document.get('run')!r}, not {run!r}")
    try:
        foundation = SpatialFoundation.from_document(document["foundation"])
        artifacts = tuple(
            ContextArtifact(ref=ArtifactRef.from_document(item), disposition=_disposition(item))
            for item in document["artifacts"]
        )
        selection = document["observation_selection"]
        plan_digest = str(document["plan_digest"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContextRunError(f"{path} is not a valid ContextRun record: {error}") from error
    identity = _identity(foundation, selection, artifacts)
    if document.get("context_run_id") != identity:
        raise ContextRunError(
            f"{path} does not match its identity: it records {document.get('context_run_id')!r} "
            f"but its content gives {identity!r}"
        )
    return ContextRun(
        identity=identity,
        foundation=foundation,
        observation_selection=selection,
        artifacts=artifacts,
        run=run,
        plan_digest=plan_digest,
    )


def _plan_selection(execution: ExecutionPlan) -> Mapping[str, Any] | None:
    """Return the observation selection of the observation-scoped context stage, run or selected.

    A selected perception is not among the executed stages, but the selection its evidence
    covers is still the plan's; :func:`check_context_selection` holds the two together.
    """
    present = {stage.stage_id for stage in execution.stages} | set(execution.reused)
    for stage in execution.plan.stages:
        if stage.stage_id in present and stage.observation_selection is not None:
            return stage.observation_selection
    return None


def _signature(ref: ArtifactRef) -> tuple[str, str, str, str | None]:
    """What makes two references name the same artifact; the location is only a locator."""
    return (ref.stage_id, ref.contract, ref.artifact_id, ref.content_hash)


def _disposition(item: Mapping[str, Any]) -> Disposition:
    value = item.get("disposition")
    if value == "produced":
        return "produced"
    if value == "reused":
        return "reused"
    raise ValueError(f"unknown artifact disposition {value!r}")


def _identity(
    foundation: SpatialFoundation,
    selection: Mapping[str, Any] | None,
    artifacts: tuple[ContextArtifact, ...],
) -> ContextRunId:
    """Identify a ContextRun by its foundation, its selection and its artifacts' content."""
    document = {
        "foundation": foundation.identity,
        "observation_selection": selection,
        "artifacts": sorted(
            [artifact.ref.stage_id, artifact.ref.contract, artifact.ref.content_hash]
            for artifact in artifacts
        ),
    }
    return ContextRunId(f"sha256:{hashlib.sha256(_canonical(document).encode()).hexdigest()}")


def _canonical(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
