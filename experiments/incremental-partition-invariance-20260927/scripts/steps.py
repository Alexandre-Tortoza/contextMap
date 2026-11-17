"""Passos compartilhados pelos harnesses de contexto incremental.

Cada passo executa a CLI em outro processo e registra resposta, tempo e memória."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextmap.evaluation import (
    BuildState,
    encode_incremental_context,
    measure_child_process,
    stage_dispositions,
)
from contextmap.ingestion import SequenceArtifactReader
from contextmap.runtime import resolve_effective_config

CONTEXT_STEP = Path(__file__).resolve().with_name("context_step.py")
PERCEPTION_COMPONENTS = (
    "region_discovery",
    "dense_features",
    "region_features",
    "semantic_interpretation",
)
BUILD_STAGES = (
    "semantic_fusion",
    "semantic_mapping",
    "entity_resolution",
    "spatial_relations",
    "context_map",
)
"""The artifacts a build materializes, in pipeline order."""
STRIP_PROVIDERS = "resources.providers={}"
"""A build runs no perception, but the CLI composes every stage its configuration can: without
providers, a build's composition loads no perception model."""

PROVIDER_HINT = (
    "visual_perception could not be composed: SAM2 and Qwen ship no model loader, so the "
    "configuration must declare a RuntimeProvider for each under resources.providers "
    '("visual_perception.region_discovery" and "visual_perception.semantic_interpretation" '
    'as "module:attribute" targets importable on this machine), with the vision extra '
    "(torch, transformers, Pillow) installed. Nothing was faked: see the README."
)


class StepFailed(Exception):
    """A step of the harness did not complete; the run stops there.

    Attributes:
        step: The step that failed.
        message: The CLI's error, or why no answer could be read.
        problems: The problems the CLI listed, as ``"<path>: <message>"``.
        record: Exit status, wall time and peak memory of the failed process.
        stderr: The end of what the process wrote to its standard error.
    """

    def __init__(
        self,
        step: str,
        message: str,
        *,
        problems: Sequence[str] = (),
        record: dict[str, Any] | None = None,
        stderr: str = "",
    ) -> None:
        """Keep everything the report needs to explain the failure."""
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message
        self.problems = tuple(problems)
        self.record = record or {}
        self.stderr = stderr

    def to_document(self) -> dict[str, Any]:
        """Return the failure as the report records it, with a hint when perception is missing."""
        perception = any("visual_perception" in item for item in (self.message, *self.problems))
        return {
            "step": self.step,
            "message": "; ".join((self.message, *self.problems)),
            "hint": PROVIDER_HINT if perception else None,
            **self.record,
            "stderr_tail": self.stderr[-2000:],
        }


def unexpected_failure(error: Exception) -> dict[str, Any]:
    """Return the report's failure for an error raised outside a command, before re-raising it."""
    return {"step": None, "message": f"{type(error).__name__}: {error}", "hint": None}


class Portable:
    """Keep machine paths out of what a harness report records about a failure."""

    def __init__(self, *, workspace: Path, outputs: Path, checkout: Path) -> None:
        """Replace the longest roots first: the workspace may live under the outputs."""
        roots = {
            str(workspace): "<workspace>",
            str(outputs): "<outputs>",
            str(checkout): "<checkout>",
            str(Path.home()): "~",
        }
        self._roots = sorted(roots.items(), key=lambda item: -len(item[0]))

    def text(self, value: str) -> str:
        """Return ``value`` with every known machine root replaced by its label."""
        for root, label in self._roots:
            value = value.replace(root, label)
        return value

    def document(self, failure: dict[str, Any]) -> dict[str, Any]:
        """Return a failure record whose text fields cite no machine path."""
        return {
            key: self.text(value) if isinstance(value, str) else value
            for key, value in failure.items()
        }

    def failure(self, failure: StepFailed, steps_dir: Path) -> dict[str, Any]:
        """Record a failed step; its standard error, full of machine paths, stays in ``steps/``.

        Args:
            failure: The step that failed.
            steps_dir: Where the arm's steps keep their requests and answers,
                ``<harness>/steps/<arm>``; the record cites the error file from ``<harness>``.
        """
        document = failure.to_document()
        stderr = steps_dir / f"{failure.step}.stderr.txt"
        stderr.write_text(str(document.pop("stderr_tail")), encoding="utf-8")
        return self.document({**document, "stderr": relative(stderr, steps_dir.parents[1])})


@dataclass(frozen=True)
class Partition:
    """The images of one experiment and how they are split.

    Attributes:
        whole: The selected images, in canonical order.
        chunks: Consecutive, equal slices of ``whole`` that partition it.
        rerun: A frame range of the canonical index covering exactly ``whole``: the same images
            under another selection, so a second whole run is another inference execution.
    """

    whole: tuple[str, ...]
    chunks: tuple[tuple[str, ...], ...]
    rerun: dict[str, Any] = field(hash=False)


@dataclass(frozen=True, kw_only=True)
class StepResult:
    """What one completed step answered, and what it cost.

    Attributes:
        name: The step.
        document: The CLI's JSON answer.
        record: Exit status, wall time (s) and peak RSS (MiB) of the step's process.
    """

    name: str
    document: dict[str, Any]
    record: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class ContextCli:
    """The ``contextmap`` commands of one harness run, with their shared configuration.

    Attributes:
        workspace: The runtime workspace, the one holding the foundation run.
        configs: The configuration files, in order.
        overrides: ``PATH=VALUE`` overrides applied to every step.
        steps_dir: Where each step's request and answer are kept (outside Git).
        code_identity: The ``--code-identity`` of every context run and build: the code their
            runs record, from which the CLI also derives the semantic map's code digest.
        dry_run_executors: ``"module:attribute"`` of the executors a dry run injects; ``None``
            on the GPU machine, where every stage is composed from the configuration.
    """

    workspace: Path
    configs: tuple[Path, ...]
    overrides: tuple[str, ...]
    steps_dir: Path
    code_identity: str
    dry_run_executors: str | None = None

    def branch(self, name: str, *, foundation_run: Path) -> StepResult:
        """Create a branch over the foundation of a completed run."""
        argv = ["context", "branch", "create", name, "--from-run", str(foundation_run)]
        return self.run(f"branch-{name}", argv)

    def context_run(self, name: str, *, branch: str, selection: dict[str, Any]) -> StepResult:
        """Run perception and association over one selection and append the ContextRun."""
        return self.run(
            name,
            ["context", "run", "--branch", branch, "--code-identity", self.code_identity],
            overrides=(f"inputs.observation_selection={json.dumps(selection)}",),
        )

    def build(
        self,
        name: str,
        *,
        branch: str,
        revision: int,
        members: Sequence[str] = (),
        overrides: Sequence[str] = (),
    ) -> StepResult:
        """Freeze a branch revision (or some of its members) and materialize its ContextMap."""
        argv = ["context", "build", "--branch", branch, "--revision", str(revision)]
        argv += ["--code-identity", self.code_identity]
        for member in members:
            argv += ["--member", member]
        return self.run(name, argv, overrides=(STRIP_PROVIDERS, *overrides))

    def run(self, name: str, argv: Sequence[str], *, overrides: Sequence[str] = ()) -> StepResult:
        """Run one CLI command in a fresh, measured process and read its JSON answer.

        Raises:
            StepFailed: If the command did not complete or its answer cannot be read.
        """
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        answer = self.steps_dir / f"{name}.json"
        request = {
            "argv": [*argv, "--workspace", str(self.workspace), "--json"],
            "configs": [str(path) for path in self.configs],
            "overrides": [*self.overrides, *overrides],
            "dry_run_executors": self.dry_run_executors,
            "answer": str(answer),
        }
        request_path = self.steps_dir / f"{name}.request.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        measured = measure_child_process([sys.executable, str(CONTEXT_STEP), str(request_path)])
        record = measured.to_record()
        try:
            document: dict[str, Any] = json.loads(answer.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StepFailed(
                name, f"no readable answer ({error})", record=record, stderr=measured.stderr
            ) from None
        if measured.returncode != 0 or not document.get("ok", False):
            listed = document.get("problems", [])
            problems = [f"{item['path']}: {item['message']}" for item in listed]
            raise StepFailed(
                name,
                str(document.get("error", "the command failed")),
                problems=problems,
                record=record,
                stderr=measured.stderr,
            )
        return StepResult(name=name, document=document, record=record)


def step_record(step: StepResult, workspace: Path) -> dict[str, Any]:
    """Record a step that ran a plan: its cost, its run and how each stage got its artifact."""
    run = Path(step.document["run_directory"])
    return {
        **step.record,
        "run": relative(run, workspace),
        "stages": stage_dispositions(run),
        "storage_bytes": directory_bytes(run),
    }


def build_record(state: BuildState, build_run: Path, workspace: Path) -> dict[str, Any]:
    """Record a materialized build: identities, counts, validation and bytes per artifact."""
    fields = (
        "context_build_id",
        "context_run_ids",
        "context_map_id",
        "counts",
        "validation",
        "integrity_problems",
    )
    return {
        **encode_incremental_context({name: getattr(state, name) for name in fields}),
        "run": relative(build_run, workspace),
        "artifact_bytes": {
            stage: directory_bytes(build_run.parent.parent / "artifacts" / stage / build_run.name)
            for stage in BUILD_STAGES
            if (build_run.parent.parent / "artifacts" / stage / build_run.name).is_dir()
        },
        "fusion_layer_two": encode_incremental_context(state.layer_two.fusion),
    }


def configuration_record(configs: Sequence[Path], overrides: Sequence[str]) -> dict[str, Any]:
    """Return the effective configuration a run freezes, without any machine path.

    Returns:
        ``configuration`` (digest, file names, overrides and the effective document without
        ``resources.workspace``), ``perception`` (each perception component: backend and
        parameters, the checkpoints included) and ``providers`` (the declared targets).
    """
    effective = resolve_effective_config(files=list(configs), overrides=list(overrides))
    document = effective.config.to_document()
    # O workspace é um caminho da máquina: fica fora do registro versionável.
    document.get("resources", {}).pop("workspace", None)
    perception = document.get("components", {}).get("visual_perception", {})
    return {
        "configuration": {
            "digest": effective.digest,
            "files": [path.name for path in configs],
            "overrides": list(overrides),
            "document": document,
        },
        "perception": {slot: perception.get(slot) for slot in PERCEPTION_COMPONENTS},
        "providers": document.get("resources", {}).get("providers", {}),
    }


def image_positions(sequence: Path) -> list[tuple[int, str]]:
    """Return every image of a SequenceArtifact as ``(canonical index position, id)``.

    The position is the one a ``frame_range`` selection counts, over every modality.
    """
    return [
        (position, str(entry.observation_id))
        for position, entry in enumerate(SequenceArtifactReader(sequence).iter_index())
        if entry.modality == "image"
    ]


def plan_partition(
    images: Sequence[tuple[int, str]], *, first: int, count: int, chunks: int
) -> Partition:
    """Select ``count`` consecutive images from ``first`` and split them into equal chunks.

    Args:
        images: Every image of the sequence, as :func:`image_positions` returns them.
        first: Index, among the images, of the first one selected.
        count: Number of images selected.
        chunks: Number of consecutive chunks, each of ``count / chunks`` images.

    Returns:
        The partition.

    Raises:
        ValueError: If the selection does not fit the sequence or the chunks are not equal.
    """
    if first < 0:
        raise ValueError(f"the first image must be a non-negative index, got {first}")
    if count < 1 or chunks < 1:
        raise ValueError(f"count and chunks must be positive, got {count} and {chunks}")
    if count % chunks:
        raise ValueError(f"{count} images do not split into {chunks} equal chunks")
    if first + count > len(images):
        raise ValueError(
            f"images [{first}, {first + count}) do not fit: the sequence has only "
            f"{len(images)} images"
        )
    selected = images[first : first + count]
    whole = tuple(observation_id for _, observation_id in selected)
    size = count // chunks
    return Partition(
        whole=whole,
        chunks=tuple(whole[start : start + size] for start in range(0, count, size)),
        rerun={
            "kind": "frame_range",
            "start_frame_index": selected[0][0],
            "end_frame_index": selected[-1][0] + 1,
        },
    )


def explicit(observation_ids: Sequence[str]) -> dict[str, Any]:
    """Return the canonical selection document of an explicit set of observations."""
    return {"kind": "explicit_ids", "observation_ids": sorted(observation_ids)}


def git_identity(checkout: Path) -> dict[str, Any]:
    """Return the commit of a checkout and whether its tree has changes, or ``None`` values."""
    status = _git(checkout, "status", "--porcelain")
    return {
        "commit": _git(checkout, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def resolve_code_identity(
    code: Mapping[str, Any], *, checkout: Path, evidence_class: str, requested: str | None
) -> str:
    """Return the code identity every step of a harness run records, or refuse a real run.

    A real run is evidence, so it runs exactly the commit it records: the checkout needs a
    commit and a tree known to be clean, and an explicit identity must name that very commit.
    Git normalizes the explicit value in the checkout, so an abbreviated SHA of the checked-out
    commit is accepted, and the identity recorded is always the commit's full SHA. A dry run
    (CI, development) may run from a dirty tree or outside git, and records what it is given.

    Args:
        code: The checkout's state as :func:`git_identity` read it, the one the run records.
        checkout: The checkout the harness runs from, where ``requested`` is normalized.
        evidence_class: ``"real"`` or ``"dry_run"``.
        requested: The ``--code-identity`` given, or ``None`` when omitted.

    Returns:
        For a real run, the full SHA of the checked-out commit. For a dry run, ``requested`` as
        given, otherwise the commit, otherwise ``"unknown"``.

    Raises:
        ValueError: For a real run, with every reason that applies and how to fix it: no commit
            identifies the checkout; its tree is dirty or ``git status`` is unavailable;
            ``requested`` does not name a commit, or names another than the checked-out one.
            Also for an evidence class that is neither ``"real"`` nor ``"dry_run"``.
    """
    if evidence_class == "dry_run":
        # O dry run, da CI e do desenvolvimento, roda de árvores com mudanças e só registra o
        # estado delas em ``code``.
        if requested is not None:
            return requested
        return str(code["commit"] or "unknown")
    if evidence_class != "real":
        raise ValueError(f"unknown evidence class {evidence_class!r}: a run is 'real' or 'dry_run'")
    commit = code["commit"]
    checkout_problems = []
    if commit is None:
        checkout_problems.append(
            "no commit identifies the checkout (not a git repository, or no commit)"
        )
    if code["dirty"] is True:
        checkout_problems.append("the working tree is dirty")
    elif code["dirty"] is not False:
        checkout_problems.append(
            "git status is unavailable, so the working tree is not known to be clean"
        )
    identity_problems = []
    if requested is not None:
        # Só o git sabe se o valor nomeia o commit do checkout: nunca se comparam abreviações.
        named = _git(checkout, "rev-parse", "--verify", "--quiet", f"{requested}^{{commit}}")
        if named is None:
            identity_problems.append(
                f"--code-identity {requested!r} does not name a commit of the checkout"
            )
        elif named != commit:
            identity_problems.append(
                f"--code-identity {requested!r} names commit {named}, not the checked-out "
                f"commit {commit}"
            )
    fixes = []
    if checkout_problems:
        fixes.append("commit or stash the changes and rerun on a clean checkout")
    if identity_problems:
        fixes.append("pass the checked-out commit to --code-identity, or omit it")
    if fixes:
        raise ValueError(
            "a real run must run exactly the commit it records, but "
            + "; ".join([*checkout_problems, *identity_problems])
            + ": "
            + "; ".join(fixes)
        )
    return str(commit)


def directory_bytes(directory: Path) -> int:
    """Return the bytes of every file under a directory."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def relative(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, written with ``/``, never a machine path."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def map_location(build: Any) -> str:
    """Return ``<map_id>/<NNNNNN>``, where a build published its map, relative to the maps root.

    The maps root is a sibling of the run store, so this is never a path under the workspace;
    the build record of the CLI names the absolute directory, and only its two last parts are
    portable.
    """
    published = Path(build.document["map"]["location"])
    return f"{published.parent.name}/{published.name}"


def write_once(path: Path, document: dict[str, Any]) -> None:
    """Write a JSON document that must not exist yet.

    Raises:
        FileExistsError: If the file exists: a record of another run is never overwritten.
    """
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _git(checkout: Path, *arguments: str) -> str | None:
    """Return what a git command printed in a checkout, stripped, or ``None`` if it failed."""
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
