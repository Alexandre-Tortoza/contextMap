"""Índice de builds de um mapa: o livro-razão imutável do que foi publicado sob um ``map_id``.

Cada materialização de uma branch publica um ``ContextMapArtifact`` sob a raiz de mapas, em
``<maps_root>/<map_id>/<NNNNNN>``, e acrescenta uma entrada ao índice da branch, no run store,
em ``branches/<map_id>/builds/<NNNNNN>.json``. O índice é só acréscimo e sem lacunas
(``1..M``): é ele que torna cada mapa publicado localizável por ``map_id`` sem varrer runs, e
que liga cada artifact visível a um viewer ao build e ao run que o produziram.

Por isso o ordinal de um build é reservado sob um lock do índice daquele mapa
(:func:`claim_context_build`) e só registrado depois que o mapa é publicado. A resolução da
raiz de mapas (:func:`effective_maps_root`) e do diretório de um mapa a partir da sua entrada
(:func:`resolve_map_artifact`) moram aqui porque são a outra ponta do mesmo livro-razão: o
escritor e os leitores de um build precisam concordar sobre onde ele publicou. O que um build
consome, e a identidade dele, ficam em
:mod:`~contextmap.runtime.contextual_mapping.context_build`.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmap.runtime._files import publish_text
from contextmap.runtime.artifacts import ArtifactRef, artifact_directory
from contextmap.runtime.contextual_mapping.context_branch import ContextBranch, branch_directory
from contextmap.runtime.contextual_mapping.context_build import ContextBuildId
from contextmap.runtime.contextual_mapping.errors import ContextBuildError

CONTEXT_BUILD_INDEX_SCHEMA_VERSION = "0.1.0"
"""Version of a build-index entry (``branches/<map_id>/builds/<NNNNNN>.json``)."""

BUILDS_DIRECTORY = "builds"
"""Where a branch's build index lives: ``<workspace>/<dataset>/branches/<name>/builds``."""

_BUILD_INDEX_LOCK_FILENAME = ".build-index.lock"


@dataclass(frozen=True, kw_only=True)
class ContextBuildEntry:
    """One materialization published under a map, recorded in the branch's build index.

    The index is orchestration provenance: an append-only, immutable ledger of the map's builds
    that makes each published ``ContextMapArtifact`` discoverable by ``map_id`` without scanning
    runs, and ties every viewer-visible artifact back to the build and run that produced it. It
    copies no evidence: the map is referenced by :class:`ArtifactRef`.

    Attributes:
        index: The build's ordinal under this map, from 1, in the order builds were recorded.
        context_build_id: The frozen input the build materialized.
        branch_revision: The branch revision the build froze; provenance.
        context_map: A handle to the published :class:`ContextMapArtifact`, with its content hash
            and its location **relative to the maps root** (``<map_id>/<NNNNNN>``).
        run: Where the build's runtime run lives, relative to the run store
            (``<dataset>/run-NNNN``).
    """

    index: int
    context_build_id: ContextBuildId
    branch_revision: int
    context_map: ArtifactRef
    run: str

    def to_document(self) -> dict[str, Any]:
        """Return the JSON-compatible form persisted in the build index."""
        return {
            "schema_version": CONTEXT_BUILD_INDEX_SCHEMA_VERSION,
            "index": self.index,
            "context_build_id": self.context_build_id,
            "branch_revision": self.branch_revision,
            "context_map": self.context_map.to_document(),
            "run": self.run,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ContextBuildEntry:
        """Rebuild an entry from its persisted form.

        Raises:
            ContextBuildError: If the record follows another schema version or is malformed.
        """
        if document.get("schema_version") != CONTEXT_BUILD_INDEX_SCHEMA_VERSION:
            raise ContextBuildError(
                f"build index entry follows schema_version {document.get('schema_version')!r}, "
                f"this runtime reads {CONTEXT_BUILD_INDEX_SCHEMA_VERSION!r}"
            )
        try:
            return cls(
                index=int(document["index"]),
                context_build_id=ContextBuildId(str(document["context_build_id"])),
                branch_revision=int(document["branch_revision"]),
                context_map=ArtifactRef.from_document(document["context_map"]),
                run=str(document["run"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContextBuildError(f"invalid build index entry: {error}") from error


def builds_directory(workspace: Path, dataset: str, name: str) -> Path:
    """Return a map's build-index directory, ``<workspace>/<dataset>/branches/<name>/builds``."""
    return branch_directory(workspace, dataset, name) / BUILDS_DIRECTORY


def record_context_build(
    workspace: Path,
    branch: ContextBranch,
    *,
    index: int,
    context_build_id: ContextBuildId,
    branch_revision: int,
    context_map: ArtifactRef,
    run: str,
) -> ContextBuildEntry:
    """Append one build to a map's index, atomically and without overwriting.

    Args:
        workspace: The run store root, where the branch (and its index) live.
        branch: The map the build belongs to.
        index: The build's ordinal, from 1; the caller allocates it from the current index.
        context_build_id: The frozen input the build materialized.
        branch_revision: The branch revision the build froze.
        context_map: A handle to the published map, located relative to the maps root.
        run: The build's run directory, relative to the run store.

    Returns:
        The recorded entry.

    Raises:
        ContextBuildError: If this ``index`` was already recorded (a concurrent build took it);
            reopen the index and record again.
    """
    entry = ContextBuildEntry(
        index=index,
        context_build_id=context_build_id,
        branch_revision=branch_revision,
        context_map=context_map,
        run=run,
    )
    directory = builds_directory(workspace, branch.dataset, branch.name)
    text = json.dumps(entry.to_document(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    try:
        publish_text(directory, f"{index:06d}.json", text)
    except FileExistsError as error:
        raise ContextBuildError(
            f"build {index} of map {branch.name!r} was recorded meanwhile; reopen the index "
            "and record again"
        ) from error
    return entry


@dataclass(frozen=True, kw_only=True)
class ClaimedContextBuild:
    """The ordinal and the map directory one build holds while it materializes.

    Attributes:
        index: The build's ordinal under this map, from 1, to record once the map is published.
        map_output: Where the build publishes its ``ContextMapArtifact``; it does not exist yet.
    """

    index: int
    map_output: Path


@contextmanager
def claim_context_build(
    workspace: Path, branch: ContextBranch, *, maps_root: Path
) -> Iterator[ClaimedContextBuild]:
    """Hold a map's next build ordinal for the duration of one build.

    The build index is a strictly contiguous ``1..M`` ledger, so the ordinal cannot be derived
    independently by two builds of the same map: both would read the same number, publish into
    the same ``<maps root>/<map_id>/NNNNNN`` and one would lose the directory it had already
    written. The ordinal is therefore allocated and held under a lock on the map's index, the
    same way the run store allocates a run number, and released when the build ends.

    Builds of *different* maps never contend: the lock is per map.

    Args:
        workspace: The run store root, where the branch and its index live.
        branch: The map being built.
        maps_root: Where published maps live; the build's directory is allocated inside it.

    Yields:
        The held ordinal and the directory the map must be published into.

    Raises:
        ContextBuildError: If another build of this map holds the ordinal, or if a map already
            sits at the claimed directory without an entry recording it.
    """
    directory = builds_directory(workspace, branch.dataset, branch.name)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / _BUILD_INDEX_LOCK_FILENAME
    with lock_path.open("a", encoding="ascii") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ContextBuildError(
                f"another build of map {branch.name!r} is in progress; builds of one map are "
                "serialized because its index has no gaps"
            ) from error
        index = len(read_context_builds(workspace, dataset=branch.dataset, name=branch.name)) + 1
        map_output = maps_root / branch.name / f"{index:06d}"
        # NOTE: Um mapa publicado que nenhuma entrada registra vem de um build interrompido. O
        # próximo build reivindica o mesmo ordinal, porque o índice não tem lacunas, e remover
        # ou substituir um artifact publicado nunca é automático: quem opera decide.
        if map_output.exists():
            raise ContextBuildError(
                f"an unrecorded map already exists at {map_output} (an interrupted build); "
                "remove it and run the build again"
            )
        yield ClaimedContextBuild(index=index, map_output=map_output)


def read_context_builds(
    workspace: Path, *, dataset: str, name: str
) -> tuple[ContextBuildEntry, ...]:
    """Read a map's build index, in order, from the run store.

    Args:
        workspace: The run store root.
        dataset: The dataset the map lives in.
        name: The ``map_id`` (branch name).

    Returns:
        The recorded builds ordered by index; empty when the map has none.

    Raises:
        ContextBuildError: If a record is unreadable or malformed, or the indices are not
            exactly ``1..M``.
    """
    directory = builds_directory(workspace, dataset, name)
    records = sorted(directory.glob("[0-9]*.json")) if directory.is_dir() else []
    entries: list[ContextBuildEntry] = []
    for expected, record in enumerate(records, start=1):
        if record.stem != f"{expected:06d}":
            raise ContextBuildError(
                f"map {name!r} lacks the record of build {expected}; found {record.name}"
            )
        try:
            document = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ContextBuildError(f"cannot read {record}: {error}") from error
        if not isinstance(document, dict):
            raise ContextBuildError(f"{record} is not a build index entry")
        entry = ContextBuildEntry.from_document(document)
        if entry.index != expected:
            raise ContextBuildError(f"{record.name} records build {entry.index}, not {expected}")
        entries.append(entry)
    return tuple(entries)


def effective_maps_root(workspace: Path, configured: str | Path | None) -> Path:
    """Return the maps root a ``context build`` publishes into.

    ``resources.maps_root`` wins when set. Otherwise the default is a clean sibling of the run
    store, ``<workspace>/../maps``, so a build always publishes each map by ``map_id`` apart from
    the run store. It is the single definition: the writer (the CLI) and the readers of a build
    must agree on where an unconfigured build put its map.
    """
    return Path(configured) if configured is not None else workspace.parent / "maps"


def resolve_map_artifact(maps_root: Path, entry: ContextBuildEntry) -> Path:
    """Return the directory of a build's ``ContextMapArtifact`` inside the maps root.

    Raises:
        ValueError: If the entry's handle has no location inside the maps root.
    """
    return artifact_directory(maps_root, entry.context_map)
