"""Atomic, self-describing run directories shared by every run artifact.

``docs/ARTIFACTS.md`` fixes the rules every run artifact follows regardless of
its payload: an interrupted write must never look like a finished run, a
finished run is immutable and never overwritten, its manifest inventories
every contractual file with size and hash so corruption is detected, and run
indexes are monotonic per capability and sequence and computed from valid runs
on disk (a registry file is only a convenience). This module implements those
rules once. It knows nothing about what a run contains; the owning capability
decides the files, the manifest fields and what makes a run valid.

Publication is durable as well as atomic: every file and directory of a run is
forced to stable storage (``fsync``) before the rename that makes it visible,
and the parent directory right after it. The rename alone only protects against
a process that dies; without the syncs a power loss could persist the rename
before the data and leave a published run with empty or truncated files.
Syncing a directory opens it read-only, which is POSIX behavior (Linux, the
platform the project runs on). An owner whose files come from writers of its own
publishes through :func:`publish_run_directory`, which applies the same sequence to
a directory it staged.

Publication never replaces anything at the final path. ``rename(2)`` silently
replaces an empty directory, so a check followed by a rename would leave a window
in which a directory created by another process is lost. The run is therefore
renamed with ``renameat2(..., RENAME_NOREPLACE)``, which the kernel refuses when
the final path exists in any form. That call is Linux only (kernel 3.15, glibc
2.28) and needs a filesystem that implements the flag (ext4, xfs, btrfs, tmpfs
and overlayfs do; NFS and 9p refuse it). Where it is unavailable publication
fails; it never falls back to a rename that could replace a target.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import functools
import hashlib
import io
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, BinaryIO
from uuid import uuid4

DEBUG_DIRECTORY = "__DEBUG__"
"""Nome do diretório de evidência humana de qualquer run artifact.

Todo artifact grava o que é apenas diagnóstico sob este nome, e **nada** contratual
mora aqui: o diretório nunca entra no inventário do manifest, então removê-lo não pode
invalidar um run, e nenhum estágio a jusante pode depender do que está dentro.

O nome é maiúsculo e cercado por sublinhados para que seja impossível confundi-lo com
um diretório contratual (``outputs/``, ``metrics/``) numa listagem, num `rsync` ou num
`.gitignore`. Ele vive aqui, e não em cada capability, porque é a mesma regra para todas:
um writer novo usa esta constante em vez de repetir o literal.
"""


_MANIFEST_FILENAME = "manifest.json"
_README_FILENAME = "README.md"
_TEMPORARY_PREFIX = ".tmp-"

# Constantes de <fcntl.h> e <linux/fs.h>: caminhos relativos ao cwd e rename sem substituir.
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1

# Arquivos grandes (a geometria de um mapa pode ter gigabytes) nunca são lidos por inteiro.
_HASH_CHUNK_BYTES = 1 << 20


class RunDirectoryError(Exception):
    """Raised when a run directory cannot be written or published safely."""


@dataclass(frozen=True, kw_only=True)
class FileEntry:
    """One file in a run's inventory.

    Attributes:
        path: Path relative to the run directory, with ``/`` separators.
        size_bytes: Size of the file in bytes.
        content_hash: ``"sha256:<hex digest>"`` of the content.
    """

    path: str
    size_bytes: int
    content_hash: str


def file_entry(relative_path: str, data: bytes) -> FileEntry:
    """Build the inventory entry of a file's content.

    Args:
        relative_path: Path relative to the run directory.
        data: The file content.

    Returns:
        The entry with the content's size and SHA-256.
    """
    return FileEntry(
        path=relative_path,
        size_bytes=len(data),
        content_hash=f"sha256:{hashlib.sha256(data).hexdigest()}",
    )


def check_file_inventory(root: Path, inventory: Iterable[FileEntry]) -> list[str]:
    """Compare an inventory with what is actually on disk.

    Files are hashed in chunks, so a file larger than memory can be checked.

    Args:
        root: The run directory.
        inventory: The entries the manifest promises.

    Returns:
        Human-readable problems (a path that is not inside the run, missing
        file, size mismatch, hash mismatch); empty means the inventory matches.
        A path outside the run is reported and never opened.
    """
    problems: list[str] = []
    for entry in inventory:
        # Um manifest adulterado nunca faz o leitor abrir um arquivo fora do run.
        if not is_run_relative_path(entry.path):
            problems.append(f"invalid path in manifest: {entry.path!r}")
            continue
        file_path = root / entry.path
        if not file_path.is_file():
            problems.append(f"missing file referenced by manifest: {entry.path}")
            continue
        size_bytes, content_hash = hash_file(file_path)
        if size_bytes != entry.size_bytes:
            problems.append(
                f"size mismatch for {entry.path}: expected {entry.size_bytes}, found {size_bytes}"
            )
            continue
        if content_hash != entry.content_hash:
            problems.append(f"content hash mismatch for {entry.path}")
    return problems


def hash_file(path: Path) -> tuple[int, str]:
    """Measure a file on disk the way an inventory entry records it, in bounded memory.

    The file is read in chunks, so a payload larger than memory can be hashed.

    Args:
        path: The file to read.

    Returns:
        Its size in bytes and its ``"sha256:<hex digest>"`` content hash, the
        ``size_bytes`` and ``content_hash`` of a :class:`FileEntry`.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, f"sha256:{digest.hexdigest()}"


class _HashingRawWriter(io.RawIOBase):
    """Raw writer that hashes and counts exactly the bytes it hands to the file."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = memoryview(data)
        written = self._handle.write(view) or 0
        self.digest.update(view[:written])
        self.size_bytes += written
        return written


class AtomicRunDirectory:
    """Builds a run directory in a temporary sibling and publishes it atomically.

    Use it as a context manager: files are written under a hidden temporary
    directory, and only :meth:`publish` renames it to its final path, after
    verifying the inventory. Leaving the block without publishing, or with an
    exception, removes the temporary directory, so the final path never exists
    half written. Every file and directory reaches stable storage before that
    rename, so this also holds after a power loss, and the rename never replaces
    anything already at the final path (see :meth:`publish`).
    """

    def __init__(self, final_dir: Path, *, refuse_existing_early: bool = True) -> None:
        """Prepare a run directory.

        Args:
            final_dir: Where the finished run will live.
            refuse_existing_early: ``True`` (the default) refuses an existing
                ``final_dir`` here, before anything is written. An owner that
                documents detecting the collision only when it publishes, such as
                the streaming sequence artifact writer, passes ``False``, and
                nothing on disk is read here. :meth:`publish` never replaces an
                existing ``final_dir`` either way.

        Raises:
            RunDirectoryError: If ``final_dir`` already exists and
                ``refuse_existing_early`` is set; a finished run is immutable,
                and rerunning creates a new run instead.
        """
        if refuse_existing_early and final_dir.exists():
            raise RunDirectoryError(f"run directory already exists: {final_dir}")
        self._final_dir = final_dir
        self._tmp_dir = final_dir.parent / f"{_TEMPORARY_PREFIX}{final_dir.name}-{uuid4().hex[:8]}"
        self._entries: dict[str, FileEntry] = {}
        self._written: set[str] = set()
        self._open_streams = 0
        self._failed_write = False
        self._discarded = False
        self._published = False

    def __enter__(self) -> AtomicRunDirectory:
        """Create the temporary directory."""
        self._tmp_dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard the temporary directory unless the run was published."""
        if not self._published:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def discard(self) -> None:
        """Remove the temporary directory now, reporting a removal that fails.

        For an owner that keeps a run open across calls and must know when its
        cleanup did not happen: unlike leaving the ``with`` block, a failure is
        raised, and calling it again retries the removal. The run can no longer
        be written or published. It has no effect on a published run.

        Raises:
            OSError: If the temporary directory could not be removed.
        """
        if self._published:
            return
        self._discarded = True
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self._tmp_dir)

    def write_text(self, relative_path: str, text: str, *, contractual: bool = True) -> None:
        """Write a UTF-8 text file; see :meth:`write_bytes`."""
        self.write_bytes(relative_path, text.encode("utf-8"), contractual=contractual)

    def write_bytes(self, relative_path: str, data: bytes, *, contractual: bool = True) -> None:
        """Write a file, recording it in the inventory when it is contractual.

        The file is forced to stable storage (``fsync``) before the call returns.
        A write that fails removes its partial file and leaves the run
        unpublishable, like a failed stream of :meth:`open_binary`: a caller that
        catches the error and goes on can never publish a truncated file.

        Args:
            relative_path: Path relative to the run directory.
            data: The file content.
            contractual: ``True`` for data a downstream consumer may depend on:
                it enters the inventory, so its loss is detected. ``False`` for
                human-only evidence, which must live under :data:`DEBUG_DIRECTORY`:
                it is written but never inventoried, so removing it cannot
                invalidate the run.

        Raises:
            RunDirectoryError: If the path is not a plain relative path inside
                the run, or the same path was already written.
            OSError: If the file cannot be written or forced to stable storage;
                the partial file is removed and the run can then no longer be
                published.
        """
        self._require_not_discarded()
        _require_relative_path(relative_path)
        _require_debug_placement(relative_path, contractual=contractual)
        if relative_path in self._written:
            raise RunDirectoryError(f"path already written in this run: {relative_path}")
        target = self._tmp_dir / relative_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_synced(target, data)
        except BaseException as error:
            self._discard_failed_write(target, error)
            raise
        self._written.add(relative_path)
        if contractual:
            self._entries[relative_path] = file_entry(relative_path, data)

    @contextmanager
    def open_binary(self, relative_path: str, *, contractual: bool = True) -> Iterator[BinaryIO]:
        """Stream a file that may be larger than memory.

        The bytes are hashed and counted as they are written, so the inventory
        entry costs no second read. Use it like :meth:`write_bytes` for payloads
        that are produced incrementally. The file is forced to stable storage
        (``fsync``) when the stream is closed.

        Args:
            relative_path: Path relative to the run directory.
            contractual: See :meth:`write_bytes`.

        Yields:
            A writable binary stream. It must be closed (leave the ``with``
            block) before :meth:`publish`.

        Raises:
            RunDirectoryError: If the path is not a plain relative path inside
                the run, or the same path was already written.
            OSError: If the file cannot be written or forced to stable storage.
                Like any exception raised in the ``with`` block, it removes the
                partial file and the run can then no longer be published (the
                failure contract of :meth:`write_bytes`).
        """
        self._require_not_discarded()
        _require_relative_path(relative_path)
        _require_debug_placement(relative_path, contractual=contractual)
        if relative_path in self._written:
            raise RunDirectoryError(f"path already written in this run: {relative_path}")
        target = self._tmp_dir / relative_path
        self._written.add(relative_path)
        self._open_streams += 1
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                raw = _HashingRawWriter(handle)
                stream = io.BufferedWriter(raw, buffer_size=_HASH_CHUNK_BYTES)
                try:
                    yield stream
                finally:
                    stream.close()
                _sync_file(handle)
        except BaseException as error:
            self._discard_failed_write(target, error)
            raise
        finally:
            self._open_streams -= 1
        if contractual:
            self._entries[relative_path] = FileEntry(
                path=relative_path,
                size_bytes=raw.size_bytes,
                content_hash=f"sha256:{raw.digest.hexdigest()}",
            )

    def _require_not_discarded(self) -> None:
        """Refuse to write or publish once :meth:`discard` removed the run."""
        if self._discarded:
            raise RunDirectoryError(f"the run was discarded: {self._final_dir}")

    def _discard_failed_write(self, target: Path, error: BaseException) -> None:
        """Make the run unpublishable and remove the partial file of a write that failed.

        ``error`` stays the exception the caller sees: a partial file that cannot
        be removed is only reported in a note attached to it.
        """
        # Um arquivo parcial nunca pode ser publicado nem parecer inventariado.
        self._failed_write = True
        try:
            target.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"the partial file {target} could not be removed: {cleanup_error}")

    def written_path(self, relative_path: str) -> Path:
        """Return where a file written in this run currently lives.

        For reading back what was written before the run is published, for
        example to open a payload and derive human-only evidence from it. Do not
        write through it: files are only added with :meth:`write_bytes` and
        :meth:`open_binary`, which keep the inventory honest.

        Args:
            relative_path: Path relative to the run directory.

        Raises:
            RunDirectoryError: If no such file was written in this run.
        """
        if relative_path not in self._written:
            raise RunDirectoryError(f"path has not been written in this run: {relative_path}")
        return self._tmp_dir / relative_path

    def inventory(self) -> tuple[FileEntry, ...]:
        """Return the inventory recorded so far, sorted by path.

        It is exactly what :meth:`publish` adds to the manifest: every contractual file written
        with :meth:`write_bytes`, and every contractual stream of :meth:`open_binary` once it is
        closed, with the size and hash counted while it was written. An owner whose manifest
        depends on the inventory (for example a content identity computed over the file hashes)
        builds it from here without reading any file again.

        Returns:
            The entries recorded when this call is made; a stream still open is not among them.
        """
        return tuple(sorted(self._entries.values(), key=lambda entry: entry.path))

    def publish(self, *, manifest: Mapping[str, Any], readme: str | None) -> None:
        """Write the manifest, verify the inventory and make the run visible.

        The manifest and README are not part of their own inventory. The manifest
        is ``manifest.json``: the owner's fields plus ``file_inventory``, indented
        and key-sorted. The README is ``README.md``, unless the owner's layout has
        none.

        Publication is durable: the payloads were forced to stable storage when
        written, the manifest and README are forced here, then every directory of
        the run, and only then the run is renamed to its final path, after which
        the parent directory is forced too. After a power loss the final path
        either does not exist or holds the complete run.

        Cost: one ``fsync`` per file of the run (paid by :meth:`write_bytes` and
        :meth:`open_binary` as each file is written), one per directory of the
        run and one for the parent. Each waits for the storage device, so the
        cost grows with the number of files and with the data not yet flushed,
        not with anything the owning capability computes.

        Publication never replaces the final path: if anything exists there when
        the run is renamed, even an empty directory created after the run was
        opened, the rename itself fails and the existing entry is left intact.
        There is no window between a check and the rename (see the module
        docstring for the platform this requires).

        Args:
            manifest: Manifest fields from the owning capability; the
                ``file_inventory`` is added here.
            readme: Human-readable summary of the run, or ``None`` for a layout
                without ``README.md`` (the sequence artifact has none).

        Raises:
            RunDirectoryError: If the run was discarded, a stream is open, a
                write failed, the inventory does not match the files on disk,
                the final path exists when the run is renamed, or the platform
                or filesystem cannot rename without replacing; the run is then
                not published.
            OSError: If a file or directory cannot be forced to stable storage.
                Before the rename the run is not published; when the parent
                directory fails after it, the run is visible but its durability
                is not confirmed.
        """
        self._require_not_discarded()
        if self._open_streams:
            raise RunDirectoryError("a stream is still open; close it before publishing the run")
        if self._failed_write:
            raise RunDirectoryError("a write failed in this run; the run cannot be published")
        inventory = self.inventory()
        record = {
            **manifest,
            "file_inventory": [
                {
                    "path": entry.path,
                    "size_bytes": entry.size_bytes,
                    "content_hash": entry.content_hash,
                }
                for entry in inventory
            ],
        }
        _write_synced(
            self._tmp_dir / _MANIFEST_FILENAME,
            json.dumps(record, indent=2, sort_keys=True).encode("utf-8"),
        )
        if readme is not None:
            _write_synced(self._tmp_dir / _README_FILENAME, readme.encode("utf-8"))

        problems = check_file_inventory(self._tmp_dir, inventory)
        if problems:
            raise RunDirectoryError(f"internal consistency check failed before publish: {problems}")
        # Os dados já estão no disco; faltam os nomes de cada diretório do run.
        _rename_synced(self._tmp_dir, self._final_dir, self._run_directories())
        self._published = True
        _sync_path(self._final_dir.parent)

    def _run_directories(self) -> list[Path]:
        """Return every directory of the run: its root and each parent of a written file."""
        parents = {parent for path in self._written for parent in PurePosixPath(path).parents}
        return sorted({self._tmp_dir, *(self._tmp_dir / parent for parent in parents)})


def publish_run_directory(staging_dir: Path, final_dir: Path) -> None:
    """Publish a run its owner staged with its own writers, durably and never replacing.

    :class:`AtomicRunDirectory` is how a run is built. This is for an owner whose
    files come from writers of its own that take a directory, such as the mask and
    feature stores of a perception run, so they cannot go through
    :meth:`AtomicRunDirectory.write_bytes`: the run still gets the same publication.
    Every file and directory under ``staging_dir`` is forced to stable storage
    (``fsync``), ``staging_dir`` is renamed to ``final_dir`` without ever replacing
    anything there, and the parent directory is forced after the rename. The owner
    has already written its manifest and checked its inventory; nothing is read or
    verified here.

    Cost: one ``fsync`` per file and per directory of the run, all paid here instead
    of as each file is written, and one for the parent.

    Args:
        staging_dir: The complete run, in a temporary sibling of ``final_dir``.
        final_dir: Where the run becomes visible.

    Raises:
        RunDirectoryError: If ``final_dir`` exists when the run is renamed, or the
            platform or filesystem cannot rename without replacing; the run is then
            not published.
        OSError: If a file or directory cannot be listed or forced to stable storage.
            Before the rename the run is not published; when the parent directory
            fails after it, the run is visible but its durability is not confirmed.
    """
    directories: list[Path] = []
    for root, _, filenames in os.walk(staging_dir, onerror=_raise):
        directory = Path(root)
        for filename in filenames:
            _sync_path(directory / filename)
        directories.append(directory)
    _rename_synced(staging_dir, final_dir, directories)
    _sync_path(final_dir.parent)


def is_run_relative_path(relative_path: str) -> bool:
    """Tell whether a path is a plain relative path that stays inside a run directory.

    Args:
        relative_path: Path with ``/`` separators, as a manifest or an index records it.

    Returns:
        ``False`` for an empty path, an absolute path, a path with a ``..`` part or ``.``
        alone: joining any of them to the run directory could leave it or name it.
    """
    path = PurePosixPath(relative_path)
    return bool(relative_path) and not (
        path.is_absolute() or ".." in path.parts or path.parts == (".",)
    )


def _rename_synced(staging_dir: Path, final_dir: Path, directories: Iterable[Path]) -> None:
    """Rename a run whose files are already on disk to ``final_dir``, never replacing.

    The names in each directory of the run reach the disk before the rename, or the
    rename could survive a power loss without the files. The caller syncs the
    parent directory right after, so the rename itself reaches the disk.
    """
    for directory in directories:
        _sync_path(directory)
    _rename_no_replace(staging_dir, final_dir)


def _raise(error: OSError) -> None:
    """Make ``os.walk`` fail on a directory it cannot list instead of skipping it."""
    raise error


def _rename_no_replace(source: Path, target: Path) -> None:
    """Rename ``source`` to ``target`` atomically, failing if ``target`` exists in any form.

    Raises:
        RunDirectoryError: If ``target`` exists, or the platform or the filesystem
            cannot rename without replacing (nothing is renamed then).
        OSError: For any other failure of the rename.
    """
    renameat2 = _renameat2() if sys.platform.startswith("linux") else None
    if renameat2 is None:
        raise RunDirectoryError(
            f"cannot publish {target} without risking replacing it: renameat2(RENAME_NOREPLACE) "
            "is only available on Linux with glibc 2.28 or newer"
        )
    if renameat2(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(target), _RENAME_NOREPLACE):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RunDirectoryError(f"run directory already exists: {target}")
        if code in (errno.EINVAL, errno.ENOSYS):
            raise RunDirectoryError(
                f"cannot publish {target}: the kernel or the filesystem refused "
                "renameat2(RENAME_NOREPLACE); publish on a local filesystem that supports it, "
                "such as ext4, xfs, btrfs or tmpfs"
            )
        raise OSError(code, os.strerror(code), os.fspath(source), None, os.fspath(target))


@functools.cache
def _renameat2() -> Callable[[int, bytes, int, bytes, int], int] | None:
    """Return glibc's ``renameat2``, or ``None`` when the C library does not provide it."""
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError:
        return None
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    return function


def _write_synced(path: Path, data: bytes) -> None:
    """Write ``data`` to a new file and force it to stable storage before returning."""
    with path.open("wb") as handle:
        handle.write(data)
        _sync_file(handle)


def _sync_file(handle: BinaryIO) -> None:
    """Flush a file's buffers and force its content to stable storage."""
    handle.flush()
    os.fsync(handle.fileno())


def _sync_path(path: Path) -> None:
    """Force a file's content, or a directory's entries, to stable storage.

    Raises:
        OSError: If the path cannot be opened or synced; opening a directory
            read-only to sync it is POSIX and fails on platforms that forbid it.
    """
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_debug_placement(relative_path: str, *, contractual: bool) -> None:
    """Require that diagnostic evidence lives under the debug directory, and nothing else does.

    Sem esta checagem a regra seria só convenção: um writer novo poderia gravar diagnóstico
    como contratual (e então perdê-lo invalidaria o run) ou gravar dado contratual sob
    ``__DEBUG__/`` (e então apagar o diretório quebraria um consumidor a jusante).

    Args:
        relative_path: Path relative to the run directory.
        contractual: Whether the caller declared the file contractual.

    Raises:
        RunDirectoryError: If a contractual file is placed under the debug directory, or a
            non-contractual file is placed outside it.
    """
    inside = relative_path == DEBUG_DIRECTORY or relative_path.startswith(f"{DEBUG_DIRECTORY}/")
    if contractual and inside:
        raise RunDirectoryError(
            f"contractual file must not live under {DEBUG_DIRECTORY}/, which is never "
            f"inventoried and may be removed: {relative_path}"
        )
    if not contractual and not inside:
        raise RunDirectoryError(
            f"non-contractual file must live under {DEBUG_DIRECTORY}/, so that removing "
            f"diagnostics cannot invalidate the run: {relative_path}"
        )


def _require_relative_path(relative_path: str) -> None:
    if not is_run_relative_path(relative_path):
        raise RunDirectoryError(
            f"expected a relative path inside the run directory, got {relative_path!r}"
        )
