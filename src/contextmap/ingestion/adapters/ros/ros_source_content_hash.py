"""Hash incremental do conteúdo ROS efetivamente lido pelo adapter.

O acumulador incorpora tópico, timestamp de gravação e bytes crus de cada mensagem durante a
leitura, evitando uma segunda passada quando há janela. Assim, a identidade cobre exatamente o
trecho processado sem transformar hashing em custo proporcional à fonte inteira.
"""

from __future__ import annotations

import hashlib


class StreamingContentHash:
    """Accumulates a content hash over exactly the raw messages an adapter reads.

    ``sequence_provenance.compute_source_content_hash`` re-reads a whole
    file/directory in its own separate pass — appropriate once, for a full
    ingestion, but it would cost O(source size) even when a
    :class:`~contextmap.ingestion.recorded_source_adapter.SourceWindow` limits
    reading to a small fraction of the source. This accumulates a hash
    incrementally, one message at a time, as an adapter's own
    ``read_observations()`` already reads each message — at no extra I/O —
    so the resulting hash covers exactly what was read: the whole source
    when no window is configured, or only the window otherwise.
    """

    def __init__(self) -> None:
        """Create an empty accumulator (:meth:`hexdigest` returns ``None`` until updated)."""
        self._digest = hashlib.sha256()
        self._touched = False

    def update(self, *, topic: str, timestamp_nanoseconds: int, rawdata: bytes) -> None:
        """Fold one message's identity and raw bytes into the running hash.

        Args:
            topic: The message's source topic.
            timestamp_nanoseconds: The message's recording timestamp.
            rawdata: The message's raw, still-serialized bytes.
        """
        self._touched = True
        self._digest.update(topic.encode("utf-8"))
        self._digest.update(timestamp_nanoseconds.to_bytes(8, "big", signed=True))
        self._digest.update(rawdata)

    def hexdigest(self) -> str | None:
        """Return the accumulated hash, or ``None`` if :meth:`update` was never called.

        Returns:
            ``"sha256:<hex digest>"``, or ``None``.
        """
        return f"sha256:{self._digest.hexdigest()}" if self._touched else None
