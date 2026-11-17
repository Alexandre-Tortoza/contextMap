"""Os digests de identidade de Semantic Mapping: configuração efetiva e conjunto de suporte.

Toda identidade persistida desta capability — o fingerprint de uma política, o digest do
artifact de fusão de origem, o digest do conjunto de geometria de um resumo — é um SHA-256
sobre uma forma canônica. A forma canônica é a regra que realmente importa: sem ordenação de
chaves e sem separadores fixos, a mesma configuração produziria fingerprints diferentes entre
execuções, e `EntityProvenance.configuration_fingerprint` deixaria de identificar uma
configuração.

Este módulo é folha: ele não conhece nenhum contrato da capability, não decide política e não
escolhe limiar. Quem monta o registro a ser digerido é a política dona dele.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from contextmap.geometric_mapping import GeometryReference


def canonical_digest(record: Mapping[str, object]) -> str:
    """Digest a record under the canonical JSON form used by every identity of this capability.

    Args:
        record: The record to digest, holding JSON primitives only.

    Returns:
        ``sha256:`` followed by the hex digest of the canonical JSON of ``record``.
    """
    # NOTE: `sort_keys` e separadores sem espaço são o contrato; mudá-los mudaria todo
    # fingerprint já gravado em provenance.
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def geometry_set_digest(references: Sequence[GeometryReference]) -> str:
    """Digest the exact set of geometry identities of a support.

    Args:
        references: The support, sorted by ``geometry_id``.

    Returns:
        ``sha256:`` followed by the digest of the identities, so a summary can be tied to the
        exact set of geometry it was computed from.
    """
    # NOTE: as identidades são unidas por newline, não por JSON: o digest é de um conjunto de
    # ids, não de um registro, e a forma não pode mudar sem invalidar resumos já gravados.
    joined = "\n".join(reference.geometry_id for reference in references)
    return f"sha256:{hashlib.sha256(joined.encode()).hexdigest()}"
