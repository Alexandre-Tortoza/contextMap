"""Identidade da regra de junção, separada da regra e da persistência.

A política responde "o que esta derivação computou", não "sobre qual grade". A grade varia
entre execuções e viaja no seu próprio fingerprint, emprestado da agregação; o que esta
política versiona é a atribuição em si -- uma unidade de suporte por ponto bruto da entidade
que cai no voxel -- e a semântica do peso.

Ela mora fora de ``service`` porque a regra e o run persistido precisam nomeá-la, e
``run_artifact`` não pode depender da regra. Ela não calcula nada e não lê configuração.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

ENTITY_VOXEL_SUPPORT_POLICY_ID = "entity-voxel-support"
"""Identity of the join rule, independent of the grid it runs on."""

ENTITY_VOXEL_SUPPORT_POLICY_VERSION = "0.1.0"
"""Version of what the join computes; a change of semantics requires a new version."""

_ATTRIBUTION = "aggregation-membership"
_WEIGHT_SEMANTICS = "measurement-count"


@dataclass(frozen=True, kw_only=True)
class EntityVoxelSupportPolicy:
    """Attribute one unit of support per raw point of an entity that falls in a voxel.

    This version has no parameter: the voxel a point belongs to is decided by the
    aggregation's own membership, never by re-binning, so the only thing that varies
    between executions -- the grid -- belongs to the aggregation and travels in its
    fingerprint. A change to what the join computes is a change of
    :data:`ENTITY_VOXEL_SUPPORT_POLICY_VERSION`, never a silent change of behaviour.
    """

    @property
    def rule(self) -> str:
        """The rule a derived support declares."""
        return ENTITY_VOXEL_SUPPORT_POLICY_ID

    def to_record(self) -> dict[str, Any]:
        """Describe everything that changes the derived support.

        Returns:
            The policy document a run persists, including the weight semantics: a count of
            measurements, never a probability, and never mixed with a score.
        """
        return {
            "policy_id": ENTITY_VOXEL_SUPPORT_POLICY_ID,
            "policy_version": ENTITY_VOXEL_SUPPORT_POLICY_VERSION,
            "attribution": _ATTRIBUTION,
            "weight_semantics": _WEIGHT_SEMANTICS,
            "rule": self.rule,
        }

    def fingerprint(self) -> str:
        """Hash the policy: equal for equal semantics, different when any of them changes.

        Returns:
            ``"sha256:<hex>"`` of :meth:`to_record`.
        """
        payload = json.dumps(self.to_record(), sort_keys=True)
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
