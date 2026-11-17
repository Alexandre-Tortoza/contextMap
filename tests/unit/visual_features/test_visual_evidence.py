"""Invariantes de identidade de uma feature visual.

A decomposição reescreveu ``visual_evidence`` e, no caminho, a validação que recusava
``feature_id`` vazio desapareceu: a auditoria do histórico mostrou a contagem de ``raise``
do módulo caindo de 9 para 6 em ``6a6ca6e4``, e duas das três mensagens perdidas apenas
mudaram de texto. Esta ficou sem rede.

Uma feature sem identidade entra no artifact indistinguível de outra, e o índice de
payloads é chaveado por ela — e ``Region2D`` já recusa o equivalente, então aceitá-la aqui
deixava as duas capabilities incoerentes sobre a mesma regra.
"""

from __future__ import annotations

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.shared import BackendProvenance
from contextmap.visual_perception.visual_features import FeatureScope, VisualFeature

_PROVENANCE = BackendProvenance(
    backend_id="dinov2",
    capability="feature_extractor",
    provider="meta",
    model="dinov2-base",
    version="1",
    configuration_fingerprint=None,
)


def _feature(feature_id: str) -> VisualFeature:
    """Build a global-scope feature whose only varying field is its identity."""
    return VisualFeature(
        source_observation_id=SourceObservationId("frame-1"),
        feature_id=feature_id,  # type: ignore[arg-type]
        scope=FeatureScope.GLOBAL,
        embedding_space_id="space-0",
        shape=(2,),
        dtype="float32",
        payload_reference="features/f.npy",
        provenance=_PROVENANCE,
    )


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
def test_a_feature_without_an_identity_is_refused(blank: str) -> None:
    """An empty or blank feature_id must not reach the artifact."""
    with pytest.raises(ValueError, match="feature_id must not be empty"):
        _feature(blank)


def test_a_feature_with_an_identity_is_accepted() -> None:
    """The guard rejects only a missing identity, not a valid one."""
    assert str(_feature("feature-0001").feature_id) == "feature-0001"
