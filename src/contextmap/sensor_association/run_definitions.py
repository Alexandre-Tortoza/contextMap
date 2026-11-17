"""Versões de definições que identificam um run de Sensor Association.

Centraliza o vínculo entre versões dos donos, fingerprint e manifest.
A grafia persistida das chaves é preservada para manter a identidade da evidência.
"""

from __future__ import annotations

from collections.abc import Mapping

from contextmap.sensor_association.dense_sampling import SAMPLING_POLICY_ID
from contextmap.sensor_association.diagnostics import DIAGNOSTICS_DEFINITIONS_VERSION
from contextmap.sensor_association.membership import COVERAGE_DEFINITIONS_VERSION
from contextmap.sensor_association.quality import QUALITY_DEFINITIONS_VERSION

DEFINITION_VERSIONS: Mapping[str, str] = {
    "coverage": COVERAGE_DEFINITIONS_VERSION,
    "quality": QUALITY_DEFINITIONS_VERSION,
    "diagnostics": DIAGNOSTICS_DEFINITIONS_VERSION,
    "dense_sampling": SAMPLING_POLICY_ID,
}
"""Versão de cada conjunto de definições, com os nomes que o manifest do run grava."""

# NOTE: O fingerprint escreve três dessas chaves com outro nome. A grafia é arbitrária, mas é
# hasheada dentro da proveniência de observações já persistidas, então mudá-la invalidaria a
# comparação com todo run existente. O mapeamento fica explícito aqui em vez de ser derivado
# por sufixo, que esconderia a exceção de `dense_sampling`.
_FINGERPRINT_KEYS: Mapping[str, str] = {
    "coverage": "coverage_definitions",
    "quality": "quality_definitions",
    "diagnostics": "diagnostics_definitions",
    "dense_sampling": "dense_sampling",
}


def fingerprint_definitions() -> dict[str, str]:
    """Return the definition versions under the key names the fingerprint hashes.

    Returns:
        One entry per definition, named as the configuration fingerprint has always named it.

    Raises:
        KeyError: If a definition was added to :data:`DEFINITION_VERSIONS` without deciding how
            the fingerprint names it. The fingerprint is persisted provenance, so the name is a
            decision, never a default.
    """
    return {_FINGERPRINT_KEYS[name]: version for name, version in DEFINITION_VERSIONS.items()}
