"""Referência a um payload persistido e o escalar JSON que descreve metadados.

Os dois contratos aqui atravessam capabilities com significado idêntico e não têm
dono entre elas.

``ArtifactReference`` é a forma única de um contrato público apontar para um payload
persistido sem embuti-lo: ``docs/ARTIFACTS.md`` fixa a mesma regra para todo artifact
("payloads grandes podem ser referenciados de forma tardia, com metadados e hash"), e
quem a usa hoje nomeia payloads de naturezas diferentes — a imagem preparada e cada
transformação aplicada em visual_perception, a máscara de um candidato em
region_perception, os arquivos que o composition root referencia. É o par de contrato
de ``shared.run_directory.FileEntry``, que já mora aqui e mede um arquivo do inventário
de um run com o mesmo tamanho-e-hash.

``JsonScalar`` é o vocabulário de um valor escalar JSON, usado onde um contrato preserva
metadados nativos inspecionáveis (``RegionCandidate.native_metadata``,
``TransformationRecord.parameters``, os parâmetros de política e de prompt de
semantic_interpretation). Não é conceito de região nem de percepção: é a fronteira
serializável, e a mesma em toda capability que a declara.

Nenhum dos dois entrou aqui para quebrar ciclo: ambos moravam em
``visual_perception.region_models``, onde não eram contrato de região nem de
visual_perception, e era essa a inversão de ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Reference a persisted payload without embedding it in a public contract.

    Attributes:
        uri: Location of the payload, relative to the run that owns it.
        sha256: Content digest, as 64 lowercase hexadecimal characters.
        media_type: Media type of the payload bytes.
    """

    uri: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        """Validate the content-addressed reference."""
        if not self.uri:
            raise ValueError("artifact uri must not be empty")
        invalid_character = any(character not in "0123456789abcdef" for character in self.sha256)
        if len(self.sha256) != 64 or invalid_character:
            raise ValueError("artifact sha256 must contain 64 lowercase hexadecimal characters")
        if not self.media_type:
            raise ValueError("artifact media_type must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {"uri": self.uri, "sha256": self.sha256, "media_type": self.media_type}

    @classmethod
    def from_dict(cls, value: object) -> ArtifactReference:
        """Restore an artifact reference from serialized data.

        Args:
            value: The serialized mapping.

        Returns:
            The restored reference.

        Raises:
            TypeError: If the value is not an object with string keys, or a field has
                the wrong type.
            ValueError: If a field is empty or the digest is malformed.
        """
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError("artifact reference must be an object with string keys")
        fields = {}
        for key in ("uri", "sha256", "media_type"):
            field_value = value.get(key)
            if not isinstance(field_value, str):
                raise TypeError(f"{key} must be a string")
            fields[key] = field_value
        return cls(uri=fields["uri"], sha256=fields["sha256"], media_type=fields["media_type"])
