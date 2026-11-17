"""Leitura de um run finalizado: manifest, suporte reconstruído e integridade.

O reader abre um diretório publicado sem NumPy e sem o mapa bruto. Reconstruir o suporte
passa pelos construtores de domínio, então uma linha adulterada falha ao decodificar em vez
de virar uma célula inválida, e ``verify_integrity`` confronta o inventário de arquivos com
o que está em disco antes de olhar o conteúdo.

Ele nunca escreve, nunca repara um run e não conhece a regra de junção.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from contextmap.geometric_mapping import MapId
from contextmap.ingestion import FrameId
from contextmap.shared import check_file_inventory
from contextmap.spatial_support.models import SemanticSpatialSupport
from contextmap.spatial_support.run_artifact.layout import (
    CELLS,
    CONFIG,
    METRICS,
    SUPPORT,
    SpatialSupportArtifactError,
    cell_of,
    read_json_document,
)
from contextmap.spatial_support.run_artifact.manifest import (
    SpatialSupportManifest,
    load_manifest,
)


class SpatialSupportArtifactReader:
    """Reads a finalized semantic spatial support from its own directory."""

    def __init__(self, run_dir: Path) -> None:
        """Open a derivation.

        Raises:
            SpatialSupportArtifactError: If the manifest is missing or its schema
                version is not understood.
        """
        self._root = run_dir
        self._manifest = load_manifest(run_dir)

    @property
    def manifest(self) -> SpatialSupportManifest:
        """The derivation's manifest."""
        return self._manifest

    def support(self) -> SemanticSpatialSupport:
        """Reconstruct the persisted support, revalidated by its own decoding.

        Raises:
            SpatialSupportArtifactError: If a payload is missing or inconsistent.
        """
        header = self._read_record(SUPPORT)
        frame_id = FrameId(header["frame_id"])
        cells = tuple(cell_of(json.loads(line), frame_id) for line in self._cell_lines())
        try:
            return SemanticSpatialSupport(
                source_map_id=MapId(header["source_map_id"]),
                frame_id=frame_id,
                grid_fingerprint=header["grid_fingerprint"],
                aggregation_rule=header["aggregation_rule"],
                cells=cells,
            )
        except ValueError as error:
            raise SpatialSupportArtifactError(
                f"the persisted support is inconsistent: {error}"
            ) from error

    def verify_integrity(self) -> list[str]:
        """Check the file inventory and that the support reconstructs.

        Returns:
            Human-readable problems; empty means the derivation is intact.
        """
        problems = check_file_inventory(self._root, self._manifest.file_inventory)
        if problems:
            return problems
        try:
            support = self.support()
        except SpatialSupportArtifactError as error:
            return [str(error)]
        header = self._read_record(SUPPORT)
        config = self._read_record(CONFIG)
        metrics = self._read_record(METRICS)
        expected = {
            "source_map_id": str(support.source_map_id),
            "frame_id": str(support.frame_id),
            "grid_fingerprint": support.grid_fingerprint,
            "aggregation_rule": support.aggregation_rule,
            "cell_count": len(support.cells),
            "entity_count": len(
                {item.entity_id for cell in support.cells for item in cell.entity_support}
            ),
            "total_support_weight": sum(
                item.support_weight for cell in support.cells for item in cell.entity_support
            ),
        }
        for field, value in expected.items():
            if getattr(self._manifest, field) != value:
                problems.append(f"manifest {field} differs from the persisted support")
            for name, record in (("support", header), ("config", config), ("metrics", metrics)):
                if field in record and record[field] != value:
                    problems.append(f"{name} {field} differs from the persisted support")
        if config["policy_fingerprint"] != self._manifest.policy_fingerprint:
            problems.append("manifest policy_fingerprint differs from config")
        if config["schema_version"] != self._manifest.schema_version:
            problems.append("manifest schema_version differs from config")
        return problems

    def _cell_lines(self) -> list[str]:
        path = self._root / CELLS
        if not path.is_file():
            raise SpatialSupportArtifactError(f"missing {CELLS} in {self._root}")
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line]

    def _read_record(self, relative_path: str) -> dict[str, Any]:
        path = self._root / relative_path
        if not path.is_file():
            raise SpatialSupportArtifactError(f"missing {relative_path} in {self._root}")
        return read_json_document(path)
