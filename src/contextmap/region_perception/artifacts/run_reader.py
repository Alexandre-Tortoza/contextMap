"""Leitura de uma execução de region_perception já publicada.

Dá acesso tipado ao manifest, às regiões, às máscaras e à auditoria de descoberta de um run
existente, sem recomputar nada: a descoberta já aconteceu, e aqui ela é apenas lida. As
regiões voltam como `Region2D`, não como `dict`, para que uma seleção de upstream errada
falhe no contrato e não adiante um registro truncado.

A seleção errada falha logo na construção: um diretório que não existe, ou que existe sem
`manifest.json`, não é um artifact, e tratá-lo como um run vazio esconderia o erro a
jusante (AGENTS.md §26). Pela mesma razão, um run sem o stream de auditoria e uma máscara
que a região reivindica mas o store não tem são erros nomeados, nunca `None` silencioso.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmap.region_perception.artifacts.mask_store import (
    MaskStoreError,
    MaskStoreReader,
)
from contextmap.region_perception.artifacts.run_manifest import (
    RegionPerceptionRunError,
    RegionPerceptionRunManifest,
)
from contextmap.region_perception.artifacts.run_writer import (
    FRAMES_FILENAME,
    GROUNDING_FILENAME,
    MASK_REFERENCE_PREFIX,
    MASKS_DIRNAME,
    REFINEMENT_FILENAME,
    REGION_DISCOVERY_AUDIT_FILENAME,
    REGIONS_FILENAME,
)

if TYPE_CHECKING:
    from contextmap.ingestion import SourceObservationId
    from contextmap.region_perception.artifacts.frame import RegionPerceptionFrame
    from contextmap.region_perception.artifacts.mask_store import RegionMaskResolver
    from contextmap.region_perception.discovery.region_discovery import RegionDiscoveryAudit
    from contextmap.region_perception.grounding.region_grounding import RegionGroundingExecution
    from contextmap.region_perception.refinement.region_refinement import RegionRefinementExecution
    from contextmap.region_perception.region_evidence import Region2D
    from contextmap.region_perception.region_geometry import InlineMask


class _PublishedMaskResolver:
    """Resolves a published run's ``mask_reference`` values against its own mask store.

    The mapping comes from the published index, never from parsing the reference back into
    a key: an observation id with a separator in it would otherwise decide which mask is
    loaded. Nothing is cached, so the pixels live only as long as the caller holds them.
    """

    def __init__(self, store: MaskStoreReader) -> None:
        """Index this run's masks by the reference the regions of the run carry."""
        self._store = store
        self._keys = {
            f"{MASK_REFERENCE_PREFIX}{store.entry(observation_id, region_id).payload_reference}": (
                observation_id,
                region_id,
            )
            for observation_id, region_id in store.region_keys()
        }

    def resolve_mask(self, reference: str) -> InlineMask:
        """Load and hash-verify the mask one published reference names.

        Raises:
            MaskStoreError: If this run's mask index holds no such reference.
            MaskPayloadIntegrityError: If the payload fails its hash check.
        """
        try:
            observation_id, region_id = self._keys[reference]
        except KeyError:
            raise MaskStoreError(
                f"mask_reference {reference!r} names no mask of this region perception run; "
                f"the run's {MASKS_DIRNAME}/ index holds {len(self._keys)} masks"
            ) from None
        return self._store.load(observation_id, region_id)


class RegionPerceptionRunReader:
    """Read region perception run from artifact directory."""

    def __init__(self, artifact_dir: str | Path) -> None:
        """Bind the reader to an existing, published run directory.

        Args:
            artifact_dir: Directory of the run to read, holding ``manifest.json`` and
                ``outputs/``.

        Raises:
            RegionPerceptionRunError: If the directory does not exist, holds no manifest,
                or the manifest cannot be decoded — so a wrong upstream selection fails
                here instead of returning an empty run.
        """
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.exists():
            raise RegionPerceptionRunError(f"artifact directory not found: {artifact_dir}")

        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise RegionPerceptionRunError(
                f"manifest not found in artifact: {manifest_path}; a directory is not a run"
            )

        try:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RegionPerceptionRunError(f"cannot read manifest: {error}") from error

        self._manifest = RegionPerceptionRunManifest.from_dict(record)
        self._integrity_issues: list[str] | None = None

    @property
    def manifest(self) -> RegionPerceptionRunManifest:
        """The manifest of this run, decoded when the reader was built."""
        return self._manifest

    def read_manifest(self) -> RegionPerceptionRunManifest:
        """Return the run manifest.

        Returns:
            The typed manifest, including the inventory the publication recorded.
        """
        return self._manifest

    def verify_integrity(self) -> list[str]:
        """Compare the published files with the hashes the manifest promises.

        Returns:
            Human-readable problems (missing file, size or hash mismatch); empty means
            the run on disk is exactly the one that was published. The result is cached,
            since hashing the outputs costs a full read.
        """
        if self._integrity_issues is None:
            from contextmap.shared import check_file_inventory

            self._integrity_issues = check_file_inventory(
                self.artifact_dir, self._manifest.file_inventory
            )
        return self._integrity_issues

    def iter_frames(self) -> Iterator[RegionPerceptionFrame]:
        """Yield one frame at a time, resolving only its audited spatial constraints."""
        path = self.artifact_dir / FRAMES_FILENAME
        if not path.exists():
            raise RegionPerceptionRunError(f"regional frame stream missing: {path}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield self._decode_frame(json.loads(line))

    def frame(self, source_observation_id: SourceObservationId) -> RegionPerceptionFrame:
        """Load one selected frame without hydrating masks belonging to other frames."""
        if str(source_observation_id) not in self.manifest.source_observation_ids:
            raise RegionPerceptionRunError(f"unselected frame: {source_observation_id}")
        path = self.artifact_dir / FRAMES_FILENAME
        if not path.exists():
            raise RegionPerceptionRunError(f"regional frame stream missing: {path}")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["image"]["source_observation_id"] == source_observation_id:
                    return self._decode_frame(record)
        raise RegionPerceptionRunError(f"selected frame missing: {source_observation_id}")

    def read_frames(self) -> Sequence[RegionPerceptionFrame]:
        """Read the full frame stream; streaming consumers should use ``iter_frames``."""
        return tuple(self.iter_frames())

    def _decode_frame(self, record: dict[str, Any]) -> RegionPerceptionFrame:
        """Hydrate preparation constraints for exactly one persisted frame record."""
        from contextmap.region_perception.artifacts.frame import RegionPerceptionFrame
        from contextmap.region_perception.grounding.region_grounding import (
            decode_prepared_image_reference,
        )

        image = record["image"]
        constraints = (
            [image["valid_region"]] if image["valid_region"] is not None else []
        ) + image["exclusion_regions"]
        resolver = self.mask_resolver()
        for constraint in constraints:
            constraint["mask"] = resolver.resolve_mask(constraint.pop("mask_reference")).to_dict()
        return RegionPerceptionFrame(
            image=decode_prepared_image_reference(image),
            outcome=record["outcome"],
            failure_detail=record["failure_detail"],
        )

    def read_grounding(self) -> tuple[RegionGroundingExecution, ...]:
        """Read grounding execution evidence, verifying the separately persisted raw answers."""
        from contextmap.region_perception.grounding.region_grounding import (
            decode_region_grounding_execution,
        )

        path = self.artifact_dir / GROUNDING_FILENAME
        if not path.exists():
            raise RegionPerceptionRunError(f"grounding stream missing: {path}")
        executions = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                record = json.loads(line)
                response_path = (self.artifact_dir / record["raw_response_reference"]).resolve()
                if not response_path.is_relative_to(self.artifact_dir.resolve()):
                    raise ValueError("grounding response reference escapes artifact")
                executions.append(
                    decode_region_grounding_execution(
                        record, raw_response=response_path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise RegionPerceptionRunError(
                    f"{path}:{number} is not a decodable grounding: {error}"
                ) from error
        return tuple(executions)

    def read_refinement(self) -> tuple[RegionRefinementExecution, ...]:
        """Read refinement prompts, accepted lazy regions, rejection reasons and native scores."""
        from contextmap.region_perception.refinement.region_refinement import (
            decode_region_refinement_execution,
        )

        path = self.artifact_dir / REFINEMENT_FILENAME
        if not path.exists():
            raise RegionPerceptionRunError(f"refinement stream missing: {path}")
        executions = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                executions.append(decode_region_refinement_execution(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise RegionPerceptionRunError(
                    f"{path}:{number} is not a decodable refinement: {error}"
                ) from error
        return tuple(executions)

    def read_regions(self) -> Sequence[Region2D]:
        """Read every published region as a typed contract.

        A decoded region's ``mask`` is always ``None``: pixel data is never inlined in the
        record (#378). A caller that needs the pixels loads them through the mask store,
        hash-verified, from ``mask_reference``.

        Returns:
            The regions in publication order; empty when the run discovered none, which
            is a valid run and distinct from a run that was never published.

        Raises:
            RegionPerceptionRunError: If a published line is not a decodable region.
        """
        from contextmap.region_perception.artifacts.region_serialization import decode_region

        regions_path = self.artifact_dir / REGIONS_FILENAME
        if not regions_path.exists():
            return []

        regions: list[Region2D] = []
        with regions_path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    regions.append(decode_region(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise RegionPerceptionRunError(
                        f"{regions_path}:{number} is not a decodable region: {error}"
                    ) from error

        return regions

    def read_region_discovery_audits(self) -> Sequence[RegionDiscoveryAudit]:
        """Read how every audited frame's canonical regions were decided, in publication order.

        Returns:
            One audit per frame the run discovered regions in. Empty means the run audited no
            frame, which is distinct from a run that records no audit at all.

        Raises:
            RegionPerceptionRunError: If the run publishes no audit stream — a consumer that
                replays these regions cannot invent the evidence behind them — or a published
                record is not a decodable audit.
        """
        from contextmap.region_perception.artifacts.region_serialization import (
            decode_discovery_audit,
        )

        audit_path = self.artifact_dir / REGION_DISCOVERY_AUDIT_FILENAME
        if not audit_path.exists():
            raise RegionPerceptionRunError(
                f"region perception run {self._manifest.run_id!r} publishes no "
                f"{REGION_DISCOVERY_AUDIT_FILENAME}; the evidence behind its regions is not in "
                "the artifact and cannot be reconstructed from it"
            )

        audits: list[RegionDiscoveryAudit] = []
        with audit_path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    audits.append(
                        decode_discovery_audit(json.loads(line), run_id=self.manifest.run_id)
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise RegionPerceptionRunError(
                        f"{audit_path}:{number} is not a decodable region discovery audit: {error}"
                    ) from error

        return audits

    def mask_store(self) -> MaskStoreReader:
        """Return keyed mask access for consumers that cite physical frame and region identity."""
        return MaskStoreReader.open(self.artifact_dir / MASKS_DIRNAME)

    def mask_resolver(self) -> RegionMaskResolver:
        """Return where this run's ``mask_reference`` values are turned back into pixels.

        Returns:
            A resolver over the published mask store, hash-verifying every payload it loads
            and refusing a reference the run's index does not hold.
        """
        return _PublishedMaskResolver(MaskStoreReader.open(self.artifact_dir / MASKS_DIRNAME))
