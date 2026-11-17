"""Publicação de uma execução de region_perception.

As regiões entram tipadas — `Region2D`, não `dict` — e só alcançam o disco no `finalize`,
que grava tudo em um diretório temporário e o renomeia atomicamente
(:class:`~contextmap.shared.AtomicRunDirectory`). Assim o caminho final nunca existe pela
metade: ou não existe, ou contém um run completo, com hash por arquivo no inventário do
manifest (AGENTS.md §8).

As máscaras são a exceção ao "só no finalize": elas saem da RAM na fronteira do produtor
(`externalize_region_mask`), para um store de staging, e a região segue carregando apenas
`mask_reference`. Um frame 640x480 custa ~38 KB empacotado, mas um run de 360 frames com
dezenas de regiões por frame não cabe em memória até o fim. A auditoria de descoberta, ao
contrário, é metadado leve e fica retida até o `finalize`.

Um run já publicado é imutável: reexecutar cria outra identidade, nunca sobrescreve a
anterior, e o `finalize` recusa um diretório final que já exista.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from contextmap.region_perception.artifacts.frame import RegionPerceptionFrame
from contextmap.region_perception.artifacts.mask_store import (
    MASK_INDEX_FILENAME,
    MaskStoreError,
    MaskStoreWriter,
    write_mask_index,
)
from contextmap.region_perception.artifacts.run_manifest import (
    REGION_PERCEPTION_SCHEMA_VERSION,
    RegionPerceptionRunError,
    RegionPerceptionRunManifest,
)

if TYPE_CHECKING:
    from contextmap.ingestion import SourceObservationId
    from contextmap.region_perception.discovery.region_discovery import RegionDiscoveryAudit
    from contextmap.region_perception.grounding.region_grounding import RegionGroundingExecution
    from contextmap.region_perception.refinement.region_refinement import RegionRefinementExecution
    from contextmap.region_perception.region_evidence import Region2D
    from contextmap.region_perception.region_geometry import InlineMask
    from contextmap.shared import AtomicRunDirectory
    from contextmap.visual_perception import PreparedImage

GROUNDING_FILENAME = "outputs/grounding.jsonl"
REFINEMENT_FILENAME = "outputs/refinement.jsonl"

FRAMES_FILENAME = "outputs/frames.jsonl"

REGIONS_FILENAME = "outputs/regions.jsonl"
"""Contractual stream of the regions a run published, in publication order."""

REGION_DISCOVERY_AUDIT_FILENAME = "outputs/region-discovery-audit.jsonl"
"""Contractual stream of one discovery audit per frame, in processing order."""

MASKS_DIRNAME = "outputs/masks"
"""Root of the run's mask store, inside the published artifact."""

MASK_REFERENCE_PREFIX = f"{MASKS_DIRNAME}/"
"""Prefix every ``Region2D.mask_reference`` of a published run carries."""


class RegionPerceptionRunWriter:
    """Write region perception run (regions, masks, audits, lineage) to an artifact directory.

    It may be used as a context manager, and an executor should: leaving the block without
    :meth:`finalize` discards the staged mask payloads, so an interrupted run leaves nothing
    behind that a later run would trip over.
    """

    SCHEMA_VERSION = REGION_PERCEPTION_SCHEMA_VERSION
    """Schema version this writer publishes."""

    def __init__(self, output_dir: str | Path) -> None:
        """Bind the writer to the path the finished run will live at.

        Nothing is created here: the final directory only comes into existence when
        :meth:`finalize` publishes it, so an interrupted run leaves no trace that a
        consumer could mistake for a finished artifact.

        Args:
            output_dir: Where the published run will live. It must not exist yet when
                :meth:`finalize` runs.
        """
        self.output_dir = Path(output_dir)
        self.regions: list[Region2D] = []
        self.manifest: RegionPerceptionRunManifest | None = None
        self._groundings: dict[tuple[str, str], RegionGroundingExecution] = {}
        self._refinements: dict[tuple[str, str], RegionRefinementExecution] = {}
        self._frames: dict[str, dict[str, Any]] = {}
        self._audits: dict[str, RegionDiscoveryAudit] = {}
        self._mask_store: MaskStoreWriter | None = None
        self._mask_staging = (
            self.output_dir.parent / f".tmp-{self.output_dir.name}-masks-{uuid4().hex[:8]}"
        )
        self._finalized = False
        self._externalized_masks: dict[tuple[str, str], str] = {}
        self._discarded = False

    def __enter__(self) -> RegionPerceptionRunWriter:
        """Return this writer; the mask staging directory is created on first use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Discard the staged mask payloads unless :meth:`finalize` published the run."""
        if not self._finalized:
            self._discard_staging()

    def _discard_staging(self) -> None:
        """Remove the mask staging directory, so an unpublished run leaves nothing behind."""
        shutil.rmtree(self._mask_staging, ignore_errors=True)
        self._mask_store = None
        self._discarded = True

    def write_manifest(self, manifest: RegionPerceptionRunManifest) -> None:
        """Record the run's identity and lineage for publication.

        Args:
            manifest: Identity, lineage and backend provenance of this run. It is held
                until :meth:`finalize`, which publishes it with the inventory it
                actually wrote.
        """
        if self._finalized:
            raise RegionPerceptionRunError("cannot write manifest after finalize()")
        self.manifest = manifest

    def write_frame(
        self,
        image: PreparedImage,
        *,
        outcome: Literal["succeeded", "failed"] = "succeeded",
        failure_detail: str | None = None,
    ) -> None:
        """Record preparation and outcome, staging constraints without inline mask pixels."""
        from contextmap.region_perception.grounding.region_grounding import (
            encode_prepared_image_reference,
        )
        from contextmap.region_perception.region_evidence import RegionId

        if self._finalized:
            raise RegionPerceptionRunError("cannot write frame after finalize()")
        frame = RegionPerceptionFrame(image=image, outcome=outcome, failure_detail=failure_detail)
        key = str(image.source_observation_id)
        if key in self._frames:
            raise RegionPerceptionRunError(f"duplicate regional frame: {key!r}")
        record = encode_prepared_image_reference(image)
        constraints = ([record["valid_region"]] if image.valid_region is not None else []) + record[
            "exclusion_regions"
        ]
        masks = ([image.valid_region.mask] if image.valid_region is not None else []) + [
            constraint.mask for constraint in image.exclusion_regions
        ]
        for index, (constraint, mask) in enumerate(zip(constraints, masks, strict=True)):
            if self._mask_store is None:
                self._mask_store = MaskStoreWriter(self._mask_staging / MASKS_DIRNAME)
            entry = self._mask_store.write(
                region_id=RegionId(f"frame-constraint-{index:04d}"),
                source_observation_id=image.source_observation_id,
                mask=mask,
            )
            constraint.pop("mask")
            constraint["mask_reference"] = f"{MASK_REFERENCE_PREFIX}{entry.payload_reference}"
        self._frames[key] = {
            "image": record,
            "outcome": frame.outcome,
            "failure_detail": frame.failure_detail,
        }

    def _require_execution_owner(self, run_id: str, observation_id: SourceObservationId) -> None:
        """Reject a mismatched producer or observation before accepting scientific evidence."""
        if self._finalized:
            raise RegionPerceptionRunError("cannot write execution after finalize()")
        if self.manifest is None:
            raise RegionPerceptionRunError("write manifest before regional executions")
        if run_id != self.manifest.run_id:
            raise RegionPerceptionRunError("regional execution belongs to another run")
        if str(observation_id) not in self.manifest.source_observation_ids:
            raise RegionPerceptionRunError("regional execution observation is outside selection")

    def write_grounding(self, execution: RegionGroundingExecution) -> None:
        """Keep the exact request, prompt, raw answer, outputs and diagnostics for publication."""
        self._require_execution_owner(
            execution.request.region_perception_run_id, execution.request.source_observation_id
        )
        key = (str(execution.request.source_observation_id), str(execution.request_id))
        if key in self._groundings:
            raise RegionPerceptionRunError("duplicate grounding execution")
        self._groundings[key] = execution

    def write_refinement(self, execution: RegionRefinementExecution) -> RegionRefinementExecution:
        """Stage accepted masks and return the outcome records sharing those lazy references."""
        self._require_execution_owner(
            execution.request.region_perception_run_id, execution.request.source_observation_id
        )
        key = (str(execution.request.source_observation_id), str(execution.request_id))
        if key in self._refinements:
            raise RegionPerceptionRunError("duplicate refinement execution")
        outcomes = tuple(
            replace(
                outcome,
                region=(
                    None
                    if outcome.region is None
                    else self.externalize_region_mask(
                        execution.request.source_observation_id, outcome.region
                    )
                ),
            )
            for outcome in execution.outcomes
        )
        externalized = replace(execution, outcomes=outcomes)
        self._refinements[key] = externalized
        return externalized

    def externalize_region_mask(
        self, source_observation_id: SourceObservationId, region: Region2D
    ) -> Region2D:
        """Stage a region's inline mask now and return the region carrying only its reference.

        This is the producer boundary: the discovery backend's pixels reach the staging mask
        store right away instead of living in RAM until the run is published, and the region
        continues through the executor with ``mask_reference`` only. The reference is the one
        :meth:`finalize` publishes, so a consumer resolves it against the finished artifact.

        Args:
            source_observation_id: The physical observation the region was discovered in.
            region: The region produced; a box-only region (no ``mask``) is returned as is.

        Returns:
            The region with ``mask=None`` and ``mask_reference`` pointing into this run's
            ``outputs/masks/``.

        Raises:
            RegionPerceptionRunError: If called after :meth:`finalize`.
            MaskStoreError: If a previously staged region is given different mask pixels.
            OSError: If staging fails; the writer is then discarded.
        """
        if self._finalized:
            raise RegionPerceptionRunError("cannot externalize region masks after finalize()")
        if self._discarded:
            raise RegionPerceptionRunError("regional writer was discarded after a failed write")
        if region.mask is None:
            return region
        if (
            region.source_observation_id is not None
            and region.source_observation_id != source_observation_id
        ):
            raise RegionPerceptionRunError("region mask belongs to another observation")
        key = (str(source_observation_id), str(region.region_id))
        reference = self._externalized_masks.get(key)
        if reference is not None:
            if self.resolve_mask(reference) != region.mask:
                raise MaskStoreError("conflicting mask pixels for an already staged region")
            return replace(region, mask=None, mask_reference=reference)
        if self._mask_store is None:
            self._mask_store = MaskStoreWriter(self._mask_staging / MASKS_DIRNAME)
        try:
            entry = self._mask_store.write(
                region_id=region.region_id,
                source_observation_id=source_observation_id,
                mask=region.mask,
            )
        except OSError:
            self._discard_staging()
            raise
        reference = f"{MASK_REFERENCE_PREFIX}{entry.payload_reference}"
        self._externalized_masks[key] = reference
        return replace(region, mask=None, mask_reference=reference)

    def resolve_mask(self, reference: str) -> InlineMask:
        """Resolve and hash-check this execution's staged mask by its public reference."""
        if self._mask_store is None or not reference.startswith(MASK_REFERENCE_PREFIX):
            raise MaskStoreError("reference is not a mask of this run")
        return self._mask_store.load(reference.removeprefix(MASK_REFERENCE_PREFIX))

    def add_region_discovery_audit(self, audit: RegionDiscoveryAudit) -> None:
        """Record how one frame's canonical regions were decided, for publication.

        The audit — every pass with its backend diagnostics, the pass-level and normalization
        rejections, the merge lineage and the effective discovery identity — is the evidence
        behind the regions, and the run that discovered them is the run that must carry it.
        A consumer that replays these regions reads the audit back instead of inventing one.

        Args:
            audit: The audit the canonical region discovery reported for one frame.

        Raises:
            RegionPerceptionRunError: If called after :meth:`finalize`, or if this run already
                carries an audit for the same ``source_observation_id``.
        """
        if self._finalized:
            raise RegionPerceptionRunError("cannot add region discovery audits after finalize()")
        key = str(audit.source_observation_id)
        if key in self._audits:
            raise RegionPerceptionRunError(
                f"duplicate region discovery audit in region perception run: {key!r}"
            )
        self._audits[key] = audit

    def write_region(self, region: Region2D) -> None:
        """Accumulate one region for publication.

        Args:
            region: The validated region to publish. Its mask pixels are never inlined;
                only ``mask_reference`` is persisted (#378), so a region whose pixels this
                run owns is handed through :meth:`externalize_region_mask` first.

        Raises:
            TypeError: If the argument is not a :class:`Region2D`. The boundary used to
                take a raw ``dict``, which let a renamed field cross it silently.
            RegionPerceptionRunError: If the region still carries inline mask pixels: they
                would be dropped by the record's encoding, and the published region would
                claim no mask at all.
        """
        from contextmap.region_perception.region_evidence import Region2D as Region2DClass

        if not isinstance(region, Region2DClass):
            raise TypeError(
                f"write_region expects a Region2D, got {type(region).__name__}; "
                "encode nothing by hand — the writer owns the wire format"
            )
        if self._finalized:
            raise RegionPerceptionRunError("cannot write region after finalize()")
        if any(written.region_id == region.region_id for written in self.regions):
            raise RegionPerceptionRunError("duplicate region identity in published run")
        if region.mask is not None:
            raise RegionPerceptionRunError(
                f"region {region.region_id!r} still carries inline mask pixels, which the "
                "published record cannot hold; stage them with externalize_region_mask() first"
            )
        if region.mask_reference is not None:
            reference = region.mask_reference
            if not reference.startswith(MASK_REFERENCE_PREFIX) or self._mask_store is None:
                raise RegionPerceptionRunError("region mask_reference was not staged by this run")
            try:
                entry = self._mask_store.entry_for(reference.removeprefix(MASK_REFERENCE_PREFIX))
                if entry.region_id != str(region.region_id) or entry.source_observation_id != str(
                    region.source_observation_id
                ):
                    raise MaskStoreError("mask_reference belongs to another regional identity")
            except MaskStoreError as error:
                raise RegionPerceptionRunError(
                    "region mask_reference was not staged by this run"
                ) from error
        self.regions.append(region)

    def finalize(self) -> RegionPerceptionRunManifest:
        """Publish the run atomically and return the manifest that was written.

        Returns:
            The published manifest, carrying the inventory of the contractual files with
            their sizes and SHA-256 hashes. A caller that derives the run's content
            identity from the inventory builds it from this value.

        Raises:
            RegionPerceptionRunError: If no manifest was written, the final directory
                already exists — a finished run is immutable (AGENTS.md §8) — or a region
                claims a mask this run never staged.
        """
        import json

        from contextmap.region_perception.artifacts.region_serialization import (
            encode_discovery_audit,
            encode_region,
        )
        from contextmap.region_perception.grounding.region_grounding import (
            encode_region_grounding_execution,
        )
        from contextmap.region_perception.refinement.region_refinement import (
            encode_region_refinement_execution,
        )
        from contextmap.shared import AtomicRunDirectory, RunDirectoryError

        if self.manifest is None:
            raise RegionPerceptionRunError(
                "manifest must be written before finalizing: regions with no run identity "
                "could not be audited"
            )
        if set(self._frames) != set(self.manifest.source_observation_ids):
            raise RegionPerceptionRunError(
                "regional frame coverage does not match explicit selection"
            )
        for region in self.regions:
            if str(region.source_observation_id) not in self.manifest.source_observation_ids:
                raise RegionPerceptionRunError("region observation is outside explicit selection")
        self._require_execution_materialization()
        self._require_staged_masks()

        try:
            atomic_run = AtomicRunDirectory(self.output_dir, refuse_existing_early=True)
        except RunDirectoryError as error:
            raise RegionPerceptionRunError(str(error)) from error

        with atomic_run as atomic:
            atomic.write_bytes(
                FRAMES_FILENAME,
                "".join(
                    f"{json.dumps(frame, sort_keys=True)}\n" for frame in self._frames.values()
                ).encode(),
                contractual=True,
            )
            with atomic.open_binary(REGIONS_FILENAME, contractual=True) as stream:
                for region in self.regions:
                    line = json.dumps(encode_region(region), separators=(",", ":"))
                    stream.write((line + "\n").encode("utf-8"))

            grounding_records = []
            for index, execution in enumerate(self._groundings.values()):
                reference = f"outputs/grounding-responses/response-{index:06d}.txt"
                atomic.write_bytes(
                    reference, execution.raw_response.encode("utf-8"), contractual=True
                )
                grounding_records.append(
                    encode_region_grounding_execution(execution, raw_response_reference=reference)
                )
            atomic.write_bytes(
                GROUNDING_FILENAME,
                "".join(
                    f"{json.dumps(record, sort_keys=True)}\n" for record in grounding_records
                ).encode(),
                contractual=True,
            )
            atomic.write_bytes(
                REFINEMENT_FILENAME,
                "".join(
                    f"{json.dumps(encode_region_refinement_execution(execution), sort_keys=True)}\n"
                    for execution in self._refinements.values()
                ).encode(),
                contractual=True,
            )

            # Gravada SEMPRE, mesmo vazia: a ausência de frames auditados nunca se confunde
            # com "este run não registra auditoria", que faria um consumidor adivinhar.
            audit_records = [
                encode_discovery_audit(audit, run_id=self.manifest.run_id)
                for audit in self._audits.values()
            ]
            atomic.write_bytes(
                REGION_DISCOVERY_AUDIT_FILENAME,
                "".join(
                    f"{json.dumps(record, sort_keys=True)}\n" for record in audit_records
                ).encode("utf-8"),
                contractual=True,
            )
            self._publish_masks(atomic)

            # NOTE: o inventário publicado também entra no manifest devolvido. A publicação
            # o injeta no JSON em disco, e sem ele aqui o manifest tipado divergiria do
            # artifact: quem deriva a identidade do run dos hashes receberia lista vazia, e
            # dois runs de conteúdos diferentes teriam o mesmo content_hash.
            published = replace(
                self.manifest, region_count=len(self.regions), file_inventory=atomic.inventory()
            )
            atomic.publish(
                manifest=published.to_dict(),
                readme=(
                    "# Region Perception Run\n\n"
                    f"Run ID: {published.run_id}\n"
                    f"Schema Version: {published.schema_version}\n"
                    f"Regions: {len(self.regions)}\n"
                    f"Audited frames: {len(self._audits)}\n"
                ),
            )

        self._finalized = True
        self._discard_staging()
        return published

    def _require_execution_materialization(self) -> None:
        """Require published outcomes and refinement prompts to match their source evidence."""
        from contextmap.region_perception.refinement.region_refinement import (
            refinement_prompts_from,
        )

        published = {region.region_id: region for region in self.regions}
        executions: list[RegionGroundingExecution | RegionRefinementExecution] = [
            *self._groundings.values(),
            *self._refinements.values(),
        ]
        for execution in executions:
            for region in execution.regions:
                if published.get(region.region_id) != region:
                    raise RegionPerceptionRunError("regional execution region is not materialized")
        for execution in self._refinements.values():
            observation = str(execution.request.source_observation_id)
            for prompt in execution.request.prompts:
                grounding = self._groundings.get((observation, str(prompt.grounding_request_id)))
                if grounding is None:
                    raise RegionPerceptionRunError("refinement prompt names missing grounding")
                if prompt not in refinement_prompts_from(grounding):
                    raise RegionPerceptionRunError(
                        "refinement prompt differs from its grounding output"
                    )

    def _require_staged_masks(self) -> None:
        """Hold every region's own mask reference to a mask this run actually staged.

        Raises:
            RegionPerceptionRunError: If a region points into this run's mask store but names
                no payload this run staged for that very region. Publishing it would hand a
                consumer a reference that resolves to nothing, and a resolver that answered
                ``None`` for it would turn a lost mask into a box-only region (AGENTS.md §26).
        """
        staged = (
            set()
            if self._mask_store is None
            else {entry.payload_reference for entry in self._mask_store.entries()}
        )
        for region in self.regions:
            reference = region.mask_reference
            if reference is None or not reference.startswith(MASK_REFERENCE_PREFIX):
                continue
            if reference.removeprefix(MASK_REFERENCE_PREFIX) not in staged:
                raise RegionPerceptionRunError(
                    f"region {region.region_id!r} has mask_reference {reference!r}, which this "
                    "run did not stage"
                )
        referenced = {
            region.mask_reference.removeprefix(MASK_REFERENCE_PREFIX)
            for region in self.regions
            if region.mask_reference is not None
        }
        for frame in self._frames.values():
            image = frame["image"]
            constraints = (
                [image["valid_region"]] if image["valid_region"] is not None else []
            ) + image["exclusion_regions"]
            referenced.update(
                constraint["mask_reference"].removeprefix(MASK_REFERENCE_PREFIX)
                for constraint in constraints
            )
        if staged - referenced:
            raise RegionPerceptionRunError(
                "staged mask has no published region references or preparation constraint"
            )

    def _publish_masks(self, atomic: AtomicRunDirectory) -> None:
        """Move the staged mask payloads and their index into the run being published.

        Each payload enters the inventory under its published path, so a lost mask is
        detectable; the index is written from the same entries, in staging order, which is
        the order the regions were externalized in.
        """
        if self._mask_store is None:
            return
        entries = self._mask_store.entries()
        staging_root = self._mask_staging / MASKS_DIRNAME
        for entry in entries:
            atomic.write_bytes(
                f"{MASKS_DIRNAME}/{entry.payload_reference}",
                (staging_root / entry.payload_reference).read_bytes(),
                contractual=True,
            )
        write_mask_index(staging_root, entries)
        atomic.write_bytes(
            f"{MASKS_DIRNAME}/{MASK_INDEX_FILENAME}",
            (staging_root / MASK_INDEX_FILENAME).read_bytes(),
            contractual=True,
        )
