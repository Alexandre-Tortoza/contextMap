"""Contrato do artifact de region_perception: tipado na fronteira e publicado atomicamente.

``region_perception`` é um stage: seu run é um artifact público que `semantic_scoring` e o
encadeamento ponta a ponta selecionam como upstream. Antes desta branch ele era o único stage
que publicava com ``json.dumps`` direto — sem staging, sem hash por arquivo, sem inventário e
sem ``schema_version`` —, e trocava `Region2D` por `dict` na entrada e na saída.

Os dois defeitos são o mesmo: o artifact não sabia o que carregava. Sem contrato tipado, um
campo renomeado atravessa a fronteira calado; sem publicação atômica, uma execução
interrompida deixa um diretório que parece um run completo. Estes testes fixam as duas
garantias, iguais às de ``semantic_scoring.artifacts``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    InlineMask,
    Region2D,
    RegionDiscoveryAudit,
    RegionDiscoveryIdentity,
    RegionId,
    RegionPerceptionRunError,
    RegionPerceptionRunManifest,
    RegionPerceptionRunReader,
    RegionPerceptionRunWriter,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PreparedImage

_PROVENANCE = BackendProvenance(
    backend_id="sam3",
    capability="region_discovery",
    provider="meta",
    model="sam3-huge",
    version="1.0.0",
    configuration_fingerprint="abc123",
)


def _manifest(**overrides: object) -> RegionPerceptionRunManifest:
    """Build a manifest whose required lineage and backend identity are all present."""
    fields: dict[str, object] = {
        "run_id": "region_run_001",
        "run_index": 1,
        "sequence_name": "sequence",
        "selection_id": "selection",
        "sequence_artifact_id": "seq_001",
        "source_observation_ids": ["obs_001"],
        "created_at": "2026-10-01T12:00:00Z",
        "prepared_image_id": "prep_img_001",
        "discovery_backend_id": "sam3",
        "discovery_model": "sam3-huge",
        "discovery_version": "1.0.0",
        "discovery_configuration_fingerprint": "abc123",
        "normalization_policy_fingerprint": "def456",
    }
    fields.update(overrides)
    return RegionPerceptionRunManifest(**fields)  # type: ignore[arg-type]


def _region(region_id: str, *, x: float = 0.0, y: float = 0.0) -> Region2D:
    """Build an accepted 100x100 region whose only varying field is its position."""
    return Region2D(
        region_id=RegionId(region_id),
        provenance=_PROVENANCE,
        bounding_box=BoundingBox2D(x=x, y=y, width=100.0, height=100.0),
        source_observation_id=SourceObservationId("obs_001"),
        area_pixels=10_000.0,
    )


def _audit(observation_id: str) -> RegionDiscoveryAudit:
    """An audit of a frame whose discovery rejected and merged nothing."""
    return RegionDiscoveryAudit(
        source_observation_id=SourceObservationId(observation_id),
        backend=_PROVENANCE,
        passes=(),
        diagnostics=(),
        pass_rejections=(),
        discovery_identity=RegionDiscoveryIdentity.effective(),
        normalization_rejections=(),
        merge_decisions=(),
    )


class TestManifest:
    """The manifest carries the run's identity, lineage and schema version."""

    def test_the_manifest_declares_its_schema_version(self) -> None:
        """A consumer must be able to tell which schema it is reading."""
        assert _manifest().to_dict()["schema_version"] == "0.2.0"

    def test_the_manifest_round_trips_through_its_own_dict(self) -> None:
        """from_dict is the inverse of to_dict, so a published run reads back typed."""
        original = _manifest(region_count=2, grounding_backend_id="locateanything")
        restored = RegionPerceptionRunManifest.from_dict(original.to_dict())
        assert restored == original

    def test_an_unsupported_schema_version_is_refused(self) -> None:
        """Reading a schema this code does not know must fail, not guess (AGENTS.md §26)."""
        record = _manifest().to_dict()
        record["schema_version"] = "9.9.9"
        with pytest.raises(RegionPerceptionRunError, match="unsupported schema_version"):
            RegionPerceptionRunManifest.from_dict(record)


class TestTypedBoundary:
    """Regions cross the artifact boundary as Region2D, never as a bare dict."""

    def test_a_written_region_reads_back_as_the_same_region(self, tmp_path: Path) -> None:
        """The round trip preserves the contract, not just the JSON."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest(region_count=2))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        first = _region("region_001")
        second = _region("region_002", x=50.0, y=50.0)
        writer.write_region(first)
        writer.write_region(second)
        writer.finalize()

        regions = RegionPerceptionRunReader(run_dir).read_regions()
        assert [region.region_id for region in regions] == ["region_001", "region_002"]
        assert regions == [first, second]

    def test_the_writer_refuses_anything_that_is_not_a_region(self, tmp_path: Path) -> None:
        """A dict used to pass straight through; now the boundary names the type it wants."""
        writer = RegionPerceptionRunWriter(tmp_path / "run")
        writer.write_manifest(_manifest())
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        with pytest.raises(TypeError, match="write_region expects a Region2D"):
            writer.write_region({"region_id": "region_001"})  # type: ignore[arg-type]

    def test_an_empty_run_reads_back_as_no_regions(self, tmp_path: Path) -> None:
        """A run that discovered nothing is a valid run, distinct from a missing one."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest(region_count=0))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        writer.finalize()

        reader = RegionPerceptionRunReader(run_dir)
        assert reader.read_manifest().region_count == 0
        assert list(reader.read_regions()) == []


class TestAtomicPublication:
    """The run directory either does not exist or holds a complete, verifiable run."""

    def test_nothing_reaches_the_final_path_before_finalize(self, tmp_path: Path) -> None:
        """An interrupted run must not leave a directory that looks like a finished one."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest())
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        writer.write_region(_region("region_001"))

        assert not run_dir.exists()

    def test_finalize_publishes_the_manifest_and_the_regions(self, tmp_path: Path) -> None:
        """The published layout is the one the reader and the inventory promise."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest(region_count=1))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        writer.write_region(_region("region_001"))
        writer.finalize()

        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "outputs" / "regions.jsonl").exists()
        lines = (run_dir / "outputs" / "regions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["region_id"] == "region_001"

    def test_the_manifest_inventories_every_contractual_file(self, tmp_path: Path) -> None:
        """Without an inventory the loss of a file is undetectable (AGENTS.md §8)."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest(region_count=1))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        writer.write_region(_region("region_001"))
        published = writer.finalize()

        paths = [entry.path for entry in published.file_inventory]
        # A auditoria é gravada mesmo vazia: "nenhum frame auditado" nunca se confunde com
        # "este run não registra auditoria", que faria a leitura adivinhar.
        assert paths == [
            "outputs/frames.jsonl",
            "outputs/grounding.jsonl",
            "outputs/refinement.jsonl",
            "outputs/region-discovery-audit.jsonl",
            "outputs/regions.jsonl",
        ]
        assert all(entry.content_hash.startswith("sha256:") for entry in published.file_inventory)

    def test_integrity_verification_detects_a_tampered_output(self, tmp_path: Path) -> None:
        """The hash in the manifest is what makes the artifact auditable."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest(region_count=1))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        writer.write_region(_region("region_001"))
        writer.finalize()

        reader = RegionPerceptionRunReader(run_dir)
        assert reader.verify_integrity() == []

        (run_dir / "outputs" / "regions.jsonl").write_text("{}\n")
        assert RegionPerceptionRunReader(run_dir).verify_integrity() != []

    def test_publishing_over_an_existing_run_is_refused(self, tmp_path: Path) -> None:
        """A finished artifact is immutable; rerunning creates another run (AGENTS.md §8)."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest())
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        with pytest.raises(RegionPerceptionRunError, match="already exists"):
            writer.finalize()

    def test_finalizing_without_a_manifest_is_refused(self, tmp_path: Path) -> None:
        """Regions with no run identity could not be audited, so they are not published."""
        writer = RegionPerceptionRunWriter(tmp_path / "run")
        writer.write_region(_region("region_001"))
        with pytest.raises(RegionPerceptionRunError, match="manifest must be written"):
            writer.finalize()


class TestMasksAndAudit:
    """A run that a consumer can replay has to carry the pixels and the evidence too."""

    def test_a_staged_mask_is_published_indexed_and_resolvable(self, tmp_path: Path) -> None:
        """The pixels leave RAM at the producer boundary and come back hash-verified."""
        run_dir = tmp_path / "run"
        pixels = np.zeros((4, 4), dtype=bool)
        pixels[1, 1:3] = True
        with RegionPerceptionRunWriter(run_dir) as writer:
            writer.write_manifest(_manifest(region_count=1))
            writer.write_frame(
                PreparedImage(
                    source_observation_id=SourceObservationId("obs_001"),
                    payload_reference="source-observation:obs_001",
                    width=200,
                    height=200,
                )
            )
            region = writer.externalize_region_mask(
                SourceObservationId("obs_001"),
                replace(_region("region_001"), mask=InlineMask(pixels)),
            )
            # A região segue sem pixels, carregando só a referência do store do run.
            assert region.mask is None
            assert region.mask_reference == "outputs/masks/obs_001/region_001.npy"
            writer.write_region(region)
            published = writer.finalize()

        assert {entry.path for entry in published.file_inventory} == {
            "outputs/frames.jsonl",
            "outputs/grounding.jsonl",
            "outputs/refinement.jsonl",
            "outputs/regions.jsonl",
            "outputs/region-discovery-audit.jsonl",
            "outputs/masks/mask-index.jsonl",
            "outputs/masks/obs_001/region_001.npy",
        }
        reader = RegionPerceptionRunReader(run_dir)
        assert reader.verify_integrity() == []
        resolved = reader.mask_resolver().resolve_mask("outputs/masks/obs_001/region_001.npy")
        assert np.array_equal(resolved.as_array(), pixels)

    def test_an_abandoned_run_leaves_no_staged_mask_behind(self, tmp_path: Path) -> None:
        """Leaving the block without finalize must not leave a directory for a later run."""
        run_dir = tmp_path / "run"
        with RegionPerceptionRunWriter(run_dir) as writer:
            writer.write_manifest(_manifest())
            writer.write_frame(
                PreparedImage(
                    source_observation_id=SourceObservationId("obs_001"),
                    payload_reference="source-observation:obs_001",
                    width=200,
                    height=200,
                )
            )
            writer.externalize_region_mask(
                SourceObservationId("obs_001"),
                replace(_region("region_001"), mask=InlineMask(np.ones((2, 2), dtype=bool))),
            )

        assert not run_dir.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_region_that_still_carries_pixels_is_refused(self, tmp_path: Path) -> None:
        """The record cannot hold pixels, so dropping them silently is not an option."""
        writer = RegionPerceptionRunWriter(tmp_path / "run")
        writer.write_manifest(_manifest())
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        with pytest.raises(RegionPerceptionRunError, match="still carries inline mask pixels"):
            writer.write_region(
                replace(_region("region_001"), mask=InlineMask(np.ones((2, 2), dtype=bool)))
            )

    def test_a_region_claiming_a_mask_this_run_never_staged_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A published reference that resolves to nothing would read back as box-only."""
        writer = RegionPerceptionRunWriter(tmp_path / "run")
        writer.write_manifest(_manifest(region_count=1))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        with pytest.raises(RegionPerceptionRunError, match="mask_reference"):
            writer.write_region(
                replace(
                    _region("region_001"), mask_reference="outputs/masks/obs_001/region_001.npy"
                )
            )

    def test_the_same_frame_cannot_be_audited_twice(self, tmp_path: Path) -> None:
        """One frame is decided once; two audits for it would contradict each other."""
        writer = RegionPerceptionRunWriter(tmp_path / "run")
        writer.add_region_discovery_audit(_audit("obs_001"))
        with pytest.raises(RegionPerceptionRunError, match="duplicate region discovery audit"):
            writer.add_region_discovery_audit(_audit("obs_001"))

    def test_the_audits_read_back_in_publication_order(self, tmp_path: Path) -> None:
        """The audit is the evidence behind the regions, and it survives the round trip."""
        run_dir = tmp_path / "run"
        writer = RegionPerceptionRunWriter(run_dir)
        writer.write_manifest(_manifest())
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_001"),
                payload_reference="source-observation:obs_001",
                width=200,
                height=200,
            )
        )
        writer.write_manifest(_manifest(source_observation_ids=("obs_001", "obs_002")))
        writer.write_frame(
            PreparedImage(
                source_observation_id=SourceObservationId("obs_002"),
                payload_reference="source-observation:obs_002",
                width=200,
                height=200,
            )
        )
        writer.add_region_discovery_audit(_audit("obs_002"))
        writer.add_region_discovery_audit(_audit("obs_001"))
        writer.finalize()

        audits = RegionPerceptionRunReader(run_dir).read_region_discovery_audits()
        assert [str(audit.source_observation_id) for audit in audits] == ["obs_002", "obs_001"]
        assert audits[0].backend == _PROVENANCE


class TestReader:
    """A wrong upstream selection fails at the reader, not downstream as empty data."""

    def test_a_missing_run_directory_is_refused(self, tmp_path: Path) -> None:
        """Pointing at a run that was never published must fail here."""
        with pytest.raises(RegionPerceptionRunError, match="not found"):
            RegionPerceptionRunReader(tmp_path / "never-published")

    def test_a_run_directory_without_a_manifest_is_refused(self, tmp_path: Path) -> None:
        """A directory is not an artifact; the manifest is what makes it one."""
        (tmp_path / "run").mkdir()
        with pytest.raises(RegionPerceptionRunError, match="manifest not found"):
            RegionPerceptionRunReader(tmp_path / "run")


def test_manifest_without_schema_is_rejected() -> None:
    """A missing schema declaration cannot be interpreted as the current version."""
    record = _manifest().to_dict()
    record.pop("schema_version")
    with pytest.raises(RegionPerceptionRunError, match="schema_version"):
        RegionPerceptionRunManifest.from_dict(record)
