import json
from dataclasses import replace
from hashlib import sha256

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    AuditedRegions,
    BackendDiagnostics,
    BoundingBox,
    DiscoveryPassConfig,
    HintContribution,
    MergeKind,
    NativeRegionText,
    NormalizationConfig,
    RegionCandidate,
    RegionDiscoveryAudit,
    RegionDiscoveryIdentity,
    RegionProvenance,
    RejectionReason,
)
from contextmap.region_perception.discovery.region_discovery import (
    DiscoveryInput,
    DiscoveryOutput,
    discover_canonical_regions,
)
from contextmap.shared import ArtifactReference, BackendProvenance
from contextmap.visual_perception import PreparedImage

_PASSES = DiscoveryPassConfig(max_candidates_per_pass=3)
_NORMALIZATION = NormalizationConfig(minimum_area_pixels=2.0)
_DIAGNOSTICS = BackendDiagnostics(
    duration_ms=2.5,
    proposal_count=4,
    warnings=("fake warning",),
    metadata=(("raw_proposal_count", 5),),
)


def _image() -> PreparedImage:
    return PreparedImage(
        source_observation_id=SourceObservationId("frame-audit"),
        payload_reference="sequence/frames/frame-audit.png",
        payload_artifact=ArtifactReference(
            uri="sequence/frames/frame-audit.png",
            sha256=sha256(b"audit").hexdigest(),
            media_type="image/png",
        ),
        width=4,
        height=3,
        transformations=(),
    )


class _ProposalsToAudit:
    """Proposes on every pass a region, its duplicate, a sliver and one proposal over budget.

    ``texts`` gives some proposals the text a task such as Florence-2 ``<OD>`` returns with them.
    """

    def __init__(self, texts: dict[str, str] | None = None) -> None:
        self._texts = texts or {}

    def backend_provenance(self) -> BackendProvenance:
        return BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="test",
            model="none",
            version="1",
            configuration_fingerprint="sha256:fake",
        )

    def discover_candidates(self, discovery_input: DiscoveryInput) -> DiscoveryOutput:
        width = discovery_input.discovery_pass.input_width
        height = discovery_input.discovery_pass.input_height
        boxes = {
            "p-1": BoundingBox(0, 0, width, height),
            "p-2": BoundingBox(0, 0, width, height),
            "p-3": BoundingBox(0, 0, 1, 1),
            "p-4": BoundingBox(0, 0, 2, 2),
        }
        return DiscoveryOutput(
            candidates=tuple(
                RegionCandidate(
                    candidate_id=candidate_id,
                    source_observation_id=discovery_input.prepared_image.source_observation_id,
                    perception_run_id=discovery_input.perception_run_id,
                    perception_result_id=discovery_input.perception_result_id,
                    image_width=width,
                    image_height=height,
                    bounding_box=box,
                    provenance=RegionProvenance(
                        backend_id="fake",
                        backend_version="1",
                        checkpoint="none",
                        config_digest="sha256:fake",
                        discovery_pass_id=discovery_input.discovery_pass.pass_id,
                        native_proposal_id=candidate_id,
                    ),
                    native_text=(
                        NativeRegionText(task="<OD>", text=self._texts[candidate_id])
                        if candidate_id in self._texts
                        else None
                    ),
                )
                for candidate_id, box in boxes.items()
            ),
            diagnostics=_DIAGNOSTICS,
        )


def _audited(texts: dict[str, str] | None = None) -> AuditedRegions:
    return discover_canonical_regions(
        _image(),
        _ProposalsToAudit(texts),
        pass_config=_PASSES,
        normalization_config=_NORMALIZATION,
    )


_TEXTS = {"p-1": "chair", "p-2": "armchair", "p-3": "speck", "p-4": "lamp"}


def test_canonical_discovery_reports_every_pass_rejection_and_merge() -> None:
    audited = _audited()

    audit = audited.audit
    assert audit.source_observation_id == "frame-audit"
    assert audit.backend == _ProposalsToAudit().backend_provenance()
    assert [discovery_pass.pass_id for discovery_pass in audit.passes] == ["full-frame"]
    assert audit.diagnostics == (_DIAGNOSTICS,)
    assert [(item.candidate_id, item.reason) for item in audit.pass_rejections] == [
        ("full-frame/p-4", RejectionReason.REGION_BUDGET_EXCEEDED)
    ]
    assert [(item.candidate_id, item.reason) for item in audit.normalization_rejections] == [
        ("full-frame/p-3", RejectionReason.AREA_BELOW_MINIMUM),
        ("full-frame/p-2", RejectionReason.MERGED_DUPLICATE),
    ]
    assert [
        (decision.representative_candidate_id, decision.merged_candidate_id, decision.kind)
        for decision in audit.merge_decisions
    ] == [("full-frame/p-1", "full-frame/p-2", MergeKind.IOU_DUPLICATE)]
    # O frame nomeia a configuração e os algoritmos efetivos que decidiram suas regiões (#636).
    assert audit.discovery_identity == RegionDiscoveryIdentity.effective(_PASSES, _NORMALIZATION)
    # A região sobrevivente e a auditoria nomeiam os mesmos candidatos.
    (region,) = audited.regions
    assert region.contributor_candidate_ids == ("full-frame/p-1", "full-frame/p-2")


def test_the_audit_record_round_trips_through_json() -> None:
    audit = _audited().audit

    record = json.loads(json.dumps(audit.to_dict()))

    assert set(record) == {
        "source_observation_id",
        "backend",
        "passes",
        "pass_rejections",
        "discovery_identity",
        "normalization_rejections",
        "merge_decisions",
        "semantic_hints",
    }
    # Cada pass carrega o diagnóstico do backend naquele pass.
    assert record["passes"][0]["diagnostics"] == _DIAGNOSTICS.to_dict()
    assert RegionDiscoveryAudit.from_dict(record) == audit


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_dimensions", [1, 1]),
        ("unexpected", "value"),
    ],
)
def test_a_record_that_is_not_the_canonical_encoding_is_refused(field: str, value: object) -> None:
    record = json.loads(json.dumps(_audited().audit.to_dict()))
    record["passes"][0][field] = value

    with pytest.raises(ValueError, match="canonical"):
        RegionDiscoveryAudit.from_dict(record)


def test_every_pass_needs_its_backend_diagnostics() -> None:
    audit = _audited().audit

    with pytest.raises(ValueError, match="one diagnostic record"):
        RegionDiscoveryAudit(
            source_observation_id=audit.source_observation_id,
            backend=audit.backend,
            passes=audit.passes,
            diagnostics=(),
            pass_rejections=audit.pass_rejections,
            discovery_identity=audit.discovery_identity,
            normalization_rejections=audit.normalization_rejections,
            merge_decisions=audit.merge_decisions,
        )


def test_native_text_reaches_the_audit_linked_by_the_merge_lineage() -> None:
    """#523: the text of each normalized proposal is auditable evidence of its frame."""
    audited = _audited(_TEXTS)
    (region,) = audited.regions

    hints = audited.audit.semantic_hints

    assert [
        (hint.candidate_id, hint.native_text.text, hint.contribution, hint.region_id)
        for hint in hints
    ] == [
        ("full-frame/p-1", "chair", HintContribution.REPRESENTATIVE, region.region_id),
        ("full-frame/p-2", "armchair", HintContribution.MERGED, region.region_id),
        ("full-frame/p-3", "speck", HintContribution.REJECTED, None),
    ]
    # O texto nunca entra na região, e o proposal cortado pelo orçamento do pass não chegou à
    # normalização: não há vínculo a registrar para ele.
    assert audited.regions == _audited().regions
    assert all(hint.source_observation_id == "frame-audit" for hint in hints)


def test_geometry_only_discovery_records_no_semantic_hint() -> None:
    assert _audited().audit.semantic_hints == ()


def test_semantic_hints_round_trip_through_the_audit_record() -> None:
    audit = _audited(_TEXTS).audit

    record = json.loads(json.dumps(audit.to_dict()))

    assert [item["native_text"]["text"] for item in record["semantic_hints"]] == [
        "chair",
        "armchair",
        "speck",
    ]
    assert RegionDiscoveryAudit.from_dict(record) == audit


def test_an_audit_recorded_without_semantic_hints_reads_as_empty() -> None:
    """A 0.6.0 audit written before the field existed recorded no native text."""
    audit = _audited().audit
    record = json.loads(json.dumps(audit.to_dict()))
    del record["semantic_hints"]

    assert RegionDiscoveryAudit.from_dict(record) == audit


@pytest.mark.parametrize(
    ("index", "changes", "message"),
    [
        (1, {"contribution": HintContribution.REPRESENTATIVE}, "merged"),
        (0, {"contribution": HintContribution.MERGED}, "merged"),
        (2, {"contribution": HintContribution.MERGED}, "merged"),
        (0, {"source_observation_id": "frame-other"}, "observation"),
    ],
)
def test_a_hint_contradicting_the_audit_lineage_is_refused(
    index: int, changes: dict[str, object], message: str
) -> None:
    audit = _audited(_TEXTS).audit
    hints = list(audit.semantic_hints)
    hint = hints[index]
    if changes.get("contribution") is HintContribution.REPRESENTATIVE or (
        changes.get("contribution") is HintContribution.MERGED and hint.region_id is None
    ):
        changes = {**changes, "region_id": audit.semantic_hints[0].region_id}
    hints[index] = replace(hint, **changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        replace(audit, semantic_hints=tuple(hints))


def test_a_rejected_hint_must_name_a_rejected_candidate() -> None:
    audit = _audited(_TEXTS).audit
    first = audit.semantic_hints[0]
    stray = replace(
        first,
        candidate_id="full-frame/p-9",
        contribution=HintContribution.REJECTED,
        region_id=None,
    )

    with pytest.raises(ValueError, match="rejected"):
        replace(audit, semantic_hints=(*audit.semantic_hints, stray))
