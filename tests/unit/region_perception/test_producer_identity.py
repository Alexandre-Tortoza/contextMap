"""Producer-owned regional identity and assembly contracts."""

from dataclasses import replace

import pytest

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    Region2D,
    RegionId,
    namespace_regions,
)
from contextmap.shared import BackendProvenance


def _region() -> Region2D:
    """Return one frame-local normalized region."""
    return Region2D(
        region_id=RegionId("region-0000"),
        source_observation_id=SourceObservationId("frame-a"),
        bounding_box=BoundingBox2D(x=0, y=0, width=2, height=2),
        provenance=BackendProvenance(
            backend_id="fake",
            capability="region_discovery",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="cfg",
        ),
    )


def test_published_identity_is_owned_by_regional_run_and_frame() -> None:
    """Same local region from another run or observation must be distinct."""
    region = _region()
    first = namespace_regions(
        run_id="run-a", source_observation_id=SourceObservationId("frame-a"), regions=(region,)
    )
    repeated = namespace_regions(
        run_id="run-a", source_observation_id=SourceObservationId("frame-a"), regions=(region,)
    )
    other_run = namespace_regions(
        run_id="run-b", source_observation_id=SourceObservationId("frame-a"), regions=(region,)
    )
    other_frame = namespace_regions(
        run_id="run-a",
        source_observation_id=SourceObservationId("frame-b"),
        regions=(replace(region, source_observation_id=SourceObservationId("frame-b")),),
    )
    assert first == repeated
    assert (
        len(
            {
                first.regions[0].region_id,
                other_run.regions[0].region_id,
                other_frame.regions[0].region_id,
            }
        )
        == 3
    )
    assert first.region_id_map[region.region_id] == first.regions[0].region_id
    assert first.regions[0].bounding_box == region.bounding_box
    assert region.region_id == "region-0000"


def test_namespace_refuses_wrong_physical_frame() -> None:
    """Publication must not silently reassign observation provenance."""
    with pytest.raises(ValueError, match="observation"):
        namespace_regions(
            run_id="run", source_observation_id=SourceObservationId("frame-b"), regions=(_region(),)
        )


def test_region_assembly_preserves_all_sources_in_order_without_merging() -> None:
    """Discovery, grounded boxes and refined masks remain separate evidence."""
    from contextmap.region_perception import assemble_regions

    from .test_regional_execution_streams import _grounding, _refinement

    grounding = _grounding()
    refinement = _refinement(grounding)
    result = assemble_regions(
        discovered=(_region(),), groundings=(grounding,), refinements=(refinement,)
    )
    assert result == (_region(), *grounding.regions, *refinement.regions)


def test_region_assembly_refuses_duplicate_identities() -> None:
    """An identical ID from two sources must fail instead of being deduplicated silently."""
    from contextmap.region_perception import assemble_regions

    with pytest.raises(ValueError, match="twice"):
        assemble_regions(discovered=(_region(), _region()))


def test_namespace_remaps_native_text_region_link_and_preserves_candidate_lineage() -> None:
    """A regional hint cites the qualified region without rewriting normalization decisions."""
    from dataclasses import replace

    from contextmap.region_perception.region_geometry import NativeRegionText, RegionProvenance
    from contextmap.region_perception.region_semantic_hint import (
        HintContribution,
        RegionSemanticHint,
    )

    from .test_artifacts import _audit

    region = _region()
    hint = RegionSemanticHint(
        candidate_id="candidate",
        source_observation_id="frame-a",
        perception_run_id="local-run",
        perception_result_id="local-result",
        provenance=RegionProvenance(
            backend_id="fake",
            backend_version="1",
            checkpoint="fake",
            config_digest="cfg",
            discovery_pass_id="full",
            native_proposal_id="proposal",
        ),
        native_text=NativeRegionText(task="OD", text="chair"),
        contribution=HintContribution.REPRESENTATIVE,
        region_id=region.region_id,
    )
    audit = replace(_audit("frame-a"), semantic_hints=(hint,))
    published = namespace_regions(
        run_id="regional-run",
        source_observation_id=SourceObservationId("frame-a"),
        regions=(region,),
        audit=audit,
    )
    assert published.audit is not None
    assert published.audit.semantic_hints[0].region_id == published.regions[0].region_id
    assert published.audit.semantic_hints[0].candidate_id == "candidate"
    assert published.audit.semantic_hints[0].perception_run_id == "regional-run"
    from contextmap.region_perception.artifacts.region_serialization import (
        decode_discovery_audit,
        encode_discovery_audit,
    )

    record = encode_discovery_audit(published.audit, run_id="regional-run")
    assert "perception_result_id" not in str(record)
    assert "perception_run_id" not in str(record)
    assert decode_discovery_audit(record, run_id="regional-run") == published.audit


def test_published_hint_uses_regional_artifact_identity_without_fused_result() -> None:
    """The wire contract cannot make an independent region depend on a fused result."""
    from contextmap.region_perception.artifacts.region_serialization import encode_discovery_audit

    from .test_artifacts import _audit

    record = encode_discovery_audit(_audit("frame-a"), run_id="region-run")
    assert record["region_artifact_id"] == "region-run"
