"""The belief and the per-hypothesis support persist in the resolution run, versioned.

``EntityResolutionRunArtifact`` versions its content. Version 0.2.0 added optional belief
and per-hypothesis physical support; 0.3.0 added exact member and geometry sources for each
aggregated attribute. Version 0.4.0 qualifies visual producer lineage. The reader accepts the
published historical forms.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from mapping_builders import make_hypothesis
from resolution_entity_builders import attribute, entity_at, fused_id
from resolution_run_fixtures import RUN, RunInputs, build_inputs, resolution_of, write_run

from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    EntityResolutionRunReader,
    ResolutionOutcome,
    ResolvedBeliefPolicy,
    RunArtifactError,
    SupportStatus,
    materialize_resolved_entities,
    retrieve_candidate_sets,
)
from contextmap.ingestion import SourceObservationId
from contextmap.semantic_mapping import Entity


def _supported_entity(name: str, index: int, frame: str) -> Entity:
    """An entity whose one hypothesis records the frame that supports it."""
    hypothesis = dataclasses.replace(
        make_hypothesis(fused_evidence_id=fused_id(index + 1)),
        supporting_physical_observations=(SourceObservationId(frame),),
    )
    return entity_at(
        name,
        (index * 1.0, 0.0, 0.0),
        support_number=index + 1,
        spatial=(f"spatial--{name}",),
        hypotheses=(hypothesis,),
    )


def _belief_inputs() -> RunInputs:
    entities = {
        "a": _supported_entity("a", 0, "frame-0010"),
        "b": _supported_entity("b", 1, "frame-0020"),
    }
    sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    resolutions = (resolution_of(entities["a"], entities["b"], ResolutionOutcome.MATCH),)
    materialization = materialize_resolved_entities(
        entities.values(),
        [item.decision for item in resolutions],
        resolution_run_id=RUN,
        belief_policy=ResolvedBeliefPolicy(min_supporting_physical_observations=2),
    )
    return RunInputs(entities, sets, resolutions, materialization)


def _manifest_version(directory: Path) -> str:
    version: str = json.loads((directory / "manifest.json").read_text())["schema_version"]
    return version


def _set_historical_manifest_version(directory: Path, version: str) -> None:
    """Present the lineage shape persisted before visual producers were qualified."""
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = version
    lineage = manifest["lineage"]
    lineage["perception_run_ids"] = lineage.pop("visual_artifact_ids")
    path.write_text(json.dumps(manifest))


def test_a_run_with_a_belief_is_recorded_under_the_new_version_and_reads_back(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "artifact"
    inputs, _ = write_run(directory, _belief_inputs())

    assert _manifest_version(directory) == "0.4.0"
    reader = EntityResolutionRunReader(directory)
    (resolved,) = reader.resolved_entities().entities
    assert resolved == inputs.materialization.resolved.entities[0]
    assert resolved.semantic_state.belief is not None
    assert resolved.semantic_state.belief.support_status is SupportStatus.SUPPORTED


def test_a_run_without_belief_uses_the_same_current_producer_schema(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    write_run(directory, build_inputs())

    assert _manifest_version(directory) == "0.4.0"
    assert EntityResolutionRunReader(directory).resolved_entities().entities


def test_a_run_with_attribute_sources_is_recorded_as_0_4_0_and_reads_back(
    tmp_path: Path,
) -> None:
    entity = entity_at("a", attributes=(attribute("color", "blue"),))
    candidate_sets = retrieve_candidate_sets(
        (entity,), CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    materialization = materialize_resolved_entities((entity,), (), resolution_run_id=RUN)
    inputs = RunInputs({"a": entity}, candidate_sets, (), materialization)
    directory = tmp_path / "artifact"

    write_run(directory, inputs)
    (resolved,) = EntityResolutionRunReader(directory).resolved_entities().entities

    assert _manifest_version(directory) == "0.4.0"
    (source,) = resolved.semantic_state.attribute_sources
    assert source.member_entity_refs == (entity.reference,)
    assert source.geometry_refs == entity.geometry.geometry_refs


def test_reader_accepts_published_0_3_0_attribute_sources(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    inputs = _single_entity_inputs(with_attribute=True)
    write_run(directory, inputs)
    _set_historical_manifest_version(directory, "0.3.0")

    reader = EntityResolutionRunReader(directory)
    assert reader.manifest.schema_version == "0.3.0"
    assert reader.manifest.lineage.visual_artifact_ids == ()
    assert reader.resolved_entities() == inputs.materialization.resolved


def _single_entity_inputs(*, with_attribute: bool) -> RunInputs:
    """Build one resolved entity, with an attribute only when the test needs schema 0.3.0."""
    entity = entity_at("a", attributes=(attribute("color", "blue"),) if with_attribute else ())
    candidate_sets = retrieve_candidate_sets(
        (entity,), CandidateRetrievalPolicy(centroid_radius_m=20.0, bounds_margin_m=0.1)
    )
    materialization = materialize_resolved_entities((entity,), (), resolution_run_id=RUN)
    return RunInputs({"a": entity}, candidate_sets, (), materialization)


def _replace_resolved_record(directory: Path, record: dict[str, object]) -> None:
    """Change one stored record and its lazy-read index to exercise both reader paths."""
    data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (directory / "outputs/resolved-entities.jsonl").write_bytes(data)
    index = {"resolved_entity_id": record["resolved_entity_id"], "offset": 0, "length": len(data)}
    (directory / "outputs/resolved-entity-index.jsonl").write_text(json.dumps(index) + "\n")


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0"])
@pytest.mark.parametrize("read_one", [False, True])
def test_historical_versions_refuse_attribute_sources(
    tmp_path: Path, version: str, read_one: bool
) -> None:
    directory = tmp_path / "artifact"
    inputs = _single_entity_inputs(with_attribute=False) if version == "0.1.0" else _belief_inputs()
    write_run(directory, inputs)
    _set_historical_manifest_version(directory, version)
    record = json.loads((directory / "outputs/resolved-entities.jsonl").read_text())
    record["semantic_state"]["attribute_sources"] = []
    _replace_resolved_record(directory, record)

    reader = EntityResolutionRunReader(directory)
    with pytest.raises(RunArtifactError, match="attribute_sources"):
        if read_one:
            reader.resolved_entity(inputs.materialization.resolved.entities[0].reference)
        else:
            reader.resolved_entities()


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0"])
@pytest.mark.parametrize("read_one", [False, True])
def test_a_0_3_payload_cannot_be_read_under_a_historical_manifest(
    tmp_path: Path, version: str, read_one: bool
) -> None:
    directory = tmp_path / "artifact"
    inputs = _single_entity_inputs(with_attribute=True)
    write_run(directory, inputs)
    _set_historical_manifest_version(directory, version)

    reader = EntityResolutionRunReader(directory)
    with pytest.raises(RunArtifactError, match="attribute_sources"):
        if read_one:
            reader.resolved_entity(inputs.materialization.resolved.entities[0].reference)
        else:
            reader.resolved_entities()


@pytest.mark.parametrize("version", ["0.3.0", "0.4.0"])
@pytest.mark.parametrize("source_shape", ["missing", "empty", "wrong_attribute"])
@pytest.mark.parametrize("read_one", [False, True])
def test_version_0_3_and_0_4_require_exact_attribute_sources(
    tmp_path: Path, version: str, source_shape: str, read_one: bool
) -> None:
    directory = tmp_path / "artifact"
    inputs = _single_entity_inputs(with_attribute=True)
    write_run(directory, inputs)
    if version == "0.3.0":
        _set_historical_manifest_version(directory, version)
    else:
        assert _manifest_version(directory) == version
    record = json.loads((directory / "outputs/resolved-entities.jsonl").read_text())
    state = record["semantic_state"]
    if source_shape == "missing":
        del state["attribute_sources"]
    elif source_shape == "empty":
        state["attribute_sources"] = []
    else:
        state["attribute_sources"][0]["name"] = "unrelated"
    _replace_resolved_record(directory, record)

    reader = EntityResolutionRunReader(directory)
    with pytest.raises(RunArtifactError, match="attribute_sources"):
        if read_one:
            reader.resolved_entity(inputs.materialization.resolved.entities[0].reference)
        else:
            reader.resolved_entities()


def test_a_run_of_an_unknown_version_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    write_run(directory, build_inputs())
    manifest = directory / "manifest.json"
    record = json.loads(manifest.read_text())
    record["schema_version"] = "0.5.0"
    manifest.write_text(json.dumps(record))

    with pytest.raises(RunArtifactError, match=r"0\.5\.0"):
        EntityResolutionRunReader(directory)
