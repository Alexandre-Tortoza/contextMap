"""Assembly carries what the claims stated and the belief after resolution into the map.

The chain is real from the claim to the map: claims are fused by Semantic Fusion, mapped into an
entity semantic state by Semantic Mapping, merged (or not) and believed by Entity Resolution's own
materialization, written by its run writer, and translated by ``assemble_context_map``. Only the
geometry of each entity is a fixture box inside the real geometric-map artifact.

* the attributes the claims stated (``claim-attribute-v1``) reach the map with an origin that cites
  the fused evidence they were accumulated in; derived attributes (``class``, which restates a
  member-level primary) stay in the resolution run;
* the belief reaches the map only when Entity Resolution derived one: a support status, a primary
  label only when supported, and a count of distinct physical observations per label. With one
  frame nothing is ever supported (B1 regression).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from context_map_builders import entity_capabilities
from context_map_serialization_geometry import build_geometry_artifact
from fusion_run_fixtures import LINEAGE as BASE_FUSION_LINEAGE
from mapping_fusion import ClaimSpec, View, fuse
from resolution_entity_builders import entity_at, fused_id
from resolution_run_fixtures import LINEAGE as BASE_RESOLUTION_LINEAGE
from resolution_run_fixtures import resolution_of
from test_context_map_assembly import CONTEXT_MAP_ID, _assemble, _matching_metadata

from contextmap.artifact import (
    ContextAttributeOrigin,
    ContextEntity,
    ContextMap,
    ContextMapArtifactReader,
    ContextMapArtifactWriter,
    ContextSupportStatus,
    DerivationKind,
    PolicyRef,
    ProvenanceError,
    ValidationStatus,
    context_map_from_record,
    context_map_to_record,
    validate_context_map_artifact,
)
from contextmap.artifact.references import UpstreamRecordRef
from contextmap.entity_resolution import (
    CandidateRetrievalPolicy,
    EntityResolutionRunId,
    EntityResolutionRunWriter,
    ResolutionOutcome,
    ResolvedBeliefPolicy,
    materialize_resolved_entities,
    retrieve_candidate_sets,
)
from contextmap.geometric_mapping import (
    GeometricMapArtifactManifest,
    GeometricMapArtifactReader,
    geometry_id_for,
    geometry_index_of,
)
from contextmap.semantic_fusion import FusionOutcome, SemanticFusionRunId, SemanticFusionRunWriter
from contextmap.semantic_interpretation import ClaimId
from contextmap.semantic_mapping import (
    AttributeOrigin,
    Entity,
    EntityAttribute,
    EntityMaterializationPolicy,
    ExternalKnowledgeSource,
    GeometrySummaryPolicy,
    InteractionRadiusPolicy,
    SemanticMapId,
    materialize_entities,
    semantic_state_from_fused_evidence,
)
from contextmap.shared import Vector3

SEMANTIC_MAP_ID = SemanticMapId("semantic-map--semantics-0001")
RUN_ID = EntityResolutionRunId("entity-resolution--semantics-run-0001")
BELIEF = ResolvedBeliefPolicy(min_supporting_physical_observations=2)
BLUE_METAL_DOOR = ClaimSpec(
    "blue metal door", category="door", attributes=(("color", "blue"), ("material", "metal"))
)


def _entity(
    manifest: GeometricMapArtifactManifest,
    name: str,
    number: int,
    second: int,
    claims: Sequence[ClaimSpec],
    *,
    center: Vector3,
) -> Entity:
    """An entity whose semantic state is what the real fusion and mapping make of the claims."""
    _, fused = fuse([View("run-a", f"frame-{second:04d}", tuple(claims))])
    state = semantic_state_from_fused_evidence(fused)
    return entity_at(
        name,
        center,
        support_number=number,
        seconds=(second,),
        hypotheses=tuple(
            dataclasses.replace(item, fused_evidence_id=fused_id(number))
            for item in state.hypotheses
        ),
        attributes=state.attributes,
        spatial=(f"spatial--{name}",),
        map_id=manifest.map_id,
        semantic_map_id=SEMANTIC_MAP_ID,
        first_index=100 * (number - 1),
    )


def _entity_with_interaction_radius(
    tmp_path: Path,
    manifest: GeometricMapArtifactManifest,
    *,
    interaction_factor: float = 1.5,
) -> Entity:
    """Materialize one entity whose radius follows the real Semantic Mapping path."""
    claim = dataclasses.replace(BLUE_METAL_DOOR, interaction_factor=interaction_factor)
    support, fused = fuse([View("run-a", "frame-0010", (claim,))])
    support = dataclasses.replace(
        support,
        geometric_map_id=manifest.map_id,
        geometry_support=tuple(
            dataclasses.replace(
                reference,
                map_id=manifest.map_id,
                geometry_id=geometry_id_for(
                    map_id=manifest.map_id,
                    index=geometry_index_of(
                        map_id=reference.map_id,
                        geometry_id=reference.geometry_id,
                    ),
                ),
            )
            for reference in support.geometry_support
        ),
    )
    outcome = FusionOutcome(support=support, evidence=fused)
    fusion_manifest = SemanticFusionRunWriter(
        output_dir=tmp_path / "semantic-fusion",
        sequence_name="sequence-0001",
        run_id=SemanticFusionRunId("fusion-run-0001"),
        run_index=1,
        lineage=dataclasses.replace(BASE_FUSION_LINEAGE, geometric_map_id=manifest.map_id),
        code_version="test",
    ).write((outcome,))
    geometry_dir = tmp_path / "geometry-workspace" / "geometric_mapping"
    with GeometricMapArtifactReader(geometry_dir) as reader:
        materialization = materialize_entities(
            (outcome,),
            fusion_manifest=fusion_manifest,
            geometry=reader.geometry(),
            semantic_map_id=SEMANTIC_MAP_ID,
            policy=EntityMaterializationPolicy(
                geometry=GeometrySummaryPolicy(
                    sparse_point_threshold=3,
                    connectivity_radius_m=0.5,
                )
            ),
            code_version="test",
        )
    assert materialization.rejections == ()
    (entity,) = materialization.entities
    return entity


def _assembled(
    tmp_path: Path,
    build: Callable[[GeometricMapArtifactManifest], dict[str, Entity]],
    *,
    matches: Sequence[tuple[str, str]] = (),
    belief: ResolvedBeliefPolicy | None = BELIEF,
) -> tuple[ContextMap, dict[str, Path]]:
    """The assembled map, and where each upstream artifact it cites lives, for a writer."""
    geometry_dir, manifest = build_geometry_artifact(tmp_path / "geometry-workspace")
    entities = build(manifest)
    candidate_sets = retrieve_candidate_sets(
        entities.values(), CandidateRetrievalPolicy(centroid_radius_m=50.0, bounds_margin_m=0.1)
    )
    resolutions = tuple(
        resolution_of(entities[first], entities[second], ResolutionOutcome.MATCH)
        for first, second in matches
    )
    materialization = materialize_resolved_entities(
        entities.values(),
        [item.decision for item in resolutions],
        resolution_run_id=RUN_ID,
        belief_policy=belief,
    )
    resolution_dir = tmp_path / "entity-resolution"
    EntityResolutionRunWriter(
        output_dir=resolution_dir,
        run_id=RUN_ID,
        lineage=dataclasses.replace(
            BASE_RESOLUTION_LINEAGE,
            geometric_map_id=manifest.map_id,
            semantic_map_ids=(SEMANTIC_MAP_ID,),
            semantic_mapping_artifact_digest="sha256:" + "ab" * 32,
        ),
        code_version="test",
    ).write(candidate_sets=candidate_sets, resolutions=resolutions, materialization=materialization)
    context_map = _assemble(
        manifest,
        context_map_id=CONTEXT_MAP_ID,
        metadata=_matching_metadata(manifest, capabilities=entity_capabilities()),
        geometric_map_location=geometry_dir,
        entity_resolution_location=resolution_dir,
    )
    return context_map, {str(manifest.map_id): geometry_dir, str(RUN_ID): resolution_dir}


def _by_member(context_map: ContextMap) -> dict[str, ContextEntity]:
    return {str(item.member_entities[0].entity_id): item for item in context_map.entities}


# --- attributes (C7a) ----------------------------------------------------------------------------


def test_a_blue_metal_door_reaches_the_map_with_its_label_and_stated_attributes(
    tmp_path: Path,
) -> None:
    context_map, _ = _assembled(
        tmp_path,
        lambda manifest: {
            "door": _entity(manifest, "door", 1, 10, (BLUE_METAL_DOOR,), center=(0.0, 0.0, 0.0))
        },
    )

    (entity,) = context_map.entities
    state = entity.semantic_state
    assert [item.label for item in state.hypotheses] == ["blue metal door"]
    assert [(item.name, item.value) for item in state.attributes] == [
        ("category", "door"),
        ("color", "blue"),
        ("material", "metal"),
    ]
    fused = (UpstreamRecordRef(artifact_id="fusion-run-0001", record_id=str(fused_id(1))),)
    for attribute in state.attributes:
        assert attribute.attribute_origin is ContextAttributeOrigin.OBSERVED
        assert attribute.origin.kind is DerivationKind.MULTIVIEW_FUSED
        assert attribute.origin.derived_from == fused
        assert attribute.origin.policy == PolicyRef(policy_id="claim-attribute", version="v1")


def test_a_derived_attribute_the_belief_does_not_restate_reaches_the_map(tmp_path: Path) -> None:
    # Um atributo derivado como interaction_radius_m é uma propriedade semântica com regra
    # versionada e evidência própria. Descartá-lo obrigaria quem lê o mapa público a reabrir o run
    # de Entity Resolution só para saber que ele existe, e o mapa é o produto público.
    def build(manifest: GeometricMapArtifactManifest) -> dict[str, Entity]:
        return {"door": _entity_with_interaction_radius(tmp_path, manifest)}

    context_map, _ = _assembled(tmp_path, build)

    (entity,) = context_map.entities
    attributes = {item.name: item for item in entity.semantic_state.attributes}
    # `class` continua fora: a crença do mapa já carrega o mesmo rótulo em primary_label.
    assert "class" not in attributes
    radius = attributes["interaction_radius_m"]
    assert float(radius.value) > 0
    assert radius.origin.kind is DerivationKind.MULTIVIEW_FUSED
    assert radius.origin.policy == PolicyRef(
        policy_id="interaction-radius-from-factor", version="v1"
    )
    assert radius.derivation_id == InteractionRadiusPolicy().derivation_id
    assert radius.attribute_origin is ContextAttributeOrigin.DERIVED
    assert {ref.artifact_id for ref in radius.origin.derived_from} == {
        "fusion-run-0001",
        str(entity.geometry_refs[0].map_id),
        str(entity.member_entities[0].semantic_map_id),
    }
    assert context_map_from_record(context_map_to_record(context_map)) == context_map


def test_a_derived_attribute_cites_only_the_member_and_geometry_that_produced_it(
    tmp_path: Path,
) -> None:
    source: dict[str, Entity] = {}

    def build(manifest: GeometricMapArtifactManifest) -> dict[str, Entity]:
        radius_member = _entity_with_interaction_radius(tmp_path, manifest)
        plain_member = entity_at(
            "plain-door",
            (1.0, 0.0, 0.0),
            support_number=2,
            seconds=(20,),
            hypotheses=(),
            map_id=manifest.map_id,
            semantic_map_id=SEMANTIC_MAP_ID,
            first_index=100,
        )
        source.update(radius=radius_member, plain=plain_member)
        return source

    context_map, _ = _assembled(tmp_path, build, matches=(("radius", "plain"),))

    (resolved,) = context_map.entities
    radius = next(
        item for item in resolved.semantic_state.attributes if item.name == "interaction_radius_m"
    )
    semantic_records = {
        ref for ref in radius.origin.derived_from if ref.artifact_id == str(SEMANTIC_MAP_ID)
    }
    geometry_records = {
        ref
        for ref in radius.origin.derived_from
        if ref.artifact_id == str(source["radius"].geometry.geometric_map_id)
    }
    assert semantic_records == {
        UpstreamRecordRef(
            artifact_id=str(source["radius"].semantic_map_id),
            record_id=str(source["radius"].entity_id),
        )
    }
    assert geometry_records == {
        UpstreamRecordRef(artifact_id=str(ref.map_id), record_id=str(ref.geometry_id))
        for ref in source["radius"].geometry.geometry_refs
    }


def test_a_derived_attribute_whose_provenance_cannot_be_reconstructed_is_refused(
    tmp_path: Path,
) -> None:
    # A regra da política v2 não é "aceitar DERIVED", e sim "carregar o derivado cuja proveniência
    # a montagem consegue reconstruir". Qualquer atributo derivado que não chegue aos claims e à
    # evidência fundida que o originaram é recusado, nunca carregado sem grounding nem descartado
    # em silêncio -- o que vale para qualquer atributo novo, não só os de hoje.
    def build(manifest: GeometricMapArtifactManifest) -> dict[str, Entity]:
        entity = _entity(manifest, "door", 1, 10, (BLUE_METAL_DOOR,), center=(0.0, 0.0, 0.0))
        state = entity.semantic_state
        restated = next(item for item in state.attributes if item.name == "class")
        untraceable = dataclasses.replace(
            restated,
            name="derived_property",
            value="1.5",
            derivation_id="some-derivation-v1",
            evidence=tuple(
                dataclasses.replace(reference, claim_id=ClaimId("claim-no-hypothesis-lists"))
                for reference in restated.evidence
            ),
        )
        attributes = tuple(
            sorted(
                (*state.attributes, untraceable),
                key=lambda item: (item.name, item.value, item.origin.value, item.derivation_id),
            )
        )
        return {
            "door": dataclasses.replace(
                entity, semantic_state=dataclasses.replace(state, attributes=attributes)
            )
        }

    with pytest.raises(ProvenanceError, match="derived_property"):
        _assembled(tmp_path, build)


def test_an_external_knowledge_attribute_is_excluded_by_the_schema_s_own_rule(
    tmp_path: Path,
) -> None:
    # Conhecimento externo não é evidência sobre este mapa e pode chegar sem nenhuma citação,
    # enquanto um EvidenceOrigin exige ao menos uma: a exclusão é uma decisão nomeada, não um
    # descarte acidental.
    def build(manifest: GeometricMapArtifactManifest) -> dict[str, Entity]:
        entity = _entity(manifest, "door", 1, 10, (BLUE_METAL_DOOR,), center=(0.0, 0.0, 0.0))
        state = entity.semantic_state
        imagined = EntityAttribute(
            name="usual_use",
            value="storage",
            origin=AttributeOrigin.EXTERNAL_KNOWLEDGE,
            derivation_id="external-knowledge-v1",
            evidence=(),
            external_source=ExternalKnowledgeSource(
                source_id="warehouse-ontology", source_version="2026.1", entry_id="usual_use"
            ),
        )
        attributes = tuple(
            sorted(
                (*state.attributes, imagined),
                key=lambda item: (item.name, item.value, item.origin.value, item.derivation_id),
            )
        )
        return {
            "door": dataclasses.replace(
                entity, semantic_state=dataclasses.replace(state, attributes=attributes)
            )
        }

    context_map, _ = _assembled(tmp_path, build)

    (entity,) = context_map.entities
    assert "usual_use" not in {item.name for item in entity.semantic_state.attributes}


def test_divergent_stated_values_both_reach_the_map_and_none_wins(tmp_path: Path) -> None:
    context_map, _ = _assembled(
        tmp_path,
        lambda manifest: {
            "a": _entity(
                manifest,
                "a",
                1,
                10,
                (ClaimSpec("door", attributes=(("color", "blue"),)),),
                center=(0.0, 0.0, 0.0),
            ),
            "b": _entity(
                manifest,
                "b",
                2,
                20,
                (ClaimSpec("door", attributes=(("color", "gray"),)),),
                center=(0.2, 0.0, 0.0),
            ),
        },
        matches=(("a", "b"),),
    )

    (entity,) = context_map.entities
    colors = [item.value for item in entity.semantic_state.attributes if item.name == "color"]
    assert colors == ["blue", "gray"]


def test_equal_stated_values_from_distinct_derivations_reach_the_map(tmp_path: Path) -> None:
    def build(manifest: GeometricMapArtifactManifest) -> dict[str, Entity]:
        members: dict[str, Entity] = {}
        for index, name in enumerate(("a", "b"), start=1):
            member = _entity(
                manifest,
                name,
                index,
                index * 10,
                (ClaimSpec("door", attributes=(("color", "blue"),)),),
                center=(float(index), 0.0, 0.0),
            )
            attributes = tuple(
                dataclasses.replace(
                    attribute,
                    derivation_id=f"claim-attribute-v1@sha256:{index:064x}",
                )
                if attribute.name == "color"
                else attribute
                for attribute in member.semantic_state.attributes
            )
            members[name] = dataclasses.replace(
                member,
                semantic_state=dataclasses.replace(member.semantic_state, attributes=attributes),
            )
        return members

    context_map, _ = _assembled(tmp_path, build, matches=(("a", "b"),))

    (resolved,) = context_map.entities
    colors = [item for item in resolved.semantic_state.attributes if item.name == "color"]
    assert [(item.value, item.derivation_id) for item in colors] == [
        ("blue", f"claim-attribute-v1@sha256:{index:064x}") for index in (1, 2)
    ]
    assert context_map_from_record(context_map_to_record(context_map)) == context_map


# --- belief (C9) ---------------------------------------------------------------------------------


def test_one_frame_gives_tentative_entities_and_no_primary_label(tmp_path: Path) -> None:
    """B1 regression: every entity of a single frame has one physical observation."""
    context_map, _ = _assembled(
        tmp_path,
        lambda manifest: {
            name: _entity(
                manifest,
                name,
                number,
                55,
                claims,
                center=(20.0 * number, 0.0, 0.0),
            )
            for number, (name, claims) in enumerate(
                (
                    ("door", (ClaimSpec("door"),)),
                    ("pole", (ClaimSpec("metal pole"),)),
                    ("ambiguous", (ClaimSpec("door"), ClaimSpec("wall"))),
                ),
                start=1,
            )
        },
    )

    states = [item.semantic_state for item in context_map.entities]
    assert [state.support_status for state in states] == [ContextSupportStatus.TENTATIVE] * 3
    assert all(state.primary_label is None for state in states)
    assert {
        hypothesis.supporting_physical_observation_count
        for state in states
        for hypothesis in state.hypotheses
    } == {1}


def test_two_frames_with_the_same_label_give_a_supported_primary_label(tmp_path: Path) -> None:
    context_map, _ = _assembled(
        tmp_path,
        lambda manifest: {
            "a": _entity(manifest, "a", 1, 10, (ClaimSpec("door"),), center=(0.0, 0.0, 0.0)),
            "b": _entity(manifest, "b", 2, 20, (ClaimSpec("door"),), center=(0.2, 0.0, 0.0)),
        },
        matches=(("a", "b"),),
    )

    (entity,) = context_map.entities
    assert entity.semantic_state.support_status is ContextSupportStatus.SUPPORTED
    assert entity.semantic_state.primary_label == "door"
    (door,) = entity.semantic_state.hypotheses
    assert door.supporting_physical_observation_count == 2


def test_belief_and_attributes_survive_the_artifact_and_its_full_validation(
    tmp_path: Path,
) -> None:
    context_map, locations = _assembled(
        tmp_path,
        lambda manifest: {
            "a": _entity(manifest, "a", 1, 10, (BLUE_METAL_DOOR,), center=(0.0, 0.0, 0.0)),
            "b": _entity(manifest, "b", 2, 20, (BLUE_METAL_DOOR,), center=(0.2, 0.0, 0.0)),
        },
        matches=(("a", "b"),),
    )
    output_dir = tmp_path / "out" / "context_map"

    ContextMapArtifactWriter(output_dir=output_dir).write(context_map, upstream_locations=locations)

    with ContextMapArtifactReader.open(output_dir) as reader:
        reopened = reader.context_map()
    assert reopened == context_map
    (entity,) = reopened.entities
    assert entity.semantic_state.primary_label == "blue metal door"
    assert len(entity.semantic_state.attributes) == 3
    report = validate_context_map_artifact(output_dir, dependency_paths=locations)
    assert report.status is ValidationStatus.VERIFIED, report


def test_without_a_belief_the_semantic_state_is_written_as_before(tmp_path: Path) -> None:
    context_map, _ = _assembled(
        tmp_path,
        lambda manifest: {
            "a": _entity(manifest, "a", 1, 10, (ClaimSpec("door"),), center=(0.0, 0.0, 0.0))
        },
        belief=None,
    )

    (entity,) = context_map_to_record(context_map)["entities"]
    assert set(entity["semantic_state"]) == {"status", "hypotheses"}
    assert set(entity["semantic_state"]["hypotheses"][0]) == {"label", "origin"}
