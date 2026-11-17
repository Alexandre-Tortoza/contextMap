"""``region-interaction/v2`` and ``v3``: the interaction factor, shown where the schema accepts it.

Measured on the 10:31 frame with ``region-interaction/v1``: Qwen put ``interaction_factor`` next to
``attributes`` instead of inside it in 31 of 61 regions, and the strict parser rejected each of
those claims whole. v2 showed the claim's shape, with placeholders instead of example values so no
number is suggested; Qwen then returned that bare claim as the whole response, without the
``abstained``/``claims`` envelope, in 59 of 61 regions. v3 shows the whole response, and its
example, once the placeholders are filled in, is a response the strict parser accepts.
"""

from __future__ import annotations

import re

from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import RegionId
from contextmap.semantic_interpretation import (
    SEMANTIC_PROMPT_TEMPLATES,
    SemanticConfidencePolicy,
    SemanticInferenceProvenance,
    SemanticInterpretationMode,
    SemanticInterpretationRequest,
    SemanticInterpretationRunId,
    SemanticRequestId,
    SemanticVisualView,
    VisualViewKind,
    parse_semantic_response,
)
from contextmap.semantic_interpretation.prompt_policy import (
    SEMANTIC_RESPONSE_SCHEMA,
)
from contextmap.shared import BackendProvenance

V3 = "region-interaction/v3"
_PLACEHOLDER_VALUES = {
    '"<bare noun>"': '"door"',
    '"<coarse class or null>"': "null",
    '"<thing, stuff or null>"': '"thing"',
    '"<color>"': '"gray"',
    "<number>": "1.0",
}


def test_the_v2_template_is_a_region_template_of_the_same_response_schema() -> None:
    template = SEMANTIC_PROMPT_TEMPLATES["region-interaction/v2"]

    assert template.mode is SemanticInterpretationMode.REGION
    assert template.output_schema_version == SEMANTIC_RESPONSE_SCHEMA


def test_the_v2_template_places_the_factor_inside_attributes_without_suggesting_a_value() -> None:
    text = SEMANTIC_PROMPT_TEMPLATES["region-interaction/v2"].instructions

    assert '"attributes": {' in text and '"interaction_factor": <number>' in text
    assert "never next to" in text
    for anchored in ("0.8", "1.5", "2.0"):
        assert anchored not in text


def test_v1_is_kept_verbatim() -> None:
    assert "interaction_factor" in SEMANTIC_PROMPT_TEMPLATES["region-interaction/v1"].instructions
    assert "<number>" not in SEMANTIC_PROMPT_TEMPLATES["region-interaction/v1"].instructions


def _request() -> SemanticInterpretationRequest:
    region_id = RegionId("region-0007")
    return SemanticInterpretationRequest(
        request_id=SemanticRequestId("region-request-0001"),
        source_observation_id=SourceObservationId("frame-0124"),
        semantic_interpretation_run_id=SemanticInterpretationRunId("run-0001--frame-0124"),
        mode=SemanticInterpretationMode.REGION,
        region_id=region_id,
        visual_views=(
            SemanticVisualView(
                view_id="view-0001",
                kind=VisualViewKind.MASK_OUTLINE_CROP,
                payload_reference="outputs/semantic-views/input.png",
                source_observation_id=SourceObservationId("frame-0124"),
                region_id=region_id,
                sha256="0" * 64,
            ),
        ),
        prompt_template_id=V3,
        requested_output_schema=SEMANTIC_RESPONSE_SCHEMA,
        configuration_fingerprint="sha256:config",
    )


def _provenance() -> SemanticInferenceProvenance:
    return SemanticInferenceProvenance(
        backend=BackendProvenance(
            backend_id="fake",
            capability="semantic_interpreter",
            provider="fake",
            model="fake",
            version="1",
            configuration_fingerprint="sha256:config",
        ),
        task_identity="region-interpretation",
        prompt_template_id=V3,
        output_schema_version=SEMANTIC_RESPONSE_SCHEMA,
        raw_response_reference="__DEBUG__/raw-response.txt",
    )


def _example_response(instructions: str) -> str:
    """The response the instructions show, from its opening brace to the matching close."""
    start = instructions.index('{"abstained": false')
    depth = 0
    for end, character in enumerate(instructions[start:], start):
        depth += {"{": 1, "}": -1}.get(character, 0)
        if depth == 0:
            return instructions[start : end + 1]
    raise AssertionError("the example response is not closed")


def test_the_v3_template_is_a_region_template_of_the_same_response_schema() -> None:
    template = SEMANTIC_PROMPT_TEMPLATES[V3]

    assert template.mode is SemanticInterpretationMode.REGION
    assert template.output_schema_version == SEMANTIC_RESPONSE_SCHEMA


def test_the_v3_example_is_the_whole_response_and_suggests_no_value() -> None:
    example = _example_response(SEMANTIC_PROMPT_TEMPLATES[V3].instructions)

    assert set(re.findall(r"<[^<>]+>", example)) == {
        "<bare noun>",
        "<coarse class or null>",
        "<thing, stuff or null>",
        "<color>",
        "<number>",
    }
    assert re.search(r"\d", example) is None


def test_the_v3_example_filled_in_is_a_response_the_strict_parser_accepts() -> None:
    example = _example_response(SEMANTIC_PROMPT_TEMPLATES[V3].instructions)
    for placeholder, value in _PLACEHOLDER_VALUES.items():
        example = example.replace(placeholder, value)

    parsed = parse_semantic_response(
        example,
        _request(),
        _provenance(),
        confidence_policy=SemanticConfidencePolicy.UNSCORED_ONLY,
    )

    (claim,) = parsed.claims
    assert claim.hypothesis == "door"
    assert {a.name: a.value for a in claim.attributes}["interaction_factor"] == 1.0
