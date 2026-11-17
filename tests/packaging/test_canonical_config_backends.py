"""The canonical configuration examples against what each visual-perception backend accepts.

``examples/v0.1.0/canonical-config.toml`` resolves (``contextmap inspect config``), but its
visual-perception blocks name the model with ``model``, a parameter SAM2, DINOv2 and CLIP do not
have, and leave out parameters they and Qwen require: the stage cannot be composed from it
(#643). The file stays frozen as the record of what v0.1.0 shipped, and the erratum lives in
``docs/releases/v0.1.0.md``; ``examples/v0.2.0`` is the corrected template for the next release.

The template fills in only values the repository records. Every other value is a
``{ PLACEHOLDER = "<NAME>" }`` table, which no backend parameter accepts, so each one alone keeps
the stage from being composed -- even where a placeholder text would pass, since SAM2
``model_version`` and Qwen ``precision`` only have to be non-empty. The tests fill them as a
user can, with ``--set``-style overrides, using values that are only syntactically valid.

Composing validates each backend's parameters with the backend's own configuration class and
constructs the adapters. Every model runtime is a fake and nothing is loaded, so this runs on a
base install.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from contextmap.region_perception.adapters.sam2 import Sam2RegionDiscovery
from contextmap.runtime import resolve_effective_config
from contextmap.runtime.composition import ComposedRuntime, compose
from contextmap.runtime.configuration import ConfigurationError
from contextmap.runtime.errors import BackendConfigurationError
from contextmap.semantic_interpretation import (
    SemanticPromptPolicy,
    SemanticRequestPolicy,
    SemanticViewPolicy,
    VisualViewKind,
)
from contextmap.semantic_interpretation.adapters.qwen import (
    QwenSemanticInterpreter,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
FROZEN = EXAMPLES / "v0.1.0" / "canonical-config.toml"
TEMPLATE = EXAMPLES / "v0.2.0" / "canonical-config.toml"
TEMPLATE_README = EXAMPLES / "v0.2.0" / "README.md"

REGION_DISCOVERY = "region_perception.discovery"
DENSE_FEATURES = "visual_features.dense"
REGION_FEATURES = "visual_features.region"
SEMANTIC_INTERPRETATION = "semantic_interpretation.interpreter"
VISUAL_PERCEPTION = (REGION_DISCOVERY, DENSE_FEATURES, REGION_FEATURES, SEMANTIC_INTERPRETATION)
REQUEST_POLICY_GROUPS = frozenset({"prompt_policy", "view_policy"})

# Como a v0.1.0 nomeava cada componente: o arquivo congelado é um artifact histórico, anterior
# à decomposição, e seus ids não mudam porque o catálogo mudou.
FROZEN_COMPONENT_IDS = {
    REGION_DISCOVERY: "visual_perception.region_discovery",
    DENSE_FEATURES: "visual_perception.dense_features",
    REGION_FEATURES: "visual_perception.region_features",
    SEMANTIC_INTERPRETATION: "visual_perception.semantic_interpretation",
}

# Um valor por placeholder do modelo, só sintaticamente válido. Nenhum é recomendação: os valores
# reais não estão registrados no repositório (examples/v0.2.0/README.md).
FILL: dict[str, object] = {
    "components.region_perception.discovery.sam2.model_version": "test-model-version",
    "components.visual_features.dense.dinov2.revision": "a" * 40,
    "components.visual_features.region.clip.revision": "b" * 40,
    "components.semantic_interpretation.interpreter.qwen.revision": "c" * 40,
    "components.semantic_interpretation.interpreter.qwen.precision": "bfloat16",
    "components.semantic_interpretation.interpreter.qwen.max_new_tokens": 1,
}


def _fake_runtime(config: object, secrets: object) -> object:
    """Stand in for a model runtime: composing constructs the adapters and never runs them."""
    return object()


def _compose_visual_perception(
    files: Sequence[Path], values: Mapping[str, object] | None = None
) -> ComposedRuntime:
    """Compose the perception stages, with ``values`` set as ``--set`` would.

    Os três produtores independentes compõem os mesmos quatro backends da amostra.
    """
    overrides = [f"{path}={json.dumps(value)}" for path, value in (values or {}).items()]
    effective = resolve_effective_config(files=files, overrides=overrides)
    return compose(
        effective,
        stages=["region_perception", "visual_perception", "semantic_interpretation"],
        providers={component_id: _fake_runtime for component_id in VISUAL_PERCEPTION},
        environ={},
    )


def _placeholders(node: object, path: tuple[str, ...] = ()) -> dict[str, str]:
    """Map the dotted path of every ``{ PLACEHOLDER = "<NAME>" }`` table to its name."""
    if not isinstance(node, dict):
        return {}
    if set(node) == {"PLACEHOLDER"}:
        return {".".join(path): str(node["PLACEHOLDER"])}
    found: dict[str, str] = {}
    for key, value in node.items():
        found.update(_placeholders(value, (*path, key)))
    return found


def _component_block(document: Mapping[str, object], component_id: str) -> dict[str, object]:
    """Return a component's selection and its selected backend's block, as the file has them."""
    capability, slot = component_id.split(".")
    components = document["components"]
    assert isinstance(components, dict)
    component = components[capability][slot]
    backend = component["backend"]
    return {"backend": backend, backend: component[backend]}


def _refusals(error: BackendConfigurationError, prefix: str) -> list[str]:
    """Return the problems a backend reported that start with ``prefix``."""
    return [problem for problem in error.problems if problem.startswith(prefix)]


def test_every_value_the_template_leaves_open_is_a_named_placeholder_its_readme_lists() -> None:
    placeholders = _placeholders(tomllib.loads(TEMPLATE.read_text(encoding="utf-8")))

    assert placeholders.keys() == FILL.keys()
    readme = TEMPLATE_README.read_text(encoding="utf-8")
    assert [name for name in placeholders.values() if f"`{name}`" not in readme] == []


def test_the_filled_template_composes_every_visual_perception_backend() -> None:
    composed = _compose_visual_perception([TEMPLATE], FILL)

    assert composed.stages == (
        "region_perception",
        "visual_perception",
        "semantic_interpretation",
    )
    # Discovery e interpretação são construídos por run, sobre o rascunho dele (#661).
    assert composed.region_discovery is not None
    assert isinstance(composed.region_discovery.factory(Path("run")), Sam2RegionDiscovery)
    assert composed.dense_features is not None
    assert composed.region_features is not None
    assert composed.semantic_interpreter is not None
    assert isinstance(composed.semantic_interpreter(Path("run")), QwenSemanticInterpreter)
    # A política de request explícita que reproduz o request da v0.1.0: os prompts anteriores à
    # #542 e uma única view, o recorte justo da região, anterior à #524.
    assert composed.semantic_request_policy == SemanticRequestPolicy.from_prompt_policy(
        SemanticPromptPolicy(scene="scene/v1", region="region/v1"),
        views=SemanticViewPolicy(region_views=(VisualViewKind.TIGHT_CROP,)),
    )


@pytest.mark.parametrize("placeholder", sorted(FILL))
def test_each_placeholder_left_as_committed_keeps_the_stage_from_composing(
    placeholder: str,
) -> None:
    """The template cannot be run by accident, not even with all but one placeholder filled."""
    others = {path: value for path, value in FILL.items() if path != placeholder}

    with pytest.raises(BackendConfigurationError) as refused:
        _compose_visual_perception([TEMPLATE], others)

    segments = placeholder.split(".")
    assert refused.value.component_id == ".".join(segments[1:3])
    assert _refusals(refused.value, f"{segments[-1]}:")


def test_the_frozen_v0_1_0_canonical_configuration_cannot_compose_visual_perception() -> None:
    """The erratum of docs/releases/v0.1.0.md, on the unmodified file.

    Since the decomposition the file is refused one layer earlier than the erratum describes:
    it names the capability ``visual_perception`` for components the catalog now splits across
    ``region_perception``, ``visual_features`` and ``semantic_interpretation``, so resolution
    rejects it before any backend sees its parameters. The erratum itself — sam2 refusing
    ``model`` — stays covered by
    :func:`test_each_frozen_v0_1_0_visual_perception_block_is_refused_by_its_backend`, which
    transplants each frozen block into the current template under the id that owns it.
    """
    with pytest.raises(ConfigurationError) as refused:
        _compose_visual_perception([FROZEN])

    assert "components.visual_perception: unknown capability" in str(refused.value)


@pytest.mark.parametrize(
    ("component_id", "problem"),
    [
        (REGION_DISCOVERY, "model: unknown parameter"),
        (DENSE_FEATURES, "model: unknown parameter"),
        (REGION_FEATURES, "model: unknown parameter"),
        (
            SEMANTIC_INTERPRETATION,
            "missing required parameter(s): precision, max_new_tokens, temperature",
        ),
    ],
)
def test_each_frozen_v0_1_0_visual_perception_block_is_refused_by_its_backend(
    component_id: str, problem: str
) -> None:
    """Each frozen block, moved alone into the filled template, is refused by its own backend.

    The semantic request policy groups did not exist in v0.1.0 and have had no default since
    #542/#524, so the frozen semantic block takes the template's: the refusal is then the defect
    the erratum names, not the groups added after the release.
    """
    frozen = tomllib.loads(FROZEN.read_text(encoding="utf-8"))
    block = _component_block(frozen, FROZEN_COMPONENT_IDS[component_id])
    if component_id == SEMANTIC_INTERPRETATION:
        template = _component_block(
            tomllib.loads(TEMPLATE.read_text(encoding="utf-8")), component_id
        )
        backend = str(block["backend"])
        frozen_parameters, template_parameters = block[backend], template[backend]
        assert isinstance(frozen_parameters, dict) and isinstance(template_parameters, dict)
        assert not REQUEST_POLICY_GROUPS & frozen_parameters.keys()
        block[backend] = {
            **frozen_parameters,
            **{group: template_parameters[group] for group in REQUEST_POLICY_GROUPS},
        }
    transplant = {
        **{path: value for path, value in FILL.items() if f".{component_id}." not in path},
        f"components.{component_id}": block,
    }

    with pytest.raises(BackendConfigurationError) as refused:
        _compose_visual_perception([TEMPLATE], transplant)

    assert refused.value.component_id == component_id
    assert _refusals(refused.value, problem)
