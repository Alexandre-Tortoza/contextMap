"""The arm matrix of the perception pilot (#665) and the checks its configurations must pass.

Seven arms, each its own context branch and its own ContextMap over one spatial foundation:
the full 2 x 3 factorial of region discovery (SAM2 automatic, SAM3 with a concept set) and
region prompt (``region/v1``, ``region-abstention/v1``, ``region-scene-context/v1``), plus a
replica of the baseline, a separate execution of the same configuration that measures the
noise floor of the semantic backend.

An arm's configuration is the base, then the layers every arm shares, then the arm's own layer
(the runtime's layer precedence). Before anything is created, every arm must resolve through
the runtime, keep no template placeholder, and differ from every other arm only where the
variables the two assign differently allow it (``contextmap.evaluation.arm_differences``,
field by field).

The checks take the arms, the variables and the baseline as parameters, the pilot's by default:
the capability ablation campaign (#670) runs them over its own matrix.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

from contextmap.evaluation import (
    ArmDifference,
    ExperimentArm,
    ExperimentVariable,
    ResolvedTopology,
    StageImplementation,
    TopologyStage,
    VariationKind,
    arm_differences,
)
from contextmap.runtime import CONFIG_SCHEMA_VERSION, EffectiveConfig, resolve_effective_config

BASELINE = "sam2-region-v1"
REPLICA = "sam2-region-v1-replica"
PLACEHOLDER_MARKER = "REPLACE"
"""What every value the templates leave to the GPU machine contains."""

_BACKEND_VERSION = f"runtime-config/{CONFIG_SCHEMA_VERSION}"
COMPARED_SECTIONS = ("pipeline", "inputs", "policies")
"""Sections compared besides the components. ``resources`` is deployment (which process loads
which model, on which device, into which workspace), never a scientific parameter: a SAM3 arm
names another region-discovery provider than a SAM2 one."""


class ControlledArm(Protocol):
    """An arm :func:`check_controlled` compares: its identity and the level of each variable."""

    @property
    def slug(self) -> str:
        """Identity of the arm, the key of its resolved configuration."""
        ...

    @property
    def assignments(self) -> tuple[tuple[str, str], ...]:
        """Return ``(variable, level)`` for every variable of the comparison."""
        ...


@dataclass(frozen=True, kw_only=True)
class PilotArm:
    """One arm of the pilot.

    Attributes:
        slug: Identity of the arm, the name of its layer and of its branch.
        discovery: Level of the ``region_discovery`` variable.
        region_prompt: Level of the ``region_prompt`` variable: the region template.
        role: ``baseline``, ``variant`` or ``replica``.
    """

    slug: str
    discovery: str
    region_prompt: str
    role: str

    @property
    def assignments(self) -> tuple[tuple[str, str], ...]:
        """Return ``(variable, level)`` for both variables."""
        return (("region_discovery", self.discovery), ("region_prompt", self.region_prompt))


ARMS = (
    PilotArm(slug=BASELINE, discovery="sam2-automatic", region_prompt="region/v1", role="baseline"),
    PilotArm(
        slug="sam2-region-abstention-v1",
        discovery="sam2-automatic",
        region_prompt="region-abstention/v1",
        role="variant",
    ),
    PilotArm(
        slug="sam2-region-scene-context-v1",
        discovery="sam2-automatic",
        region_prompt="region-scene-context/v1",
        role="variant",
    ),
    PilotArm(
        slug="sam3-region-v1",
        discovery="sam3-concept-set",
        region_prompt="region/v1",
        role="variant",
    ),
    PilotArm(
        slug="sam3-region-abstention-v1",
        discovery="sam3-concept-set",
        region_prompt="region-abstention/v1",
        role="variant",
    ),
    PilotArm(
        slug="sam3-region-scene-context-v1",
        discovery="sam3-concept-set",
        region_prompt="region-scene-context/v1",
        role="variant",
    ),
    PilotArm(slug=REPLICA, discovery="sam2-automatic", region_prompt="region/v1", role="replica"),
)
"""Every arm, in execution order: the baseline first, its replica last."""

VARIABLES = (
    ExperimentVariable(
        name="region_discovery",
        kind=VariationKind.BACKEND,
        touches=("region_perception.discovery",),
        values=("sam2-automatic", "sam3-concept-set"),
        baseline_value="sam2-automatic",
        description="SAM2 automatic masks against SAM3 over one concept set in one run (#662)",
        configuration_fields=("backend", "sam2", "sam3"),
    ),
    ExperimentVariable(
        name="region_prompt",
        kind=VariationKind.POLICY,
        touches=("semantic_interpretation.interpreter",),
        values=("region/v1", "region-abstention/v1", "region-scene-context/v1"),
        baseline_value="region/v1",
        description="the instruction template of region requests, with scene context for #529",
        configuration_fields=(
            "qwen.prompt_policy.region",
            "qwen.prompt_policy.region_scene_context",
        ),
    ),
)


class ArmConfigurationError(ValueError):
    """The arms cannot run as a controlled comparison; nothing was created.

    Attributes:
        problems: One sentence per problem, naming the arm.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        """Keep every problem, not only the first."""
        self.problems = tuple(problems)
        super().__init__("\n".join(("the arms cannot run:", *(f"- {p}" for p in problems))))


def arm_layers(arms_dir: Path, names: Mapping[str, str] | None = None) -> dict[str, Path]:
    """Return the layer of every arm, a file of ``arms_dir``.

    Args:
        arms_dir: The directory of the filled layers.
        names: The file name of each arm's layer, by arm; each pilot arm's ``<slug>.json`` when
            omitted.

    Raises:
        ArmConfigurationError: If an arm has no layer there.
    """
    if names is None:
        names = {arm.slug: f"{arm.slug}.json" for arm in ARMS}
    layers = {slug: arms_dir / name for slug, name in names.items()}
    missing = [
        f"{slug}: no layer at {path.name}" for slug, path in layers.items() if not path.is_file()
    ]
    if missing:
        raise ArmConfigurationError(missing)
    return layers


def resolve_arm(base: Path, shared: Sequence[Path], layer: Path) -> EffectiveConfig:
    """Resolve one arm through the runtime: the base, the shared layers, then its own.

    Raises:
        ConfigurationError: If a layer cannot be read or the merged document is invalid.
    """
    return resolve_effective_config(files=[base, *shared, layer])


def refuse_placeholders(effectives: Mapping[str, EffectiveConfig]) -> None:
    """Refuse every arm whose resolved configuration still holds a template placeholder.

    A placeholder left as text can compose (a SAM model version only has to be non-empty) and
    would then be recorded in provenance as if it were the machine's value.

    Raises:
        ArmConfigurationError: Naming each arm, path and placeholder.
    """
    problems = [
        f"{slug}: {path} is still the template placeholder {value!r}"
        for slug, effective in effectives.items()
        for path, value in _leaves(effective.config.to_document())
        if isinstance(value, str) and PLACEHOLDER_MARKER in value
    ]
    if problems:
        raise ArmConfigurationError(problems)


def check_controlled(
    effectives: Mapping[str, EffectiveConfig],
    *,
    arms: Sequence[ControlledArm] = ARMS,
    variables: tuple[ExperimentVariable, ...] = VARIABLES,
    baseline: str = BASELINE,
) -> dict[str, tuple[ArmDifference, ...]]:
    """Check every pair of arms is a controlled comparison; return each arm's baseline diff.

    Every pair is checked, not only each arm against the baseline, so two arms of one discovery
    level (the SAM3 arms, whose concept set is written in each layer) cannot drift apart.

    Args:
        effectives: The resolved configuration of every arm, by arm.
        arms: The arms compared and the level each assigns; the pilot's by default.
        variables: The variables those levels belong to; the pilot's by default.
        baseline: The arm every other one is reported against; the pilot's by default.

    Returns:
        By arm, its differences from the baseline, every one declared by a variable.

    Raises:
        ArmConfigurationError: If two arms differ in anything no variable they assign differently
            declares, or an arm assigns a level its configuration does not show.
    """
    compared = {
        arm.slug: ExperimentArm(
            arm_id=arm.slug,
            assignments=arm.assignments,
            topology=arm_topology(effectives[arm.slug]),
        )
        for arm in arms
    }
    problems: list[str] = []
    for first, second in combinations(arms, 2):
        found = arm_differences(variables, compared[first.slug], compared[second.slug])
        problems += [
            f"{second.slug} differs from {first.slug} in {item.describe()}, which no variable "
            "they assign differently declares"
            for item in found
            if not item.declared
        ]
        witnessed = {name for item in found for name in item.declared_by}
        problems += [
            f"{second.slug} assigns {name} = {level!r} and {first.slug} "
            f"{dict(first.assignments)[name]!r}, but their configurations do not differ there"
            for name, level in second.assignments
            if level != dict(first.assignments)[name] and name not in witnessed
        ]
    if problems:
        raise ArmConfigurationError(problems)
    return {
        slug: arm_differences(variables, compared[baseline], compared[slug]) for slug in compared
    }


def arm_topology(effective: EffectiveConfig) -> ResolvedTopology:
    """Record an arm's resolved configuration as nodes whose fields ``arm_differences`` compares.

    One node per component (its backend and that backend's parameters, as the runtime keeps
    them), and one per compared section. Dependencies are not modeled: every arm runs the same
    ``canonical/1`` plan, and a stage toggled by a layer shows in the ``pipeline`` node.
    """
    document = effective.config.to_document()
    stages = [
        _node(f"{capability}.{slot}", capability, entry.get("backend") or "unselected", entry)
        for capability, slots in document["components"].items()
        for slot, entry in slots.items()
    ]
    stages += [
        _node(section, "runtime", section, document[section]) for section in COMPARED_SECTIONS
    ]
    return ResolvedTopology(stages=tuple(stages))


def _node(stage_id: str, capability: str, backend: str, configuration: Any) -> TopologyStage:
    """Return one node recording ``configuration`` field by field."""
    return TopologyStage(
        stage_id=stage_id,
        capability=capability,
        implementation=StageImplementation.from_configuration(
            backend_id=backend,
            backend_version=_BACKEND_VERSION,
            model=None,
            configuration=configuration,
        ),
    )


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted path, leaf)`` of a JSON document; list items are indexed ``[i]``."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    else:
        yield path, value
