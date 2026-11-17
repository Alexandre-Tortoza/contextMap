"""The arm matrix of the capability ablation campaign (#670), read from ``config/campaign.json``.

The matrix is data: the variables, and for each arm its role, group, the level of each variable
it changes, its configuration layer, how it runs and which arm's perception it reuses, plus the
parameters of the new axes with the origin of each value. Reading it checks what the harness
relies on: the baseline first, one replica, known variables and levels, and a reuse arm that
reuses an earlier arm that ran perception. The configuration checks are the pilot's
(``pilot_arms``): every pair of arms may differ only where a variable they assign differently
declares it; here every registered parameter must also be the value its arm resolves, and a
scorer must name the model of the region features it scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pilot_arms import ArmConfigurationError, check_controlled

from contextmap.evaluation import ArmDifference, ExperimentVariable
from contextmap.runtime import EffectiveConfig

MATRIX_SCHEMA = "contextmap.experiment.perception-ablation-campaign.matrix/v1"
EXECUTIONS = ("perception", "downstream_build", "association_rerun", "semantic_scoring")
"""How an arm runs.

``perception``: its own branch, a ContextRun over the selection and a build. The reuse arms run
over the perception run of another arm, never a new one: ``downstream_build`` builds that arm's
branch again under the arm's downstream layer (the #501 scenario-3 pattern);
``association_rerun`` associates that arm's perception run again over the foundation and
materializes a map in one runtime run; ``semantic_scoring`` scores that perception run in the
optional stage."""
ROLES = ("baseline", "arm", "replica")


class MatrixError(ValueError):
    """The matrix cannot drive a campaign; nothing was created.

    Attributes:
        problems: One sentence per problem.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        """Keep every problem, not only the first."""
        self.problems = tuple(problems)
        super().__init__(
            "\n".join(("the campaign matrix is invalid:", *(f"- {p}" for p in problems)))
        )


@dataclass(frozen=True, kw_only=True)
class CampaignArm:
    """One arm of the campaign.

    Attributes:
        slug: Identity of the arm, the name of its branch when it has one.
        role: ``baseline``, ``arm`` or ``replica``.
        group: The axis the arm belongs to in the protocol.
        assignments: ``(variable, level)`` for every variable, the baseline's level where the
            arm changes nothing.
        layer: File name of the arm's configuration layer.
        execution: One of :data:`EXECUTIONS`.
        reuse: The arm whose perception run this arm consumes, or ``None``.
    """

    slug: str
    role: str
    group: str
    assignments: tuple[tuple[str, str], ...]
    layer: str
    execution: str
    reuse: str | None

    def to_record(self) -> dict[str, Any]:
        """Return the protocol record of the arm."""
        return {
            "arm": self.slug,
            "role": self.role,
            "group": self.group,
            "assignments": dict(self.assignments),
            "layer": self.layer,
            "execution": self.execution,
            "reuse": self.reuse,
        }


@dataclass(frozen=True, kw_only=True)
class RegisteredParameter:
    """A value the protocol fixes for a new axis, where it is configured and where it came from.

    Attributes:
        name: Identity of the parameter.
        arm: The arm whose configuration holds it.
        path: Dotted path of the value in the arm's resolved configuration.
        value: The value, as JSON.
        origin: Where the value comes from.
    """

    name: str
    arm: str
    path: str
    value: Any
    origin: str

    def to_record(self) -> dict[str, Any]:
        """Return the protocol record of the parameter."""
        return {
            "name": self.name,
            "arm": self.arm,
            "path": self.path,
            "value": self.value,
            "origin": self.origin,
        }


@dataclass(frozen=True, kw_only=True)
class CampaignMatrix:
    """The whole matrix, in execution order.

    Attributes:
        sha256: SHA-256 of the matrix file's bytes.
        image_stride: One camera image in every ``image_stride`` of the window is perceived.
        voxel_script: The #624 driver of the voxel layers, relative to the checkout.
        voxel_cells_m: The voxel cell sizes, in meters.
        variables: The variables every arm assigns.
        arms: The arms, in execution order.
        parameters: The registered parameters of the new axes.
    """

    sha256: str
    image_stride: int
    voxel_script: str
    voxel_cells_m: tuple[float, ...]
    variables: tuple[ExperimentVariable, ...]
    arms: tuple[CampaignArm, ...]
    parameters: tuple[RegisteredParameter, ...]

    @property
    def baseline(self) -> str:
        """Return the baseline arm, the first one."""
        return self.arms[0].slug

    @property
    def replica(self) -> str:
        """Return the replica of the baseline, the one arm whose role is ``replica``."""
        return next(arm.slug for arm in self.arms if arm.role == "replica")

    def baseline_level(self, variable: str) -> str:
        """Return the level the baseline holds for a variable."""
        return next(item.baseline_value for item in self.variables if item.name == variable)

    def arm(self, slug: str) -> CampaignArm:
        """Return an arm by its slug."""
        return next(arm for arm in self.arms if arm.slug == slug)


def read_matrix(path: Path) -> CampaignMatrix:
    """Read and check the campaign matrix.

    Raises:
        MatrixError: If the matrix cannot be read or cannot drive a campaign, with every problem.
    """
    try:
        data = path.read_bytes()
        document = json.loads(data)
        if document.get("schema") != MATRIX_SCHEMA:
            raise MatrixError([f"{path.name} does not follow {MATRIX_SCHEMA!r}"])
        variables = tuple(ExperimentVariable.from_record(item) for item in document["variables"])
        baseline_levels = {variable.name: variable.baseline_value for variable in variables}
        arms = tuple(
            CampaignArm(
                slug=item["slug"],
                role=item["role"],
                group=item["group"],
                assignments=tuple({**baseline_levels, **item["assignments"]}.items()),
                layer=item["layer"],
                execution=item["execution"],
                reuse=item["reuse"],
            )
            for item in document["arms"]
        )
        parameters = tuple(RegisteredParameter(**item) for item in document["parameters"])
        matrix = CampaignMatrix(
            sha256=hashlib.sha256(data).hexdigest(),
            image_stride=int(document["selection"]["image_stride"]),
            voxel_script=str(document["voxel_layers"]["script"]),
            voxel_cells_m=tuple(float(cell) for cell in document["voxel_layers"]["cells_m"]),
            variables=variables,
            arms=arms,
            parameters=parameters,
        )
    except MatrixError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise MatrixError([f"cannot read {path.name}: {error!r}"]) from None
    problems = list(_problems(matrix))
    if problems:
        raise MatrixError(problems)
    return matrix


def check_matrix(
    matrix: CampaignMatrix, effectives: Mapping[str, EffectiveConfig]
) -> dict[str, tuple[ArmDifference, ...]]:
    """Check every pair of arms is a controlled comparison; return each arm's baseline diff.

    Raises:
        ArmConfigurationError: If two arms differ in anything no variable they assign differently
            declares, or an arm assigns a level its configuration does not show.
    """
    return check_controlled(
        effectives, arms=matrix.arms, variables=matrix.variables, baseline=matrix.baseline
    )


def check_parameters(
    matrix: CampaignMatrix, effectives: Mapping[str, EffectiveConfig]
) -> list[dict[str, Any]]:
    """Check every registered parameter is the value its arm resolves; return their records.

    A value the protocol records with an origin must be the one the arm runs with, or the
    record would cite an origin for a value nobody used.

    Raises:
        ArmConfigurationError: Naming each parameter whose arm resolves another value.
    """
    problems = []
    for parameter in matrix.parameters:
        document: Any = effectives[parameter.arm].config.to_document()
        for key in parameter.path.split("."):
            document = document.get(key) if isinstance(document, Mapping) else None
        if _canonical(document) != _canonical(parameter.value):
            problems.append(
                f"{parameter.arm}: {parameter.name} ({parameter.path}) resolves to "
                f"{_canonical(document)}, but the protocol registers {_canonical(parameter.value)}"
            )
    if problems:
        raise ArmConfigurationError(problems)
    return [parameter.to_record() for parameter in matrix.parameters]


def check_scorer_space(
    matrix: CampaignMatrix, effectives: Mapping[str, EffectiveConfig]
) -> dict[str, dict[str, Any]]:
    """Check every scoring arm's scorer names the model of the region features it scores.

    A scorer compares only in the embedding space of the region features of the perception run
    it scores, and refuses another one when the stage runs, late in the campaign. The space is
    the model's, so the scorer's backend, ``checkpoint`` and ``revision`` must be those of
    ``region_features`` in the reused arm's resolved configuration, before anything runs.

    Returns:
        By scoring arm, the backend, checkpoint and revision checked, and the reused arm.

    Raises:
        ArmConfigurationError: Naming each scoring arm, field and the two values that differ.
    """
    problems: list[str] = []
    checked: dict[str, dict[str, Any]] = {}
    for arm in matrix.arms:
        if arm.execution != "semantic_scoring" or arm.reuse is None:
            continue
        scorer = effectives[arm.slug].config.components["semantic_scoring.scorer"]
        features = effectives[arm.reuse].config.components["visual_features.region"]
        if scorer.backend != features.backend:
            problems.append(
                f"{arm.slug}: the scorer is {scorer.backend!r}, but the region features of "
                f"{arm.reuse} are {features.backend!r}"
            )
            continue
        for name in ("checkpoint", "revision"):
            mine, theirs = scorer.parameters.get(name), features.parameters.get(name)
            if mine != theirs:
                problems.append(
                    f"{arm.slug}: the scorer's {name} is {mine!r}, but the region features of "
                    f"{arm.reuse} were extracted with {theirs!r}"
                )
        checked[arm.slug] = {
            "backend": scorer.backend,
            "checkpoint": scorer.parameters.get("checkpoint"),
            "revision": scorer.parameters.get("revision"),
            "region_features_of": arm.reuse,
        }
    if problems:
        raise ArmConfigurationError(problems)
    return checked


def _problems(matrix: CampaignMatrix) -> Iterator[str]:
    """Yield every reason the matrix cannot drive a campaign."""
    slugs = [arm.slug for arm in matrix.arms]
    if len(set(slugs)) != len(slugs):
        yield f"an arm appears twice in {slugs}"
    roles = [arm.role for arm in matrix.arms]
    if not roles or roles[0] != "baseline":
        yield "the baseline must be the first arm"
    if roles.count("baseline") != 1 or roles.count("replica") != 1:
        yield "exactly one baseline and one replica are declared"
    levels = {variable.name: variable.values for variable in matrix.variables}
    ran: dict[str, str] = {}
    for arm in matrix.arms:
        if arm.role not in ROLES:
            yield f"{arm.slug}: unknown role {arm.role!r}"
        if arm.execution not in EXECUTIONS:
            yield f"{arm.slug}: unknown execution {arm.execution!r}"
        for name, level in arm.assignments:
            if name not in levels:
                yield f"{arm.slug}: assigns the unknown variable {name!r}"
            elif level not in levels[name]:
                yield f"{arm.slug}: {level!r} is not a level of {name}"
        if arm.execution == "perception" and arm.reuse is not None:
            yield f"{arm.slug}: an arm that runs perception reuses nothing"
        if arm.execution != "perception" and ran.get(str(arm.reuse)) != "perception":
            yield f"{arm.slug}: reuses {arm.reuse!r}, not an earlier arm that runs perception"
        if arm.execution != "perception" and arm.role != "arm":
            yield f"{arm.slug}: the {arm.role} runs its own perception"
        ran[arm.slug] = arm.execution
    for parameter in matrix.parameters:
        if parameter.arm not in ran:
            yield f"parameter {parameter.name}: unknown arm {parameter.arm!r}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
