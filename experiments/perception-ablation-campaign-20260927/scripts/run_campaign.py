"""Capability ablation campaign over a fixed spatial foundation.

One command runs the protocol registered in the issue (see the README) over one spatial
foundation and the window #669 ingested. Before anything is created it reads the matrix
(``config/campaign.json``), selects every sixth camera image of the window from the foundation's
sequence, checks every arm's configuration (resolution, template placeholders, only declared
differences, the registered parameters) and, for a real run, that it runs from a clean checkout
of the commit it records (``steps.resolve_code_identity``), then freezes ``protocol.json``.

It then derives the voxel layers of the foundation map once (the geometric part of the #624
driver, a measured child process) and runs the arms in the matrix's order, each command of the
``contextmap`` CLI in a fresh, measured process (the #501 steps):

- a ``perception`` arm creates its branch from the foundation, runs ``context run`` over the
  selection and ``context build``, as the pilot (#665) does;
- ``downstream_build`` builds the reused arm's branch again under the arm's downstream layer
  (the #501 scenario-3 pattern): no context stage runs;
- ``association_rerun`` runs ``contextmap run`` to ``context_map`` with the foundation and the
  reused arm's three evidence producers selected through an explicit catalog: association
  and every downstream stage are recomputed; the three evidence producers are reused;
- ``semantic_scoring`` runs ``contextmap stage semantic_scoring`` over the reused regional,
  visual and semantic producers.

Each reuse arm records the identities of its reused regional, visual and semantic producers,
and the run stops if they differ. As soon as an arm finishes, its entry is published to the
viewer index (``maps/<arm>.json`` and ``maps.json``, each written atomically). ``report.json``
holds every step, the pilot's annotation-free metrics per arm and the baseline-replica label
agreement. The run stops at the first step that fails and still writes a ``failed`` report;
only a ``complete`` report whose criterion passes, with exit status 0, is the campaign's evidence.
Nothing is overwritten: an existing run directory is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Os passos usam o harness incremental e as métricas vêm do piloto de ablação: ambos
# diretórios entram no caminho de import, como o piloto faz com os passos.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "incremental-partition-invariance-20260927"
        / "scripts"
    ),
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "perception-pilot-ablation-20260927" / "scripts")
)

from campaign_index import ViewerIndex
from campaign_matrix import (
    CampaignArm,
    CampaignMatrix,
    check_matrix,
    check_parameters,
    check_scorer_space,
    read_matrix,
)
from campaign_window import CorridorWindow, WindowSelection, read_window, select_images
from pilot_arms import COMPARED_SECTIONS, arm_layers, refuse_placeholders, resolve_arm
from pilot_metrics import REPLICA_JOIN, TOP_LABELS, perception_metrics, replica_agreement
from steps import (
    BUILD_STAGES,
    ContextCli,
    Portable,
    StepFailed,
    StepResult,
    build_record,
    configuration_record,
    directory_bytes,
    git_identity,
    map_location,
    relative,
    resolve_code_identity,
    step_record,
    unexpected_failure,
    write_once,
)

from contextmap.artifact import (
    ContextMapArtifactReader,
    Severity,
    ValidationLevel,
    validate_context_map_artifact,
)
from contextmap.evaluation import (
    DriverRun,
    ExperimentDriverError,
    measure_child_process,
    read_build_state,
)
from contextmap.runtime import (
    ArtifactRef,
    ConfigurationError,
    EffectiveConfig,
    SpatialFoundation,
    SpatialFoundationError,
    foundation_of_run,
    resolve_plan,
)
from contextmap.semantic_fusion import SemanticFusionRunReader
from contextmap.semantic_mapping import SemanticMappingRunReader
from contextmap.semantic_scoring import SemanticScoringRunReader

EXPERIMENT = "perception-ablation-campaign-20260927"
SCHEMA = "contextmap.experiment.perception-ablation-campaign/v2"
MAPS_SCHEMA = "contextmap.experiment.perception-ablation-campaign.maps/v2"
CHECKOUT = Path(__file__).resolve().parents[3]
MATRIX = Path(__file__).resolve().parents[1] / "config" / "campaign.json"
"""The matrix: data of the checkout, never of the filled configuration copy."""
_REF_FIELDS = ("stage_id", "contract", "artifact_id", "content_hash", "location")
_FOUNDATION_STAGES = {
    "ingestion": "sequence",
    "state_estimation": "state_estimation",
    "geometric_mapping": "geometry",
}
"""The foundation stage each foundation role fills, as a runtime run selects it."""


def main(argv: list[str] | None = None) -> int:
    """Run the campaign; 0 only when every arm finished and the criterion passes, 2 if refused."""
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve()
    base = args.base_config.resolve()
    shared = tuple(path.resolve() for path in args.all_arms_config or ())
    evidence_class = "real" if args.dry_run_executors is None else "dry_run"
    # Tudo que pode recusar a execução é conferido antes de criar o diretório do run.
    try:
        run = DriverRun.for_driver(
            Path(__file__),
            experiment=EXPERIMENT,
            run_number=args.run_number,
            outputs_root=None if args.outputs_root is None else args.outputs_root.resolve(),
        )
        harness = run.output_dir("harness")
        portable = Portable(workspace=workspace, outputs=run.outputs_root, checkout=CHECKOUT)
        matrix = read_matrix(MATRIX)
        foundation = foundation_of_run(workspace, args.foundation_run.resolve())
        assert foundation.sequence.location is not None
        window = read_window(args.window_file.resolve())
        selection = select_images(
            window,
            workspace / foundation.sequence.location,
            sequence_artifact_id=foundation.sequence.artifact_id,
            stride=matrix.image_stride,
            accept_gaps=args.accept_window_gaps,
        )
        layers = arm_layers(args.arms_dir.resolve(), {arm.slug: arm.layer for arm in matrix.arms})
        effectives = {slug: resolve_arm(base, shared, layer) for slug, layer in layers.items()}
        refuse_placeholders(effectives)
        differences = check_matrix(matrix, effectives)
        parameters = check_parameters(matrix, effectives)
        scorers = check_scorer_space(matrix, effectives)
        protocol = _protocol(
            run,
            matrix=matrix,
            foundation=foundation,
            window=window,
            selection=selection,
            configs={arm.slug: (base, *shared, layers[arm.slug]) for arm in matrix.arms},
            differences=differences,
            parameters=parameters,
            scorers=scorers,
            portable=portable,
            accept_window_gaps=args.accept_window_gaps,
        )
        code_identity = resolve_code_identity(
            protocol["code"],
            checkout=CHECKOUT,
            evidence_class=evidence_class,
            requested=args.code_identity,
        )
        harness.mkdir(parents=True)
    except FileExistsError:
        print(f"error: {harness} already exists: a run is never overwritten", file=sys.stderr)
        return 2
    except (ExperimentDriverError, SpatialFoundationError, ConfigurationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    protocol.update(evidence_class=evidence_class, code_identity=code_identity)
    write_once(harness / "protocol.json", protocol)
    index = ViewerIndex(harness, schema=MAPS_SCHEMA)
    index.start()

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": evidence_class,
        "protocol_sha256": hashlib.sha256((harness / "protocol.json").read_bytes()).hexdigest(),
        "status": "running",
        "failure": None,
        "voxel_layers": {"status": "not_run"},
        "arms": {
            arm.slug: {
                "status": "not_run",
                "role": arm.role,
                "group": arm.group,
                "execution": arm.execution,
                "reuse_of": arm.reuse,
                "steps": {},
            }
            for arm in matrix.arms
        },
        "replica_agreement": None,
    }
    campaign = _Campaign(
        matrix=matrix,
        run=run,
        workspace=workspace,
        harness=harness,
        foundation_run=args.foundation_run.resolve(),
        foundation=foundation,
        selection=selection,
        effectives=effectives,
        clis={
            arm.slug: ContextCli(
                workspace=workspace,
                configs=(base, *shared, layers[arm.slug]),
                overrides=(),
                steps_dir=harness / "steps" / arm.slug,
                code_identity=code_identity,
                dry_run_executors=args.dry_run_executors,
            )
            for arm in matrix.arms
        },
        report=report,
        index=index,
    )
    try:
        campaign.voxel_layers()
        campaign.arms()
        report["status"] = "complete"
    except StepFailed as failure:
        arm = _failed_arm(report)
        steps_dir = harness / "steps" / (arm or "voxel-layers")
        steps_dir.mkdir(parents=True, exist_ok=True)
        report.update(status="failed", failure={"arm": arm, **portable.failure(failure, steps_dir)})
    except Exception as error:
        # Um erro fora de um comando (ao ler ou medir) também deixa o relatório, e sobe.
        failed = {"arm": _failed_arm(report), **unexpected_failure(error)}
        report.update(status="failed", failure=portable.document(failed))
        _finish(harness, report, matrix)
        raise
    _finish(harness, report, matrix)
    print(f"{report['status']}: {harness / 'report.json'}")
    return 0 if report["criterion"]["passes"] else 1


@dataclass(frozen=True, kw_only=True)
class _Arm:
    """What a finished arm leaves for the arms that reuse it.

    Attributes:
        branch: Its context branch.
        artifacts: The artifacts of its ContextRun, by stage, as ``ArtifactRef`` documents.
        entry: Its entry in the viewer index.
    """

    branch: str
    artifacts: dict[str, dict[str, Any]]
    entry: dict[str, Any]


@dataclass(kw_only=True)
class _Campaign:
    """One run of the campaign: its inputs and what it has produced so far."""

    matrix: CampaignMatrix
    run: DriverRun
    workspace: Path
    harness: Path
    foundation_run: Path
    foundation: SpatialFoundation
    selection: WindowSelection
    effectives: dict[str, EffectiveConfig]
    clis: dict[str, ContextCli]
    report: dict[str, Any]
    index: ViewerIndex
    finished: dict[str, _Arm] = field(default_factory=dict)

    def voxel_layers(self) -> None:
        """Derive the voxel layers of the foundation map once, with the geometric multiscale driver.

        Raises:
            StepFailed: If the driver fails or reports an incomplete run.
        """
        entry = self.report["voxel_layers"]
        entry["status"] = "failed"
        output = self.harness / "voxel-layers"
        assert self.foundation.geometry.location is not None
        measured = measure_child_process(
            [
                sys.executable,
                str(CHECKOUT / self.matrix.voxel_script),
                str(output),
                "--raw-map",
                str(self.workspace / self.foundation.geometry.location),
                "--cells",
                *(repr(cell) for cell in self.matrix.voxel_cells_m),
            ]
        )
        entry["step"] = record = measured.to_record()
        if measured.returncode != 0:
            raise StepFailed(
                "voxel-layers",
                f"the #624 driver exited with status {measured.returncode}",
                record=record,
                stderr=measured.stderr,
            )
        voxel = json.loads((output / "report.json").read_text(encoding="utf-8"))
        layers = {
            f"{arm['cell_m']}": relative(output / name / "voxel_aggregation", self.harness)
            for name, arm in voxel["arms"].items()
            if arm["cell_m"] is not None
        }
        entry.update(
            status="complete",
            report=relative(output / "report.json", self.harness),
            source_map_run_id=voxel["source"]["run_id"],
            source_map_id=voxel["source"]["map_id"],
            layers=layers,
        )
        self.index.publish_voxel_layers(layers)

    def arms(self) -> None:
        """Run every arm in the matrix's order and publish each; compare the replica when it ends.

        The agreement is recorded as soon as the replica finishes, so a later arm that fails
        (Eagle 2.5, the last one) does not cost it.

        Raises:
            StepFailed: At the first command that does not complete, or a reuse arm that did not
                consume the reused arm's three evidence producers; later arms stay ``not_run``.
        """
        runs = {
            "perception": self._perception,
            "downstream_build": self._downstream_build,
            "association_rerun": self._association_rerun,
            "semantic_scoring": self._semantic_scoring,
        }
        for arm in self.matrix.arms:
            entry = self.report["arms"][arm.slug]
            # Um braço fica "failed" até o último passo dele terminar: a exceção que interrompe o
            # run deixa esse estado no relatório.
            entry["status"] = "failed"
            published = runs[arm.execution](arm, entry)
            self.index.publish_arm(arm.slug, published)
            entry["status"] = "complete"
            if arm.slug == self.matrix.replica:
                baseline = self.finished[self.matrix.baseline].artifacts
                replica = self.finished[arm.slug].artifacts
                self.report["replica_agreement"] = replica_agreement(
                    self.workspace / baseline["semantic_interpretation"]["location"],
                    self.workspace / replica["semantic_interpretation"]["location"],
                    REPLICA_JOIN,
                    baseline_regions=self.workspace / baseline["region_perception"]["location"],
                    replica_regions=self.workspace / replica["region_perception"]["location"],
                )

    def _perception(self, arm: CampaignArm, entry: dict[str, Any]) -> dict[str, Any]:
        """Run an arm's own branch, ContextRun and build, as the pilot does."""
        cli, steps = self.clis[arm.slug], entry["steps"]
        branch = f"campaign-{self.run.run_id}-{arm.slug}"
        entry["branch"] = branch
        steps["branch"] = cli.branch(branch, foundation_run=self.foundation_run).record
        context = cli.context_run(
            "context-run", branch=branch, selection=self.selection.observation_selection()
        )
        steps["context_run"] = step_record(context, self.workspace)
        artifacts = {
            item["stage_id"]: _ref(item) for item in context.document["context_run"]["artifacts"]
        }
        entry["artifacts"] = artifacts
        build = cli.build("build", branch=branch, revision=1)
        steps["build"] = step_record(build, self.workspace)
        build_run = Path(build.document["run_directory"])
        entry["build"] = build_record(
            read_build_state(build_run, workspace=self.workspace), build_run, self.workspace
        )
        entry["perception"] = perception_metrics(
            self.workspace / artifacts["semantic_interpretation"]["location"],
            region_run=self.workspace / artifacts["region_perception"]["location"],
        )
        return self._finished(
            arm,
            branch=branch,
            artifacts=artifacts,
            context_map_id=entry["build"]["context_map_id"],
            paths={
                "context_map": map_location(build),
                "fusion_run": relative(
                    build_run.parent.parent / "artifacts" / "semantic_fusion" / build_run.name,
                    self.workspace,
                ),
            },
        )

    def _downstream_build(self, arm: CampaignArm, entry: dict[str, Any]) -> dict[str, Any]:
        """Build the reused arm's branch again under this arm's downstream layer."""
        source = self._source(arm)
        cli, steps = self.clis[arm.slug], entry["steps"]
        entry["branch"] = source.branch
        build = cli.build("build", branch=source.branch, revision=1)
        steps["build"] = step_record(build, self.workspace)
        build_run = Path(build.document["run_directory"])
        provided = build.document["context_build"]["provided"]
        # Um build consome a percepção e a associação da ContextRun; o PTv3 fica nela.
        entry["reuse"] = self._reuse(
            arm,
            source,
            consumed={
                stage: [_ref(ref) for ref in provided.get(stage, [])]
                for stage in (*_EVIDENCE_INPUTS, "sensor_association")
            },
            dispositions=steps["build"]["stages"],
        )
        entry["build"] = build_record(
            read_build_state(build_run, workspace=self.workspace), build_run, self.workspace
        )
        entry["perception"] = {"reused_from": arm.reuse}
        return self._finished(
            arm,
            branch=source.branch,
            artifacts=source.artifacts,
            context_map_id=entry["build"]["context_map_id"],
            paths={
                "context_map": map_location(build),
                "fusion_run": relative(
                    build_run.parent.parent / "artifacts" / "semantic_fusion" / build_run.name,
                    self.workspace,
                ),
            },
        )

    def _association_rerun(self, arm: CampaignArm, entry: dict[str, Any]) -> dict[str, Any]:
        """Associate the reused regional/visual artifacts and materialize a new map."""
        source = self._source(arm)
        cli, steps = self.clis[arm.slug], entry["steps"]
        entry["branch"] = None
        producers = {stage: source.artifacts[stage] for stage in _EVIDENCE_INPUTS}
        selected = {
            **{
                stage: getattr(self.foundation, role).to_document()
                for stage, role in _FOUNDATION_STAGES.items()
            },
            **producers,
        }
        plan = resolve_plan(self.effectives[arm.slug])
        # O Point Representation, quando ligado, representa a associação nova deste braço.
        targets = [
            stage
            for stage in ("context_map", "point_representation")
            if plan.order is not None and stage in plan.order
        ]
        argv = ["run", *(item for stage in targets for item in ("--stage", stage))]
        argv += self._selected(arm, selected, cli=cli)
        step = cli.run(
            "association-run", argv, overrides=(self._providers(arm, targets, selected),)
        )
        steps["association_run"] = step_record(step, self.workspace)
        run_dir = Path(step.document["run_directory"])
        entry["reuse"] = self._reuse(
            arm,
            source,
            consumed={
                "region_perception": _consumed(step, "sensor_association", "regions"),
                "visual_perception": _consumed(step, "sensor_association", "visual_evidence"),
                "semantic_interpretation": _consumed(
                    step, "semantic_fusion", "semantic_interpretation"
                ),
            },
            dispositions=steps["association_run"]["stages"],
        )
        entry["materialization"] = _materialization_record(run_dir, self.workspace)
        entry["perception"] = {"reused_from": arm.reuse}
        point_representation = (
            run_dir.parent.parent / "artifacts" / "point_representation" / run_dir.name
        )
        return self._finished(
            arm,
            branch=None,
            artifacts={
                **producers,
                "sensor_association": {
                    "location": relative(
                        run_dir.parent.parent / "artifacts" / "sensor_association" / run_dir.name,
                        self.workspace,
                    )
                },
                "point_representation": {
                    "location": relative(point_representation, self.workspace)
                    if point_representation.is_dir()
                    else None
                },
            },
            context_map_id=entry["materialization"]["context_map_id"],
            paths={
                "context_map": relative(
                    run_dir.parent.parent / "artifacts" / "context_map" / run_dir.name,
                    self.workspace,
                ),
                "fusion_run": relative(
                    run_dir.parent.parent / "artifacts" / "semantic_fusion" / run_dir.name,
                    self.workspace,
                ),
            },
        )

    def _semantic_scoring(self, arm: CampaignArm, entry: dict[str, Any]) -> dict[str, Any]:
        """Score the reused regional, visual and semantic artifacts in their optional stage."""
        source = self._source(arm)
        cli, steps = self.clis[arm.slug], entry["steps"]
        entry["branch"] = source.branch
        selected = {stage: source.artifacts[stage] for stage in _EVIDENCE_INPUTS}
        argv = ["stage", "semantic_scoring"]
        argv += self._selected(arm, selected, cli=cli)
        step = cli.run(
            "semantic-scoring",
            argv,
            overrides=(self._providers(arm, ["semantic_scoring"], selected),),
        )
        steps["semantic_scoring"] = step_record(step, self.workspace)
        run_dir = Path(step.document["run_directory"])
        entry["reuse"] = self._reuse(
            arm,
            source,
            consumed={
                stage: _consumed(step, "semantic_scoring", name)
                for stage, name in _EVIDENCE_INPUTS.items()
            },
            dispositions=steps["semantic_scoring"]["stages"],
        )
        entry["scoring"] = _scoring_record(
            self.workspace / "artifacts" / "semantic_scoring" / run_dir.name, self.workspace
        )
        entry["perception"] = {"reused_from": arm.reuse}
        # Nenhuma política de fusão consome SemanticScore: o mapa é o do braço reusado.
        return {
            **source.entry,
            "arm": arm.slug,
            "role": arm.role,
            "group": arm.group,
            "execution": arm.execution,
            "paths": {**source.entry["paths"], "scoring_run": entry["scoring"]["run"]},
        }

    def _source(self, arm: CampaignArm) -> _Arm:
        """Return the finished arm whose evidence producers ``arm`` reuses."""
        assert arm.reuse is not None, "the matrix gives every reuse arm its source"
        return self.finished[arm.reuse]

    def _reuse(
        self,
        arm: CampaignArm,
        source: _Arm,
        *,
        consumed: dict[str, list[dict[str, Any]]],
        dispositions: dict[str, str],
    ) -> dict[str, Any]:
        """Record which of the reused arm's artifacts this arm consumed, or stop the run.

        The artifacts must be the same (stage, contract, id and content hash) and each must have
        been provided to the run, never executed by it.

        Raises:
            StepFailed: If a context artifact the arm consumed is not the reused arm's one.
        """
        record: dict[str, Any] = {"source_arm": arm.reuse, "dispositions": dispositions}
        holds = True
        for stage, refs in consumed.items():
            reused = source.artifacts[stage]
            same = [_identity(ref) for ref in refs] == [_identity(reused)]
            record[_REUSE_KEYS[stage]] = {
                "source": reused,
                "consumed": refs[0] if len(refs) == 1 else refs,
                "same_artifact": same,
            }
            holds = holds and same and dispositions.get(stage) == "provided"
        record["holds"] = holds
        if not holds:
            raise StepFailed(
                "reuse-check",
                f"{arm.slug} did not consume exactly the artifacts of {arm.reuse}: {record}",
            )
        return record

    def _selected(
        self, arm: CampaignArm, selected: dict[str, dict[str, Any]], *, cli: ContextCli
    ) -> list[str]:
        """Write the explicit catalog of the selected artifacts; return the selection flags.

        The catalog lists exactly the artifacts the run is given, with the lineage their
        records state: a perception run of a ContextRun was built from the foundation's
        sequence (the ContextRun refuses evidence of another foundation).
        """
        sequence = self.foundation.sequence.artifact_id
        entries = [
            {
                **ref,
                "run_index": _run_number(ref["location"]),
                "lineage": {"upstream": {"ingestion": [sequence]}}
                if stage == "region_perception"
                else {
                    "upstream": {
                        "ingestion": [sequence],
                        "region_perception": [selected["region_perception"]["artifact_id"]],
                    }
                }
                if stage in ("visual_perception", "semantic_interpretation")
                else {},
            }
            for stage, ref in selected.items()
        ]
        catalog = self.harness / "catalogs" / f"{arm.slug}.json"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        write_once(catalog, {"schema_version": "0.1.0", "entries": entries})
        flags = [
            item
            for stage, ref in selected.items()
            for item in ("--select", f"{stage}={ref['artifact_id']}")
        ]
        return [*flags, "--catalog", str(catalog), "--code-identity", cli.code_identity]

    def _providers(
        self, arm: CampaignArm, targets: list[str], selected: dict[str, dict[str, Any]]
    ) -> str:
        """Keep only the providers of the stages the run executes, as a configuration override.

        The CLI composes every stage its configuration allows and a provider loads its model
        when composed: a run over a selected perception run must not load the perception models.
        """
        effective = self.effectives[arm.slug]
        execution = resolve_plan(effective).scope(
            targets=targets,
            provided={stage: ArtifactRef.from_document(ref) for stage, ref in selected.items()},
        )
        components = {component for stage in execution.stages for component in stage.components}
        providers = {
            component: target
            for component, target in effective.config.resources.providers.items()
            if component in components
        }
        return f"resources.providers={json.dumps(providers, sort_keys=True)}"

    def _finished(
        self,
        arm: CampaignArm,
        *,
        branch: str | None,
        artifacts: dict[str, dict[str, Any]],
        context_map_id: str,
        paths: dict[str, str],
    ) -> dict[str, Any]:
        """Keep what later arms reuse and return the arm's entry in the viewer index."""
        entry = {
            "arm": arm.slug,
            "role": arm.role,
            "group": arm.group,
            "execution": arm.execution,
            "branch": branch,
            "context_map_id": context_map_id,
            "paths": {
                "context_map": paths["context_map"],
                "region_artifact": artifacts["region_perception"]["location"],
                "visual_evidence_artifact": artifacts["visual_perception"]["location"],
                "semantic_interpretation_artifact": artifacts["semantic_interpretation"][
                    "location"
                ],
                "association_run": artifacts["sensor_association"]["location"],
                "fusion_run": paths["fusion_run"],
                "point_representation_run": artifacts.get("point_representation", {}).get(
                    "location"
                ),
                "scoring_run": None,
            },
        }
        if branch is not None and arm.execution == "perception":
            self.finished[arm.slug] = _Arm(branch=branch, artifacts=artifacts, entry=entry)
        return entry


_EVIDENCE_INPUTS = {
    "region_perception": "regions",
    "visual_perception": "visual_evidence",
    "semantic_interpretation": "semantic_interpretation",
}

_REUSE_KEYS = {
    **_EVIDENCE_INPUTS,
    "sensor_association": "association",
    "point_representation": "point_representation",
}


def _ref(document: dict[str, Any]) -> dict[str, Any]:
    """Return an artifact reference as a runtime ``ArtifactRef`` document, without extras."""
    return {name: document[name] for name in _REF_FIELDS}


def _identity(ref: dict[str, Any]) -> tuple[Any, ...]:
    """What makes two references name the same artifact; the location is only a locator."""
    return tuple(ref[name] for name in _REF_FIELDS[:-1])


def _consumed(step: StepResult, stage_id: str, name: str) -> list[dict[str, Any]]:
    """Return the artifacts a stage of a completed run consumed as one of its inputs."""
    stage = next(
        item for item in step.document["execution"]["stages"] if item["stage_id"] == stage_id
    )
    return [_ref(ref) for ref in stage["inputs"][name]]


def _run_number(location: str) -> int:
    """Return the run number from ``artifacts/<stage>/run-NNNN``."""
    return int(Path(location).name.removeprefix("run-"))


def _materialization_record(run_dir: Path, workspace: Path) -> dict[str, Any]:
    """Record a map materialized by a runtime run that has no ContextBuild record.

    ``read_build_state`` reads a ContextBuild; this run selected its inputs through a catalog,
    so its map is read by the same public readers, and only its identity, counts, validation and
    integrity are recorded.
    """
    with ContextMapArtifactReader.open(
        run_dir.parent.parent / "artifacts" / "context_map" / run_dir.name
    ) as reader:
        context_map = reader.context_map()
    fusion = SemanticFusionRunReader(
        run_dir.parent.parent / "artifacts" / "semantic_fusion" / run_dir.name
    )
    mapping = SemanticMappingRunReader(
        run_dir.parent.parent / "artifacts" / "semantic_mapping" / run_dir.name
    )
    validation = validate_context_map_artifact(
        run_dir.parent.parent / "artifacts" / "context_map" / run_dir.name,
        level=ValidationLevel.FULL,
    )
    return {
        "run": relative(run_dir, workspace),
        "context_map_id": str(context_map.context_map_id),
        "counts": {
            "fusion": fusion.read_record("metrics/counts.json"),
            "semantic_entities": sum(1 for _ in mapping.iter_entities()),
            "resolved_entities": len(context_map.entities),
            "relations": len(context_map.relations),
            "relations_by_state": dict(
                sorted(Counter(item.state.value for item in context_map.relations).items())
            ),
        },
        "validation": {
            "status": validation.status.value,
            "errors": [
                f"{item.code}: {item.message}"
                for item in validation.findings
                if item.severity is Severity.ERROR
            ],
            "warnings": [
                f"{item.code}: {item.message}"
                for item in validation.findings
                if item.severity is Severity.WARNING
            ],
        },
        "integrity_problems": [
            *(f"semantic_fusion: {problem}" for problem in fusion.verify_integrity()),
            *(f"semantic_mapping: {problem}" for problem in mapping.verify_integrity()),
        ],
        "artifact_bytes": {
            stage: directory_bytes(run_dir.parent.parent / "artifacts" / stage / run_dir.name)
            for stage in ("sensor_association", *BUILD_STAGES)
            if (run_dir.parent.parent / "artifacts" / stage / run_dir.name).is_dir()
        },
    }


def _scoring_record(run_dir: Path, workspace: Path) -> dict[str, Any]:
    """Record a scoring run: what it scored, in which space, under which template."""
    reader = SemanticScoringRunReader(run_dir)
    manifest = reader.manifest
    counts = manifest.counts
    return {
        "run": relative(run_dir, workspace),
        "region_artifact_id": manifest.lineage.region_artifact_id,
        "visual_evidence_artifact_id": manifest.lineage.visual_evidence_artifact_id,
        "semantic_interpretation_artifact_id": manifest.lineage.semantic_interpretation_artifact_id,
        "scorer": manifest.scorer.backend_id,
        "embedding_space_id": manifest.embedding_space_id,
        "text_template": manifest.text_template,
        "score_type": manifest.score_type.value,
        "counts": {
            "results": counts.result_count,
            "region_claims": counts.region_claim_count,
            "scene_claims": counts.scene_claim_count,
            "scored_claims": counts.scored_claim_count,
            "unscored_region_claims": counts.unscored_region_claim_count,
            "scores": counts.score_count,
        },
        "integrity_problems": reader.verify_integrity(),
    }


def _finish(harness: Path, report: dict[str, Any], matrix: CampaignMatrix) -> None:
    """Judge the campaign's criterion and write the report, never over another run's."""
    arms = report["arms"]
    complete = [entry for entry in arms.values() if entry["status"] == "complete"]
    maps = [
        entry.get("build") or entry.get("materialization")
        for entry in complete
        if entry["execution"] != "semantic_scoring"
    ]
    verified = [
        item
        for item in maps
        if item["validation"]["status"] == "verified" and not item["integrity_problems"]
    ]
    materializing = [arm for arm in matrix.arms if arm.execution != "semantic_scoring"]
    reuse = [entry for entry in complete if entry.get("reuse", {}).get("holds")]
    # A campanha passa quando todo braço termina, todo mapa é verificado, cada braço que
    # materializa tem o seu mapa e todo braço de reuso consumiu a percepção do braço reusado.
    report["criterion"] = {
        "arms": len(arms),
        "complete_arms": len(complete),
        "verified_maps": len(verified),
        "distinct_context_maps": len({item["context_map_id"] for item in maps}),
        "reuse_holds": len(reuse),
        "passes": report["status"] == "complete"
        and len(complete) == len(arms)
        and len(verified) == len({item["context_map_id"] for item in maps}) == len(materializing)
        and len(reuse) == sum(1 for arm in matrix.arms if arm.reuse is not None),
    }
    write_once(harness / "report.json", report)


def _failed_arm(report: dict[str, Any]) -> str | None:
    """Return the arm the run stopped in, or ``None`` when it stopped outside every arm."""
    return next(
        (slug for slug, entry in report["arms"].items() if entry["status"] == "failed"), None
    )


def _protocol(
    run: DriverRun,
    *,
    matrix: CampaignMatrix,
    foundation: SpatialFoundation,
    window: CorridorWindow,
    selection: WindowSelection,
    configs: dict[str, tuple[Path, ...]],
    differences: dict[str, Any],
    parameters: list[dict[str, Any]],
    scorers: dict[str, dict[str, Any]],
    portable: Portable,
    accept_window_gaps: bool,
) -> dict[str, Any]:
    """Return the record frozen before any step runs: what the protocol fixes for this run."""
    return {
        "schema": SCHEMA,
        "issue": 670,
        "run": run.run_id,
        "code": git_identity(CHECKOUT),
        "matrix": {"file": relative(MATRIX, CHECKOUT), "sha256": matrix.sha256},
        "foundation": foundation.to_document(),
        "window": {
            "file": window.file,
            "sha256": window.sha256,
            "document": _portable_leaves(window.document, portable),
            "accept_window_gaps": accept_window_gaps,
        },
        "selection": selection.to_document(),
        "variables": [variable.to_record() for variable in matrix.variables],
        "parameters": parameters,
        "scorer_space": scorers,
        "arms": [
            {
                **arm.to_record(),
                "branch": f"campaign-{run.run_id}-{arm.slug}"
                if arm.execution == "perception"
                else None,
                **configuration_record(configs[arm.slug], ()),
                "differences_from_baseline": [item.to_record() for item in differences[arm.slug]],
            }
            for arm in matrix.arms
        ],
        "configuration_check": {
            "method": "contextmap.evaluation.arm_differences over every pair of arms, field by "
            "field, on the resolved configuration; every registered parameter is the value its "
            "arm resolves; every scorer names the backend, checkpoint and revision of the region "
            "features it scores",
            "compared": ["components", *COMPARED_SECTIONS],
            "excluded": {"resources": "deployment: model providers, device and workspace"},
        },
        "voxel_layers": {
            "script": matrix.voxel_script,
            "cells_m": list(matrix.voxel_cells_m),
            "raw_map": foundation.geometry.to_document(),
        },
        "replica_join": {
            "policy_id": REPLICA_JOIN.policy_id,
            "min_iou": REPLICA_JOIN.min_iou,
            "baseline": matrix.baseline,
            "replica": matrix.replica,
            "regions": "accepted regions; masks loaded from each run's mask store",
            "label": "the set of a region's primary region-claim hypotheses, exact text",
            "agreement": "agreeing / compared, over joined pairs labeled on both sides",
        },
        "region_labels": {
            "top": TOP_LABELS,
            "label": "the set of a region's primary region-claim hypotheses, exact text",
        },
    }


def _portable_leaves(value: Any, portable: Portable) -> Any:
    """Return a JSON document whose text leaves cite no machine root."""
    if isinstance(value, dict):
        return {key: _portable_leaves(item, portable) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_leaves(item, portable) for item in value]
    return portable.text(value) if isinstance(value, str) else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", type=Path, required=True, help="the runtime workspace")
    parser.add_argument(
        "--foundation-run",
        type=Path,
        required=True,
        help="completed run whose sequence, trajectory and map are the one foundation",
    )
    parser.add_argument(
        "--window-file",
        type=Path,
        required=True,
        help="the window.json #669 wrote (contextmap.corridor-window/v1)",
    )
    parser.add_argument(
        "--accept-window-gaps",
        action="store_true",
        help="run a window outside its pose/LiDAR gap thresholds; the protocol records it",
    )
    parser.add_argument(
        "--base-config", type=Path, required=True, help="the baseline configuration, filled"
    )
    parser.add_argument(
        "--arms-dir",
        type=Path,
        required=True,
        help="directory with the filled layer of every arm, as config/campaign.json names it",
    )
    parser.add_argument(
        "--all-arms-config",
        type=Path,
        action="append",
        metavar="PATH",
        help="a layer applied to every arm, after the base and before the arm's own",
    )
    parser.add_argument(
        "--code-identity",
        help="the code every step records (default: git HEAD); a real run records the full SHA "
        "and refuses any value that is not the checked-out commit",
    )
    parser.add_argument("--run-number", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, help="default: the checkout's outputs/")
    parser.add_argument(
        "--dry-run-executors",
        metavar="MODULE:ATTRIBUTE",
        help="CI only: executors that replace perception; the report is marked dry_run",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
