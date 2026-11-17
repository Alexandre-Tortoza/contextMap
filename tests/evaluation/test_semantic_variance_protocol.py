"""The #580 variance protocol analysis: pairing, canonicalization and spread over repetitions.

The analysis lives under ``experiments/semantic-variance-protocol-20260927/scripts/`` (a
research script, outside the package, by the issue's own scope), but it decides what the
experiment reports and therefore whether H1 is corroborated: it is deterministic behavior and
is tested here, never on the GPU machine.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from contextmap.evaluation import SemanticAttemptAccounting
from contextmap.ingestion import SourceObservationId
from contextmap.region_perception import (
    BoundingBox2D,
    Region2D,
    RegionId,
    RegionPerceptionRunManifest,
    RegionPerceptionRunWriter,
)
from contextmap.semantic_interpretation import (
    ClaimId,
    HypothesisRole,
    SemanticClaim,
    SemanticInferenceProvenance,
    SemanticInterpretationRunId,
    SemanticInterpretationRunWriter,
)
from contextmap.shared import BackendProvenance
from contextmap.visual_perception import PerceptionResult, PreparedImage
from semantic_artifact_builders import write_semantic_views

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY_ROOT / "experiments" / "semantic-variance-protocol-20260927" / "scripts"
SCRIPT_15 = (
    REPOSITORY_ROOT
    / "experiments"
    / "e2e-real-canonical-run-20260923"
    / "scripts-run5"
    / "15_visual_perception_content_equivalence.py"
)

ChosenClaim = tuple[str, str, str | None, float | None]
"""``(hypothesis, role, category, confidence)`` of one synthetic claim."""


@pytest.fixture
def variance(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    yield importlib.import_module("semantic_variance")


def _script_15() -> ModuleType:
    import ast

    tree = ast.parse(SCRIPT_15.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_claim_key"
    )
    module = ModuleType("historical_claim_tuple")
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SCRIPT_15), "exec"),
        module.__dict__,
    )
    return module


def _provenance(capability: str) -> BackendProvenance:
    return BackendProvenance(
        backend_id=f"fake_{capability}",
        capability=capability,
        provider="fake",
        model="fake",
        version="0.1",
    )


def _claim(
    frame: str, region: str | None, index: int, chosen: ChosenClaim, run_id: str = "run"
) -> SemanticClaim:
    hypothesis, role, category, confidence = chosen
    return SemanticClaim(
        claim_id=ClaimId(f"claim-{index:04d}"),
        source_observation_id=SourceObservationId(frame),
        semantic_interpretation_run_id=SemanticInterpretationRunId(run_id),
        hypothesis=hypothesis,
        role=HypothesisRole(role),
        provenance=SemanticInferenceProvenance(
            backend=_provenance("semantic_interpreter"),
            task_identity="qwen-region-interpretation",
            prompt_template_id="region/v1",
            output_schema_version="semantic-response/1",
        ),
        category=category,
        confidence=confidence,
        region_id=None if region is None else RegionId(region),
    )


def _region(region: str, x: float = 0.0) -> Region2D:
    return Region2D(
        region_id=RegionId(region),
        bounding_box=BoundingBox2D(x=x, y=0, width=10, height=10),
        provenance=_provenance("region_discovery"),
        region_kind="object",
    )


def _result(
    frame: str,
    claims: Mapping[str, Sequence[ChosenClaim]],
    *,
    run_id: str = "run",
    region_x: float = 0.0,
) -> PerceptionResult:
    regions = tuple(_region(region, x=region_x) for region in claims)
    built = [
        _claim(frame, region, index, chosen, run_id)
        for index, (region, chosen) in enumerate(
            (region, chosen) for region, chosens in claims.items() for chosen in chosens
        )
    ]
    return PerceptionResult(
        source_observation_id=SourceObservationId(frame),
        region_artifact_id="regional",
        semantic_interpretation_artifact_id=run_id,
        sequence_artifact_id="sequence-20",
        regions=regions,
        claims=tuple(built),
    )


CHAIR = ("chair", "primary", "furniture", None)
STOOL = ("stool", "alternative", "furniture", None)
SEAT = ("seat", "alternative", "furniture", None)
DOOR = ("door", "primary", None, None)


def _attempts(parsed: int, failed: int = 0) -> SemanticAttemptAccounting:
    return SemanticAttemptAccounting(attempted=parsed + failed, parsed=parsed, parse_failed=failed)


def _run(
    variance: ModuleType,
    label: str,
    results: Sequence[PerceptionResult],
    *,
    digest: str = "sha256:config",
    parsed: int = 2,
) -> object:
    return variance.run_content(
        label, results, configuration_digest=digest, attempts=_attempts(parsed)
    )


class TestCanonicalization:
    def test_a_claim_key_is_the_canonical_tuple_of_script_15(self, variance: ModuleType) -> None:
        script_15 = _script_15()
        claims = [
            _claim("frame-1", "region-0001", 0, CHAIR),
            _claim("frame-1", "region-0001", 1, STOOL),
            _claim("frame-2", None, 2, ("a corridor", "primary", "scene", 0.25)),
        ]

        for claim in claims:
            assert tuple(variance.claim_key(claim)) == script_15._claim_key(claim)

    def test_every_projection_is_keyed_by_frame_and_region(self, variance: ModuleType) -> None:
        assert variance.PROJECTIONS == {
            "tuple": ("hypothesis", "role", "category", "confidence"),
            "hypothesis": ("hypothesis",),
            "role": ("role",),
            "category": ("category",),
            "confidence": ("confidence",),
            "hypothesis+role+category": ("hypothesis", "role", "category"),
            "slot": (),
        }


class TestPairAgreement:
    def test_each_projection_counts_matches_over_the_union(self, variance: ModuleType) -> None:
        first = _run(
            variance,
            "A-01",
            [
                _result("frame-1", {"region-0001": [CHAIR, STOOL]}),
                _result("frame-2", {"region-0002": [DOOR]}),
            ],
        )
        # O frame-2 do segundo run falhou no parser: nenhuma claim, mesma região.
        second = _run(
            variance,
            "A-02",
            [
                _result("frame-1", {"region-0001": [CHAIR, SEAT]}),
                _result("frame-2", {"region-0002": []}),
            ],
        )

        pair = variance.compare_pair(first, second)

        agreements = {name: (a.matched, a.union, a.rate) for name, a in pair.agreements.items()}
        assert agreements == {
            "tuple": (1, 4, 0.25),
            "hypothesis": (1, 4, 0.25),
            "role": (2, 3, 2 / 3),
            "category": (1, 2, 0.5),
            "confidence": (1, 2, 0.5),
            "hypothesis+role+category": (1, 4, 0.25),
            "slot": (1, 2, 0.5),
        }
        assert (pair.first, pair.second) == ("A-01", "A-02")
        assert pair.frames_compared == 2
        assert pair.region_mismatches == 0
        assert pair.missing_frames == 0

    def test_identical_runs_agree_exactly(self, variance: ModuleType) -> None:
        results = [_result("frame-1", {"region-0001": [CHAIR, STOOL]})]
        pair = variance.compare_pair(
            _run(variance, "B-01", results), _run(variance, "B-02", results)
        )

        assert {name: a.rate for name, a in pair.agreements.items()} == dict.fromkeys(
            variance.PROJECTIONS, 1.0
        )

    def test_no_claim_on_either_side_is_undefined_not_perfect(self, variance: ModuleType) -> None:
        # O script 15 devolvia 1.0 para 0/0; aqui 0/0 é indefinido, nunca concordância perfeita.
        empty = [_result("frame-1", {"region-0001": []})]
        pair = variance.compare_pair(_run(variance, "A-01", empty), _run(variance, "A-02", empty))

        assert pair.agreements["tuple"].union == 0
        assert pair.agreements["tuple"].rate is None

    def test_differing_regions_and_missing_frames_are_counted_like_script_15(
        self, variance: ModuleType
    ) -> None:
        first = _run(
            variance,
            "A-01",
            [
                _result("frame-1", {"region-0001": [CHAIR]}),
                _result("frame-2", {"region-0002": [DOOR]}),
            ],
        )
        second = _run(
            variance,
            "A-02",
            [_result("frame-1", {"region-0001": [CHAIR]}, region_x=3.0)],
        )

        pair = variance.compare_pair(first, second)

        assert pair.region_mismatches == 1
        assert pair.missing_frames == 1
        assert pair.frames_compared == 1
        # Como no script 15, as claims de um frame ausente de um dos lados não entram na conta.
        assert (pair.agreements["tuple"].matched, pair.agreements["tuple"].union) == (1, 1)


class TestSpread:
    def test_min_median_and_max_over_the_defined_values(self, variance: ModuleType) -> None:
        assert variance.spread([1.0, 0.25, 0.5]) == {
            "count": 3,
            "undefined": 0,
            "min": 0.25,
            "median": 0.5,
            "max": 1.0,
        }

    def test_an_even_count_takes_the_mean_of_the_middle_values(self, variance: ModuleType) -> None:
        assert variance.spread([0.2, 0.4, None])["median"] == pytest.approx(0.3)
        assert variance.spread([0.2, 0.4, None])["undefined"] == 1

    def test_nothing_defined_has_no_statistic(self, variance: ModuleType) -> None:
        assert variance.spread([None]) == {
            "count": 1,
            "undefined": 1,
            "min": None,
            "median": None,
            "max": None,
        }


class TestArm:
    def _three(self, variance: ModuleType) -> list[object]:
        return [
            _run(variance, "A-01", [_result("frame-1", {"region-0001": [CHAIR, STOOL]})], parsed=2),
            _run(variance, "A-02", [_result("frame-1", {"region-0001": [CHAIR, STOOL]})], parsed=2),
            _run(variance, "A-03", [_result("frame-1", {"region-0001": [CHAIR, SEAT]})], parsed=1),
        ]

    def test_every_pair_is_compared_once_in_label_order(self, variance: ModuleType) -> None:
        runs = self._three(variance)

        arm = variance.compare_arm("A", list(reversed(runs)))

        assert [(pair.first, pair.second) for pair in arm.pairs] == [
            ("A-01", "A-02"),
            ("A-01", "A-03"),
            ("A-02", "A-03"),
        ]
        assert arm.spreads["tuple"] == {
            "count": 3,
            "undefined": 0,
            "min": 1 / 3,
            "median": 1 / 3,
            "max": 1.0,
        }
        assert arm.parsed_executions == {
            "count": 3,
            "undefined": 0,
            "min": 1,
            "median": 2,
            "max": 2,
        }

    def test_a_variance_needs_at_least_two_runs(self, variance: ModuleType) -> None:
        with pytest.raises(variance.VarianceProtocolError, match="at least two"):
            variance.compare_arm("A", self._three(variance)[:1])

    def test_runs_of_one_arm_must_share_one_configuration(self, variance: ModuleType) -> None:
        runs = self._three(variance)
        other = _run(variance, "A-04", [_result("frame-1", {})], digest="sha256:other")

        with pytest.raises(variance.VarianceProtocolError, match="configuration"):
            variance.compare_arm("A", [*runs, other])

    def test_a_label_names_one_run_only(self, variance: ModuleType) -> None:
        runs = self._three(variance)

        with pytest.raises(variance.VarianceProtocolError, match="A-01"):
            variance.compare_arm("A", [*runs, runs[0]])


class TestHypotheses:
    def _arm(self, variance: ModuleType, *frames: Mapping[str, Sequence[ChosenClaim]]) -> object:
        runs = [
            _run(variance, f"B-{index:02d}", [_result("frame-1", claims)])
            for index, claims in enumerate(frames, start=1)
        ]
        return variance.compare_arm("B", runs)

    def test_h1_is_corroborated_only_when_every_pair_agrees_exactly(
        self, variance: ModuleType
    ) -> None:
        same = {"region-0001": [CHAIR]}

        assert variance.h1_verdict(self._arm(variance, same, same, same)) == "corroborated"
        assert (
            variance.h1_verdict(self._arm(variance, same, same, {"region-0001": [STOOL]}))
            == "rejected"
        )

    def test_h1_cannot_be_decided_without_any_claim(self, variance: ModuleType) -> None:
        empty: Mapping[str, Sequence[ChosenClaim]] = {"region-0001": []}

        assert variance.h1_verdict(self._arm(variance, empty, empty, empty)) == "undetermined"

    def test_h2_reports_the_gap_between_the_tuple_and_its_content_components(
        self, variance: ModuleType
    ) -> None:
        arm = self._arm(
            variance,
            {"region-0001": [("chair", "primary", "furniture", 0.5)]},
            {"region-0001": [("chair", "primary", "furniture", 0.75)]},
            {"region-0001": [("chair", "primary", "furniture", 0.5)]},
        )

        evidence = variance.h2_evidence(arm)

        assert evidence["median_tuple"] == 0.0
        assert evidence["median_hypothesis_role_category"] == 1.0
        assert evidence["median_difference"] == 1.0
        assert evidence["claims_with_confidence"] == 3
        # A issue não quantifica "materialmente": o relatório dá os números, não um veredito.
        assert "verdict" not in evidence


def _region_dir(run_dir: Path) -> Path:
    return run_dir.with_name(run_dir.name + "-regions")


def _write_run(
    run_dir: Path, results: Sequence[PerceptionResult], *, digest: str = "sha256:config"
) -> Path:
    regional = RegionPerceptionRunWriter(output_dir=_region_dir(run_dir))
    regional.write_manifest(
        RegionPerceptionRunManifest(
            run_id="regional",
            run_index=1,
            sequence_name="corridor-02",
            selection_id="sha256:selection",
            sequence_artifact_id="sequence-20",
            source_observation_ids=tuple(str(item.source_observation_id) for item in results),
            created_at="fixture",
            prepared_image_id="prepared",
            discovery_backend_id="fake",
            discovery_model="fake",
            discovery_version="1",
            discovery_configuration_fingerprint="cfg",
            normalization_policy_fingerprint="norm",
        )
    )
    for result in results:
        regional.write_frame(
            PreparedImage(
                source_observation_id=result.source_observation_id,
                payload_reference="source.png",
                width=20,
                height=20,
            )
        )
        for region in result.regions:
            regional.write_region(
                replace(region, source_observation_id=result.source_observation_id)
            )
    regional.finalize()
    writer = SemanticInterpretationRunWriter(
        output_dir=run_dir,
        sequence_name="corridor-02",
        run_id=SemanticInterpretationRunId("run"),
        run_index=1,
        sequence_artifact_id="sequence-20",
        selection_id="sha256:selection",
        region_artifact_id="regional",
        configuration_digest=digest,
    )
    write_semantic_views(writer, results, "run")
    writer.finalize()
    return run_dir


class TestPersistedRuns:
    def test_a_run_is_read_back_from_its_perception_artifact(
        self, variance: ModuleType, tmp_path: Path
    ) -> None:
        run_dir = _write_run(
            tmp_path / "A-01",
            [
                _result("frame-1", {"region-0001": [CHAIR, STOOL]}),
                _result("frame-2", {"region-0002": [DOOR]}),
            ],
        )

        run = variance.load_run("A-01", run_dir, region_run_dir=_region_dir(run_dir))

        assert run.label == "A-01"
        assert run.configuration_digest == "sha256:config"
        assert run.run_id == "run"
        assert run.content_hash.startswith("sha256:")
        assert sorted(run.claims) == ["frame-1", "frame-2"]
        assert len(run.claims["frame-1"]) == 2
        assert run.attempts.attempted == 2

    def test_a_run_that_does_not_verify_is_refused(
        self, variance: ModuleType, tmp_path: Path
    ) -> None:
        run_dir = _write_run(tmp_path / "A-01", [_result("frame-1", {"region-0001": [CHAIR]})])
        results = run_dir / "outputs" / "semantic-observations.jsonl"
        results.write_bytes(results.read_bytes() + b"\n")

        with pytest.raises(variance.VarianceProtocolError, match="integrity"):
            variance.load_run("A-01", run_dir, region_run_dir=_region_dir(run_dir))

    def test_the_report_command_writes_every_arm_and_both_hypotheses(
        self, variance: ModuleType, tmp_path: Path
    ) -> None:
        same = [_result("frame-1", {"region-0001": [CHAIR, STOOL]})]
        other = [_result("frame-1", {"region-0001": [CHAIR, SEAT]})]
        dirs = {
            name: _write_run(tmp_path / name, results)
            for name, results in (
                ("A-01", same),
                ("A-02", other),
                ("A-03", same),
                ("B-01", same),
                ("B-02", same),
                ("B-03", same),
            )
        }
        output = tmp_path / "report"

        code = variance.main(
            [
                "--arm",
                f"A={dirs['A-01']},{dirs['A-02']},{dirs['A-03']}",
                "--arm",
                f"B={dirs['B-01']},{dirs['B-02']},{dirs['B-03']}",
                "--region-arm",
                "A=" + ",".join(str(_region_dir(dirs[f"A-{i:02d}"])) for i in range(1, 4)),
                "--region-arm",
                "B=" + ",".join(str(_region_dir(dirs[f"B-{i:02d}"])) for i in range(1, 4)),
                "--output",
                str(output),
            ]
        )

        assert code == 0
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert set(report["arms"]) == {"A", "B"}
        assert report["arms"]["B"]["spreads"]["tuple"]["min"] == 1.0
        assert report["arms"]["A"]["spreads"]["tuple"]["min"] == pytest.approx(1 / 3)
        assert report["hypotheses"]["H1"]["verdict"] == "corroborated"
        assert report["hypotheses"]["H2"]["claims_with_confidence"] == 0
        assert report["arms_share_configuration"] is True
        assert (output / "report.md").is_file()

    def test_the_report_refuses_to_overwrite_an_earlier_one(
        self, variance: ModuleType, tmp_path: Path
    ) -> None:
        same = [_result("frame-1", {"region-0001": [CHAIR]})]
        first, second = _write_run(tmp_path / "1", same), _write_run(tmp_path / "2", same)
        output = tmp_path / "report"
        output.mkdir()

        with pytest.raises(variance.VarianceProtocolError, match="exists"):
            variance.main(
                [
                    "--arm",
                    f"A={first},{second}",
                    "--region-arm",
                    f"A={_region_dir(first)},{_region_dir(second)}",
                    "--output",
                    str(output),
                ]
            )
