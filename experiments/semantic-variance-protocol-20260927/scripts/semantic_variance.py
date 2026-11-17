"""Variance of canonical semantic claims across repeated Visual Perception runs (issue #580).

The analysis reads the ``SemanticInterpretationArtifact`` of every repetition
and its explicitly selected regional artifact. It pairs every two repetitions of an arm
and reports, per pair and as min/median/max over all pairs, how many canonical claims agree.

Canonicalization is exactly the one of
``experiments/e2e-real-canonical-run-20260923/scripts-run5/15_visual_perception_content_equivalence.py``,
which measured the 29/90 baseline of run0005: a claim is the tuple
``(source_observation_id, region_id, hypothesis, role, category, confidence)``, each run's
claims of one frame form a set, and a pair agrees on ``matched / (matched + only_first +
only_second)`` summed over the frames present in both runs. Two deliberate differences, both
about reporting and neither about the tuple:

- ``0/0`` (no claim on either side) is undefined, never ``1.0``: a pair that produced nothing
  did not agree on anything;
- a frame missing from one run and a frame whose regions differ are counted separately
  (script 15 added both to one ``region mismatches`` count).

Besides the full tuple, every claim is projected onto each component (``hypothesis``, ``role``,
``category``, ``confidence``), onto ``hypothesis+role+category`` (H2's comparison) and onto
``slot`` (just the frame and region that received a claim), always keyed by frame and region,
and agreement is computed the same way over the projected sets.

Usage:
    python semantic_variance.py --arm A=DIR,DIR,DIR --arm B=DIR,DIR,DIR --output DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

from contextmap.evaluation import SemanticAttemptAccounting, account_semantic_attempts
from contextmap.region_perception import (
    BoundingBox2D,
    Region2D,
    RegionId,
    RegionPerceptionRunReader,
)
from contextmap.runtime.executors import inventory_digest
from contextmap.semantic_interpretation import SemanticClaim, SemanticInterpretationRunReader
from contextmap.visual_perception import PerceptionResult

EXPERIMENT = "semantic-variance-protocol-20260927"
ISSUE = 580

COMPONENTS = ("hypothesis", "role", "category", "confidence")
"""Content components of a canonical claim, in the order of script 15's tuple."""

PROJECTIONS: dict[str, tuple[str, ...]] = {
    "tuple": COMPONENTS,
    "hypothesis": ("hypothesis",),
    "role": ("role",),
    "category": ("category",),
    "confidence": ("confidence",),
    "hypothesis+role+category": ("hypothesis", "role", "category"),
    "slot": (),
}
"""Components each agreement is computed over; every projection is keyed by frame and region."""

EXACT_PROJECTION = "tuple"
CONTENT_PROJECTION = "hypothesis+role+category"


class VarianceProtocolError(Exception):
    """Raised when runs cannot be compared as repetitions of one configuration."""


class ClaimKey(NamedTuple):
    """The canonical identity of one claim, exactly script 15's ``_claim_key``."""

    source_observation_id: str
    region_id: str | None
    hypothesis: str
    role: str
    category: str | None
    confidence: float | None


RegionSignature = tuple[BoundingBox2D, str | None, bool]
"""What script 15 compared of a region: its box, its kind and whether it was accepted."""


def claim_key(claim: SemanticClaim) -> ClaimKey:
    """Return the canonical identity script 15 gave a claim."""
    return ClaimKey(
        source_observation_id=str(claim.source_observation_id),
        region_id=claim.region_id,
        hypothesis=claim.hypothesis,
        role=claim.role.value,
        category=claim.category,
        confidence=claim.confidence,
    )


def region_signature(region: Region2D) -> RegionSignature:
    """Return what script 15 compared of a region; embeddings and masks are left out."""
    return (region.bounding_box, region.region_kind, region.is_accepted)


@dataclass(frozen=True, kw_only=True)
class RunContent:
    """The canonical content of one repetition, reduced to what the protocol compares.

    Attributes:
        label: Name of the repetition inside the report, for example ``"A-01"``.
        configuration_digest: Resolved pipeline configuration digest the run recorded.
        attempts: Semantic requests the run attempted, parsed and failed to parse.
        claims: Canonical claims per ``source_observation_id``.
        regions: Region signatures per ``source_observation_id``, in the run's order.
        run_id: The run id its manifest records, when read from an artifact.
        content_hash: Digest of its contractual file inventory, when read from an artifact.
        location: Where it was read from, when read from an artifact.
    """

    label: str
    configuration_digest: str
    attempts: SemanticAttemptAccounting
    claims: Mapping[str, frozenset[ClaimKey]]
    regions: Mapping[str, tuple[RegionSignature, ...]]
    run_id: str | None = None
    content_hash: str | None = None
    location: str | None = None

    @property
    def claims_with_confidence(self) -> int:
        """Count the canonical claims that carry a confidence at all."""
        return sum(1 for keys in self.claims.values() for key in keys if key.confidence is not None)


def run_content(
    label: str,
    results: Iterable[PerceptionResult],
    *,
    configuration_digest: str,
    attempts: SemanticAttemptAccounting,
    run_id: str | None = None,
    content_hash: str | None = None,
    location: str | None = None,
) -> RunContent:
    """Reduce the results of one run to its canonical content.

    Raises:
        VarianceProtocolError: If one frame has more than one result.
    """
    claims: dict[str, frozenset[ClaimKey]] = {}
    regions: dict[str, tuple[RegionSignature, ...]] = {}
    for result in results:
        frame = str(result.source_observation_id)
        if frame in claims:
            raise VarianceProtocolError(f"{label}: frame {frame!r} has more than one result")
        claims[frame] = frozenset(claim_key(claim) for claim in result.claims)
        regions[frame] = tuple(region_signature(region) for region in result.regions)
    return RunContent(
        label=label,
        configuration_digest=configuration_digest,
        attempts=attempts,
        claims=claims,
        regions=regions,
        run_id=run_id,
        content_hash=content_hash,
        location=location,
    )


def load_run(
    label: str, run_dir: Path, *, region_run_dir: Path, location: str | None = None
) -> RunContent:
    """Read one repetition from its ``SemanticInterpretationArtifact``.

    Args:
        label: Name of the repetition inside the report.
        run_dir: The semantic artifact directory.
        region_run_dir: Explicit regional artifact named by the semantic lineage.
        location: What the report records as its location, for example a path relative to the
            outputs root that holds no machine path; defaults to ``run_dir`` as given.

    Raises:
        VarianceProtocolError: If the artifact does not pass its own integrity check: a run
            that does not verify is never evidence.
    """
    reader = SemanticInterpretationRunReader(run_dir)
    regions = RegionPerceptionRunReader(region_run_dir)
    problems = reader.verify_integrity()
    if problems:
        raise VarianceProtocolError(f"{label}: {run_dir} fails its integrity check: {problems}")
    manifest = reader.manifest
    if regions.manifest.run_id != manifest.region_artifact_id:
        raise VarianceProtocolError("selected region producer does not match semantic lineage")
    problems = regions.verify_integrity()
    if problems:
        raise VarianceProtocolError(f"regional artifact fails integrity check: {problems}")
    views = []
    all_regions = regions.read_regions()
    for item in reader.iter_observations():
        observed = tuple(
            region
            for region in all_regions
            if region.source_observation_id == item.source_observation_id
        )
        slots = {
            region.region_id: RegionId(f"region-slot-{index:04d}")
            for index, region in enumerate(observed)
        }
        # A identidade global do produtor difere por repetição; a comparação usa a posição
        # regional explícita e verifica sua geometria separadamente, como o protocolo original.
        views.append(
            PerceptionResult(
                source_observation_id=item.source_observation_id,
                sequence_artifact_id=manifest.sequence_artifact_id,
                region_artifact_id=manifest.region_artifact_id,
                semantic_interpretation_artifact_id=str(manifest.run_id),
                claims=tuple(
                    replace(
                        claim, region_id=None if claim.region_id is None else slots[claim.region_id]
                    )
                    for claim in item.claims
                ),
                scene_context=item.scene_context,
                regions=tuple(
                    replace(region, region_id=slots[region.region_id]) for region in observed
                ),
            )
        )
    return run_content(
        label,
        views,
        configuration_digest=manifest.configuration_digest,
        attempts=account_semantic_attempts(reader),
        run_id=str(manifest.run_id),
        content_hash=inventory_digest(manifest.file_inventory),
        location=str(run_dir) if location is None else location,
    )


@dataclass(frozen=True)
class Agreement:
    """Set agreement of one projection between two runs."""

    matched: int
    only_first: int
    only_second: int

    @property
    def union(self) -> int:
        """Return the size of the union of both sets."""
        return self.matched + self.only_first + self.only_second

    @property
    def rate(self) -> float | None:
        """Return ``matched / union``, or ``None`` when neither run had anything to compare."""
        return None if self.union == 0 else self.matched / self.union

    @property
    def exact(self) -> bool:
        """Whether both runs produced the same non-empty set."""
        return self.union > 0 and self.matched == self.union


@dataclass(frozen=True, kw_only=True)
class PairComparison:
    """Every agreement between two repetitions.

    Attributes:
        first: Label of the first run.
        second: Label of the second run.
        frames_compared: Frames present in both runs; only they contribute claims.
        region_mismatches: Compared frames whose regions differ, which makes pairing claims by
            ``region_id`` meaningless for that frame.
        missing_frames: Frames present in only one run.
        agreements: Agreement per projection of :data:`PROJECTIONS`.
    """

    first: str
    second: str
    frames_compared: int
    region_mismatches: int
    missing_frames: int
    agreements: Mapping[str, Agreement]


def _project(
    keys: frozenset[ClaimKey], components: tuple[str, ...]
) -> frozenset[tuple[object, ...]]:
    """Project claims onto some components, always keyed by frame and region."""
    return frozenset(
        (
            key.source_observation_id,
            key.region_id,
            *(getattr(key, component) for component in components),
        )
        for key in keys
    )


def compare_pair(first: RunContent, second: RunContent) -> PairComparison:
    """Compare two repetitions frame by frame, over every projection."""
    counts = {name: [0, 0, 0] for name in PROJECTIONS}
    frames_compared = region_mismatches = missing_frames = 0
    for frame in sorted(set(first.claims) | set(second.claims)):
        if frame not in first.claims or frame not in second.claims:
            missing_frames += 1
            continue
        frames_compared += 1
        if first.regions[frame] != second.regions[frame]:
            region_mismatches += 1
        for name, components in PROJECTIONS.items():
            ours = _project(first.claims[frame], components)
            theirs = _project(second.claims[frame], components)
            counts[name][0] += len(ours & theirs)
            counts[name][1] += len(ours - theirs)
            counts[name][2] += len(theirs - ours)
    return PairComparison(
        first=first.label,
        second=second.label,
        frames_compared=frames_compared,
        region_mismatches=region_mismatches,
        missing_frames=missing_frames,
        agreements={name: Agreement(*values) for name, values in counts.items()},
    )


def spread(values: Sequence[float | int | None]) -> dict[str, Any]:
    """Return min, median and max over the defined values, and how many were undefined.

    The median of an even count is the mean of the two middle values
    (:func:`statistics.median`).
    """
    defined = [value for value in values if value is not None]
    return {
        "count": len(values),
        "undefined": len(values) - len(defined),
        "min": min(defined) if defined else None,
        "median": statistics.median(defined) if defined else None,
        "max": max(defined) if defined else None,
    }


@dataclass(frozen=True, kw_only=True)
class ArmComparison:
    """All pairs of one arm and their spread.

    Attributes:
        name: The arm, for example ``"A"``.
        runs: Its repetitions, ordered by label.
        pairs: Every unordered pair of repetitions, once, in label order.
        spreads: Min/median/max of each projection's agreement over the pairs.
        parsed_executions: Min/median/max over the runs of the parsed semantic executions.
        attempted_executions: Min/median/max over the runs of the attempted semantic requests.
    """

    name: str
    runs: tuple[RunContent, ...]
    pairs: tuple[PairComparison, ...]
    spreads: Mapping[str, dict[str, Any]]
    parsed_executions: dict[str, Any]
    attempted_executions: dict[str, Any]

    @property
    def configuration_digest(self) -> str:
        """Return the configuration every run of the arm shares."""
        return self.runs[0].configuration_digest


def compare_arm(name: str, runs: Sequence[RunContent]) -> ArmComparison:
    """Compare every pair of repetitions of one arm.

    Raises:
        VarianceProtocolError: If the arm has fewer than two runs, a label repeats, or the runs
            do not share one configuration digest (they would not be repetitions).
    """
    labels = [run.label for run in runs]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if repeated:
        raise VarianceProtocolError(f"arm {name}: labels name more than one run: {repeated}")
    if len(runs) < 2:
        raise VarianceProtocolError(
            f"arm {name}: a variance needs at least two runs, got {len(runs)}"
        )
    digests = sorted({run.configuration_digest for run in runs})
    if len(digests) != 1:
        raise VarianceProtocolError(
            f"arm {name}: its runs do not share one configuration ({digests}), so they are not "
            "repetitions of one arm"
        )
    ordered = tuple(sorted(runs, key=lambda run: run.label))
    pairs = tuple(compare_pair(first, second) for first, second in combinations(ordered, 2))
    return ArmComparison(
        name=name,
        runs=ordered,
        pairs=pairs,
        spreads={
            projection: spread([pair.agreements[projection].rate for pair in pairs])
            for projection in PROJECTIONS
        },
        parsed_executions=spread([run.attempts.parsed for run in ordered]),
        attempted_executions=spread([run.attempts.attempted for run in ordered]),
    )


def h1_verdict(arm: ArmComparison) -> str:
    """Apply the issue's H1 criterion to the arm with the determinism flags.

    H1 is corroborated when every pair of the arm agrees exactly on the full tuple and rejected
    as soon as any pair diverges. With no claim to compare it cannot be decided.

    Returns:
        ``"corroborated"``, ``"rejected"`` or ``"undetermined"``.
    """
    agreements = [pair.agreements[EXACT_PROJECTION] for pair in arm.pairs]
    if any(agreement.union > 0 and not agreement.exact for agreement in agreements):
        return "rejected"
    if agreements and all(agreement.exact for agreement in agreements):
        return "corroborated"
    return "undetermined"


def h2_evidence(arm: ArmComparison) -> dict[str, Any]:
    """Return the numbers H2 is judged on, in the arm without the determinism flags.

    The issue asks whether ``hypothesis+role+category`` agrees "materially" more than the full
    tuple and does not quantify "materially", so this reports the numbers and no verdict.
    ``claims_with_confidence`` makes the confidence component's contribution visible: when it is
    zero, the tuple and ``hypothesis+role+category`` coincide by construction.
    """
    tuple_median = arm.spreads[EXACT_PROJECTION]["median"]
    content_median = arm.spreads[CONTENT_PROJECTION]["median"]
    return {
        "arm": arm.name,
        "median_tuple": tuple_median,
        "median_hypothesis_role_category": content_median,
        "median_difference": (
            None
            if tuple_median is None or content_median is None
            else content_median - tuple_median
        ),
        "pairs": [
            {
                "first": pair.first,
                "second": pair.second,
                "tuple": pair.agreements[EXACT_PROJECTION].rate,
                "hypothesis+role+category": pair.agreements[CONTENT_PROJECTION].rate,
            }
            for pair in arm.pairs
        ],
        "claims_with_confidence": sum(run.claims_with_confidence for run in arm.runs),
    }


def _agreement_record(agreement: Agreement) -> dict[str, Any]:
    return {
        "matched": agreement.matched,
        "only_first": agreement.only_first,
        "only_second": agreement.only_second,
        "union": agreement.union,
        "rate": agreement.rate,
    }


def _run_record(run: RunContent) -> dict[str, Any]:
    return {
        "label": run.label,
        "location": run.location,
        "run_id": run.run_id,
        "content_hash": run.content_hash,
        "configuration_digest": run.configuration_digest,
        "frames": len(run.claims),
        "claims": sum(len(keys) for keys in run.claims.values()),
        "claims_with_confidence": run.claims_with_confidence,
        "semantic_attempts": {
            "attempted": run.attempts.attempted,
            "parsed": run.attempts.parsed,
            "parse_failed": run.attempts.parse_failed,
            "measurement_status": run.attempts.measurement_status.value,
        },
    }


def _arm_record(arm: ArmComparison) -> dict[str, Any]:
    return {
        "configuration_digest": arm.configuration_digest,
        "runs": [_run_record(run) for run in arm.runs],
        "pairs": [
            {
                "first": pair.first,
                "second": pair.second,
                "frames_compared": pair.frames_compared,
                "region_mismatches": pair.region_mismatches,
                "missing_frames": pair.missing_frames,
                "agreements": {
                    name: _agreement_record(agreement)
                    for name, agreement in pair.agreements.items()
                },
            }
            for pair in arm.pairs
        ],
        "spreads": dict(arm.spreads),
        "parsed_executions": arm.parsed_executions,
        "attempted_executions": arm.attempted_executions,
    }


def build_report(arms: Mapping[str, ArmComparison]) -> dict[str, Any]:
    """Assemble the JSON-compatible report of every compared arm.

    H1 is decided on arm ``B`` and H2's numbers come from arm ``A``, when those arms are present.
    """
    hypotheses: dict[str, Any] = {}
    if "B" in arms:
        hypotheses["H1"] = {"arm": "B", "verdict": h1_verdict(arms["B"])}
    if "A" in arms:
        hypotheses["H2"] = h2_evidence(arms["A"])
    return {
        "experiment": EXPERIMENT,
        "issue": ISSUE,
        "canonicalization": (
            "(source_observation_id, region_id, hypothesis, role, category, confidence) as in "
            "experiments/e2e-real-canonical-run-20260923/scripts-run5/"
            "15_visual_perception_content_equivalence.py; agreement = matched / (matched + "
            "only_first + only_second) over the frames present in both runs; 0/0 is undefined"
        ),
        "projections": {name: list(components) for name, components in PROJECTIONS.items()},
        "arms": {name: _arm_record(arm) for name, arm in arms.items()},
        "arms_share_configuration": len({arm.configuration_digest for arm in arms.values()}) == 1,
        "hypotheses": hypotheses,
    }


def _number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the report as a short human summary; ``report.json`` stays the record."""
    lines = [
        f"# Protocolo de variância semântica (issue #{report['issue']})",
        "",
        f"Canonicalização: {report['canonicalization']}.",
        "",
    ]
    for name, arm in report["arms"].items():
        runs = arm["runs"]
        lines += [
            f"## Braço {name}",
            "",
            f"{len(runs)} repetições, {len(arm['pairs'])} pares, configuração "
            f"`{arm['configuration_digest']}`.",
            "",
            "| Projeção | mín | mediana | máx | pares indefinidos |",
            "|---|---|---|---|---|",
        ]
        for projection, values in arm["spreads"].items():
            lines.append(
                f"| {projection} | {_number(values['min'])} | {_number(values['median'])} | "
                f"{_number(values['max'])} | {values['undefined']} |"
            )
        lines += [
            "",
            "| Par | tupla | regiões divergentes | frames ausentes |",
            "|---|---|---|---|",
        ]
        for pair in arm["pairs"]:
            exact = pair["agreements"][EXACT_PROJECTION]
            lines.append(
                f"| {pair['first']} vs {pair['second']} | {exact['matched']}/{exact['union']} | "
                f"{pair['region_mismatches']} | {pair['missing_frames']} |"
            )
        lines += [
            "",
            "| Repetição | tentadas | parseadas | falhas de parser |",
            "|---|---|---|---|",
        ]
        for run in runs:
            attempts = run["semantic_attempts"]
            lines.append(
                f"| {run['label']} | {attempts['attempted']} | {attempts['parsed']} | "
                f"{attempts['parse_failed']} |"
            )
        lines.append("")
    hypotheses = report["hypotheses"]
    lines += ["## Hipóteses", ""]
    if "H1" in hypotheses:
        lines.append(f"- H1 (braço {hypotheses['H1']['arm']}): **{hypotheses['H1']['verdict']}**.")
    if "H2" in hypotheses:
        h2 = hypotheses["H2"]
        lines.append(
            f"- H2 (braço {h2['arm']}): mediana da tupla {_number(h2['median_tuple'])}, de "
            f"hypothesis+role+category {_number(h2['median_hypothesis_role_category'])}, "
            f"diferença {_number(h2['median_difference'])}; claims com confidence: "
            f"{h2['claims_with_confidence']}. Sem veredito automático: a issue não quantifica "
            "'materialmente'."
        )
    shared = "sim" if report["arms_share_configuration"] else "não"
    lines.append(f"- Braços com a mesma configuração: {shared}.")
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], output: Path) -> None:
    """Write ``report.json`` and ``report.md`` into a directory that must not exist yet.

    Raises:
        VarianceProtocolError: If ``output`` exists: an earlier report is never overwritten.
    """
    if output.exists():
        raise VarianceProtocolError(f"{output} already exists: a report is never overwritten")
    output.mkdir(parents=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")


def _arm_argument(value: str) -> tuple[str, list[Path]]:
    name, separator, directories = value.partition("=")
    if not separator or not name or not directories:
        raise argparse.ArgumentTypeError(f"expected NAME=DIR[,DIR...], got {value!r}")
    return name, [Path(directory) for directory in directories.split(",")]


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the given runs, arm by arm, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        action="append",
        type=_arm_argument,
        required=True,
        help="NAME=DIR[,DIR...]: explicit semantic artifacts for one arm's repetitions",
    )
    parser.add_argument(
        "--region-arm",
        action="append",
        type=_arm_argument,
        required=True,
        help="NAME=DIR[,DIR...]: explicit regional artifacts paired with semantic repetitions",
    )
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.output.exists():
        raise VarianceProtocolError(
            f"{options.output} already exists: a report is never overwritten"
        )
    regional = dict(options.region_arm)
    for name, directories in options.arm:
        if name not in regional or len(regional[name]) != len(directories):
            raise VarianceProtocolError(
                f"arm {name!r} needs one explicit regional artifact per repetition"
            )
    arms = {
        name: compare_arm(
            name,
            [
                load_run(f"{name}-{index:02d}", directory, region_run_dir=regional[name][index - 1])
                for index, directory in enumerate(directories, start=1)
            ],
        )
        for name, directories in options.arm
    }
    write_report(build_report(arms), options.output)
    print(f"report: {options.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
