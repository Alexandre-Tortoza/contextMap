"""C3: valida o deskew in-repo (``pose-interpolation-deskew-v1``) na janela 628-637 s.

Entradas (somente leitura): a SequenceArtifact e a run de State Estimation de
``workspace-old/frame-10_31-raw`` (corridor-02, ingestão da janela 631 s ± 3 s), com a mesma
política de lookup e de correção que o mapa E0 usou (lidas do ``config.json`` do run).

Três braços, a mesma métrica da investigação (``p3_map_consistency`` /
``p3_time_offset_deskew``: planos locais PCA k=12 no scan de referência mais próximo da imagem
55, planaridade < 0,02, raio 0,25 m, pontos a < 12 m do sensor, scans a ≤ 3 s da referência,
só scans com ≥ 200 correspondências):

* ``in_repo_raw``: ``assemble_geometry_inputs_from_artifacts`` sem deskew (tem de reproduzir o
  ``geometry.bin`` do E0 byte a byte);
* ``in_repo_deskew``: o mesmo caminho com ``deskew=PoseInterpolationDeskewPolicy`` (bins de
  1 ms, arredondamento ao múltiplo mais próximo);
* ``offline_deskew``: o ``place(..., deskew=True)`` da investigação (bins de 5 ms, centro do bin,
  pose GT sem regra de gap), recalculado aqui para comparar no mesmo ambiente.

Além disso mede o ``ScanOverlapReport`` in-repo (agora com p90) dos dois mapas in-repo,
escritos como ``GeometricMapArtifact`` em ``--work-dir`` (fora do repositório).

Uso::

    PYTHONPATH=src python \
        experiments/correction-benchmark-10_31-20260930/deskew/validate_deskew.py \
        --work-dir /tmp/deskew-work
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/semantic-geometry-investigation-20260930/scripts"))

from common import (  # noqa: E402
    IMAGE_ID,
    Trajectory,
    load_scan_payload,
    run_dir,
    seq_dir,
    sequence_index,
    static_transform,
    ts_ns,
)
from p3_map_consistency import MAX_RANGE_M, plane_model, residuals  # noqa: E402
from p3_time_offset_deskew import place  # noqa: E402

from contextmap.evaluation import (  # noqa: E402
    GeometricMappingProtocol,
    encode_geometric_mapping_report,
    evaluate_geometric_mapping,
)
from contextmap.geometric_mapping import (  # noqa: E402
    GeometricMapArtifactReader,
    GeometricMapArtifactWriter,
    GeometricMapRunId,
    GeometryInputPlan,
    MapDebugLevel,
    MapId,
    PointTimeReference,
    PointTimeUnit,
    PoseInterpolationDeskewPolicy,
    accumulate_plan,
    assemble_geometry_inputs_from_artifacts,
    transform_scans,
)
from contextmap.geometric_mapping.serialization import (  # noqa: E402
    decode_lookup_policy,
    decode_motion_correction_policy,
)
from contextmap.ingestion import FullSequenceSelection, SequenceArtifactReader  # noqa: E402
from contextmap.state_estimation import StateEstimationRunReader  # noqa: E402

RESULTS = HERE / "results"
WINDOW_S = 3.0  # scans a <= 3 s da referência, como na investigação
MIN_MATCHED = 200  # idem: scans com menos correspondências não entram no agregado
# Valores gravados pela investigação (p3_time_offset_deskew.json, delta 0).
INVESTIGATION_OFFLINE = {"median_m": 0.02161, "p90_m": 0.06852}
INVESTIGATION_RAW = {"median_m": 0.03282, "p90_m": 0.10048}
POLICY = PoseInterpolationDeskewPolicy(
    time_field="time",
    time_unit=PointTimeUnit.SECONDS,
    reference=PointTimeReference.SCAN_TIMESTAMP,
    time_bin_ns=1_000_000,
)
# Protocolo do ScanOverlapReport in-repo: declarado aqui (a avaliação não tem defaults). Todos os
# pares adjacentes do mapa; raio e planaridade próximos aos da métrica da investigação.
PROTOCOL = GeometricMappingProtocol(
    structure_point_stride=100,
    trace_sample_count=8,
    transform_tolerance_m=1e-9,
    adjacent_scan_lag=1,
    coherence_pair_count=100,
    coherence_points_per_scan=1000,
    plane_neighbour_count=12,
    max_plane_curvature=0.02,
    max_correspondence_distance_m=0.25,
    ghost_residual_m=0.05,
    density_voxel_m=0.10,
)


class _HashSink:
    """Sink binário que só calcula o sha256 do que o acumulador escreve."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.size += len(data)
        return len(data)


def _plan(deskew: PoseInterpolationDeskewPolicy | None) -> tuple[GeometryInputPlan, float]:
    config = json.loads((run_dir("raw") / "geometric_mapping/config.json").read_text())
    started = time.perf_counter()
    plan = assemble_geometry_inputs_from_artifacts(
        sequence=SequenceArtifactReader(seq_dir("raw")),
        selection=FullSequenceSelection(),
        run=StateEstimationRunReader(run_dir("raw") / "state_estimation"),
        pose_lookup=decode_lookup_policy(config["pose_lookup"]),
        motion_correction_policy=decode_motion_correction_policy(
            config["motion_correction_policy"]
        ),
        deskew=deskew,
    )
    return plan, time.perf_counter() - started


def _geometry_hash(plan: GeometryInputPlan) -> dict[str, object]:
    sink = _HashSink()
    accumulated = accumulate_plan(
        plan,
        map_id=MapId("validation"),
        sink=sink,  # type: ignore[arg-type]
        aggregation=None,
        configuration_fingerprint=None,
        code_version=None,
    )
    return {
        "sha256": f"sha256:{sink.digest.hexdigest()}",
        "size_bytes": sink.size,
        "point_count": accumulated.geometric_map.point_count,
        "scan_count": len(accumulated.scans),
    }


def _scans(plan: GeometryInputPlan, wanted: set[str]) -> dict[str, dict[str, np.ndarray]]:
    """Coordenadas de fonte e de mapa dos scans ``wanted``, indexadas pelo índice no scan."""
    out = {}
    for scan in transform_scans(plan):
        oid = str(scan.observation_id)
        if oid not in wanted:
            continue
        out[oid] = {
            "index": np.frombuffer(scan.source_point_indices, dtype=np.int64).copy(),
            "source": np.frombuffer(scan.source_coordinates_m, dtype=np.float64).reshape(-1, 3),
            "map": np.frombuffer(scan.map_coordinates_m, dtype=np.float64).reshape(-1, 3),
        }
    return out


def _metric(maps: dict[str, np.ndarray], near: dict[str, np.ndarray], ref_id: str) -> dict:
    ref = maps[ref_id][near[ref_id]]
    tree, normals, planarity = plane_model(ref)
    pooled, used = [], 0
    for oid in sorted(maps):
        if oid == ref_id:
            continue
        r = residuals(maps[oid][near[oid]], ref, tree, normals, planarity)
        if len(r) >= MIN_MATCHED:
            pooled.append(r)
            used += 1
    values = np.concatenate(pooled)
    return {
        "scans_compared": used,
        "matched": len(values),
        "median_m": round(float(np.median(values)), 5),
        "p90_m": round(float(np.percentile(values, 90)), 5),
        "p95_m": round(float(np.percentile(values, 95)), 5),
    }


def _relative(value: float, reference: float) -> float:
    return round(abs(value - reference) / reference, 4)


def _overlap(plan: GeometryInputPlan, work: Path, name: str) -> dict[str, object]:
    out = work / name
    GeometricMapArtifactWriter(
        output_dir=out,
        sequence_name="corridor-02",
        run_id=GeometricMapRunId(name),
        run_index=1,
        debug_level=MapDebugLevel.NONE,
    ).finalize(plan=plan, aggregation=None, code_version=None)
    with GeometricMapArtifactReader(out) as reader:
        report = evaluate_geometric_mapping(reader=reader, protocol=PROTOCOL)
    encoded = encode_geometric_mapping_report(report)["overlap"]
    return {
        "pair_count": len(encoded["pairs"]),
        "overlap_fraction": encoded["overlap_fraction"],
        "inconsistent_fraction": encoded["inconsistent_fraction"],
        "pooled_residual_m": encoded["pooled_residual_m"],
        "pooled_p90_residual_m": encoded["pooled_p90_residual_m"],
        "by_motion_correction": encoded["by_motion_correction"],
        "transform_trace_within_tolerance": report.transform_trace.within_tolerance,
        "integrity_problems": list(report.structure.integrity_problems),
    }


def main() -> None:
    """Mede os três braços e grava ``results/deskew_validation.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, default=None, help="onde gravar os 2 mapas")
    args = parser.parse_args()

    raw_plan, raw_s = _plan(None)
    deskew_plan, deskew_s = _plan(POLICY)
    e0 = run_dir("raw") / "geometric_mapping/outputs/geometry.bin"
    e0_hash = f"sha256:{hashlib.sha256(e0.read_bytes()).hexdigest()}"
    raw_geometry, deskew_geometry = _geometry_hash(raw_plan), _geometry_hash(deskew_plan)

    index = {e["observation_id"]: e for e in sequence_index()}
    t_img = ts_ns(index[IMAGE_ID]["timestamp"])
    headers = {
        str(item.observation_id): item.timestamp.total_nanoseconds() for item in raw_plan.inputs
    }
    ref_id = min(headers, key=lambda oid: abs(headers[oid] - t_img))
    chosen = {oid for oid, t in headers.items() if abs(t - headers[ref_id]) <= WINDOW_S * 1e9}
    deskew_ids = {str(item.observation_id) for item in deskew_plan.inputs}
    if not chosen <= deskew_ids:
        raise SystemExit(f"o deskew rejeitou scans da janela: {sorted(chosen - deskew_ids)}")

    raw_scans, deskew_scans = _scans(raw_plan, chosen), _scans(deskew_plan, chosen)
    traj = Trajectory()
    t_epson_vel = static_transform("epson", "cmu_rc1_velodyne")
    maps: dict[str, dict[str, np.ndarray]] = {
        "in_repo_raw": {},
        "in_repo_deskew": {},
        "offline_deskew": {},
    }
    near: dict[str, np.ndarray] = {}
    agreement, ranges = [], []
    for oid in sorted(chosen):
        raw, corrected = raw_scans[oid], deskew_scans[oid]
        if not np.array_equal(raw["index"], corrected["index"]):
            raise SystemExit(f"{oid}: os braços mantiveram pontos diferentes")
        # Filtro de alcance nas coordenadas BRUTAS, igual nos três braços (como na investigação).
        near[oid] = np.linalg.norm(raw["source"], axis=1) < MAX_RANGE_M
        _, payload = load_scan_payload(index[oid])
        times = payload["time"].astype(np.float64)[raw["index"]]
        offline = place(traj, t_epson_vel, headers[oid], times, raw["source"], 0, deskew=True)
        maps["in_repo_raw"][oid] = raw["map"]
        maps["in_repo_deskew"][oid] = corrected["map"]
        maps["offline_deskew"][oid] = offline
        agreement.append(np.linalg.norm(corrected["map"] - offline, axis=1))
        ranges.append(np.linalg.norm(raw["source"], axis=1))
    distances, point_range = np.concatenate(agreement), np.concatenate(ranges)

    metric = {arm: _metric(by_scan, near, ref_id) for arm, by_scan in maps.items()}
    in_repo, offline = metric["in_repo_deskew"], metric["offline_deskew"]

    with tempfile.TemporaryDirectory() as fallback:
        work = args.work_dir if args.work_dir is not None else Path(fallback)
        work.mkdir(parents=True, exist_ok=True)
        overlap = {
            "in_repo_raw": _overlap(raw_plan, work, "in-repo-raw"),
            "in_repo_deskew": _overlap(deskew_plan, work, "in-repo-deskew"),
        }

    result = {
        "experiment": "C3 deskew validation, corridor-02 628-637 s (frame 10:31)",
        "inputs": {
            "sequence_artifact": str(seq_dir("raw").relative_to(ROOT)),
            "state_estimation_run": str((run_dir("raw") / "state_estimation").relative_to(ROOT)),
            "pose_lookup_and_motion_policy_from": str(
                (run_dir("raw") / "geometric_mapping/config.json").relative_to(ROOT)
            ),
        },
        "deskew_policy": {
            "policy_id": POLICY.policy_id,
            "time_field": POLICY.time_field,
            "time_unit": POLICY.time_unit.value,
            "reference": POLICY.reference.value,
            "time_bin_ns": POLICY.time_bin_ns,
        },
        "plans": {
            arm: {
                "accepted_scan_count": len(plan.inputs),
                "rejected": {
                    str(item.observation_id): f"{item.reason.value}: {item.detail}"
                    for item in plan.rejections
                },
                "motion_correction_states": sorted(
                    {item.motion_correction.record.state.value for item in plan.inputs}
                ),
                "assembly_s": round(seconds, 2),
            }
            for arm, plan, seconds in (
                ("in_repo_raw", raw_plan, raw_s),
                ("in_repo_deskew", deskew_plan, deskew_s),
            )
        },
        "geometry": {
            "e0_geometry_bin": e0_hash,
            "in_repo_raw": raw_geometry,
            "in_repo_raw_byte_identical_to_e0": raw_geometry["sha256"] == e0_hash,
            "in_repo_deskew": deskew_geometry,
        },
        "investigation_metric": {
            "reference_scan": ref_id,
            "scans_in_window": len(chosen),
            "max_range_m": MAX_RANGE_M,
            "window_s": WINDOW_S,
            "arms": metric,
            "investigation_recorded": {
                "raw": INVESTIGATION_RAW,
                "offline_deskew": INVESTIGATION_OFFLINE,
            },
            "in_repo_vs_offline_relative_difference": {
                "median": _relative(in_repo["median_m"], offline["median_m"]),
                "p90": _relative(in_repo["p90_m"], offline["p90_m"]),
            },
            "in_repo_vs_investigation_recorded_relative_difference": {
                "median": _relative(in_repo["median_m"], INVESTIGATION_OFFLINE["median_m"]),
                "p90": _relative(in_repo["p90_m"], INVESTIGATION_OFFLINE["p90_m"]),
            },
            "plan_acceptance": {
                "median_le_0_024_m": in_repo["median_m"] <= 0.024,
                "p90_le_0_072_m": in_repo["p90_m"] <= 0.072,
                "within_5_percent_of_offline": max(
                    _relative(in_repo["median_m"], offline["median_m"]),
                    _relative(in_repo["p90_m"], offline["p90_m"]),
                )
                <= 0.05,
            },
        },
        # Os dois deskews diferem só no bin (1 ms ao múltiplo mais próximo, limitado aos pontos,
        # contra 5 ms no centro); o erro angular de tempo cresce com o alcance.
        "point_agreement_in_repo_vs_offline_deskew_m": {
            label: {
                "points": int(selected.sum()),
                "median": float(np.median(distances[selected])),
                "p99": float(np.percentile(distances[selected], 99)),
                "max": float(distances[selected].max()),
            }
            for label, selected in (
                ("all", np.ones(len(distances), dtype=bool)),
                ("range_lt_12m", point_range < MAX_RANGE_M),
                ("range_ge_12m", point_range >= MAX_RANGE_M),
            )
        },
        "scan_overlap_report": {
            "protocol": {
                "adjacent_scan_lag": PROTOCOL.adjacent_scan_lag,
                "coherence_pair_count": PROTOCOL.coherence_pair_count,
                "coherence_points_per_scan": PROTOCOL.coherence_points_per_scan,
                "plane_neighbour_count": PROTOCOL.plane_neighbour_count,
                "max_plane_curvature": PROTOCOL.max_plane_curvature,
                "max_correspondence_distance_m": PROTOCOL.max_correspondence_distance_m,
                "ghost_residual_m": PROTOCOL.ghost_residual_m,
            },
            "arms": overlap,
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "deskew_validation.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["investigation_metric"], indent=2, ensure_ascii=False))
    print(json.dumps(result["geometry"], indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
