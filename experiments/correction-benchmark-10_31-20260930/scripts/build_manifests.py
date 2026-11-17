"""Gera ``manifests/benchmark.json`` (B1) e ``manifests/b2-proposal.json`` (proposta de B2).

Só lê: dataset (sha256 dos YAMLs e do GT; o bag de 24 GB não é re-hasheado, o hash é o que a
ingestão registrou com ``hash_source=true``), artefatos dos runs baseline em
``workspace-old/frame-10_31-{raw,voxel}`` e o estado do repositório (identidade de código =
HEAD + sha256 de ``git diff HEAD`` + arquivos não rastreados relevantes).

    PYTHONPATH=src $MISE_PY \
      experiments/correction-benchmark-10_31-20260930/scripts/build_manifests.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (
    CAMERA_CALIBRATION_ID,
    CONFIG_STACK,
    E0_RUNS,
    FRAME_INDEX,
    IMAGE_ID,
    IMAGE_TIMESTAMP,
    MANIFESTS,
    NEAREST_SCAN_ID,
    POSE_SEQUENCE_ID,
    RELEVANT_UNTRACKED,
    ROOT,
    SEQUENCE_ID,
    code_identity,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)

DATASET = ROOT / "datasets/corridor-02"
# "10:31" e a janela "628-637 s" contam segundos a partir desta época de gravação
# (convenção MM_SS dos bundles frame-10_2x/10_31; 1724621084 + 631 = 1724621715).
RECORDING_EPOCH_S = 1724621084
B2_STEP = 6
B2_FRAMES = tuple(range(37, 80, B2_STEP))


def _ts_ns(stamp: dict[str, Any]) -> int:
    return int(stamp["seconds"]) * 1_000_000_000 + int(stamp["nanoseconds"])


def _sequence_dir(arm: str = "raw") -> Path:
    return E0_RUNS[arm] / f"sequences/corridor-02/{SEQUENCE_ID}"


def _index() -> list[dict[str, Any]]:
    return read_jsonl(_sequence_dir() / "index.jsonl")


def _stage_identity(run: Path, stage: str) -> dict[str, Any]:
    manifest = read_json(run / stage / "manifest.json")
    keys = (
        "run_id",
        "map_id",
        "context_map_id",
        "semantic_map_id",
        "trajectory_id",
        "configuration_fingerprint",
        "configuration_digest",
        "calibration_identity",
        "content_identity",
        "code_version",
        "schema_version",
        "created_at",
        "written_at",
    )
    return {k: manifest[k] for k in keys if k in manifest}


def _models(run: Path) -> dict[str, Any]:
    result = next(r for r in read_jsonl(run / "visual_perception/outputs/results.jsonl"))
    region = result["regions"][0]["provenance"]
    features = {}
    for feature in result["features"]:
        prov = feature["provenance"]
        features.setdefault(
            prov["backend_id"], {"model": prov["model"], "revision": prov["version"]}
        )
    interpretation = read_jsonl(run / "visual_perception/outputs/semantic-interpretations.jsonl")[0]
    scoring = read_json(run / "semantic_scoring/manifest.json")
    return {
        "region_discovery": {
            "backend": region["backend_id"],
            "model": region["model"],
            "revision": region["version"],
            "configuration_fingerprint": region["configuration_fingerprint"],
            "thresholds": {"predicted_iou_threshold": 0.0, "stability_threshold": 0.0},
            "note": "limiares = default do Sam2Config (0/0); nenhum config os sobrescreve "
            "(REPORT §7)",
        },
        "features": features,
        "semantic_interpretation": {
            **interpretation["effective_configuration"],
            "prompt_template_id": interpretation["rendered_prompt"]["template_id"],
            "view_kinds": sorted({v["kind"] for v in interpretation["request"]["visual_views"]}),
            "configuration_fingerprint": interpretation["request"]["configuration_fingerprint"],
            "scene_interpretation": "desligada (estágio removido do preset pelo driver)",
        },
        "semantic_scoring": {
            "scorer": scoring.get("scorer"),
            "text_template": scoring.get("text_template"),
            "embedding_space_id": scoring.get("embedding_space_id"),
        },
    }


def _pose_bracket(run: Path, t_ns: int) -> dict[str, Any]:
    rows = [
        r
        for r in read_jsonl(run / "state_estimation/outputs/poses.jsonl")
        if r.get("validity") == "valid"
    ]
    times = np.sort(np.array([_ts_ns(r["timestamp"]) for r in rows], dtype=np.int64))
    i = int(np.searchsorted(times, t_ns))
    if i == 0 or i >= len(times):
        return {"status": "outside trajectory"}
    gap = int(times[i] - times[i - 1])
    return {
        "bracket_gap_ms": round(gap / 1e6, 1),
        "nearest_pose_delta_ms": round(min(t_ns - times[i - 1], times[i] - t_ns) / 1e6, 2),
        "within_association_pose_gap_450ms": gap <= 450_000_000,
        "within_mapping_pose_gap_400ms": gap <= 400_000_000,
    }


def _nearest_scan(index: list[dict[str, Any]], t_ns: int) -> dict[str, Any]:
    scans = [e for e in index if e["modality"] == "lidar"]
    best = min(scans, key=lambda e: abs(_ts_ns(e["timestamp"]) - t_ns))
    return {
        "scan_id": best["observation_id"],
        "scan_header_minus_image_ms": round((_ts_ns(best["timestamp"]) - t_ns) / 1e6, 2),
    }


def benchmark_manifest() -> dict[str, Any]:
    """The B1 manifest: frame, window, dataset, calibration, baseline runs, config and code."""
    index = _index()
    images = [e for e in index if e["modality"] == "image"]
    image = images[FRAME_INDEX]
    assert image["observation_id"] == IMAGE_ID, "frame index 55 must be camera_1_image_raw-000055"
    t_image = _ts_ns(image["timestamp"])
    assert f"{t_image // 1_000_000_000}.{t_image % 1_000_000_000:09d}" == IMAGE_TIMESTAMP
    nearest = _nearest_scan(index, t_image)
    assert nearest["scan_id"] == NEAREST_SCAN_ID

    seq_prov = read_json(_sequence_dir() / "provenance/provenance.json")
    pose_prov = read_json(
        E0_RUNS["raw"] / f"sequences/corridor-02-pose/{POSE_SEQUENCE_ID}/provenance/provenance.json"
    )
    seq_manifest = read_json(_sequence_dir() / "manifest.json")
    calibration = read_json(_sequence_dir() / "calibration/calibration.json")
    window = seq_prov["ingestion_config"]["window"]
    gt = DATASET / "corridor-02-gt.txt"
    bag = DATASET / "corridor-02/corridor02.bag"
    zip_file = DATASET / "corridor-02.zip"
    geometry = {
        arm: read_json(run / "geometric_mapping/manifest.json") for arm, run in E0_RUNS.items()
    }
    runs = {}
    for arm, run in E0_RUNS.items():
        runs[arm] = {
            "run_dir": str(run.relative_to(ROOT)),
            "sequence_artifact_id": SEQUENCE_ID,
            "pose_sequence_artifact_id": POSE_SEQUENCE_ID,
            "geometry_bin_sha256": next(
                f["content_hash"]
                for f in geometry[arm]["file_inventory"]
                if f["path"] == "outputs/geometry.bin"
            ),
            "stages": {
                stage: _stage_identity(run, stage)
                for stage in (
                    "state_estimation",
                    "geometric_mapping",
                    "visual_perception",
                    "semantic_scoring",
                    "sensor_association",
                    "semantic_fusion",
                    "semantic_mapping",
                    "entity_resolution",
                    "spatial_relations",
                    "context_map",
                )
            },
        }
    extrinsic = next(
        s
        for s in calibration["static_transforms"]
        if s["parent_frame"] == "epson" and s["child_frame"] == "d"
    )
    return {
        "schema": "correction-benchmark/benchmark-manifest/v1",
        "benchmark_id": "B1",
        "description": "Um frame: 10:31 = camera_1_image_raw-000055, janela de ingestão "
        "628-637 s, mapa de 81 scans; braço raw principal, voxel secundário.",
        "frame": {
            "image_id": IMAGE_ID,
            "frame_index_in_window": FRAME_INDEX,
            "timestamp": IMAGE_TIMESTAMP,
            "timestamp_ns": t_image,
            "clock_id": image["timestamp"]["clock_id"],
            "recording_second": round(t_image / 1e9 - RECORDING_EPOCH_S, 3),
            "nearest_scan": nearest,
            "pose_bracket": _pose_bracket(E0_RUNS["raw"], t_image),
            "image_size": [image["width"], image["height"]],
            "encoding": image["encoding"],
            "rgb_payload_sha256": sha256_file(_sequence_dir() / image["payload_path"]),
        },
        "window": {
            "driver_parameters": {"start_offset_s": 631, "end_offset_s": 634, "padding_s": 3},
            "recording_time_start_s": window["start_seconds"],
            "recording_time_end_s": window["end_seconds"],
            "seconds_from_recording_epoch": [
                round(window["start_seconds"] - RECORDING_EPOCH_S, 3),
                round(window["end_seconds"] - RECORDING_EPOCH_S, 3),
            ],
            "recording_epoch_s": RECORDING_EPOCH_S,
            "clock_id": window["clock_id"],
            "timestamp_policy": seq_prov["ingestion_config"]["timestamp_policy"],
            "observations": {
                m: sum(1 for e in index if e["modality"] == m)
                for m in sorted({e["modality"] for e in index})
            },
        },
        "map": {
            arm: {
                "map_id": geometry[arm]["map_id"],
                "scan_count": geometry[arm]["scan_count"],
                "rejected_scan_count": geometry[arm]["rejected_scan_count"],
                "point_count": geometry[arm]["point_count"],
                "aggregation_rule": geometry[arm].get("aggregation_rule"),
                "time_bounds_ns": [geometry[arm]["start_time_ns"], geometry[arm]["end_time_ns"]],
            }
            for arm in E0_RUNS
        },
        "dataset": {
            "intrinsics_yaml": {
                "path": "datasets/corridor-02/corridor-02-Intrinsics.yaml",
                "sha256": sha256_file(DATASET / "corridor-02-Intrinsics.yaml"),
            },
            "extrinsics_yaml": {
                "path": "datasets/corridor-02/corridor-02-extrinsics.yaml",
                "sha256": sha256_file(DATASET / "corridor-02-extrinsics.yaml"),
            },
            "ground_truth": {
                "path": "datasets/corridor-02/corridor-02-gt.txt",
                "sha256": sha256_file(gt),
                "matches_pose_sequence_provenance": sha256_file(gt)
                == pose_prov["source_content_hash"],
            },
            "bag": {
                "path": str(bag.relative_to(ROOT)),
                "size_bytes": bag.stat().st_size,
                "sha256_recorded": seq_prov["source_content_hash"],
                "sha256_source": f"provenance da SequenceArtifact {SEQUENCE_ID} "
                "(ingestão com hash_source=true); não re-calculado aqui",
            },
            "zip": {
                "path": str(zip_file.relative_to(ROOT)),
                "size_bytes": zip_file.stat().st_size,
                "sha256_recorded": None,
                "note": "nenhum manifesto existente registra o hash do zip; "
                "não re-hasheado (15 GB)",
            },
            "image_masks": {
                "overlay": {
                    "path": "examples/v0.2.0/corridor-02-image-masks.toml",
                    "sha256": sha256_file(ROOT / "examples/v0.2.0/corridor-02-image-masks.toml"),
                },
                "valid_region": {
                    "path": "examples/v0.2.0/assets/corridor-02/camera_1-valid.npy",
                    "sha256": sha256_file(
                        ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-valid.npy"
                    ),
                },
                "ego_exclusion": {
                    "path": "examples/v0.2.0/assets/corridor-02/camera_1-ego.npy",
                    "sha256": sha256_file(
                        ROOT / "examples/v0.2.0/assets/corridor-02/camera_1-ego.npy"
                    ),
                },
            },
        },
        "calibration": {
            "calibration_identity": geometry["raw"]["calibration_identity"],
            "calibration_source_hash": seq_prov["calibration_source_hash"],
            "calibration_json_sha256": next(
                f["content_hash"]
                for f in seq_manifest["file_inventory"]
                if f["path"] == "calibration/calibration.json"
            ),
            "camera_calibration_id": CAMERA_CALIBRATION_ID,
            "camera_model": next(
                e for e in calibration["entries"] if e["calibration_id"] == CAMERA_CALIBRATION_ID
            )["camera_model"]["kind"],
            "camera_entry_content_hash": next(
                e["content_hash"]
                for e in calibration["entries"]
                if e["calibration_id"] == CAMERA_CALIBRATION_ID
            ),
            "T_epson_d": {
                "translation_m": extrinsic["translation"],
                "rotation_xyzw": extrinsic["rotation"],
            },
            "refinement": None,
            "note": "calibração BASE do dataset (sem correção yaw/pitch); "
            "o C2 cria identidade nova",
        },
        "baseline_runs": runs,
        "models": _models(E0_RUNS["raw"]),
        "configuration": {
            "config_stack": [{"path": p, "sha256": sha256_file(ROOT / p)} for p in CONFIG_STACK],
            "driver": {
                "path": "experiments/frame-10_31-raw-vs-voxel-20260930/run_two_arms.py",
                "sha256": sha256_file(
                    ROOT / "experiments/frame-10_31-raw-vs-voxel-20260930/run_two_arms.py"
                ),
                "preset": "canonical/1 sem scene_interpretation (canonical/1-labels-no-scene)",
                "tail_stages": "semantic_mapping/entity_resolution/spatial_relations construídos "
                "no driver com os valores canônicos do e2e-real-canonical-run",
                "stage_config_digests": "o driver passa config_digest literal por estágio "
                "('perc', 'geo-raw', ...), não o digest efetivo",
            },
            "effective_config_digest_now": "ver results/C4-sam2-grid/<arm>/metrics.json: "
            "arm.effective_config_digest (E0-sam2 = pilha sem override)",
            "reproduced_component_fingerprints": {
                "sam2": "sha256:9336bb272b4474c3020ce2910755a37060bd5bde67fa8a97779d51c4a4d64f69",
                "normalization_config_digest": (
                    "sha256:c2a42c0da8c7138fbdea886f92088d7a816d041c7cdbb71702b99a4efdf09faa"
                ),
                "pass_config_digest": (
                    "sha256:f1c08c0821b7243c23463304f1f5d8be5a2b3c7a328a6d83f6573e1dc1262aae"
                ),
                "evidence": "a pilha atual recompõe exatamente estes fingerprints do E0 "
                "(braço E0-sam2 do C4)",
            },
        },
        "code_identity": {
            "baseline_runs_code_version": read_json(
                E0_RUNS["raw"] / "state_estimation/manifest.json"
            )["code_version"],
            "baseline_runs_note": "o E0 rodou numa árvore suja sobre a3a175f09 (sufixo "
            ".d20260930); o diff daquele momento não foi registrado",
            "now": code_identity(RELEVANT_UNTRACKED),
        },
        "environment": {
            "python_metrics": platform.python_version(),
            "gpu": "NVIDIA RTX 3060 8 GB (1 frame por processo, workaround #689)",
            "ablation_venv": "~/.cache/contextmap2-ablation/venv (torch 2.8.0+cu126, sam2)",
            "providers": "~/.cache/contextmap2-ablation/providers/ablation_providers.py "
            "(fora do repo)",
            "hf_offline": True,
        },
    }


def b2_proposal() -> dict[str, Any]:
    """B2 proposal (decision D1 pending): same ingestion, every 6th image around frame 55."""
    index = _index()
    images = [e for e in index if e["modality"] == "image"]
    geometry = read_json(E0_RUNS["raw"] / "geometric_mapping/manifest.json")
    frames = []
    for position in B2_FRAMES:
        image = images[position]
        t = _ts_ns(image["timestamp"])
        frames.append(
            {
                "frame_index_in_window": position,
                "image_id": image["observation_id"],
                "exists_in_sequence": image["observation_id"]
                == f"camera_1_image_raw-{position:06d}",
                "timestamp_ns": t,
                "seconds_from_frame_55": round(
                    (t - _ts_ns(images[FRAME_INDEX]["timestamp"])) / 1e9, 3
                ),
                "inside_map_time_bounds": geometry["start_time_ns"] <= t <= geometry["end_time_ns"],
                "nearest_scan": _nearest_scan(index, t),
                "pose_bracket": _pose_bracket(E0_RUNS["raw"], t),
            }
        )
    return {
        "schema": "correction-benchmark/b2-proposal/v1",
        "status": "proposal",
        "decision_pending": "D1 (composição exata do B2 e esforço de anotação de identidade "
        "cross-frame)",
        "use": "só C8/C9 (multi-view); B1 continua o benchmark principal",
        "sequence_artifact_id": SEQUENCE_ID,
        "ingestion": "a mesma do B1 (janela 628-637 s); nenhuma re-ingestão",
        "selection_rule": f"imagens {B2_FRAMES[0]}..{B2_FRAMES[-1]} a cada {B2_STEP} "
        "(~0,25 s a ~24 Hz), centradas na 55",
        "frames": frames,
        "all_ids_exist": all(f["exists_in_sequence"] for f in frames),
        "caveats": [
            "frames a ~0,25 s têm pouca variação de ponto de vista (risco do plano C8)",
            "percepção multi-frame em 8 GB: rodar 1 frame por processo (#689) ou stage-major",
            "frames com bracket de pose > 450 ms são recusados pela política de pose da associação",
        ],
    }


def main() -> int:
    """Write both manifests."""
    write_json(MANIFESTS / "benchmark.json", benchmark_manifest())
    write_json(MANIFESTS / "b2-proposal.json", b2_proposal())
    print("wrote", MANIFESTS / "benchmark.json", MANIFESTS / "b2-proposal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
