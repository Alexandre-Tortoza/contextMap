"""C2: estima a correção câmera<->corpo por grade e gera o documento calibration-refinement/1.

Reexecuta o ajuste fotométrico da investigação (``p4_calibration_photometric.py``) com a MESMA
seleção de observações (7 grupos de 8 frames, nuvem com deskew offline por ponto, visibilidade
decidida com a extrínseca base, até 60 000 pontos por grupo com semente 7, grupos pares =
ajuste, ímpares = validação) e a MESMA grade (1° em ±8°, depois 0,25° em ±1,5° no ajuste;
roll = 0). O custo vem de
``contextmap.evaluation.camera_body_calibration.photometric_inconsistency``.

Dois braços, uma variável:

``--arm p4-reproduction``
    Projeção = a fórmula MEI independente de p4 (``common.mei_project``), que só exige
    ``z + xi·r > 0``. Com xi = 1,563 > 1 ela dobra para dentro da imagem pontos atrás da
    câmera (medido: 4,7 % dos pares (ponto, frame) visíveis de p4 estão a mais de 164° do eixo
    óptico). Serve só para provar que o resto do harness é idêntico: precisa reproduzir
    ``results/p4_calibration_photometric.json`` dentro de 1e-4.
``--arm estimate``
    Projeção = a do pipeline (``contextmap.sensor_association``), que recusa pontos além do
    horizonte da esfera, na visibilidade e na amostragem. É o braço que gera o documento.

Saídas (neste diretório):
  results/p4-reproduction.json                  braço p4-reproduction, comparado com p4
  results/estimate.json                         braço estimate (números, deslocamento, identidade)
  corridor-02-camera-body-refinement-v1.json    documento calibration-refinement/1 (braço estimate)

Uso (ambiente da investigação: numpy, scipy, PIL, contextmap editável):
  ~/.cache/contextmap2-audit/venv/bin/python estimate_refinement.py --arm p4-reproduction
  ~/.cache/contextmap2-audit/venv/bin/python estimate_refinement.py --arm estimate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INVESTIGATION = ROOT / "experiments/semantic-geometry-investigation-20260930"
sys.path.insert(0, str(INVESTIGATION / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from assoc import associate, load_frame  # noqa: E402
from common import (  # noqa: E402
    Trajectory,
    calibration,
    camera_model,
    load_geometry,
    load_jsonl,
    load_rgb,
    load_scan_payload,
    mei_project,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)
from p3_time_offset_deskew import place  # noqa: E402
from p4_calibration_photometric import (  # noqa: E402
    MAX_POINTS,
    MAX_RANGE_M,
    MIN_VIEWS,
    SEED,
    V1_CORRECTION,
    local_texture,
    usable_area,
)
from p4_calibration_photometric import pixels as p4_pixels  # noqa: E402
from scipy import ndimage  # noqa: E402

from contextmap.evaluation.camera_body_calibration import (  # noqa: E402
    PhotometricGroup,
    photometric_inconsistency,
    pixel_displacement,
    relative_improvement,
)
from contextmap.ingestion import (  # noqa: E402
    CalibrationReferenceId,
    CalibrationRefinement,
    FrameId,
    RefinementValidation,
    apply_calibration_refinement,
    camera_rotation_correction,
    rigid_transform_digest,
)
from contextmap.ingestion.calibration import (  # noqa: E402
    CAMERA_OPTICAL_ROTATION_CONVENTION,
    decode_calibration_set,
    encode_calibration_refinement_document,
)
from contextmap.sensor_association import (  # noqa: E402
    CameraProjection,
    PixelProjection,
    camera_projection_for,
)

REFINEMENT_ID = "corridor-02-camera-body-refinement-v1"
DOCUMENT = HERE / f"{REFINEMENT_ID}.json"
RESULTS = HERE / "results"
REFERENCE = INVESTIGATION / "results/p4_calibration_photometric.json"
DATASET = ROOT / "datasets/corridor-02"
SOURCE_FILES = ("corridor-02-Intrinsics.yaml", "corridor-02-extrinsics.yaml")
POSE_SOURCE = DATASET / "corridor-02-gt.txt"
CAMERA_ENTRY = CalibrationReferenceId("camera_1_image_raw-calib")
REPRODUCTION_TOLERANCE = 1e-4
# Código que determina o número: o estimador, a métrica, o contrato, a projeção MEI e os
# leitores/deskew da investigação que ele importa.
CODE_FILES = (
    "experiments/correction-benchmark-10_31-20260930/calibration/estimate_refinement.py",
    "src/contextmap/evaluation/camera_body_calibration.py",
    "src/contextmap/ingestion/calibration.py",
    "src/contextmap/sensor_association/camera_models.py",
    "experiments/semantic-geometry-investigation-20260930/scripts/common.py",
    "experiments/semantic-geometry-investigation-20260930/scripts/assoc.py",
    "experiments/semantic-geometry-investigation-20260930/scripts/p3_map_consistency.py",
    "experiments/semantic-geometry-investigation-20260930/scripts/p3_time_offset_deskew.py",
    "experiments/semantic-geometry-investigation-20260930/scripts/p4_calibration_photometric.py",
)

PixelFunction = Callable[[np.ndarray, tuple[int, ...]], tuple[np.ndarray, np.ndarray, np.ndarray]]


class P4FoldingProjection:
    """A fórmula MEI de p4 atrás do contrato ``CameraProjection``, sem o limite do horizonte.

    Só o braço de reprodução a usa: ela projeta (e dobra para dentro da imagem) pontos atrás
    da câmera, que a projeção do pipeline recusa.
    """

    def __init__(self, reference: CameraProjection, camera: dict[str, Any]) -> None:
        """Keep the reference camera identity and p4's camera dictionary."""
        self._identity = reference.identity
        self._camera = camera

    def project(self, points_camera_m: np.ndarray) -> PixelProjection:
        """Project with ``common.mei_project`` (valid = ``z + xi r > 1e-9``)."""
        points = np.asarray(points_camera_m, dtype=np.float64)
        uv, valid = mei_project(points, self._camera)
        valid = valid & np.isfinite(uv).all(axis=1)
        return PixelProjection(
            camera=self._identity,
            pixels=np.where(valid[:, None], uv, np.nan),
            projectable=valid,
            depth_m=points[:, 2].copy(),
            range_m=np.linalg.norm(points, axis=1),
        )


def pixel_function(projection: CameraProjection | P4FoldingProjection) -> PixelFunction:
    """Pixel inteiro mais próximo e validade, com o arredondamento e os limites de p4."""

    def pixels(points: np.ndarray, shape: tuple[int, ...]):
        uv = projection.project(points).pixels
        ok = np.isfinite(uv).all(axis=1)
        ui = np.zeros(len(uv), dtype=int)
        vi = np.zeros(len(uv), dtype=int)
        ui[ok] = np.round(uv[ok, 0]).astype(int)
        vi[ok] = np.round(uv[ok, 1]).astype(int)
        ok &= (ui >= 0) & (ui < shape[1]) & (vi >= 0) & (vi < shape[0])
        return ui, vi, ok

    return pixels


def sha256_file(path: Path) -> str:
    """Return ``sha256:<hex>`` of a file's bytes."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity() -> tuple[str, dict[str, str]]:
    """HEAD + digest dos arquivos que rodaram (a árvore tem WIP não commitado)."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    files = {name: sha256_file(ROOT / name) for name in CODE_FILES}
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return f"git:{head};files:sha256:{combined}", files


def build_groups(pixels: PixelFunction) -> list[tuple[list[str], PhotometricGroup]]:
    """Mesma seleção de p4_calibration_photometric.main (linhas 74-143), grupo a grupo.

    A única diferença possível é ``pixels``, a função que decide projeção e visibilidade.
    """
    t_epson_d = static_transform("epson", "d")
    t_epson_vel = static_transform("epson", "cmu_rc1_velodyne")
    traj = Trajectory()
    index = sequence_index()
    images = [e for e in index if e["modality"] == "image"]
    lidar = {e["observation_id"]: e for e in index if e["modality"] == "lidar"}
    geo = load_geometry("raw")
    scans = load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")
    usable = usable_area()
    rng = np.random.default_rng(SEED)

    def cloud_for(t_lo, t_hi):
        parts = []
        for s in scans:
            t0 = ts_ns(s["acquisition_timestamp"])
            if not (t_lo - 5e8 <= t0 <= t_hi + 5e8):
                continue
            seg = geo[s["first_geometry_index"] : s["first_geometry_index"] + s["geometry_count"]]
            _, raw = load_scan_payload(lidar[s["observation_id"]])
            times = raw["time"].astype(np.float64)[seg["source_index"]]
            near = np.linalg.norm(seg["source"], axis=1) < MAX_RANGE_M
            parts.append(place(traj, t_epson_vel, t0, times[near], seg["source"][near], 0, True))
        return np.concatenate(parts)

    groups = []
    for start in range(0, len(images) - 24, 27):
        chosen = images[start : start + 24 : 3]
        frames = []
        for e in chosen:
            t = ts_ns(e["timestamp"])
            try:
                t_map_epson, bracket = traj.pose(t)
            except Exception:
                continue
            if bracket["gap_ns"] > 450_000_000:
                continue
            t_map_cam = t_map_epson @ t_epson_d
            frames.append(
                {
                    "id": e["observation_id"],
                    "t": t,
                    "map_to_camera": np.linalg.inv(t_map_cam),
                    "texture": local_texture(load_rgb(e["observation_id"])),
                }
            )
        if len(frames) >= 6:
            groups.append(frames)

    out = []
    for frames in groups:
        cloud = cloud_for(frames[0]["t"], frames[-1]["t"])
        visible = np.zeros((len(cloud), len(frames)), dtype=bool)
        cams = np.zeros((len(cloud), len(frames), 3), dtype=np.float32)
        for c, f in enumerate(frames):
            m = f["map_to_camera"]
            cp = cloud @ m[:3, :3].T + m[:3, 3]
            ui, vi, ok = pixels(cp, usable.shape)
            dist = np.linalg.norm(cp, axis=1)
            ok &= dist < MAX_RANGE_M
            ok[ok] &= usable[vi[ok], ui[ok]]
            z = np.full(usable.shape, np.inf)
            np.minimum.at(z, (vi[ok], ui[ok]), dist[ok])
            near = ndimage.minimum_filter(z, size=3)
            seen = ok.copy()
            seen[ok] = dist[ok] <= near[vi[ok], ui[ok]] + 0.05 + 0.03 * dist[ok]
            visible[:, c] = seen
            cams[:, c] = cp
        keep = np.where(visible.sum(1) >= MIN_VIEWS)[0]
        if len(keep) > MAX_POINTS:
            keep = np.sort(rng.choice(keep, MAX_POINTS, replace=False))
        group = PhotometricGroup(
            camera_points_m=cams[keep],
            visible=visible[keep],
            textures=tuple(f["texture"] for f in frames),
        )
        out.append(([f["id"] for f in frames], group))
        print("group", frames[0]["id"], "points", len(keep), flush=True)
    return out


def run_grid(observations, projection) -> dict[str, Any]:
    """A grade e os números de p4, com o custo do módulo de avaliação."""
    groups = [group for _, group in observations]
    fit, held = groups[0::2], groups[1::2]

    def cost(items, yaw, pitch):
        correction = camera_rotation_correction(yaw_deg=yaw, pitch_deg=pitch, roll_deg=0.0)
        return photometric_inconsistency(items, projection=projection, correction=correction)

    def value(items, yaw, pitch):
        return cost(items, yaw, pitch).value

    def grid_min(items, yaws, pitches):
        return min((value(items, y, p), float(y), float(p)) for y in yaws for p in pitches)

    coarse = np.arange(-8, 8.1, 1.0)
    _, y0, p0 = grid_min(fit, coarse, coarse)
    _, yb, pb = grid_min(
        fit, np.arange(y0 - 1.5, y0 + 1.6, 0.25), np.arange(p0 - 1.5, p0 + 1.6, 0.25)
    )
    print("best_fit", yb, pb, flush=True)

    per_group = []
    for ids, group in observations:
        c0 = grid_min([group], coarse, coarse)
        c1 = grid_min(
            [group],
            np.arange(c0[1] - 1, c0[1] + 1.1, 0.25),
            np.arange(c0[2] - 1, c0[2] + 1.1, 0.25),
        )
        per_group.append(
            {
                "first_frame": ids[0],
                "points": len(group.visible),
                "yaw_deg": c1[1],
                "pitch_deg": c1[2],
                "cost_base": round(value([group], 0, 0), 5),
                "cost_best": round(c1[0], 5),
                "cost_v1": round(value([group], *V1_CORRECTION[:2]), 5),
            }
        )
        print(per_group[-1], flush=True)

    measured = {
        "fit_base": cost(fit, 0, 0),
        "fit_v1_correction": cost(fit, *V1_CORRECTION[:2]),
        "fit_best": cost(fit, yb, pb),
        "held_out_base": cost(held, 0, 0),
        "held_out_v1_correction": cost(held, *V1_CORRECTION[:2]),
        "held_out_best_fit": cost(held, yb, pb),
    }
    return {
        "groups": len(groups),
        "fit_groups": len(fit),
        "held_out_groups": len(held),
        "fit_ids": [i for ids, _ in observations[0::2] for i in ids],
        "held_out_ids": [i for ids, _ in observations[1::2] for i in ids],
        "best_fit": {"yaw_deg": yb, "pitch_deg": pb, "roll_deg": 0.0},
        "inconsistency": {name: round(item.value, 5) for name, item in measured.items()},
        "inconsistency_unrounded": {name: item.value for name, item in measured.items()},
        "point_counts": {name: item.point_count for name, item in measured.items()},
        "relative_improvement": {
            "held_out_best_fit": relative_improvement(
                measured["held_out_base"].value, measured["held_out_best_fit"].value
            ),
            "held_out_v1_correction": relative_improvement(
                measured["held_out_base"].value, measured["held_out_v1_correction"].value
            ),
            "fit_best_fit": relative_improvement(
                measured["fit_base"].value, measured["fit_best"].value
            ),
        },
        "per_group": per_group,
        "yaw_profile_at_best_pitch": [
            {"yaw": float(y), "cost": round(value(fit, y, pb), 5)} for y in np.arange(-6, 8.1, 1.0)
        ],
        "pitch_profile_at_best_yaw": [
            {"pitch": float(p), "cost": round(value(fit, yb, p), 5)}
            for p in np.arange(-6, 6.1, 1.0)
        ],
    }


def compare_with_p4(result: dict[str, Any]) -> dict[str, Any]:
    """Maior desvio absoluto de cada número contra results/p4_calibration_photometric.json."""
    reference = json.loads(REFERENCE.read_text())
    deviations = {
        f"inconsistency.{k}": abs(v - reference["inconsistency"][k])
        for k, v in result["inconsistency"].items()
    }
    for ours, theirs in zip(result["per_group"], reference["per_group"], strict=True):
        for key in ("points", "cost_base", "cost_best", "cost_v1", "yaw_deg", "pitch_deg"):
            deviations[f"per_group.{ours['first_frame']}.{key}"] = abs(ours[key] - theirs[key])
    for name in ("yaw_profile_at_best_pitch", "pitch_profile_at_best_yaw"):
        for ours, theirs in zip(result[name], reference[name], strict=True):
            deviations[f"{name}.{next(iter(ours.values()))}"] = abs(ours["cost"] - theirs["cost"])
    same_best = tuple(result["best_fit"].values()) == tuple(reference["best_fit"].values())
    worst = max(deviations, key=deviations.get)
    return {
        "reference": str(REFERENCE.relative_to(ROOT)),
        "tolerance": REPRODUCTION_TOLERANCE,
        "same_best_fit": same_best,
        "max_abs_deviation": deviations[worst],
        "max_abs_deviation_at": worst,
        "reproduced": same_best and deviations[worst] <= REPRODUCTION_TOLERANCE,
    }


def frame_55_displacement(best: tuple[float, float, float]) -> dict[str, Any]:
    """Deslocamento em pixel dos pontos visíveis na base, no frame 55 (B1), como p5."""
    frame = load_frame("raw")
    geometry = load_geometry("raw")
    base = associate(geometry["map"], frame)
    seen = base["visible"]
    out = {}
    for name, angles in (("best_fit", best), ("v1_correction", V1_CORRECTION)):
        corrected = associate(geometry["map"], frame, correction=angles)
        moved = np.where(corrected["in_image"][seen, None], corrected["uv"][seen], np.nan)
        stats = pixel_displacement(base["uv"][seen], moved)
        out[name] = {
            "compared_count": stats.compared_count,
            "excluded_count": stats.excluded_count,
            "median_px": round(stats.median_px, 2),
            "p10_px": round(stats.p10_px, 2),
            "p90_px": round(stats.p90_px, 2),
        }
    return out


def write_document(base, target, result, displacement, identity) -> None:
    """Monta o refinamento do braço estimate, confere que ele aplica e grava o documento."""
    best = result["best_fit"]
    raw = result["inconsistency_unrounded"]
    counts = result["point_counts"]
    refinement = CalibrationRefinement(
        refinement_id=REFINEMENT_ID,
        parent_frame=target.parent_frame,
        child_frame=target.child_frame,
        base_transform_digest=rigid_transform_digest(target),
        yaw_deg=best["yaw_deg"],
        pitch_deg=best["pitch_deg"],
        roll_deg=0.0,
        convention=CAMERA_OPTICAL_ROTATION_CONVENTION,
        method=(
            "multi-frame photometric consistency (contextmap.evaluation.camera_body_calibration"
            ".photometric_inconsistency, pipeline MEI projection) over 7 groups of 8 frames of "
            "the 628-637 s window, offline per-point deskew with the GT trajectory, visibility "
            "under the base extrinsic; grid 1 deg in +-8 deg then 0.25 deg in +-1.5 deg on the "
            "fit groups (even), roll fixed at 0"
        ),
        estimator=CODE_FILES[0],
        code_identity=identity,
        pose_source_kind="ground_truth_tum",
        pose_source_digest=sha256_file(POSE_SOURCE),
        source_files={name: sha256_file(DATASET / name) for name in SOURCE_FILES},
        validation=RefinementValidation(
            fit_ids=tuple(result["fit_ids"]),
            held_out_ids=tuple(result["held_out_ids"]),
            metrics={
                "fit_photometric_inconsistency_base": raw["fit_base"],
                "fit_photometric_inconsistency_refined": raw["fit_best"],
                "fit_point_count": float(counts["fit_base"]),
                "held_out_photometric_inconsistency_base": raw["held_out_base"],
                "held_out_photometric_inconsistency_refined": raw["held_out_best_fit"],
                "held_out_point_count": float(counts["held_out_base"]),
                "held_out_relative_improvement": result["relative_improvement"][
                    "held_out_best_fit"
                ],
                "held_out_photometric_inconsistency_v1_correction": raw["held_out_v1_correction"],
                "held_out_relative_improvement_v1_correction": result["relative_improvement"][
                    "held_out_v1_correction"
                ],
                "frame_55_pixel_displacement_median_px": displacement["best_fit"]["median_px"],
                "frame_55_pixel_displacement_p10_px": displacement["best_fit"]["p10_px"],
                "frame_55_pixel_displacement_p90_px": displacement["best_fit"]["p90_px"],
            },
        ),
    )
    # A correção precisa aplicar sobre a base do E0 antes de virar documento.
    apply_calibration_refinement(base, refinement)
    DOCUMENT.write_text(
        json.dumps(encode_calibration_refinement_document(refinement), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run one arm and write its results (and, for ``estimate``, the document)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=("p4-reproduction", "estimate"), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    base = decode_calibration_set(calibration("raw"))
    pipeline = camera_projection_for(base.entries[CAMERA_ENTRY])
    target = next(t for t in base.static_transforms if t.child_frame == FrameId("d"))
    if args.arm == "p4-reproduction":
        cam = camera_model()
        projection: CameraProjection | P4FoldingProjection = P4FoldingProjection(pipeline, cam)

        def visibility(points, shape):  # a função de p4, literalmente
            return p4_pixels(points, cam, shape)

    else:
        projection = pipeline
        visibility = pixel_function(pipeline)
    result = run_grid(build_groups(visibility), projection)
    result["reproduction"] = compare_with_p4(result)
    identity, code_files = code_identity()
    result.update({"arm": args.arm, "code_identity": identity, "code_files": code_files})
    if args.arm == "estimate":
        best = tuple(result["best_fit"].values())
        result["frame_55_pixel_displacement"] = frame_55_displacement(best)
        write_document(base, target, result, result["frame_55_pixel_displacement"], identity)
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.arm}.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(args.arm, "best", result["best_fit"], "reproduction", result["reproduction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
