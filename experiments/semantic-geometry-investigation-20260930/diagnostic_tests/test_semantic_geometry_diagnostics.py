"""Testes diagnósticos A-E da investigação semântica x geometria (frame 10:31, imagem 55).

Não fazem parte da suíte do repositório: rodam contra os artefatos reais em
``workspace-old/frame-10_31-{raw,voxel}`` e pulam quando eles não existem. Cada teste usa
os tipos e funções públicos do ContextMap2 e compara com a reimplementação independente de
``scripts/`` (numpy), para que a concordância seja evidência e não tautologia.

Rodar (venv com numpy/scipy e ``contextmap`` editável):

    PYTHONPATH=src:experiments/semantic-geometry-investigation-20260930/scripts \\
      ~/.cache/contextmap2-audit/venv/bin/python -m pytest -q \\
      experiments/semantic-geometry-investigation-20260930/diagnostic_tests
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from common import (
    IMAGE_ID,
    RESULTS,
    SEQ_ID,
    Trajectory,
    camera_model,
    load_geometry,
    load_jsonl,
    mei_project,
    mei_unproject,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)

pytestmark = pytest.mark.skipif(
    not (run_dir("raw") / "geometric_mapping/outputs/geometry.bin").is_file(),
    reason="artefatos do run frame-10_31 ausentes",
)


def _calibration_set():
    from contextmap.ingestion import SequenceArtifactReader

    return SequenceArtifactReader(run_dir("raw") / "sequences/corridor-02" / SEQ_ID).read_calibration()


def _t_map_camera() -> np.ndarray:
    image = next(e for e in sequence_index() if e["observation_id"] == IMAGE_ID)
    t_map_epson, _ = Trajectory().pose(ts_ns(image["timestamp"]))
    return t_map_epson @ static_transform("epson", "d")


def test_a_transform_round_trip_map_camera_map() -> None:
    """A: P_map -> camera -> map com as funções do pipeline; erro ~ 0 e igual à via matricial."""
    from common import matrix_to_quat

    from contextmap.shared import compose_rigid, invert_rigid, quaternion_to_rotation_matrix

    t_map_cam = _t_map_camera()
    geo = load_geometry("raw")
    rng = np.random.default_rng(0)
    points = geo["map"][rng.choice(len(geo), 50_000, replace=False)]
    # via do pipeline (frame_projection.py): P_cam = (P - t) @ R, R = R_map_cam
    q = tuple(matrix_to_quat(t_map_cam[:3, :3]))
    t = tuple(t_map_cam[:3, 3])
    r = np.array(quaternion_to_rotation_matrix(q))
    cam = (points - np.array(t)) @ r
    back = cam @ r.T + np.array(t)
    assert np.abs(back - points).max() < 1e-9
    # composição quaternion do pipeline == composição matricial independente
    image = next(e for e in sequence_index() if e["observation_id"] == IMAGE_ID)
    t_map_epson, _ = Trajectory().pose(ts_ns(image["timestamp"]))
    st = next(s for s in json.loads((run_dir("raw") / "sequences/corridor-02" / SEQ_ID / "calibration/calibration.json").read_text())["static_transforms"] if s["child_frame"] == "d")
    trans, rot = compose_rigid(
        outer_translation=tuple(t_map_epson[:3, 3]), outer_rotation=tuple(matrix_to_quat(t_map_epson[:3, :3])),
        inner_translation=tuple(st["translation"]), inner_rotation=tuple(st["rotation"]),
    )
    assert np.abs(np.array(trans) - t_map_cam[:3, 3]).max() < 1e-12
    assert np.abs(np.array(quaternion_to_rotation_matrix(rot)) - t_map_cam[:3, :3]).max() < 1e-12
    inv_t, inv_r = invert_rigid(translation=trans, rotation=rot)
    ref = np.linalg.inv(t_map_cam)
    assert np.abs(np.array(inv_t) - ref[:3, 3]).max() < 1e-9


def test_b_identity_semantics_never_moves_xyz() -> None:
    """B: uma etapa semântica falsa que anexa label='test' às referências não altera XYZ.

    Resolve cada referência das observações espaciais persistidas pelo reader público do mapa
    e compara com o geometry.bin; nenhum registro de associação carrega coordenadas próprias.
    """
    from contextmap.geometric_mapping import (
        GeometricMapArtifactReader,
        GeometryReference,
        geometry_id_for,
    )

    geo = load_geometry("raw")
    support = np.fromfile(run_dir("raw") / "sensor_association/outputs/geometry-support.u32", dtype="<u4")
    observations = load_jsonl(run_dir("raw") / "sensor_association/outputs/spatial-observations.jsonl")
    labeled: dict[int, str] = {int(i): "test" for i in support}  # a "semântica" falsa
    assert all("coordinates" not in json.dumps(o) for o in observations)
    with GeometricMapArtifactReader(run_dir("raw") / "geometric_mapping") as reader:
        packed = reader.geometry()
        map_id = packed.geometric_map.map_id
        sample = np.random.default_rng(1).choice(sorted(labeled), 2_000, replace=False)
        for index in sample:
            point = packed.get(GeometryReference(map_id=map_id, geometry_id=geometry_id_for(map_id=map_id, index=int(index))))
            assert tuple(point.coordinates_m) == tuple(geo["map"][index])
    # o context_map aponta para o mesmo mapa e contagem
    ref = json.loads((run_dir("raw") / "context_map/geometry/geometry-reference.json").read_text())
    assert ref == {"map_id": str(map_id), "point_count": len(geo)}


def test_c_projection_backprojection_round_trip() -> None:
    """C: projeção MEI do pipeline -> raio -> XYZ com a distância conhecida; e == fórmula independente."""
    from contextmap.sensor_association import camera_projection_for

    calibration = _calibration_set()
    entry = calibration.entries["camera_1_image_raw-calib"]
    projection = camera_projection_for(entry)
    t_cam_map = np.linalg.inv(_t_map_camera())
    geo = load_geometry("raw")
    cam = geo["map"] @ t_cam_map[:3, :3].T + t_cam_map[:3, 3]
    result = projection.project(cam)
    ok = result.projectable & projection.in_image(result.pixels)
    ours, _ = mei_project(cam[ok], camera_model())
    assert np.abs(ours - result.pixels[ok]).max() < 1e-6
    rays = np.asarray(projection.unproject(result.pixels[ok][:200_000]))
    rebuilt = rays * result.range_m[ok][:200_000, None]
    assert np.abs(rebuilt - cam[ok][:200_000]).max() < 1e-6
    rays_ours = mei_unproject(result.pixels[ok][:200_000], camera_model())
    assert np.abs(rays_ours - rays).max() < 1e-6


def test_d_known_pixel_metal_handle() -> None:
    """D: um ponto conhecido (barra/maçaneta, região 'metal handle') com todos os intermediários.

    Grava os valores em results/test_d_known_pixel.json e exige que a reimplementação
    reproduza o pertencimento persistido pelo pipeline para esse ponto.
    """
    from assoc import associate, load_frame, pipeline_support

    frame = load_frame("raw")
    geo = load_geometry("raw")
    base = associate(geo["map"], frame)
    corrected = associate(geo["map"], frame, correction=(4.25, -2.25, 0.0))
    persisted = pipeline_support("raw")
    region = "region-0009"
    members = persisted[region]
    mask = frame.masks[region]
    cy, cx = np.argwhere(mask).mean(0)
    chosen = int(members[np.argmin(np.hypot(base["uv"][members, 0] - cx, base["uv"][members, 1] - cy))])
    result = json.loads((run_dir("raw") / "visual_perception/outputs/results.jsonl").read_text().splitlines()[0])
    labels = {c["region_id"]: c["hypothesis"] for c in result["claims"]}
    regions_base = sorted(r for r, idx in base["per_region"].items() if chosen in set(idx.tolist()))
    regions_persisted = sorted(r for r, idx in persisted.items() if chosen in set(idx.tolist()))
    regions_corr = sorted(r for r, idx in corrected["per_region"].items() if chosen in set(idx.tolist()))
    record = {
        "geometry_index": chosen,
        "point_world_map_m": geo["map"][chosen].tolist(),
        "point_sensor_m": geo["source"][chosen].tolist(),
        "source_scan_ordinal": int(geo["scan"][chosen]),
        "point_camera_m_base": base["cam"][chosen].tolist(),
        "range_m": float(base["range"][chosen]),
        "pixel_base": base["uv"][chosen].tolist(),
        "pixel_with_v1_correction": corrected["uv"][chosen].tolist(),
        "pixel_shift_px": float(np.linalg.norm(corrected["uv"][chosen] - base["uv"][chosen])),
        "regions_base": {r: labels.get(r) for r in regions_base},
        "regions_persisted": {r: labels.get(r) for r in regions_persisted},
        "regions_with_v1_correction": {r: labels.get(r) for r in regions_corr},
        "visible_with_v1_correction": bool(corrected["visible"][chosen]),
    }
    (RESULTS / "test_d_known_pixel.json").write_text(json.dumps(record, indent=2) + "\n")
    assert regions_base == regions_persisted
    assert region in regions_base


def test_e_semantics_off_vs_on_geometry_is_identical() -> None:
    """E: mapa reconstruído sem nenhum estágio semântico == mapa referenciado pelo context_map."""
    path = RESULTS / "p2_test_e_semantics_off_vs_on.json"
    if not path.is_file():
        pytest.skip("rode scripts/p2_test_e_semantics_off.py antes (reconstrói os mapas em scratch)")
    result = json.loads(path.read_text())
    for arm in ("raw", "voxel"):
        assert result[arm]["identical"], arm
        assert result[arm]["point_count_off"] == result[arm]["point_count_on"]
        assert result[arm]["map_id_off"] == result[arm]["map_id_on"]
