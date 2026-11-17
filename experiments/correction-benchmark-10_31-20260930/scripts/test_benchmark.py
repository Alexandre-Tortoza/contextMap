# ruff: noqa: D103 - como em tests/** do repositório, testes não levam docstring própria
"""Testes do bundle: gate C10 e sanidade do E0 contra os números da investigação.

Não fazem parte da suíte do repositório (vivem no experimento, como os ``diagnostic_tests`` da
investigação). Rodam só com os ``results/`` já extraídos::

    PYTHONPATH=src $MISE_PY -m pytest -q -p no:cacheprovider \
      experiments/correction-benchmark-10_31-20260930/scripts/test_benchmark.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from gate import check, main  # noqa: E402

BUNDLE = SCRIPTS.parent
E0 = BUNDLE / "results/E0/metrics.json"
E0_VOXEL = BUNDLE / "results/E0-voxel/metrics.json"
POLICY = json.loads((BUNDLE / "manifests/gate-policy.json").read_text())

pytestmark = pytest.mark.skipif(not E0.is_file(), reason="results/E0/metrics.json not extracted")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------- sanidade do E0


def test_e0_raw_reproduces_the_investigation_numbers() -> None:
    layers = _load(E0)["layers"]
    assert layers["geometry"]["geometry_bin_sha256"].startswith("sha256:d4e4b03e")
    assert layers["geometry"]["geometry_bin_sha256_matches_manifest"] is True
    assert layers["geometry"]["point_count"] == 2_322_324
    assert layers["geometry"]["scan_count"] == 81
    assert layers["regions"]["mask_count"] == 83
    assert layers["semantics"]["claim_count"] == 77
    assert layers["semantics"]["parse_failures"] == 6
    assert round(layers["association"]["uncovered_point_fraction"] * 100, 2) == 99.07
    assert round(layers["association"]["multi_mask_point_fraction"] * 100, 1) == 92.3
    assert layers["association"]["recount"]["agrees_with_diagnostics"] is True
    assert layers["semantics"]["labels_with_color_or_material_prefix"] == 42
    assert round(layers["semantics"]["label_entropy_bits"], 2) == 4.90
    assert layers["entities"]["context_map_entities"] == 38


def test_e0_voxel_reproduces_the_investigation_numbers() -> None:
    layers = _load(E0_VOXEL)["layers"]
    assert layers["geometry"]["geometry_bin_sha256"].startswith("sha256:b243d227")
    assert round(layers["association"]["uncovered_point_fraction"] * 100, 2) == 98.97
    # Mesma percepção (hardlink): regiões e semântica idênticas ao braço raw.
    raw = _load(E0)["layers"]
    assert layers["regions"]["mask_sha256_by_region"] == raw["regions"]["mask_sha256_by_region"]
    assert layers["semantics"]["primary_by_region"] == raw["semantics"]["primary_by_region"]


# --------------------------------------------------------------------------- gate


def test_gate_passes_e0_against_itself(tmp_path: Path) -> None:
    assert main(["--baseline", str(E0), "--arm", str(E0), "--quiet"]) == 0


def test_gate_fails_an_arm_whose_geometry_bin_changed(tmp_path: Path) -> None:
    arm = copy.deepcopy(_load(E0))
    arm["arm_id"] = "tampered"
    arm["layers"]["geometry"]["geometry_bin_sha256"] = "sha256:" + "0" * 64
    path = _write(tmp_path, "tampered.json", arm)

    assert main(["--baseline", str(E0), "--arm", str(path), "--quiet"]) == 1
    report = check(_load(E0), arm, arm_kind="none", target_layer=None, policy=POLICY)
    assert any("geometry.bin changed" in v for v in report["violations"])


def test_gate_fails_the_real_voxel_map_as_an_undeclared_geometry_change() -> None:
    report = check(_load(E0), _load(E0_VOXEL), arm_kind="none", target_layer=None, policy=POLICY)
    assert not report["passed"]


def test_gate_accepts_a_declared_geometry_change_with_a_new_map_id() -> None:
    report = check(
        _load(E0), _load(E0_VOXEL), arm_kind="geometry", target_layer="geometry", policy=POLICY
    )
    assert report["passed"], report["violations"]


def test_gate_refuses_a_geometry_change_that_keeps_the_map_id() -> None:
    arm = copy.deepcopy(_load(E0))
    arm["layers"]["geometry"]["geometry_bin_sha256"] = "sha256:" + "1" * 64
    report = check(_load(E0), arm, arm_kind="geometry", target_layer="geometry", policy=POLICY)
    assert any("kept the map_id" in v for v in report["violations"])


def _calibration_arm(*, same_bytes: bool, new_identity: bool) -> dict[str, Any]:
    arm = copy.deepcopy(_load(E0))
    geometry = arm["layers"]["geometry"]
    association = arm["layers"]["association"]
    if not same_bytes:
        geometry["geometry_bin_sha256"] = "sha256:" + "2" * 64
    if new_identity:
        geometry["calibration_identity"] = "sha256:" + "c" * 64
        geometry["map_id"] = "corridor-02--" + "e" * 32
        # O projetor exige a mesma identidade no mapa e na associação.
        association["calibration_identity"] = geometry["calibration_identity"]
        association["map_id"] = geometry["map_id"]
    return arm


def test_gate_accepts_a_calibration_arm_with_same_bytes_and_new_identity() -> None:
    arm = _calibration_arm(same_bytes=True, new_identity=True)
    report = check(
        _load(E0), arm, arm_kind="calibration", target_layer="calibration", policy=POLICY
    )
    assert report["passed"], report["violations"]


def test_gate_refuses_a_calibration_arm_that_keeps_the_identity() -> None:
    arm = _calibration_arm(same_bytes=True, new_identity=False)
    report = check(
        _load(E0), arm, arm_kind="calibration", target_layer="calibration", policy=POLICY
    )
    assert any("kept the baseline calibration_identity" in v for v in report["violations"])


def test_gate_refuses_a_calibration_arm_that_changes_the_bytes() -> None:
    arm = _calibration_arm(same_bytes=False, new_identity=True)
    report = check(
        _load(E0), arm, arm_kind="calibration", target_layer="calibration", policy=POLICY
    )
    assert any("changed geometry.bin bytes" in v for v in report["violations"])


def test_gate_refuses_an_undeclared_calibration_change() -> None:
    arm = _calibration_arm(same_bytes=True, new_identity=True)
    report = check(_load(E0), arm, arm_kind="none", target_layer=None, policy=POLICY)
    assert any("calibration_identity changed" in v for v in report["violations"])


def test_gate_flags_a_regression_in_a_layer_before_the_target() -> None:
    arm = copy.deepcopy(_load(E0))
    arm["layers"]["regions"]["multiple_mask_pixel_fraction"] += 0.01
    report = check(_load(E0), arm, arm_kind="none", target_layer="semantics", policy=POLICY)
    assert any("regions.multiple_mask_pixel_fraction worsened" in v for v in report["violations"])
    # A mesma mudança na camada-alvo é só um delta informativo.
    report = check(_load(E0), arm, arm_kind="none", target_layer="regions", policy=POLICY)
    assert report["passed"]


def test_gate_flags_an_association_that_used_another_map() -> None:
    arm = copy.deepcopy(_load(E0))
    arm["layers"]["association"]["map_id"] = "corridor-02--" + "f" * 32
    report = check(_load(E0), arm, arm_kind="none", target_layer=None, policy=POLICY)
    assert any("another map_id" in v for v in report["violations"])


E0_RUN = BUNDLE.parents[1] / "workspace-old/frame-10_31-raw"


@pytest.mark.skipif(not E0_RUN.is_dir(), reason="E0 run not available")
def test_the_annotation_dependent_metrics_run_on_a_synthetic_annotation(tmp_path: Path) -> None:
    """Exercise the annotation path with values whose answer is known without a human.

    The correspondence is the investigation's test D point (geometry index 481273), whose base
    projection is (120.49, 76.33): under the E0 calibration the residual must be ~0, which also
    checks that the extractor and the pipeline share the pixel-centre convention.
    """
    from extract_metrics import extract

    annotation = {
        "schema": "correction-benchmark/weak-annotation/v1",
        "status": "annotated",
        "instances": [
            # Caixa do bbox persistido da region-0009 (x=88, y=5, 73x143): recall por caixa.
            {"instance_id": "handle-01", "class": "door_handle", "box_xyxy": [88, 5, 161, 148]},
        ],
        "scene_terms": ["flooded hallway"],
        "correspondences": [
            {
                "correspondence_id": "test-d-index",
                "pixel_uv": [120.49001521288031, 76.32646656922697],
                "geometry_index": 481273,
                "map_id": "corridor-02--5659900dc9d8f118d7ff41071f722453",
            },
            {
                "correspondence_id": "test-d-xyz",
                "pixel_uv": [120.49001521288031, 76.32646656922697],
                "point_map_m": [43.693388712255434, -90.92019656617117, -1.0064192781761085],
            },
        ],
    }
    path = _write(tmp_path, "frame-55.json", annotation)
    layers = extract(E0_RUN, arm_id="synthetic", annotation_path=path)["layers"]

    residual = layers["calibration"]["reprojection_residual_px"]
    assert residual["count"] == 2
    assert residual["median"] < 1e-3
    recall = layers["regions"]["known_object_recall"]
    assert recall["object_count"] == 1
    assert recall["matches"][0]["best_region_id"] == "region-0009"
    assert recall["recalled_count"] == 1
    scene = layers["semantics"]["scene_label_rate"]
    assert scene["count"] >= 1  # region-0005 "flooded hallway", máscara < 50 % da imagem
    assert layers["semantics"]["hallucination_rate"]["population"] >= 1


def test_gate_reports_usage_errors_with_exit_code_2(tmp_path: Path) -> None:
    assert main(["--baseline", str(tmp_path / "missing.json"), "--arm", str(E0), "--quiet"]) == 2
