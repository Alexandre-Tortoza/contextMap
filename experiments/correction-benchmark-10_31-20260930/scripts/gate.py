"""Gate C10: compara o ``metrics.json`` de um braço com o baseline (E0) e falha em violação.

Invariantes (``manifests/gate-policy.json``, ``c10-gate/v1``):

* braço comum (``arm_kind = none``, C4-C9): ``geometry.bin`` byte-idêntico (mesmo sha256),
  mesmo ``map_id`` e mesma contagem de pontos;
* braço de calibração (``calibration``, C2): ``geometry.bin`` byte-idêntico, mas
  ``calibration_identity`` e ``map_id`` diferentes (identidade nova, bytes iguais);
* braço de geometria (``geometry``, C3): os bytes podem mudar; se mudarem, o ``map_id`` tem de
  mudar. Bytes iguais num braço que declara mudar a geometria geram só um aviso;
* coerência interna do braço: a associação usa o mesmo ``map_id`` e a mesma identidade de
  calibração do mapa; checagens ``required_true`` da política;
* não-regressão: métricas declaradas das camadas ANTERIORES à camada-alvo do braço não pioram
  além da tolerância declarada (tolerância ``null`` = só reportada).

Um braço sem camada de geometria (p.ex. os braços SAM2 do C4, que não reconstroem o mapa)
não pode violar a invariante de bytes: ela sai como ``not_applicable``.

Imprime os deltas por camada. Saída: 0 = passa; 1 = violação; 2 = erro de uso.

    $MISE_PY .../scripts/gate.py --baseline results/E0/metrics.json \
        --arm results/<arm>/metrics.json \
        [--arm-kind none|calibration|geometry] [--target-layer regions] [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

POLICY = Path(__file__).resolve().parents[1] / "manifests/gate-policy.json"
ARM_KINDS = ("none", "calibration", "geometry")


def _get(document: dict[str, Any], dotted: str) -> Any:
    node: Any = document
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _measured(layers: dict[str, Any], layer: str) -> bool:
    return isinstance(layers.get(layer), dict) and layers[layer].get("status") == "measured"


def check(
    baseline: dict[str, Any],
    arm: dict[str, Any],
    *,
    arm_kind: str,
    target_layer: str | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Return the gate decision, its violations and warnings, and the per-layer deltas."""
    if arm_kind not in ARM_KINDS:
        raise ValueError(f"arm_kind must be one of {ARM_KINDS}, got {arm_kind!r}")
    order = policy["layer_order"]
    if target_layer is not None and target_layer not in order:
        raise ValueError(f"target_layer must be one of {order}, got {target_layer!r}")
    base, other = baseline["layers"], arm["layers"]
    violations: list[str] = []
    warnings: list[str] = []
    invariants: dict[str, str] = {}

    # --- invariante de geometria/identidade
    if _measured(base, "geometry") and _measured(other, "geometry"):
        bg, ag = base["geometry"], other["geometry"]
        same_bytes = bg["geometry_bin_sha256"] == ag["geometry_bin_sha256"]
        same_map = bg["map_id"] == ag["map_id"]
        same_calibration = bg["calibration_identity"] == ag["calibration_identity"]
        if arm_kind == "none":
            if not same_bytes:
                violations.append(
                    "geometry.bin changed in an arm that does not declare a geometry change"
                )
            if not same_map:
                violations.append(
                    "map_id changed in an arm that does not declare a geometry/calibration change"
                )
            if bg["point_count"] != ag["point_count"]:
                violations.append(
                    "point_count changed in an arm that does not declare a geometry change"
                )
            if not same_calibration:
                violations.append("calibration_identity changed in an arm that does not declare it")
        elif arm_kind == "calibration":
            if not same_bytes:
                violations.append(
                    "calibration arm changed geometry.bin bytes (LiDAR->body is unchanged)"
                )
            if same_calibration:
                violations.append("calibration arm kept the baseline calibration_identity")
            if same_map:
                violations.append("calibration arm kept the baseline map_id (identity must be new)")
        else:
            if not same_bytes and same_map:
                violations.append("geometry arm changed geometry.bin bytes but kept the map_id")
            if same_bytes:
                warnings.append(
                    "geometry arm declares a geometry change but geometry.bin is identical"
                )
        invariants["geometry"] = "checked"
    else:
        invariants["geometry"] = "not_applicable (a side has no geometry layer)"

    # --- coerência interna do braço
    if _measured(other, "geometry") and _measured(other, "association"):
        if other["association"]["map_id"] != other["geometry"]["map_id"]:
            violations.append("association consumed another map_id than the arm's geometry")
        if (
            other["association"]["calibration_identity"]
            != other["geometry"]["calibration_identity"]
        ):
            violations.append("association used another calibration identity than the arm's map")
    for dotted in policy.get("required_true", []):
        layer = dotted.split(".", 1)[0]
        if _measured(other, layer) and _get(other, dotted) is not True:
            violations.append(f"{dotted} is not true")

    # --- não-regressão das camadas anteriores à camada-alvo
    regressions: list[dict[str, Any]] = []
    if target_layer is not None:
        prior = set(order[: order.index(target_layer)])
        for dotted, rule in policy["non_regression"].items():
            layer = dotted.split(".", 1)[0]
            if layer not in prior or not (_measured(base, layer) and _measured(other, layer)):
                continue
            before, after = _get(base, dotted), _get(other, dotted)
            if not (_is_number(before) and _is_number(after)):
                continue
            worse = after - before if rule["better"] == "lower" else before - after
            entry = {"metric": dotted, "baseline": before, "arm": after, "worsening": worse}
            tolerance = rule.get("tolerance_abs")
            if tolerance is None:
                entry["status"] = "reported (no declared tolerance)"
            elif worse > tolerance + 1e-12:
                entry["status"] = "regression"
                violations.append(f"{dotted} worsened by {worse:.6g} (> tolerance {tolerance})")
            else:
                entry["status"] = "ok"
            regressions.append(entry)

    return {
        "gate_policy": policy["policy_id"],
        "baseline_arm": baseline.get("arm_id"),
        "arm": arm.get("arm_id"),
        "arm_kind": arm_kind,
        "target_layer": target_layer,
        "passed": not violations,
        "violations": violations,
        "warnings": warnings,
        "invariants": invariants,
        "non_regression": regressions,
        "deltas": layer_deltas(base, other, order),
    }


def layer_deltas(base: dict[str, Any], other: dict[str, Any], order: list[str]) -> dict[str, Any]:
    """Numeric deltas (arm - baseline) for every scalar metric both sides measured."""
    deltas: dict[str, Any] = {}
    for layer in order:
        if not (_measured(base, layer) and _measured(other, layer)):
            deltas[layer] = {
                "status": "not_compared",
                "baseline": (base.get(layer) or {}).get("status"),
                "arm": (other.get(layer) or {}).get("status"),
            }
            continue
        rows = {}
        for key, before in _flatten(base[layer]).items():
            after = _flatten(other[layer]).get(key)
            if _is_number(before) and _is_number(after):
                if before != after:
                    rows[key] = {"baseline": before, "arm": after, "delta": after - before}
            elif isinstance(before, str) and isinstance(after, str) and before != after:
                rows[key] = {"baseline": before, "arm": after, "delta": "changed"}
        deltas[layer] = rows
    return deltas


def _flatten(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        name = f"{prefix}{key}"
        # Tabelas por região/rótulo não são métricas escalares de camada.
        if key in {
            "mask_sha256_by_region",
            "primary_by_region",
            "label_counts",
            "lexicon",
            "masks",
        }:
            continue
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _print(report: dict[str, Any]) -> None:
    verdict = "PASS" if report["passed"] else "FAIL"
    print(
        f"[{verdict}] {report['arm']} vs {report['baseline_arm']} "
        f"(kind={report['arm_kind']}, target={report['target_layer']}, {report['gate_policy']})"
    )
    for item in report["violations"]:
        print(f"  VIOLATION: {item}")
    for item in report["warnings"]:
        print(f"  warning: {item}")
    for layer, rows in report["deltas"].items():
        if isinstance(rows, dict) and rows.get("status") == "not_compared":
            print(f"  {layer}: not compared (baseline={rows['baseline']}, arm={rows['arm']})")
            continue
        print(f"  {layer}: {len(rows)} metric(s) differ")
        for key, row in list(rows.items())[:40]:
            delta = row["delta"]
            shown = f"{delta:+.6g}" if isinstance(delta, int | float) else delta
            print(f"    {key}: {row['baseline']} -> {row['arm']} ({shown})")
    for entry in report["non_regression"]:
        print(
            f"  non-regression {entry['metric']}: {entry['status']} "
            f"(worsening {entry['worsening']:+.6g})"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--arm-kind", choices=ARM_KINDS, default=None)
    parser.add_argument("--target-layer", default=None)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return 2 if error.code else 0
    try:
        baseline = json.loads(args.baseline.read_text())
        arm = json.loads(args.arm.read_text())
        policy = json.loads(args.policy.read_text())
        declared = arm.get("arm") or {}
        report = check(
            baseline,
            arm,
            arm_kind=args.arm_kind or declared.get("arm_kind", "none"),
            target_layer=args.target_layer or declared.get("target_layer"),
            policy=policy,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"gate usage error: {error}", file=sys.stderr)
        return 2
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        _print(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
