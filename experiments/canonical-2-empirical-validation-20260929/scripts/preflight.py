"""Preflight for the canonical/2 validation bundle: refuse an unfit configuration before running.

Resolves the given config layers + overrides and refuses if any PLACEHOLDER remains, if a value
looks like an accidental machine-local absolute path, or if a recorded model identity/revision
diverges from manifests/experiment.json. It does not load models or touch the GPU.

Usage:
  preflight.py --config base.toml [--config arm.toml] [--set dotted.path=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextmap.runtime import resolve_effective_config

_BUNDLE = Path(__file__).resolve().parent.parent
_EXPERIMENT = json.loads((_BUNDLE / "manifests" / "experiment.json").read_text())


def _walk(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    effective = resolve_effective_config(files=list(args.config), overrides=list(args.overrides))
    document = effective.config.to_document()
    problems: list[str] = []

    for path, value in _walk(document):
        if isinstance(value, str):
            if "PLACEHOLDER" in value:
                problems.append(f"{path}: unresolved PLACEHOLDER ({value!r})")
            if value.startswith("/") or value.startswith("~") or "/.cache/" in value:
                problems.append(f"{path}: looks like a machine-local absolute path ({value!r})")

    # Recorded model identities must not drift from the experiment manifest.
    models = _EXPERIMENT["models"]
    checks = {
        "region_discovery.sam2.checkpoint": models["region_discovery"]["checkpoint"],
        "dense_features.dinov2.revision": models["dense_features"]["revision"],
        "region_features.clip.revision": models["region_features"]["revision"],
        "semantic_interpretation.qwen.revision": models["semantic_interpretation"]["revision"],
    }
    vp = document.get("components", {}).get("visual_perception", {})
    resolved = {
        "region_discovery.sam2.checkpoint": vp.get("region_discovery", {})
        .get("sam2", {})
        .get("checkpoint"),
        "dense_features.dinov2.revision": vp.get("dense_features", {})
        .get("dinov2", {})
        .get("revision"),
        "region_features.clip.revision": vp.get("region_features", {})
        .get("clip", {})
        .get("revision"),
        "semantic_interpretation.qwen.revision": vp.get("semantic_interpretation", {})
        .get("qwen", {})
        .get("revision"),
    }
    for key, expected in checks.items():
        if resolved[key] != expected:
            problems.append(f"{key}: {resolved[key]!r} diverges from recorded {expected!r}")

    if problems:
        print("PREFLIGHT REFUSED:")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("PREFLIGHT OK: no placeholders, no local paths, model identities match the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
