"""Concordância do Qwen entre reruns sobre as MESMAS regiões do E0 (#556, protocolo N=3).

Lê o E0 original (``workspace-old/frame-10_31-raw``) e os reruns de ``qwen_rerun.py``
(``outputs/correction-benchmark-10_31-20260930/qwen-n3/<rerun>/``). Antes de comparar saídas,
confere que a ENTRADA do modelo foi a mesma em cada região: sha256 da view enviada e
fingerprint do prompt renderizado iguais aos do E0. Uma região com entrada diferente sai da
comparação e é contada à parte.

Concordância por região (casefold-exact, sem sinonímia):

* ``outcome``: mesmo desfecho (claim vs. falha de parse);
* ``label``: mesma hipótese primária (só entre execuções que produziram claim);
* ``category``: mesma category; ``attributes``: mesmo dicionário nome->valor.

Reporta a taxa de concordância total entre os N reruns (todos iguais) e a média das taxas
par a par; a mesma conta incluindo o E0 original como uma quarta execução; e a dispersão das
métricas de semântica do extrator (claims, falhas, labels únicas, fração com modificador,
entropia) entre as execuções.

    PYTHONPATH=src $MISE_PY experiments/correction-benchmark-10_31-20260930/scripts/qwen_variance.py
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (
    ARM_OUTPUTS,
    E0_RUNS,
    RESULTS,
    read_json,
    read_jsonl,
    write_json,
)
from extract_metrics import DEFAULT_ANNOTATION, extract

OUT = RESULTS / "E0-qwen-variance"
FIELDS = ("outcome", "label", "category", "attributes")
SUMMARY_METRICS = (
    "claim_count",
    "parse_failures",
    "unique_labels",
    "labels_with_color_or_material_prefix",
    "color_or_material_prefix_fraction",
    "label_entropy_bits",
    "unique_categories",
)


RUN_ID_IN_PROMPT = re.compile(r"[0-9a-f]{32}(?=--camera_1_image_raw)")


def _prompt_modulo_run_id(text: str) -> str:
    """sha256 of the rendered prompt with the perception run id masked.

    The rendered prompt embeds ``perception_result_id`` (``<run id>--<observation>``), so two
    runs with different run ids never send byte-identical prompts, even with the same
    configuration and the same region.
    """
    return hashlib.sha256(RUN_ID_IN_PROMPT.sub("<run-id>", text).encode()).hexdigest()


def _norm(value: Any) -> Any:
    return " ".join(value.casefold().split()) if isinstance(value, str) else value


def load_execution(perception: Path) -> dict[str, dict[str, Any]]:
    """Per region: model input identity (view sha256, prompt) and the parsed primary claim."""
    records: dict[str, dict[str, Any]] = {}
    for kind, path in (
        ("claim", perception / "outputs/semantic-interpretations.jsonl"),
        ("parse_failure", perception / "outputs/semantic-interpretation-failures.jsonl"),
    ):
        for row in read_jsonl(path):
            region = row["request"]["region_id"]
            entry: dict[str, Any] = {
                "outcome": kind,
                "views": tuple(sorted(v["sha256"] for v in row["request"]["visual_views"])),
                "prompt": row["rendered_prompt"]["fingerprint"],
                "prompt_modulo_run_id": _prompt_modulo_run_id(row["rendered_prompt"]["text"]),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "label": None,
                "category": None,
                "attributes": None,
            }
            if kind == "claim":
                primary = next(c for c in row["parsed"]["claims"] if c["role"] == "primary")
                entry["label"] = _norm(primary["hypothesis"])
                entry["category"] = _norm(primary.get("category"))
                entry["attributes"] = tuple(
                    sorted(
                        (_norm(a["name"]), _norm(str(a["value"])))
                        for a in primary.get("attributes", [])
                    )
                )
            records[region] = entry
    return records


def agreement(
    executions: dict[str, dict[str, dict[str, Any]]], regions: list[str]
) -> dict[str, Any]:
    """All-equal and mean pairwise agreement per field over the given executions."""
    names = sorted(executions)
    pairs = list(itertools.combinations(names, 2))
    result: dict[str, Any] = {"executions": names, "regions": len(regions)}
    for field in FIELDS:
        all_equal = 0
        population = 0
        pair_rates = []
        for region in regions:
            values = [executions[n][region] for n in names]
            # label/category/attributes só existem quando houve claim; uma falha de parse conta
            # como desacordo de desfecho, não como rótulo diferente.
            if field != "outcome" and any(v["outcome"] != "claim" for v in values):
                continue
            population += 1
            all_equal += len({repr(v[field]) for v in values}) == 1
        for a, b in pairs:
            comparable = [
                r
                for r in regions
                if field == "outcome"
                or (
                    executions[a][r]["outcome"] == "claim"
                    and executions[b][r]["outcome"] == "claim"
                )
            ]
            if comparable:
                same = sum(executions[a][r][field] == executions[b][r][field] for r in comparable)
                pair_rates.append(
                    {
                        "pair": f"{a}~{b}",
                        "agree": same,
                        "population": len(comparable),
                        "rate": same / len(comparable),
                    }
                )
        result[field] = {
            "all_equal": all_equal,
            "population": population,
            "all_equal_rate": all_equal / population if population else None,
            "pairwise": pair_rates,
            "pairwise_mean_rate": float(np.mean([p["rate"] for p in pair_rates]))
            if pair_rates
            else None,
        }
    return result


def main() -> int:
    """Compare the E0 original and every rerun found."""
    e0 = load_execution(E0_RUNS["raw"] / "visual_perception")
    rerun_dirs = sorted(
        p for p in (ARM_OUTPUTS / "qwen-n3").iterdir() if (p / "arm.json").is_file()
    )
    reruns = {p.name: load_execution(p / "visual_perception") for p in rerun_dirs}
    arms = {p.name: read_json(p / "arm.json") for p in rerun_dirs}

    # Entrada igual = mesmas views (sha256) e mesmo prompt a menos do run id embutido nele; a
    # identidade em bytes do prompt é reportada à parte (só os reruns que replicam o run id do
    # E0 a têm).
    input_mismatch: dict[str, list[str]] = {}
    prompt_bytes_identical: dict[str, int] = {}
    for name, execution in reruns.items():
        input_mismatch[name] = sorted(
            r
            for r in set(e0) | set(execution)
            if r not in e0
            or r not in execution
            or e0[r]["views"] != execution[r]["views"]
            or e0[r]["prompt_modulo_run_id"] != execution[r]["prompt_modulo_run_id"]
        )
        prompt_bytes_identical[name] = sum(
            1 for r in e0 if r in execution and e0[r]["prompt"] == execution[r]["prompt"]
        )
    regions = sorted(
        r
        for r in e0
        if all(r not in bad for bad in input_mismatch.values())
        and all(r in x for x in reruns.values())
    )

    per_execution_metrics = {}
    for p in rerun_dirs:
        metrics = extract(
            p,
            arm_id=f"E0-qwen-{p.name}",
            annotation_path=DEFAULT_ANNOTATION,
            notes="rerun do Qwen (#556)",
        )
        metrics["arm"] = arms[p.name]
        write_json(OUT / p.name / "metrics.json", metrics)
        per_execution_metrics[p.name] = metrics["layers"]["semantics"]
    e0_semantics = read_json(RESULTS / "E0/metrics.json")["layers"]["semantics"]
    all_semantics = {"E0-original": e0_semantics, **per_execution_metrics}
    dispersion = {}
    for key in SUMMARY_METRICS:
        values = [m[key] for m in all_semantics.values()]
        dispersion[key] = {
            "values": dict(zip(all_semantics, values, strict=True)),
            "min": min(values),
            "max": max(values),
            "mean": float(np.mean(values)),
        }

    summary = {
        "protocol": "N reruns do Qwen com configuração do E0 sobre as mesmas regiões "
        "(SAM2 determinístico, conferido por sha256 de máscara)",
        "reruns": {
            name: {
                "run_id": arms[name]["perception_run_id"],
                "regions_identical_to_e0": arms[name]["regions_identical_to_e0"],
            }
            for name in arms
        },
        "model_input_identity": {
            "regions_compared": len(regions),
            "regions_with_different_input_by_rerun": input_mismatch,
            "note": "entrada = sha256 das views enviadas + prompt renderizado (a menos do run id), "
            "por região",
        },
        "prompt_byte_identical_regions_by_rerun": prompt_bytes_identical,
        "groups": {
            "r*": "run id próprio por rerun (prompt difere do E0 só no run id embutido)",
            "x*": "request idêntico ao do E0 (mesmo run id): prompt e views byte-idênticos",
        },
        "agreement_reruns_only": agreement(reruns, regions) if len(reruns) >= 2 else None,
        "agreement_by_group": {
            group: agreement(members, regions)
            for group, members in (
                ("r", {k: v for k, v in reruns.items() if k.startswith("r")}),
                ("x", {k: v for k, v in reruns.items() if k.startswith("x")}),
                (
                    "x+E0",
                    {"E0-original": e0, **{k: v for k, v in reruns.items() if k.startswith("x")}},
                ),
            )
            if len(members) >= 2
        },
        "agreement_including_e0_original": agreement({"E0-original": e0, **reruns}, regions),
        "semantic_metric_dispersion": dispersion,
        "raw_response_identical_to_e0": {
            name: sum(e0[r]["raw_response_sha256"] == x[r]["raw_response_sha256"] for r in regions)
            for name, x in reruns.items()
        },
    }
    write_json(OUT / "summary.json", summary)
    with (OUT / "per_region.csv").open("w", newline="") as handle:
        names = ["E0-original", *reruns]
        executions = {"E0-original": e0, **reruns}
        writer = csv.writer(handle)
        writer.writerow(
            ["region_id", *(f"{n}:{f}" for n in names for f in ("outcome", "label", "category"))]
        )
        for region in regions:
            writer.writerow(
                [
                    region,
                    *(
                        executions[n][region][f]
                        for n in names
                        for f in ("outcome", "label", "category")
                    ),
                ]
            )
    for scope in ("agreement_reruns_only", "agreement_including_e0_original"):
        block = summary[scope]
        if block:
            print(
                scope,
                {f: (block[f]["all_equal_rate"], block[f]["pairwise_mean_rate"]) for f in FIELDS},
            )
    print({k: (v["min"], v["max"]) for k, v in dispersion.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
