"""Resumo da interpretação semântica de um run de percepção: falhas de parse e fator de interação.

Uso: python semantic_summary.py <run>/visual_perception [--json]

Lê só `outputs/` (contratual). Conta, por template de prompt:
- regiões interpretadas e falhas de parse, agrupadas pela mensagem do parser;
- claims primárias com e sem `interaction_factor`, e a distribuição dos fatores;
- rótulos primários;
- vazamento do contorno da view: cor magenta/rosa em `attributes.color`, ou descrição
  que menciona borda/contorno rosa (o contorno é desenhado em [255, 0, 255]).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

_OUTLINE_COLOR = re.compile(r"\b(pink|magenta|purple|fuchsia)\b", re.IGNORECASE)
_OUTLINE_MENTION = re.compile(r"\b(border|outline|contour|frame)\b", re.IGNORECASE)


def _jsonl(path: Path) -> list[dict]:
    """Lê um JSONL; arquivo ausente é uma lista vazia."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise(stage_dir: Path) -> dict:
    """Resume `semantic-interpretations.jsonl` e `semantic-interpretation-failures.jsonl`."""
    outputs = stage_dir / "outputs"
    interpretations = _jsonl(outputs / "semantic-interpretations.jsonl")
    failures = _jsonl(outputs / "semantic-interpretation-failures.jsonl")
    call_failures = _jsonl(outputs / "semantic-backend-call-failures.jsonl")

    templates = Counter()
    labels = Counter()
    factors: list[float] = []
    primaries = without_factor = abstained = 0
    outline_color = outline_description = 0
    per_frame = Counter()
    for record in interpretations:
        parsed = record["parsed"]
        if parsed.get("abstained"):
            abstained += 1
        for claim in parsed["claims"]:
            templates[claim["provenance"]["prompt_template_id"]] += 1
            if claim.get("role") not in (None, "primary"):
                continue
            primaries += 1
            labels[claim["hypothesis"]] += 1
            per_frame[claim["perception_result_id"].rsplit("--", 1)[-1]] += 1
            attributes = {a["name"]: a["value"] for a in claim.get("attributes", [])}
            if "interaction_factor" in attributes:
                factors.append(float(attributes["interaction_factor"]))
            else:
                without_factor += 1
            if _OUTLINE_COLOR.search(str(attributes.get("color", ""))):
                outline_color += 1
            description = str(attributes.get("description", ""))
            if _OUTLINE_COLOR.search(description) and _OUTLINE_MENTION.search(description):
                outline_description += 1

    failure_kinds = Counter(
        re.sub(r"claim\[\d+\]", "claim[i]", f["parse_failure"]["message"]) for f in failures
    )
    attempted = len(interpretations) + len(failures) + len(call_failures)
    return {
        "stage_dir": str(stage_dir),
        "prompt_templates": dict(templates),
        "regions_attempted": attempted,
        "regions_interpreted": len(interpretations),
        "abstained": abstained,
        "parse_failures": len(failures),
        "parse_failure_rate": round(len(failures) / attempted, 4) if attempted else None,
        "parse_failure_kinds": dict(failure_kinds.most_common()),
        "backend_call_failures": len(call_failures),
        "primary_claims": primaries,
        "primary_claims_per_frame": dict(per_frame),
        "primary_claims_without_factor": without_factor,
        "interaction_factor": {
            "count": len(factors),
            "distinct": sorted(set(factors)),
            "histogram": dict(sorted(Counter(factors).items())),
            "median": statistics.median(factors) if factors else None,
            "min": min(factors) if factors else None,
            "max": max(factors) if factors else None,
            "outside_clamp_0.1_5.0": sum(1 for f in factors if not 0.1 <= f <= 5.0),
        },
        "labels": dict(labels.most_common()),
        "distinct_labels": len(labels),
        "outline_leak": {
            "color_attribute_pink_or_magenta": outline_color,
            "description_mentions_pink_border": outline_description,
        },
    }


def main() -> None:
    """Imprime o resumo de cada diretório de estágio passado."""
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    summaries = [summarise(p) for p in paths]
    print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2))


if __name__ == "__main__":
    main()
