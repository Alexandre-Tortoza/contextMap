#!/usr/bin/env python3
"""Offline relation-graph diagnostic for a canonical/2 run.

Read-only analysis that separates four competing explanations for the large
scene-graph relation count, without touching production code or the run's
artifact directories:

* **H1 entity fragmentation** - how many source entities each resolved entity
  consolidates, and the singleton fraction.
* **relation normalization** - collapse the symmetric/derived twin records so
  degree statistics are computed on canonical, unordered entity pairs (the
  ``derived_from is None`` records), exposing any double counting in the raw
  totals.
* **H2 candidate density** - how many unordered entity pairs were actually
  evaluated, as a fraction of all possible pairs.
* **H3 structural multiplier** - a keyword-based (therefore approximate)
  structural/other decomposition of the supported graph, reporting the residual
  degree distribution and density after removing structural endpoints, not only
  the drop in edge count.
* **H4 graph materialization** - whether the public context map materializes
  rejected/unresolved (and derived) records as edges, or only supported ones.

Appearance-association calibration (H5) is deliberately out of scope here.

The script reads the run's public ``outputs/*.jsonl`` payloads directly (the
serialized contract surface) for the two large streaming passes, and uses the
public :class:`ContextMapArtifactReader` for the context map. It writes a single
JSON report to the path given as the second argument and prints a summary; it
never writes into the run's own artifact directories.

Usage:
    PYTHONPATH=src python diagnose_relation_graph.py <run_workspace> <out_json>
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Nome do run cujos valores conhecidos disparam asserções duras (ver SANITY).
KNOWN_RUN_NAME = "canonical-2-real-20260929"

# Valores esperados para o run conhecido. Um desvio indica erro de
# carregamento/canonicalização e deve falhar alto (AGENTS.md 26), nunca
# prosseguir em silêncio.
EXPECTED = {
    "source_entities": 606,
    "resolved_entities": 535,
    "singletons": 484,
    "total_relations": 98568,
    "canonical_relations": 49284,
    "canonical_supported": 4962,
    "canonical_rejected": 7304,
    "canonical_unresolved": 37018,
}

# Classificação estrutural por palavra-chave sobre o texto livre das hipóteses.
# Não há campo categórico de superfície estrutural no domínio: os rótulos são
# frases descritivas abertas do modelo, então isto e aproximado e reportado como
# analise de sensibilidade (variantes strict/loose), nao como verdade de campo.
STRICT_KEYWORDS = ("wall", "floor", "ceiling", "pillar", "column")
LOOSE_EXTRA_KEYWORDS = ("structural", "surface", "beam", "slab", "girder")


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (0-100) via linear interpolation.

    Args:
        values: Sample values; may be empty.
        pct: Percentile in ``[0, 100]``.

    Returns:
        The interpolated percentile, or ``0.0`` for an empty sample.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def _describe(values: list[float]) -> dict[str, float]:
    """Summarize a numeric sample with count/mean/median/p95/max/min."""
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
    ordered = sorted(values)
    total = float(sum(ordered))
    n = len(ordered)
    median = _percentile(ordered, 50.0)
    return {
        "count": n,
        "mean": total / n,
        "median": median,
        "p95": _percentile(ordered, 95.0),
        "max": float(ordered[-1]),
        "min": float(ordered[0]),
    }


class _UnionFind:
    """Minimal union-find for connected-component sizing over string nodes."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        self._parent.setdefault(node, node)

    def find(self, node: str) -> str:
        parent = self._parent
        root = node
        while parent[root] != root:
            root = parent[root]
        # Compressão de caminho iterativa.
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def largest_component_size(self) -> int:
        if not self._parent:
            return 0
        sizes: Counter[str] = Counter(self.find(node) for node in self._parent)
        return max(sizes.values())


def _iter_jsonl(path: Path) -> Any:
    """Yield decoded JSON objects from a JSON Lines file, skipping blank lines."""
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _classify_structural(labels: list[str], keywords: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Return whether any label matches a structural keyword, and which matched.

    Matching is whole-word, case-insensitive, over the free-text hypothesis
    labels. This is an approximation: labels are descriptive sentences, so a
    match is evidence of a structural surface, not a certified type.
    """
    matched: list[str] = []
    for keyword in keywords:
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        if any(pattern.search(label) for label in labels):
            matched.append(keyword)
    return (bool(matched), matched)


def load_entities(run: Path) -> dict[str, Any]:
    """Load resolved entities: member counts and structural classification.

    Returns a dict with per-entity member counts, singleton stats, the
    source-size histogram, and structural flags under strict and loose keyword
    sets keyed by ``resolved_entity_id``.
    """
    path = run / "pipeline/entity_resolution/outputs/resolved-entities.jsonl"
    members_per_entity: list[int] = []
    size_histogram: Counter[int] = Counter()
    structural_strict: dict[str, bool] = {}
    structural_loose: dict[str, bool] = {}
    labels_by_entity: dict[str, list[str]] = {}
    matched_keywords: dict[str, list[str]] = {}

    for record in _iter_jsonl(path):
        rid = record["resolved_entity_id"]
        n_members = len(record.get("members", ()))
        members_per_entity.append(n_members)
        size_histogram[n_members] += 1
        labels = [
            hyp.get("label", "") for hyp in record.get("semantic_state", {}).get("hypotheses", ())
        ]
        labels_by_entity[rid] = labels
        strict_hit, matched = _classify_structural(labels, STRICT_KEYWORDS)
        loose_hit, matched_loose = _classify_structural(
            labels, STRICT_KEYWORDS + LOOSE_EXTRA_KEYWORDS
        )
        structural_strict[rid] = strict_hit
        structural_loose[rid] = loose_hit
        matched_keywords[rid] = matched_loose or matched

    resolved_count = len(members_per_entity)
    singletons = size_histogram.get(1, 0)
    source_entities = sum(members_per_entity)

    return {
        "resolved_entities": resolved_count,
        "source_entities": source_entities,
        "singletons": singletons,
        "singleton_fraction": (singletons / resolved_count) if resolved_count else 0.0,
        "sources_per_resolved_entity": _describe([float(v) for v in members_per_entity]),
        "size_histogram": {str(k): size_histogram[k] for k in sorted(size_histogram)},
        "structural_strict": structural_strict,
        "structural_loose": structural_loose,
        "labels_by_entity": labels_by_entity,
        "matched_keywords": matched_keywords,
    }


def _canonical_pair(subject: str, obj: str) -> tuple[str, str]:
    """Return the entity pair as a canonical unordered ``(min, max)`` tuple."""
    return (subject, obj) if subject <= obj else (obj, subject)


def load_relations(run: Path) -> dict[str, Any]:
    """Stream relations.jsonl and build canonical relation/edge structures.

    Collapses derived twins (``derived_from`` set) so statistics are computed on
    the directly-evaluated, unordered-pair canonical set. Returns raw and
    canonical state counts, the set of evaluated pairs, and the supported edge
    set, plus incidence tallies for the degree-inflation comparison.
    """
    path = run / "pipeline/spatial_relations/outputs/relations.jsonl"

    total = 0
    canonical = 0
    raw_by_state: Counter[str] = Counter()
    canonical_by_state: Counter[str] = Counter()
    canonical_by_predicate_state: dict[str, Counter[str]] = defaultdict(Counter)

    evaluated_pairs: set[tuple[str, str]] = set()
    supported_edges: set[tuple[str, str]] = set()
    supported_pair_predicate = 0
    self_loops = 0

    # Incidencia sobre TODOS os registros (inclui twins derivados): reproduz a
    # contagem inflada de "grau" que motivou a mediana suspeita de 52.
    incidence_all_supported: Counter[str] = Counter()

    for record in _iter_jsonl(path):
        total += 1
        state = record["state"]
        raw_by_state[state] += 1
        subject = record["subject_entity_ref"]["resolved_entity_id"]
        obj = record["object_entity_ref"]["resolved_entity_id"]
        if state == "supported":
            incidence_all_supported[subject] += 1
            incidence_all_supported[obj] += 1

        if record.get("derived_from") is not None:
            continue

        canonical += 1
        predicate = record["predicate"]
        canonical_by_state[state] += 1
        canonical_by_predicate_state[predicate][state] += 1
        if subject == obj:
            self_loops += 1
            continue
        pair = _canonical_pair(subject, obj)
        evaluated_pairs.add(pair)
        if state == "supported":
            supported_edges.add(pair)
            supported_pair_predicate += 1

    return {
        "total_relations": total,
        "canonical_relations": canonical,
        "raw_by_state": dict(raw_by_state),
        "canonical_by_state": dict(canonical_by_state),
        "canonical_by_predicate_state": {
            pred: dict(counts) for pred, counts in canonical_by_predicate_state.items()
        },
        "evaluated_pairs": evaluated_pairs,
        "supported_edges": supported_edges,
        "supported_pair_predicate_count": supported_pair_predicate,
        "self_loops": self_loops,
        "incidence_all_supported": incidence_all_supported,
    }


def _graph_stats(nodes: list[str], edges: set[tuple[str, str]]) -> dict[str, Any]:
    """Compute degree distribution, density, and largest component for a graph.

    Args:
        nodes: All node ids (isolated nodes included so degree-0 counts).
        edges: Undirected edges as canonical ``(min, max)`` pairs.

    Returns:
        Node/edge counts, degree summaries over all nodes and over non-isolated
        nodes, graph density ``2E/N(N-1)``, and the largest component size.
    """
    degree: Counter[str] = Counter()
    uf = _UnionFind()
    for node in nodes:
        uf.add(node)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        uf.add(a)
        uf.add(b)
        uf.union(a, b)

    n = len(nodes)
    all_degrees = [float(degree.get(node, 0)) for node in nodes]
    non_isolated = [d for d in all_degrees if d > 0]
    possible_pairs = n * (n - 1) / 2 if n > 1 else 0
    density = (len(edges) / possible_pairs) if possible_pairs else 0.0

    return {
        "nodes": n,
        "edges": len(edges),
        "possible_pairs": int(possible_pairs),
        "density": density,
        "degree_all_nodes": _describe(all_degrees),
        "degree_non_isolated": _describe(non_isolated),
        "isolated_nodes": n - len(non_isolated),
        "largest_component_size": uf.largest_component_size(),
    }


def _edge_class(pair: tuple[str, str], structural: dict[str, bool]) -> str:
    """Classify an edge by its endpoints as structural/structural, etc."""
    a_struct = structural.get(pair[0], False)
    b_struct = structural.get(pair[1], False)
    if a_struct and b_struct:
        return "structural_structural"
    if a_struct or b_struct:
        return "structural_other"
    return "other_other"


def structural_ablation(
    all_nodes: list[str],
    supported_edges: set[tuple[str, str]],
    structural: dict[str, bool],
) -> dict[str, Any]:
    """Decompose the supported graph and recompute it without structural nodes.

    Reports the edge-class decomposition, the baseline supported-graph stats,
    the residual (other-other) graph stats, and the residual-vs-baseline deltas
    so the effect is judged on residual degree/density, not only edge drop.
    """
    decomposition: Counter[str] = Counter()
    for pair in supported_edges:
        decomposition[_edge_class(pair, structural)] += 1

    baseline = _graph_stats(all_nodes, supported_edges)

    other_nodes = [n for n in all_nodes if not structural.get(n, False)]
    residual_edges = {
        pair for pair in supported_edges if _edge_class(pair, structural) == "other_other"
    }
    residual = _graph_stats(other_nodes, residual_edges)

    structural_count = sum(1 for n in all_nodes if structural.get(n, False))
    edge_drop = 1.0 - (residual["edges"] / baseline["edges"]) if baseline["edges"] else 0.0

    return {
        "structural_entities": structural_count,
        "structural_fraction": (structural_count / len(all_nodes)) if all_nodes else 0.0,
        "edge_decomposition": dict(decomposition),
        "baseline_supported_graph": baseline,
        "residual_other_only_graph": residual,
        "edge_count_drop_fraction": edge_drop,
        "median_degree_baseline_all": baseline["degree_all_nodes"]["median"],
        "median_degree_residual_all": residual["degree_all_nodes"]["median"],
        "density_baseline": baseline["density"],
        "density_residual": residual["density"],
    }


def context_map_materialization(run: Path) -> dict[str, Any]:
    """Count context-map relation records by state (H4).

    Uses the public reader so the decoded, contractual state is measured. Reports
    how much of the materialized graph is not supported, and notes that derived
    twins are carried too (supported here is the doubled figure).
    """
    from contextmap.artifact import ContextMapArtifactReader

    states: Counter[str] = Counter()
    total = 0
    with ContextMapArtifactReader.open(run / "pipeline/context_map", verify_hashes=False) as reader:
        for relation in reader.relations():
            total += 1
            state = getattr(relation, "state", None)
            states[getattr(state, "value", state)] += 1

    not_supported = total - states.get("supported", 0)
    return {
        "materialized_relations": total,
        "by_state": dict(states),
        "not_supported_fraction": (not_supported / total) if total else 0.0,
        "note": (
            "Context map materializes every relation record, including derived "
            "twins and rejected/unresolved evaluations; 'supported' here is the "
            "doubled figure (2x the canonical supported edge count)."
        ),
    }


def _sanity_checks(entities: dict[str, Any], relations: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the expected-vs-actual sanity table for the known run."""
    actual = {
        "source_entities": entities["source_entities"],
        "resolved_entities": entities["resolved_entities"],
        "singletons": entities["singletons"],
        "total_relations": relations["total_relations"],
        "canonical_relations": relations["canonical_relations"],
        "canonical_supported": relations["canonical_by_state"].get("supported", 0),
        "canonical_rejected": relations["canonical_by_state"].get("rejected", 0),
        "canonical_unresolved": relations["canonical_by_state"].get("unresolved", 0),
    }
    checks = []
    for name, expected in EXPECTED.items():
        got = actual[name]
        checks.append({"check": name, "expected": expected, "actual": got, "ok": got == expected})
    return checks


def main(argv: list[str]) -> int:
    """Run the diagnostic and write the JSON report.

    Args:
        argv: ``[run_workspace, out_json]``.

    Returns:
        Process exit code; non-zero on bad arguments or a failed sanity check
        for the known run.
    """
    if len(argv) != 2:
        print(__doc__)
        print("error: expected exactly <run_workspace> <out_json>", file=sys.stderr)
        return 2
    run = Path(argv[0]).resolve()
    out_path = Path(argv[1]).resolve()
    if not run.is_dir():
        print(f"error: run workspace not found: {run}", file=sys.stderr)
        return 2

    entities = load_entities(run)
    relations = load_relations(run)

    all_nodes = sorted(entities["structural_strict"].keys())
    evaluated_pairs = relations["evaluated_pairs"]
    supported_edges = relations["supported_edges"]

    # H2: densidade de candidatos sobre pares nao ordenados unicos.
    n_entities = entities["resolved_entities"]
    possible_pairs = n_entities * (n_entities - 1) // 2 if n_entities > 1 else 0
    unique_supported_pairs = len(supported_edges)

    # Comparacao de grau: canonico (dedup) vs incidencia sobre todos os registros.
    incidence = relations["incidence_all_supported"]
    incidence_degrees = [float(incidence.get(node, 0)) for node in all_nodes]

    ablation_strict = structural_ablation(all_nodes, supported_edges, entities["structural_strict"])
    ablation_loose = structural_ablation(all_nodes, supported_edges, entities["structural_loose"])

    h4 = context_map_materialization(run)

    sanity = _sanity_checks(entities, relations)

    report: dict[str, Any] = {
        "run_workspace": str(run),
        "run_name": run.name,
        "scope_note": (
            "Offline, read-only diagnostic. Appearance-association calibration "
            "(H5) is out of scope. Structural classification (H3) is keyword-based "
            "over free-text labels and reported as a sensitivity result."
        ),
        "sanity_checks": sanity,
        "h1_entity_fragmentation": {
            "resolved_entities": entities["resolved_entities"],
            "source_entities": entities["source_entities"],
            "singletons": entities["singletons"],
            "singleton_fraction": entities["singleton_fraction"],
            "sources_per_resolved_entity": entities["sources_per_resolved_entity"],
            "size_histogram": entities["size_histogram"],
        },
        "relation_normalization": {
            "total_relations": relations["total_relations"],
            "canonical_relations": relations["canonical_relations"],
            "twin_ratio": (
                relations["total_relations"] / relations["canonical_relations"]
                if relations["canonical_relations"]
                else 0.0
            ),
            "raw_by_state": relations["raw_by_state"],
            "canonical_by_state": relations["canonical_by_state"],
            "canonical_by_predicate_state": relations["canonical_by_predicate_state"],
            "self_loops": relations["self_loops"],
            "degree_comparison": {
                "canonical_supported_degree_all_nodes": ablation_strict["baseline_supported_graph"][
                    "degree_all_nodes"
                ],
                "canonical_supported_degree_non_isolated": ablation_strict[
                    "baseline_supported_graph"
                ]["degree_non_isolated"],
                "incidence_supported_degree_all_records": _describe(incidence_degrees),
                "note": (
                    "canonical_* counts distinct supported neighbors on the "
                    "undirected graph; incidence_* counts supported relation "
                    "records touching each entity over ALL records (derived twins "
                    "included) and is inflated relative to the true node degree."
                ),
            },
        },
        "h2_candidate_density": {
            "resolved_entities": n_entities,
            "possible_unordered_pairs": possible_pairs,
            "unique_evaluated_pairs": len(evaluated_pairs),
            "evaluated_pair_fraction": (
                len(evaluated_pairs) / possible_pairs if possible_pairs else 0.0
            ),
            "canonical_pair_predicate_candidates": relations["canonical_relations"],
            "unique_supported_pairs": unique_supported_pairs,
            "supported_pair_predicate_count": relations["supported_pair_predicate_count"],
            "note": (
                "canonical_pair_predicate_candidates counts (pair, predicate) "
                "evaluations (49284); unique_evaluated_pairs collapses predicates "
                "to unordered entity pairs. The two differ because next_to and "
                "touching can both be candidates for the same pair."
            ),
        },
        "h3_structural_multiplier": {
            "keyword_sets": {
                "strict": list(STRICT_KEYWORDS),
                "loose": list(STRICT_KEYWORDS + LOOSE_EXTRA_KEYWORDS),
            },
            "strict": ablation_strict,
            "loose": ablation_loose,
            "caveat": (
                "There is no structural-surface type in the domain. These flags "
                "come from whole-word keyword matches over free-text hypothesis "
                "labels and are a sensitivity estimate, not ground truth."
            ),
        },
        "h4_graph_materialization": h4,
        "entity_structural_classification": {
            rid: {
                "structural_strict": entities["structural_strict"][rid],
                "structural_loose": entities["structural_loose"][rid],
                "matched_keywords": entities["matched_keywords"][rid],
                "labels": entities["labels_by_entity"][rid],
            }
            for rid in all_nodes
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Resumo no stdout.
    print(f"run: {run.name}")
    print(f"report written: {out_path}")
    failed = [c for c in sanity if not c["ok"]]
    print("sanity:", "OK" if not failed else f"FAILED {[c['check'] for c in failed]}")
    h1 = report["h1_entity_fragmentation"]
    print(
        f"H1  resolved={h1['resolved_entities']} source={h1['source_entities']} "
        f"singletons={h1['singletons']} ({h1['singleton_fraction']:.1%})"
    )
    norm = report["relation_normalization"]
    print(
        f"NORM total={norm['total_relations']} canonical={norm['canonical_relations']} "
        f"(x{norm['twin_ratio']:.2f}) canonical_by_state={norm['canonical_by_state']}"
    )
    dc = norm["degree_comparison"]
    print(
        "     supported-degree median: "
        f"canonical={dc['canonical_supported_degree_all_nodes']['median']:.1f} "
        f"non_isolated={dc['canonical_supported_degree_non_isolated']['median']:.1f} "
        f"incidence(all records)={dc['incidence_supported_degree_all_records']['median']:.1f}"
    )
    h2 = report["h2_candidate_density"]
    print(
        f"H2  unique_evaluated_pairs={h2['unique_evaluated_pairs']} "
        f"of {h2['possible_unordered_pairs']} ({h2['evaluated_pair_fraction']:.1%}); "
        f"unique_supported_pairs={h2['unique_supported_pairs']}"
    )
    for variant in ("strict", "loose"):
        a = report["h3_structural_multiplier"][variant]
        print(
            f"H3[{variant}] structural={a['structural_entities']} "
            f"({a['structural_fraction']:.1%}); edges {a['baseline_supported_graph']['edges']}"
            f"->{a['residual_other_only_graph']['edges']} "
            f"(drop {a['edge_count_drop_fraction']:.1%}); "
            f"density {a['density_baseline']:.4f}->{a['density_residual']:.4f}; "
            f"median-deg(all) {a['median_degree_baseline_all']:.1f}"
            f"->{a['median_degree_residual_all']:.1f}"
        )
    print(
        f"H4  materialized={h4['materialized_relations']} by_state={h4['by_state']} "
        f"not_supported={h4['not_supported_fraction']:.1%}"
    )

    if run.name == KNOWN_RUN_NAME and failed:
        detail = "; ".join(
            f"{c['check']} expected {c['expected']} got {c['actual']}" for c in failed
        )
        raise SystemExit(f"sanity checks failed for known run {run.name}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
