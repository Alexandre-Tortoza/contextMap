"""O comando que deriva um suporte espacial semântico pela linha de comando.

A capability não é etapa do DAG: nada a jusante consome o seu artifact, e o consumidor real
é o viewer, fora deste repositório. O caminho de produção suportado é este comando, que
substitui o script datado de experimento. Ele seleciona três artifacts a montante
explicitamente e delega à capability.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from chain import synthetic_chain
from voxel_builders import write_raw_artifact

from contextmap.geometric_mapping import (
    InterScanVoxelPolicy,
    VoxelAggregationArtifactWriter,
    VoxelAggregationRunId,
    VoxelGridSpec,
)
from contextmap.runtime.cli import main
from contextmap.spatial_support import SpatialSupportArtifactReader


def _upstreams(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a ContextMap chain, its raw map and a voxel aggregation of that map."""
    with synthetic_chain(tmp_path / "chain") as chain:
        raw_dir = tmp_path / "chain" / "geometric_mapping"
        frame = chain.geometry.geometry().geometric_map.frame_id
        voxel_dir = tmp_path / "voxel"
        VoxelAggregationArtifactWriter(
            output_dir=voxel_dir, run_id=VoxelAggregationRunId("voxel-0.05m")
        ).finalize(
            source_dir=raw_dir,
            policy=InterScanVoxelPolicy(
                grid=VoxelGridSpec(frame_id=frame, origin_m=(0.0, 0.0, 0.0), cell_m=0.05)
            ),
            code_version="test",
        )
        return chain.context_map_dir, voxel_dir, raw_dir


FOREIGN_SCANS = (((0.25, 0.25, 0.25), (1.50, 0.50, 0.50)), ((0.40, 0.40, 0.40),))


def test_the_command_derives_an_intact_support_from_three_upstreams(tmp_path: Path) -> None:
    context_map_dir, voxel_dir, raw_dir = _upstreams(tmp_path)
    output = tmp_path / "support"
    out, err = io.StringIO(), io.StringIO()

    code = main(
        [
            "derive-spatial-support",
            "--context-map",
            str(context_map_dir),
            "--voxel-aggregation",
            str(voxel_dir),
            "--raw-geometry",
            str(raw_dir),
            "--output",
            str(output),
            "--run-id",
            "support-cli-0001",
            "--code-version",
            "test",
            "--json",
        ],
        stdout=out,
        stderr=err,
    )

    assert code == 0, err.getvalue()
    document = json.loads(out.getvalue())
    reader = SpatialSupportArtifactReader(output)
    assert reader.verify_integrity() == []
    assert document["run_id"] == "support-cli-0001"
    assert document["cell_count"] == reader.manifest.cell_count
    assert document["entity_count"] == reader.manifest.entity_count
    assert document["policy_fingerprint"] == reader.manifest.policy_fingerprint


def test_the_command_reports_three_upstreams_that_do_not_share_one_raw_map(
    tmp_path: Path,
) -> None:
    context_map_dir, voxel_dir, _ = _upstreams(tmp_path)
    foreign_raw_dir = write_raw_artifact(tmp_path / "foreign", FOREIGN_SCANS)
    out, err = io.StringIO(), io.StringIO()

    code = main(
        [
            "derive-spatial-support",
            "--context-map",
            str(context_map_dir),
            "--voxel-aggregation",
            str(voxel_dir),
            "--raw-geometry",
            str(foreign_raw_dir),
            "--output",
            str(tmp_path / "refused"),
            "--run-id",
            "support-cli-0002",
        ],
        stdout=out,
        stderr=err,
    )

    assert code == 1
    assert "do not share one raw map" in err.getvalue()
    assert not (tmp_path / "refused").exists()


def test_the_command_reports_a_missing_upstream_without_a_traceback(tmp_path: Path) -> None:
    _, voxel_dir, raw_dir = _upstreams(tmp_path)
    missing_context_map = tmp_path / "missing-context-map"
    out, err = io.StringIO(), io.StringIO()

    code = main(
        [
            "derive-spatial-support",
            "--context-map",
            str(missing_context_map),
            "--voxel-aggregation",
            str(voxel_dir),
            "--raw-geometry",
            str(raw_dir),
            "--output",
            str(tmp_path / "refused"),
            "--run-id",
            "support-cli-0003",
        ],
        stdout=out,
        stderr=err,
    )

    assert code == 1
    assert "missing-context-map" in err.getvalue()
    assert "Traceback" not in err.getvalue()
    assert not (tmp_path / "refused").exists()
