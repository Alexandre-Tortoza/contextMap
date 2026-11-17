"""Parte 2: a semântica altera a geometria? Snapshots 01..08 com estatísticas e deltas.

Somente leitura. Recalcula de forma independente a cadeia velodyne -> epson -> map de um scan
e compara com o geometry.bin persistido; depois segue as referências de geometria por
associação, fusão, mapeamento semântico, resolução e context_map, sempre resolvendo cada
referência de volta ao XYZ autoritativo e verificando se alguém carrega XYZ próprio.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from common import (
    IMAGE_ID,
    Trajectory,
    apply,
    cloud_stats,
    deltas,
    dump,
    load_geometry,
    load_jsonl,
    load_scan_payload,
    rigid,
    run_dir,
    sequence_index,
    static_transform,
    ts_ns,
)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def decode_deltas(deltas_list) -> np.ndarray:
    return np.cumsum(np.asarray(deltas_list, dtype=np.int64))


def geometry_index(geometry_id: str) -> int:
    return int(geometry_id.rsplit("--geom-", 1)[1])


def main() -> None:
    out: dict = {}
    index = sequence_index()
    image = next(e for e in index if e["observation_id"] == IMAGE_ID)
    t_img = ts_ns(image["timestamp"])
    lidar_entries = [e for e in index if e["modality"] == "lidar"]
    traj = Trajectory()

    for arm in ("raw", "voxel"):
        geo = load_geometry(arm)
        scans = load_jsonl(run_dir(arm) / "geometric_mapping/outputs/source-index.jsonl")
        by_obs = {s["observation_id"]: s for s in scans}
        # scan do mapa mais próximo do instante da imagem
        nearest = min(scans, key=lambda s: abs(ts_ns(s["acquisition_timestamp"]) - t_img))
        entry = next(e for e in lidar_entries if e["observation_id"] == nearest["observation_id"])
        first, count = nearest["first_geometry_index"], nearest["geometry_count"]
        seg = geo[first : first + count]
        arm_out: dict = {
            "reference_scan": nearest["observation_id"],
            "reference_scan_minus_image_ms": (ts_ns(nearest["acquisition_timestamp"]) - t_img) / 1e6,
        }

        # 01 nuvem bruta do sensor (payload da sequência, frame cmu_rc1_velodyne)
        raw_xyz, _ = load_scan_payload(entry)
        arm_out["01_raw_sensor_cloud"] = cloud_stats(raw_xyz)

        # 02 após extrínseca estática (epson <- velodyne), calculada por nós
        t_epson_vel = static_transform("epson", "cmu_rc1_velodyne")
        body_xyz = apply(t_epson_vel, raw_xyz)
        arm_out["02_cloud_after_sensor_transform"] = cloud_stats(body_xyz)

        # 03 mundo: pose GT interpolada por nós (lerp+slerp) no instante do scan
        t_scan = ts_ns(nearest["acquisition_timestamp"])
        t_map_epson, bracket = traj.pose(t_scan)
        world_ours = apply(t_map_epson, body_xyz)
        arm_out["03_cloud_world"] = cloud_stats(world_ours)
        arm_out["03_pose_bracket"] = bracket
        # cadeia registrada no source-index (o que o geometric_mapping de fato aplicou)
        chain = nearest["transform_chain"]
        t_recorded = rigid(chain[0]["translation_m"], chain[0]["rotation"]) @ rigid(
            chain[1]["translation_m"], chain[1]["rotation"]
        )
        if arm == "raw":
            # correspondência por índice de ponto no scan: o raw preserva a ordem do payload
            src_idx = seg["source_index"]
            arm_out["check_source_xyz_equals_payload"] = deltas(seg["source"], raw_xyz[src_idx])
            arm_out["check_map_xyz_vs_our_chain"] = deltas(seg["map"], world_ours[src_idx])
            arm_out["check_map_xyz_vs_recorded_chain"] = deltas(
                seg["map"], apply(t_recorded, seg["source"])
            )
            arm_out["raw_scan_points_dropped"] = int(len(raw_xyz) - count)
        else:
            # 04 voxelização intra-scan: centroides; recalculamos a partir do raw do mesmo scan
            raw_geo = load_geometry("raw")
            raw_scans = {s["observation_id"]: s for s in load_jsonl(run_dir("raw") / "geometric_mapping/outputs/source-index.jsonl")}
            rs = raw_scans[nearest["observation_id"]]
            raw_seg = raw_geo[rs["first_geometry_index"] : rs["first_geometry_index"] + rs["geometry_count"]]
            arm_out["04_cloud_after_geometric_voxelization_scan"] = cloud_stats(seg["map"])
            arm_out["04_counts_per_voxel_point"] = {
                "min": int(seg["count"].min()), "max": int(seg["count"].max()),
                "sum": int(seg["count"].sum()), "raw_points_in_scan": int(rs["geometry_count"]),
            }
            from scipy.spatial import cKDTree

            tree = cKDTree(raw_seg["map"])
            d, _ = tree.query(seg["map"])
            arm_out["04_voxel_point_to_nearest_raw_point_m"] = {
                "median": float(np.median(d)), "p95": float(np.percentile(d, 95)), "max": float(d.max()),
            }
            # recomputação independente do centroide de voxel no frame do SENSOR? no mapa? testamos ambos
            cfg = json.loads((run_dir("voxel") / "geometric_mapping/config.json").read_text())
            arm_out["04_aggregation_config"] = cfg.get("aggregation")
        arm_out["05_cloud_before_semantics_full_map"] = cloud_stats(geo["map"])
        arm_out["05_geometry_bin_sha256"] = sha256(run_dir(arm) / "geometric_mapping/outputs/geometry.bin")
        man = json.loads((run_dir(arm) / "geometric_mapping/manifest.json").read_text())
        arm_out["05_manifest_geometry_hash"] = next(
            f["content_hash"] for f in man["file_inventory"] if f["path"] == "outputs/geometry.bin"
        )

        # 06 após associação 2D<->3D: só referências (u32) ao mapa
        sa = run_dir(arm) / "sensor_association/outputs"
        support = np.fromfile(sa / "geometry-support.u32", dtype="<u4")
        obs = load_jsonl(sa / "spatial-observations.jsonl")
        assoc_idx = np.unique(support)
        arm_out["06_association_payload"] = {
            "support_references": len(support),
            "unique_points": len(assoc_idx),
            "unique_fraction_of_map": float(len(assoc_idx) / len(geo)),
            "spatial_observations": len(obs),
            "records_with_xyz_fields": sorted({k for o in obs for k in o if "coord" in k or "xyz" in k}),
        }
        arm_out["06_cloud_after_2d3d_association"] = cloud_stats(geo["map"][assoc_idx])

        # 07 após fusão: supports referenciam índices; centroide/bounds recalculados vs gravados
        sup = load_jsonl(run_dir(arm) / "semantic_fusion/outputs/fusion-supports.jsonl")
        worst = 0.0
        fused_idx = []
        for s in sup:
            idx = decode_deltas(s["geometry"]["deltas"])
            fused_idx.append(idx)
            xyz = geo["map"][idx]
            worst = max(
                worst,
                float(np.abs(xyz.mean(axis=0) - np.array(s["centroid_m"])).max()),
                float(np.abs(xyz.min(axis=0) - np.array(s["bounds"]["minimum_m"])).max()),
                float(np.abs(xyz.max(axis=0) - np.array(s["bounds"]["maximum_m"])).max()),
            )
        fused_all = np.unique(np.concatenate(fused_idx))
        arm_out["07_cloud_after_semantic_fusion"] = cloud_stats(geo["map"][fused_all])
        arm_out["07_support_summary_vs_recomputed_max_abs_m"] = worst
        arm_out["07_fused_points_subset_of_associated"] = bool(np.isin(fused_all, assoc_idx).all())

        # 07b entidades do semantic_mapping: geometria = referências + resumo derivado
        ents = load_jsonl(run_dir(arm) / "semantic_mapping/outputs/entities.jsonl")
        worst_e = 0.0
        for e in ents:
            idx = decode_deltas(e["geometry"]["deltas"])
            xyz = geo["map"][idx]
            worst_e = max(worst_e, float(np.abs(xyz.mean(axis=0) - np.array(e["geometry"]["centroid_m"])).max()))
        arm_out["07b_entity_centroid_vs_recomputed_max_abs_m"] = worst_e

        # 08 context_map: referência ao mapa (locator + hash) e geometry_refs por índice
        cm = run_dir(arm) / "context_map"
        cm_manifest = json.loads((cm / "manifest.json").read_text())
        geometry_ref = json.loads((cm / "geometry/geometry-reference.json").read_text())
        cm_ents = load_jsonl(cm / "entities/entities.jsonl")
        refs = np.array(
            sorted({geometry_index(g["geometry_id"]) for e in cm_ents for g in e["record"]["geometry_refs"]}),
            dtype=np.int64,
        )
        arm_out["08_context_map_geometry_reference"] = geometry_ref
        arm_out["08_context_map_dependency_hashes"] = [
            {k: d.get(k) for k in ("kind", "artifact_id", "content_hash", "locator")}
            for d in cm_manifest.get("dependencies", cm_manifest.get("lineage", []))
            if isinstance(d, dict)
        ][:10]
        arm_out["08_entity_points_in_context_map"] = cloud_stats(geo["map"][refs])
        arm_out["08_entity_points_subset_of_associated"] = bool(np.isin(refs, assoc_idx).all())
        # o que o viewer desenha opaco quando há cobertura de evidência: pontos com suporte
        arm_out["08_viewer_opaque_fraction_with_coverage"] = float(len(assoc_idx) / len(geo))
        out[arm] = arm_out
        print(arm, "done", flush=True)
    dump("p2_geometry_snapshots.json", out)


if __name__ == "__main__":
    main()
