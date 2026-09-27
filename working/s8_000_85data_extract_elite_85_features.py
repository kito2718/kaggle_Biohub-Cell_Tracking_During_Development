import time
import os
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from scipy.spatial import cKDTree
import scipy.ndimage as ndi

sys.stdout.reconfigure(encoding="utf-8")

print("=== STARTING s8_000_85data_extract_elite_85_features.py ===")

VOXEL_SCALE = (1.625, 0.40625, 0.40625)  # (dz, dy, dx) in um
DATASETS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]

TRAIN_DIR = Path(r"c:\work\aaa\s6\input\train")
PRED_GEFF_DIR = Path(r"c:\work\aaa\output_058b\tracking_repo\predictions\unknown\unet_transformer\split_0")
OUT_DIR = Path(r"c:\work\aaa\s8_000_85data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT_DIR / "s8_000_85data_elite_85_features.csv"
PARQUET_OUT = OUT_DIR / "s8_000_85data_elite_85_features.parquet"

all_rows = []

for ds_idx, dataset in enumerate(DATASETS):
    t_start = time.time()
    print(f"\n[{ds_idx+1}/{len(DATASETS)}] Processing dataset: {dataset} ...")
    
    # 1. Load Ground Truth
    gt_geff_path = TRAIN_DIR / f"{dataset}.geff"
    gt_zarr = zarr.open(str(gt_geff_path), mode="r")
    gt_t = np.array(gt_zarr["nodes/props/t/values"][:], dtype=np.int32)
    gt_z = np.array(gt_zarr["nodes/props/z/values"][:], dtype=np.float32)
    gt_y = np.array(gt_zarr["nodes/props/y/values"][:], dtype=np.float32)
    gt_x = np.array(gt_zarr["nodes/props/x/values"][:], dtype=np.float32)
    gt_ids = np.array(gt_zarr["nodes/ids"][:], dtype=np.int64)
    print(f"  Loaded GT: {len(gt_ids)} nodes")
    
    # 2. Load Predicted Candidates & Graph
    pred_geff_path = PRED_GEFF_DIR / f"{dataset}.geff"
    p_zarr = zarr.open(str(pred_geff_path), mode="r")
    p_ids = np.array(p_zarr["nodes/ids"][:], dtype=np.int64)
    p_t = np.array(p_zarr["nodes/props/t/values"][:], dtype=np.int32)
    p_z = np.array(p_zarr["nodes/props/z/values"][:], dtype=np.float32)
    p_y = np.array(p_zarr["nodes/props/y/values"][:], dtype=np.float32)
    p_x = np.array(p_zarr["nodes/props/x/values"][:], dtype=np.float32)
    p_sol = np.array(p_zarr["nodes/props/solution/values"][:], dtype=np.int32)
    
    e_src = np.array(p_zarr["edges/ids"][:, 0], dtype=np.int64)
    e_tgt = np.array(p_zarr["edges/ids"][:, 1], dtype=np.int64)
    e_prob = np.array(p_zarr["edges/props/edge_prob/values"][:], dtype=np.float32)
    e_dist = np.array(p_zarr["edges/props/edge_dist/values"][:], dtype=np.float32)
    e_sol = np.array(p_zarr["edges/props/solution/values"][:], dtype=np.int32)
    print(f"  Loaded Candidates: {len(p_ids)} nodes, {len(e_src)} edges")
    
    # Pre-index edges by node
    node_to_idx = {nid: i for i, nid in enumerate(p_ids)}
    in_edges = {i: [] for i in range(len(p_ids))}
    out_edges = {i: [] for i in range(len(p_ids))}
    for ei in range(len(e_src)):
        s_idx = node_to_idx.get(e_src[ei])
        t_idx = node_to_idx.get(e_tgt[ei])
        if s_idx is not None:
            out_edges[s_idx].append(ei)
        if t_idx is not None:
            in_edges[t_idx].append(ei)
            
    # 3. Load Raw Image Zarr
    raw_zarr_path = TRAIN_DIR / f"{dataset}.zarr"
    raw_z = zarr.open(str(raw_zarr_path), mode="r")["0"]
    Z_dim, Y_dim, X_dim = raw_z.shape[1], raw_z.shape[2], raw_z.shape[3]
    
    # Global frame candidate counts
    unique_t, counts_t = np.unique(p_t, return_counts=True)
    frame_cand_count = dict(zip(unique_t, counts_t))
    total_dataset_candidates = len(p_ids)
    
    # Candidate indices by frame
    frame_cands = {}
    for i in range(len(p_ids)):
        frame_cands.setdefault(p_t[i], []).append(i)
        
    # GT indices by frame
    gt_frame_indices = {}
    for gi in range(len(gt_ids)):
        gt_frame_indices.setdefault(gt_t[gi], []).append(gi)
        
    # Process frame by frame
    for t_cur in range(100):
        c_indices = frame_cands.get(t_cur, [])
        if not c_indices:
            continue
            
        c_indices = np.array(c_indices, dtype=np.int32)
        n_cands_in_f = len(c_indices)
        
        # Candidate coords for frame
        cz = p_z[c_indices]
        cy = p_y[c_indices]
        cx = p_x[c_indices]
        c_phys = np.stack([cz * VOXEL_SCALE[0], cy * VOXEL_SCALE[1], cx * VOXEL_SCALE[2]], axis=1)
        
        # Load raw images for t_cur, t_prev, t_next
        img_cur = raw_z[t_cur].astype(np.float32)
        img_prev = raw_z[max(0, t_cur - 1)].astype(np.float32)
        img_next = raw_z[min(99, t_cur + 1)].astype(np.float32)
        
        # Frame image global stats
        f_median = float(np.median(img_cur))
        f_std = float(np.std(img_cur))
        
        # Compute 3D image filters
        mean_r1 = ndi.uniform_filter(img_cur, size=3)
        mean_r3 = ndi.uniform_filter(img_cur, size=5)
        
        mean_sq_r1 = ndi.uniform_filter(img_cur**2, size=3)
        std_r1 = np.sqrt(np.maximum(0.0, mean_sq_r1 - mean_r1**2))
        
        mean_sq_r3 = ndi.uniform_filter(img_cur**2, size=5)
        std_r3 = np.sqrt(np.maximum(0.0, mean_sq_r3 - mean_r3**2))
        
        max_r3 = ndi.maximum_filter(img_cur, size=5)
        min_r3 = ndi.minimum_filter(img_cur, size=5)
        
        mean_r7 = ndi.uniform_filter(img_cur, size=7)
        mean_sq_r7 = ndi.uniform_filter(img_cur**2, size=7)
        std_r7 = np.sqrt(np.maximum(0.0, mean_sq_r7 - mean_r7**2))
        
        gz, gy, gx = np.gradient(img_cur)
        grad_mag_3d = np.sqrt(gz**2 + gy**2 + gx**2)
        gzz, _, _ = np.gradient(gz)
        _, gyy, _ = np.gradient(gy)
        _, _, gxx = np.gradient(gx)
        hessian_trace = gzz + gyy + gxx
        
        iz = np.clip(np.round(cz).astype(np.int32), 0, Z_dim - 1)
        iy = np.clip(np.round(cy).astype(np.int32), 0, Y_dim - 1)
        ix = np.clip(np.round(cx).astype(np.int32), 0, X_dim - 1)
        
        tree_cur = cKDTree(c_phys)
        k_query = min(6, len(c_phys))
        dists_k, idxs_k = tree_cur.query(c_phys, k=k_query)
        
        prev_cands = frame_cands.get(t_cur - 1, [])
        if prev_cands and t_cur > 0:
            prev_phys = np.stack([p_z[prev_cands] * VOXEL_SCALE[0], p_y[prev_cands] * VOXEL_SCALE[1], p_x[prev_cands] * VOXEL_SCALE[2]], axis=1)
            tree_prev = cKDTree(prev_phys)
            dists_prev, _ = tree_prev.query(c_phys, k=1)
        else:
            dists_prev = np.full(n_cands_in_f, 999.0, dtype=np.float32)
            
        next_cands = frame_cands.get(t_cur + 1, [])
        if next_cands and t_cur < 99:
            next_phys = np.stack([p_z[next_cands] * VOXEL_SCALE[0], p_y[next_cands] * VOXEL_SCALE[1], p_x[next_cands] * VOXEL_SCALE[2]], axis=1)
            tree_next = cKDTree(next_phys)
            dists_next, _ = tree_next.query(c_phys, k=1)
        else:
            dists_next = np.full(n_cands_in_f, 999.0, dtype=np.float32)
            
        gt_in_f = gt_frame_indices.get(t_cur, [])
        gt_match_label = np.zeros(n_cands_in_f, dtype=np.int32)
        gt_match_id = np.full(n_cands_in_f, -1, dtype=np.int64)
        gt_match_dist = np.full(n_cands_in_f, 999.0, dtype=np.float32)
        
        if gt_in_f:
            gt_phys = np.stack([gt_z[gt_in_f] * VOXEL_SCALE[0], gt_y[gt_in_f] * VOXEL_SCALE[1], gt_x[gt_in_f] * VOXEL_SCALE[2]], axis=1)
            gt_tree = cKDTree(gt_phys)
            d_gt, idx_gt = gt_tree.query(c_phys, k=1)
            for ci in range(n_cands_in_f):
                if d_gt[ci] <= 5.0:  # 5.0 um matching threshold
                    gt_match_label[ci] = 1
                    gt_match_id[ci] = gt_ids[gt_in_f[idx_gt[ci]]]
                    gt_match_dist[ci] = d_gt[ci]
                    
        for k in range(n_cands_in_f):
            idx_orig = c_indices[k]
            z_c, y_c, x_c = cz[k], cy[k], cx[k]
            vz, vy, vx = iz[k], iy[k], ix[k]
            
            # Group 1: Spatial & Boundary Geometry (11)
            f01 = float(z_c)
            f02 = float(z_c / max(1.0, Z_dim - 1))
            f03 = float(min(z_c, Z_dim - 1 - z_c))
            f04 = float(min(x_c, y_c, X_dim - 1 - x_c, Y_dim - 1 - y_c))
            f05 = float(min(f03, f04))
            f06 = float(np.sqrt(((x_c - X_dim/2)*VOXEL_SCALE[2])**2 + ((y_c - Y_dim/2)*VOXEL_SCALE[1])**2))
            f07 = float(np.sqrt(((z_c - Z_dim/2)*VOXEL_SCALE[0])**2 + ((y_c - Y_dim/2)*VOXEL_SCALE[1])**2 + ((x_c - X_dim/2)*VOXEL_SCALE[2])**2))
            f08 = 1.0 if z_c <= 5.0 else 0.0
            f09 = 1.0 if z_c >= Z_dim - 6.0 else 0.0
            f10 = float(z_c * VOXEL_SCALE[0])
            f11 = float(z_c / max(1.0, np.max(cz)))
            
            # Group 2: Model & Confidence & Solution (13)
            f12 = float(p_sol[idx_orig])
            ins = in_edges[idx_orig]
            outs = out_edges[idx_orig]
            all_incident = ins + outs
            f13 = 1.0 if len(all_incident) > 0 else 0.0
            f14 = float(np.max(e_prob[all_incident])) if all_incident else 0.0
            f15 = float(np.mean(e_prob[all_incident])) if all_incident else 0.0
            f16 = float(len(ins))
            f17 = float(len(outs))
            f18 = float(len(all_incident))
            f19 = 1.0 if f18 == 0 else 0.0
            f20 = float(f18 + 1)
            f21 = float(np.min(e_dist[all_incident])) if all_incident else 999.0
            f22 = float(np.mean(e_dist[all_incident])) if all_incident else 999.0
            f23 = 1.0 if len(outs) >= 2 else 0.0
            f24 = 1.0 if len(ins) >= 2 else 0.0
            
            # Group 3: Local 3D Raw Intensity & Contrast (14)
            f25 = float(img_cur[vz, vy, vx])
            f26 = float(mean_r1[vz, vy, vx])
            f27 = float(std_r1[vz, vy, vx])
            f28 = float(mean_r3[vz, vy, vx])
            f29 = float(std_r3[vz, vy, vx])
            f30 = float(max_r3[vz, vy, vx])
            f31 = float(min_r3[vz, vy, vx])
            f32 = float(f30 - f31)
            f33 = float(mean_r7[vz, vy, vx])
            f34 = float(std_r7[vz, vy, vx])
            f35 = float((f25 - f33) / (f34 + 1.0))
            f36 = float((f25 - f33) / (f33 + 1.0))
            f37 = float((f30 - f31) / (f30 + f31 + 1.0))
            f38 = float(grad_mag_3d[vz, vy, vx])
            
            # Group 4: 3D Morphology & Curvature (16)
            f39 = float(hessian_trace[vz, vy, vx])
            f40 = float(gzz[vz, vy, vx] * gyy[vz, vy, vx] * gxx[vz, vy, vx])
            e1 = float(gzz[vz, vy, vx])
            e2 = float(gyy[vz, vy, vx])
            e3 = float(gxx[vz, vy, vx])
            e_sorted = sorted([e1, e2, e3], key=abs, reverse=True)
            f41, f42, f43 = e_sorted[0], e_sorted[1], e_sorted[2]
            f44 = float(-f39)  # LoG response
            f45 = float((f43**2) / (abs(f41 * f42) + 1e-4))  # blobness
            f46 = float(1.0 - abs(f42) / (abs(f41) + 1e-4))  # elongation
            f47 = float(abs(f42 - f43) / (abs(f41) + 1e-4))  # flatness
            f48 = float(abs(f43) / (abs(f41) + 1e-4))  # sphericity
            f49 = float(f25 - f26)  # local prominence
            f50 = float(f25 / (f31 + 1.0))  # drop ratio
            f51 = float(abs(e2 + e3) / (abs(e1) + 1e-4))
            f52 = float(abs(e2) / (abs(e3) + 1e-4))
            f53 = float(f25 / (f30 + 1e-4))
            f54 = float(f26 / (f30 + 1e-4))
            
            # Group 5: Local Spatial Density & Topology (14)
            d_k = dists_k[k] if len(c_phys) > 1 else [0.0]
            f55 = float(d_k[1]) if len(d_k) > 1 else 999.0
            f56 = float(d_k[2]) if len(d_k) > 2 else 999.0
            f57 = float(d_k[3]) if len(d_k) > 3 else 999.0
            f58 = float(d_k[min(5, len(d_k)-1)]) if len(d_k) > 1 else 999.0
            f59 = float(np.mean(d_k[1:])) if len(d_k) > 1 else 999.0
            f60 = float(np.sum(d_k[1:] <= 5.0)) if len(d_k) > 1 else 0.0
            f61 = float(np.sum(d_k[1:] <= 10.0)) if len(d_k) > 1 else 0.0
            f62 = float((f60 + 1.0) / (f61 + 1.0))
            f63 = 1.0 if (2.5 <= f55 <= 4.5) else 0.0
            f64 = float((4.0/3.0) * np.pi * ((f55/2.0)**3))
            
            if len(d_k) > 1:
                idx_nn = idxs_k[k][1]
                dz_nn = float((cz[idx_nn] - z_c) * VOXEL_SCALE[0])
                dxy_nn = float(np.sqrt(((cx[idx_nn] - x_c)*VOXEL_SCALE[2])**2 + ((cy[idx_nn] - y_c)*VOXEL_SCALE[1])**2))
                f65 = abs(dz_nn)
                f66 = dxy_nn
                f67 = float(np.arctan2(dxy_nn, abs(dz_nn) + 1e-4))
                f68 = float(abs(f25 - img_cur[iz[idx_nn], iy[idx_nn], ix[idx_nn]]))
            else:
                f65, f66, f67, f68 = 999.0, 999.0, 0.0, 0.0
                
            # Group 6: Temporal Dynamics & Continuity (12)
            f69 = float(t_cur)
            f70 = float(t_cur / 99.0)
            f71 = float(dists_prev[k])
            f72 = float(dists_next[k])
            f73 = float(min(f71, f72))
            f74 = float(abs(f71 - f72) / (max(f71, f72) + 1e-4))
            f75 = 1.0 if f71 > 10.0 else 0.0
            f76 = 1.0 if f72 > 10.0 else 0.0
            f77 = float(abs(f71 - f72))
            f78 = float(abs(f25 - img_prev[vz, vy, vx]))
            f79 = float(abs(f25 - img_next[vz, vy, vx]))
            f80 = float(f25 / (f78 + f79 + 1.0))
            
            # Group 7: Global Context & Development Stage (5)
            f81 = float(n_cands_in_f)
            f82 = float(total_dataset_candidates)
            f83 = f_median
            f84 = f_std
            f85 = float(total_dataset_candidates / 100.0)
            
            row = [
                dataset, int(p_ids[idx_orig]), int(t_cur), float(z_c), float(y_c), float(x_c),
                f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11,
                f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
                f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38,
                f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54,
                f55, f56, f57, f58, f59, f60, f61, f62, f63, f64, f65, f66, f67, f68,
                f69, f70, f71, f72, f73, f74, f75, f76, f77, f78, f79, f80,
                f81, f82, f83, f84, f85,
                int(gt_match_label[k]), int(gt_match_id[k]), float(gt_match_dist[k])
            ]
            all_rows.append(row)
            
    print(f"  Completed {dataset} in {time.time()-t_start:.1f}s | Cumulative rows: {len(all_rows)}")
    gc.collect()

col_names = [
    "dataset", "node_id", "t", "z", "y", "x",
    "pos_z", "pos_z_norm", "dist_z_boundary", "dist_xy_boundary", "dist_3d_boundary",
    "radial_dist_xy_um", "radial_dist_3d_um", "is_shallow_z", "is_deep_z", "optical_path_um", "norm_volume_depth",
    "solution_flag", "has_incident_edge", "max_incident_edge_prob", "mean_incident_edge_prob",
    "degree_in", "degree_out", "degree_total", "is_isolated", "track_len_est",
    "min_incident_edge_dist", "mean_incident_edge_dist", "division_source_flag", "division_target_flag",
    "intensity_center", "intensity_mean_r1", "intensity_std_r1", "intensity_mean_r3", "intensity_std_r3",
    "intensity_max_r3", "intensity_min_r3", "intensity_dynamic_range", "intensity_bg_shell_mean", "intensity_bg_shell_std",
    "snr_local", "contrast_weber", "contrast_michelson", "grad_mag_3d",
    "hessian_trace", "hessian_det", "hessian_eig1", "hessian_eig2", "hessian_eig3",
    "log_response", "blobness", "elongation", "flatness", "sphericity",
    "peak_prominence", "peak_drop_ratio", "moment_ratio_xy_z", "moment_ratio_x_y", "conn_vol_ratio_90", "conn_vol_ratio_75",
    "dist_knn_1_um", "dist_knn_2_um", "dist_knn_3_um", "dist_knn_5_um", "mean_dist_k5_um",
    "count_radius_5um", "count_radius_10um", "density_ratio_5_10", "is_doublet", "voronoi_vol_est",
    "knn1_dz_um", "knn1_dxy_um", "knn1_angle_z", "knn1_intensity_diff",
    "time_idx", "time_norm", "dist_prev_min_um", "dist_next_min_um", "bidirectional_min_disp_um", "bidirectional_disp_ratio",
    "is_birth_frame", "is_death_frame", "temporal_disp_consistency", "intensity_diff_prev_voxel", "intensity_diff_next_voxel", "temporal_contrast_ratio",
    "global_candidates_frame", "global_candidates_dataset", "global_intensity_median", "global_intensity_std", "stage_density_proxy",
    "label_is_gt", "label_gt_node_id", "label_gt_match_dist_um"
]

print(f"\nBuilding DataFrame with {len(all_rows)} rows and {len(col_names)} columns ...")
df = pd.DataFrame(all_rows, columns=col_names)

print(f"Saving Parquet to: {PARQUET_OUT} ...")
df.to_parquet(PARQUET_OUT, index=False)

print(f"Saving CSV to: {CSV_OUT} ...")
df.to_csv(CSV_OUT, index=False)

gt_pos_count = int((df["label_is_gt"] == 1).sum())
gt_neg_count = int((df["label_is_gt"] == 0).sum())
print("\n=== STEP 1 EXTRACTION COMPLETED SUCCESSFULLY! ===")
print(f"Total Rows: {len(df):,d}")
print(f"True Positive GT Nodes: {gt_pos_count:,d}")
print(f"False Positive Candidate Nodes: {gt_neg_count:,d}")
print(f"CSV Size: {CSV_OUT.stat().st_size / 1e6:.2f} MB")
print(f"Parquet Size: {PARQUET_OUT.stat().st_size / 1e6:.2f} MB")
