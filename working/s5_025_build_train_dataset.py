"""
s5_025_build_train_dataset.py
全199データセット・全100フレームの全トラックから、
GTベース正解ラベル (is_gt) 付きの包括的学習用データセットを生成する。
出力: C:\work\aaa\s5\github\working\s5_025_track_training_data.parquet
"""
import sys
import os
import time
import datetime
from pathlib import Path

import zarr
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog
from joblib import Parallel, delayed

sys.stdout.reconfigure(encoding='utf-8')

SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
DATA_DIR = Path(r"C:\work\aaa\s5\input\train")
OUTPUT_PARQUET = Path(r"C:\work\aaa\s5\github\working\s5_025_track_training_data.parquet")

# 024/025 標準検出 & トラッキング パラメータ
BLOBDOG_MIN_SIGMA = (1.0, 2.0, 2.0)
BLOBDOG_MAX_SIGMA = (2.5, 6.0, 6.0)
BLOBDOG_SIGMA_RATIO = 1.6
BLOBDOG_TH_MIN = 0.035
BLOBDOG_TH_MAX = 0.068
BLOBDOG_TH_SLOPE = 0.0020
BLOBDOG_OVERLAP_DENSE = 0.50
BLOBDOG_OVERLAP_SPARSE = 0.35
D_MAX_MNN_UM = 6.0
D_MAX_DEAD_RECKONING_UM = 5.0
MATCH_THRESHOLD_UM = 7.0

def analyze_frame_optics(frame: np.ndarray) -> dict:
    sub = frame[::2, ::2, ::2]
    p01, p25, p50, p75, p995 = np.percentile(sub, [1.0, 25.0, 50.0, 75.0, 99.5])
    bg_noise = float((p75 - p25) / 1.349)
    snr_proxy = float((p995 - p50) / (bg_noise + 1e-5))
    fg_ratio = float((sub > (p50 + 3.0 * bg_noise)).mean())
    return {
        "p01": float(p01),
        "p995": float(p995),
        "bg_median": float(p50),
        "bg_noise": float(bg_noise),
        "snr_proxy": float(snr_proxy),
        "fg_ratio": float(fg_ratio),
    }

def process_single_dataset(geff_path: Path) -> list[dict]:
    ds_name = geff_path.stem
    zarr_path = DATA_DIR / f"{ds_name}.zarr"
    if not zarr_path.exists():
        return []

    zg_img = zarr.open(str(zarr_path), mode='r')['0']
    zg_geff = zarr.open(str(geff_path), mode='r')

    # GT nodes
    gt_t = zg_geff['nodes']['props']['t']['values'][:]
    gt_z = zg_geff['nodes']['props']['z']['values'][:]
    gt_y = zg_geff['nodes']['props']['y']['values'][:]
    gt_x = zg_geff['nodes']['props']['x']['values'][:]
    gt_coords = np.column_stack([gt_z, gt_y, gt_x]) * SCALE

    # 1. 021/024 Cell 8 異方性DoG検出 (全100フレーム)
    all_pred_nodes = []
    node_id_counter = 0
    n_frames = min(100, zg_img.shape[0])

    for t in range(n_frames):
        frame = zg_img[t]
        optics = analyze_frame_optics(frame)
        snr_proxy = optics["snr_proxy"]
        bg_median = optics["bg_median"]
        bg_noise = max(1e-3, optics["bg_noise"])

        is_triggered = (snr_proxy <= 12.0) or (bg_median >= 80.0)
        if not is_triggered:
            p_low = max(optics["p01"], optics["bg_median"] - 2.0 * optics["bg_noise"])
            p_high = optics["p995"]
            th = float(np.clip(BLOBDOG_TH_MIN + BLOBDOG_TH_SLOPE * (snr_proxy - 2.0), BLOBDOG_TH_MIN, BLOBDOG_TH_MAX))
            target = frame
        else:
            bg = gaussian_filter(frame, sigma=(4.0, 16.0, 16.0))
            vol_sub = np.maximum(0.0, frame.astype(np.float32, copy=False) - bg)
            p_low = np.percentile(vol_sub, 1.0)
            p_high = np.percentile(vol_sub, 99.8)
            th = 0.025
            target = vol_sub

        ov = BLOBDOG_OVERLAP_DENSE if optics["fg_ratio"] > 0.08 else BLOBDOG_OVERLAP_SPARSE
        if p_high > p_low:
            img_norm = np.clip((target.astype(np.float32, copy=False) - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0).astype(np.float32)
        else:
            img_norm = np.zeros_like(frame, dtype=np.float32)

        blobs = blob_dog(
            img_norm,
            min_sigma=BLOBDOG_MIN_SIGMA,
            max_sigma=BLOBDOG_MAX_SIGMA,
            sigma_ratio=BLOBDOG_SIGMA_RATIO,
            threshold=th,
            overlap=ov
        )

        z_max, y_max, x_max = frame.shape
        for b in blobs:
            bz, by, bx = float(b[0]), float(b[1]), float(b[2])
            iz = int(np.clip(round(bz), 0, z_max - 1))
            iy = int(np.clip(round(by), 0, y_max - 1))
            ix = int(np.clip(round(bx), 0, x_max - 1))
            raw_int = float(frame[iz, iy, ix])
            local_snr = float(max(0.0, (raw_int - bg_median) / bg_noise))
            dog_resp = float(img_norm[iz, iy, ix] - optics["bg_median"])

            # 簡易真球度/スケール
            if len(b) >= 6:
                sigma_z, sigma_y, sigma_x = float(b[3]), float(b[4]), float(b[5])
            else:
                sigma_z, sigma_y, sigma_x = 1.0, 2.0, 2.0
            sigma_xy = (sigma_y + sigma_x) / 2.0

            all_pred_nodes.append({
                'node_id': node_id_counter,
                't': t, 'z': bz, 'y': by, 'x': bx,
                'raw_intensity': raw_int,
                'local_snr': local_snr,
                'dog_response': dog_resp,
                'sigma_z': sigma_z,
                'sigma_xy': sigma_xy
            })
            node_id_counter += 1

    pred_df = pd.DataFrame(all_pred_nodes)
    if pred_df.empty:
        return []

    # 2. 022 MNN + Dead Reckoning トラッキング
    frames = sorted(pred_df['t'].unique())
    edges = []

    for i in range(len(frames) - 1):
        t1, t2 = frames[i], frames[i + 1]
        if t2 - t1 != 1:
            continue
        df1 = pred_df[pred_df['t'] == t1]
        df2 = pred_df[pred_df['t'] == t2]
        if df1.empty or df2.empty:
            continue

        p1 = df1[['z', 'y', 'x']].values * SCALE
        p2 = df2[['z', 'y', 'x']].values * SCALE
        ids1 = df1['node_id'].values
        ids2 = df2['node_id'].values

        tree2 = KDTree(p2)
        d_fwd, idx_fwd = tree2.query(p1, k=min(2, len(p2)))
        tree1 = KDTree(p1)
        d_bwd, idx_bwd = tree1.query(p2, k=min(2, len(p1)))

        if len(p2) == 1:
            d_fwd = np.column_stack([d_fwd, np.full(len(p1), 999.0)])
            idx_fwd = np.column_stack([idx_fwd, np.full(len(p1), -1)])
        if len(p1) == 1:
            d_bwd = np.column_stack([d_bwd, np.full(len(p2), 999.0)])
            idx_bwd = np.column_stack([idx_bwd, np.full(len(p2), -1)])

        for u_idx in range(len(p1)):
            best_v = idx_fwd[u_idx, 0]
            d_val = d_fwd[u_idx, 0]
            if d_val > D_MAX_MNN_UM:
                continue
            if idx_bwd[best_v, 0] == u_idx:
                edges.append({
                    'source_id': int(ids1[u_idx]),
                    'target_id': int(ids2[best_v]),
                    'dist': float(d_val),
                    'pass': 1
                })

    # Pass 2: Dead Reckoning (gap=2)
    pass1_df = pd.DataFrame(edges)
    if not pass1_df.empty:
        out_deg = set(pass1_df['source_id'].values)
        in_deg = set(pass1_df['target_id'].values)
        for i in range(len(frames) - 2):
            t1, t3 = frames[i], frames[i + 2]
            if t3 - t1 != 2:
                continue
            df1 = pred_df[(pred_df['t'] == t1) & (~pred_df['node_id'].isin(out_deg))]
            df3 = pred_df[(pred_df['t'] == t3) & (~pred_df['node_id'].isin(in_deg))]
            if df1.empty or df3.empty:
                continue

            p1 = df1[['z', 'y', 'x']].values * SCALE
            p3 = df3[['z', 'y', 'x']].values * SCALE
            ids1 = df1['node_id'].values
            ids3 = df3['node_id'].values

            tree3 = KDTree(p3)
            dists, idxs = tree3.query(p1, distance_upper_bound=D_MAX_DEAD_RECKONING_UM * 1.5)
            for u_idx, (d, v_idx) in enumerate(zip(dists, idxs)):
                if d <= D_MAX_DEAD_RECKONING_UM and v_idx < len(ids3):
                    src_id = int(ids1[u_idx])
                    tgt_id = int(ids3[v_idx])
                    if src_id not in out_deg and tgt_id not in in_deg:
                        edges.append({
                            'source_id': src_id,
                            'target_id': tgt_id,
                            'dist': float(d),
                            'pass': 2
                        })
                        out_deg.add(src_id)
                        in_deg.add(tgt_id)

    # 3. グラフ構築 & トラック特徴量抽出
    G = nx.Graph()
    G.add_nodes_from(pred_df['node_id'].values)
    for e in edges:
        G.add_edge(e['source_id'], e['target_id'])

    node_map = pred_df.set_index('node_id')
    comps = list(nx.connected_components(G))

    # GTマッチング木
    gt_tree_by_t = {}
    for t_val in np.unique(gt_t):
        t_mask = (gt_t == t_val)
        gt_tree_by_t[t_val] = (KDTree(gt_coords[t_mask]), gt_coords[t_mask])

    track_records = []
    for c in comps:
        c_len = len(c)
        sub_df = node_map.loc[list(c)].sort_values('t')
        sub_coords = sub_df[['z', 'y', 'x']].values * SCALE

        # GTマッチ判定 (厳密にGTノードと時空間突合 <= 7.0um)
        is_gt_track = False
        gt_matched_count = 0
        for _, row in sub_df.iterrows():
            t_val = int(row['t'])
            if t_val in gt_tree_by_t:
                tree_t, _ = gt_tree_by_t[t_val]
                p_um = np.array([row['z'], row['y'], row['x']]) * SCALE
                d_gt, _ = tree_t.query(p_um)
                if d_gt <= MATCH_THRESHOLD_UM:
                    gt_matched_count += 1

        # トラック内ノードの過半数または1点以上がGTと一致
        is_gt_track = (gt_matched_count >= max(1, c_len * 0.3))

        if c_len == 1:
            diffs = np.array([])
            mean_step = 0.0
            max_step = 0.0
            total_disp = 0.0
            path_len = 0.0
            straightness = 0.0
            net_velocity = 0.0
            confinement = 0.0
            step_vel_std = 0.0
            pca_aspect = 1.0
            directed_motion = 0.0
        else:
            diffs = np.linalg.norm(np.diff(sub_coords, axis=0), axis=1)
            mean_step = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
            max_step = float(np.max(diffs)) if len(diffs) > 0 else 0.0
            path_len = float(np.sum(diffs))
            total_disp = float(np.linalg.norm(sub_coords[-1] - sub_coords[0]))
            straightness = float(total_disp / (path_len + 1e-3))
            net_velocity = float(total_disp / max(1, c_len - 1))
            disp_from_start = np.linalg.norm(sub_coords - sub_coords[0], axis=1)
            confinement = float(np.max(disp_from_start) / (path_len + 1e-3))
            step_vel_std = float(np.std(diffs) / (mean_step + 1e-5)) if len(diffs) > 1 else 0.0

            if c_len >= 3:
                centered = sub_coords - np.mean(sub_coords, axis=0)
                cov = np.cov(centered, rowvar=False)
                eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                pca_aspect = float(eigvals[0] / (eigvals[1] + 1e-3)) if len(eigvals) >= 2 else 1.0
            else:
                pca_aspect = 1.0

            directed_motion = float(straightness * np.sqrt(c_len) * net_velocity)

        # 構成ノードの形態学集計値
        raw_int_mean = float(sub_df['raw_intensity'].mean())
        raw_int_min = float(sub_df['raw_intensity'].min())
        raw_int_max = float(sub_df['raw_intensity'].max())

        snr_mean = float(sub_df['local_snr'].mean())
        snr_min = float(sub_df['local_snr'].min())
        snr_max = float(sub_df['local_snr'].max())

        dog_resp_mean = float(sub_df['dog_response'].mean())
        sigma_z_mean = float(sub_df['sigma_z'].mean())
        sigma_xy_mean = float(sub_df['sigma_xy'].mean())

        track_records.append({
            'dataset': ds_name,
            'is_gt': int(is_gt_track),
            'track_length': c_len,
            'straightness_ratio': straightness,
            'net_velocity_um': net_velocity,
            'mean_step_um': mean_step,
            'max_step_um': max_step,
            'total_disp_um': total_disp,
            'path_length_um': path_len,
            'confinement_ratio': confinement,
            'trajectory_aspect_ratio': pca_aspect,
            'step_velocity_cv': step_vel_std,
            'directed_motion_index': directed_motion,
            'mean_raw_intensity': raw_int_mean,
            'min_raw_intensity': raw_int_min,
            'max_raw_intensity': raw_int_max,
            'mean_local_snr': snr_mean,
            'min_local_snr': snr_min,
            'max_local_snr': snr_max,
            'mean_dog_response': dog_resp_mean,
            'mean_sigma_z': sigma_z_mean,
            'mean_sigma_xy': sigma_xy_mean,
        })

    return track_records

def main():
    t0_all = time.time()
    print("=" * 75)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] s5_025_build_train_dataset: 全199データセットGT教師データ生成 開始")
    print("=" * 75)

    geff_files = sorted(list(DATA_DIR.glob("*.geff")))
    print(f"Found {len(geff_files)} geff files. Starting 16-parallel extraction...")

    results = Parallel(n_jobs=16, backend='loky', verbose=5)(
        delayed(process_single_dataset)(gf) for gf in geff_files
    )

    all_tracks = [item for sublist in results for item in sublist]
    df_all = pd.DataFrame(all_tracks)
    print(f"\nExtracted {len(df_all):,} tracks across 199 datasets!")
    print(f"  - Positive (GT-matched cells): {df_all['is_gt'].sum():,} ({df_all['is_gt'].mean()*100:.2f}%)")
    print(f"  - Negative (Noise tracks): {(~df_all['is_gt'].astype(bool)).sum():,} ({(1-df_all['is_gt'].mean())*100:.2f}%)")

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Successfully saved GT training dataset to {OUTPUT_PARQUET} ({os.path.getsize(OUTPUT_PARQUET)/1024/1024:.2f} MB)")
    print(f"Total time: {(time.time() - t0_all)/60:.2f} minutes")

if __name__ == "__main__":
    main()
