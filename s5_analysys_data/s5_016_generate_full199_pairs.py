# -*- coding: utf-8 -*-
"""
s5_016_generate_full199_pairs.py
全 199 データセットの 015DYNRADIUS 検出ノード (pred_nodes) と GT 正解データ (gt_nodes, gt_edges) から、
全 199 データセットの真の正例 (Label = 1) と、近傍の誤結合候補 (Hard Negatives, Label = 0) をサンプリングし、
23 次元のペア特徴量を算出して Parquet 形式で保存するスクリプト。
"""
import glob
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"

GT_NODES_PATH = WORKING_DIR / "s5_gt_nodes.csv"
GT_EDGES_PATH = WORKING_DIR / "s5_gt_edges.csv"
OUTPUT_PARQUET = DATA_DIR / "s5_016_train_pairs_all199.parquet"
OUTPUT_EVIDENCE_CSV = DATA_DIR / "s5_016_train_pairs_all199_evidence.csv"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
MAX_SEARCH_RADIUS_UM = 7.0
MAX_NEG_PER_SOURCE = 3  # 正例 1 に対し負例最大 3 件サンプリング (学習バランスとデータ容量の最適化)

print("=" * 80)
print(">>> s5_016: 全 199 データセット LightGBM 教師データ (ペア特徴量) 生成開始")
print("=" * 80)

t_start = time.time()

# 1. GT データのロード
print(f"[*] GT データロード中: {GT_NODES_PATH.name}, {GT_EDGES_PATH.name}")
df_gt_nodes_all = pd.read_csv(GT_NODES_PATH)
df_gt_edges_all = pd.read_csv(GT_EDGES_PATH)

datasets = sorted(df_gt_edges_all['dataset'].unique())
print(f"  - 全 GT データセット数: {len(datasets)} 件")
print(f"  - 全 GT ノード数: {len(df_gt_nodes_all):,} 件")
print(f"  - 全 GT エッジ数: {len(df_gt_edges_all):,} 本")

# 2. 015DYNRADIUS 検出ノードのロード
pred_files = sorted(glob.glob(str(WORKING_DIR / "s5_015DYNRADIUS_01_detect_nodes_pred_blobdog_lgbm_*.csv")))
print(f"[*] 015DYNRADIUS 検出ノードファイル読込中 ({len(pred_files)} 分割ファイル)...")
pred_list = [pd.read_csv(f) for f in pred_files]
df_pred_all = pd.concat(pred_list, ignore_index=True)
print(f"  - 全 199 データセット検出ノード総数: {len(df_pred_all):,} 件")

FEATURE_COLS_4D = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']

all_pair_records = []
ds_summary_records = []

for idx, ds in enumerate(datasets, 1):
    t_ds = time.time()
    gt_n_ds = df_gt_nodes_all[df_gt_nodes_all["dataset"] == ds]
    gt_e_ds = df_gt_edges_all[df_gt_edges_all["dataset"] == ds]
    pred_ds = df_pred_all[df_pred_all["dataset"] == ds].copy().reset_index(drop=True)

    if pred_ds.empty or gt_n_ds.empty or gt_e_ds.empty:
        print(f"[{idx}/{len(datasets)}] {ds}: データ不足のためスキップ")
        continue

    gt_edge_set = set(zip(gt_e_ds["source_id"].astype(int), gt_e_ds["target_id"].astype(int)))

    # (1) フレーム単位で GT と pred の空間突き合わせ (公式基準 max_distance = 7.0 um)
    pred_ds["gt_node_id"] = -1
    pred_ds["is_tp"] = False

    frames = sorted(pred_ds["t"].unique())
    n_tp_nodes = 0

    for t in frames:
        sub_p = pred_ds[pred_ds["t"] == t]
        sub_g = gt_n_ds[gt_n_ds["t"] == t]
        if sub_p.empty or sub_g.empty:
            continue

        pos_p = sub_p[["z", "y", "x"]].values * SCALE_VEC
        pos_g = sub_g[["z", "y", "x"]].values * SCALE_VEC

        dist_mat = cdist(pos_p, pos_g)
        row_ind, col_ind = linear_sum_assignment(dist_mat)

        p_indices = sub_p.index.values
        g_node_ids = sub_g["node_id"].values

        for r, c in zip(row_ind, col_ind):
            if dist_mat[r, c] <= MAX_SEARCH_RADIUS_UM:
                pred_ds.loc[p_indices[r], "gt_node_id"] = int(g_node_ids[c])
                pred_ds.loc[p_indices[r], "is_tp"] = True
                n_tp_nodes += 1

    gt_recall_node = n_tp_nodes / len(gt_n_ds) if len(gt_n_ds) > 0 else 0.0

    # 4D Mahalanobis の共分散行列準備
    avail_feats = [c for c in FEATURE_COLS_4D if c in pred_ds.columns]
    feats_all = pred_ds[avail_feats].values.astype(np.float32)
    cov_matrix = np.cov(feats_all, rowvar=False)
    inv_cov = np.linalg.pinv(cov_matrix)

    # (2) フレーム t -> t+1 のペア探索 & 特徴量算出
    pos_dict = {}
    for t, group in pred_ds.groupby("t"):
        pos_dict[t] = group[["z", "y", "x"]].values * SCALE_VEC

    ds_pos_count = 0
    ds_neg_count = 0

    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i+1]
        if t_next != t_curr + 1:
            continue

        df_c = pred_ds[pred_ds["t"] == t_curr]
        df_n = pred_ds[pred_ds["t"] == t_next]
        if df_c.empty or df_n.empty:
            continue

        p_curr = pos_dict[t_curr]
        p_next = pos_dict[t_next]

        spatial_dist = cdist(p_curr, p_next)
        mask = (spatial_dist <= MAX_SEARCH_RADIUS_UM)
        r_idx, c_idx = np.where(mask)
        if len(r_idx) == 0:
            continue

        fc = df_c[avail_feats].values.astype(np.float32)
        fn = df_n[avail_feats].values.astype(np.float32)
        try:
            feat_dist = cdist(fc, fn, metric="mahalanobis", VI=inv_cov)
        except Exception:
            feat_dist = cdist(fc, fn, metric="cityblock")

        sp_d = spatial_dist[r_idx, c_idx]
        mh_d = feat_dist[r_idx, c_idx]
        total_cost_4d = sp_d + mh_d

        dz = (df_n["z"].values[c_idx] - df_c["z"].values[r_idx]) * SCALE_VEC[0]
        dy = (df_n["y"].values[c_idx] - df_c["y"].values[r_idx]) * SCALE_VEC[1]
        dx = (df_n["x"].values[c_idx] - df_c["x"].values[r_idx]) * SCALE_VEC[2]
        d_xy = np.sqrt(dy**2 + dx**2)

        int1 = df_c["mean_intensity"].values[r_idx]
        int2 = df_n["mean_intensity"].values[c_idx]
        snr1 = df_c["snr"].values[r_idx]
        snr2 = df_n["snr"].values[c_idx]
        rad1 = df_c["estimated_radius_um"].values[r_idx]
        rad2 = df_n["estimated_radius_um"].values[c_idx]
        vol1 = df_c["volume_um3"].values[r_idx]
        vol2 = df_n["volume_um3"].values[c_idx]
        zdep1 = df_c["z_depth_ratio"].values[r_idx] if "z_depth_ratio" in df_c.columns else np.zeros(len(r_idx))
        zdep2 = df_n["z_depth_ratio"].values[c_idx] if "z_depth_ratio" in df_n.columns else np.zeros(len(c_idx))
        dens1 = df_c["local_density_r15"].values[r_idx] if "local_density_r15" in df_c.columns else np.zeros(len(r_idx))
        dens2 = df_n["local_density_r15"].values[c_idx] if "local_density_r15" in df_n.columns else np.zeros(len(c_idx))

        min_sp_dist = np.min(spatial_dist, axis=1)
        sp_margin = sp_d - min_sp_dist[r_idx]

        ranks = np.zeros(len(r_idx), dtype=np.int32)
        for r in np.unique(r_idx):
            match_k = np.where(r_idx == r)[0]
            sorted_k = match_k[np.argsort(sp_d[match_k])]
            ranks[sorted_k] = np.arange(1, len(sorted_k) + 1)

        src_gt = df_c["gt_node_id"].values[r_idx]
        tgt_gt = df_n["gt_node_id"].values[c_idx]
        src_tp = df_c["is_tp"].values[r_idx]
        tgt_tp = df_n["is_tp"].values[c_idx]

        labels = np.zeros(len(r_idx), dtype=np.int8)
        for k in range(len(r_idx)):
            if src_tp[k] and tgt_tp[k]:
                if (int(src_gt[k]), int(tgt_gt[k])) in gt_edge_set:
                    labels[k] = 1

        # 各 source ごとにサンプリング (正例はすべて保持、負例は最大 MAX_NEG_PER_SOURCE 件)
        sampled_indices = []
        for r in np.unique(r_idx):
            k_indices = np.where(r_idx == r)[0]
            pos_k = k_indices[labels[k_indices] == 1]
            neg_k = k_indices[labels[k_indices] == 0]

            sampled_indices.extend(pos_k)
            if len(neg_k) > 0:
                # 距離が近い難関負例 (Hard Negatives) を優先
                sorted_neg = neg_k[np.argsort(sp_d[neg_k])][:MAX_NEG_PER_SOURCE]
                sampled_indices.extend(sorted_neg)

        if not sampled_indices:
            continue

        sampled_indices = np.array(sampled_indices)
        ds_pos_count += int(np.sum(labels[sampled_indices] == 1))
        ds_neg_count += int(np.sum(labels[sampled_indices] == 0))

        chunk_df = pd.DataFrame({
            "dataset": ds,
            "t": t_curr,
            "source_id": df_c["node_id"].values[r_idx[sampled_indices]],
            "target_id": df_n["node_id"].values[c_idx[sampled_indices]],
            "label": labels[sampled_indices],
            "spatial_dist": sp_d[sampled_indices],
            "spatial_dist_xy": d_xy[sampled_indices],
            "delta_z_scaled": dz[sampled_indices],
            "delta_y_scaled": dy[sampled_indices],
            "delta_x_scaled": dx[sampled_indices],
            "abs_delta_z": np.abs(dz[sampled_indices]),
            "spatial_rank": ranks[sampled_indices],
            "spatial_margin": sp_margin[sampled_indices],
            "int_diff": np.abs(int1[sampled_indices] - int2[sampled_indices]),
            "int_ratio": int1[sampled_indices] / (int2[sampled_indices] + 1e-5),
            "snr_diff": np.abs(snr1[sampled_indices] - snr2[sampled_indices]),
            "snr_ratio": snr1[sampled_indices] / (snr2[sampled_indices] + 1e-5),
            "snr_min": np.minimum(snr1[sampled_indices], snr2[sampled_indices]),
            "radius_diff": np.abs(rad1[sampled_indices] - rad2[sampled_indices]),
            "radius_ratio": rad1[sampled_indices] / (rad2[sampled_indices] + 1e-5),
            "volume_diff": np.abs(vol1[sampled_indices] - vol2[sampled_indices]),
            "volume_ratio": vol1[sampled_indices] / (vol2[sampled_indices] + 1e-5),
            "z_depth_diff": np.abs(zdep1[sampled_indices] - zdep2[sampled_indices]),
            "density_source": dens1[sampled_indices],
            "density_target": dens2[sampled_indices],
            "density_diff": np.abs(dens1[sampled_indices] - dens2[sampled_indices]),
            "mahalanobis_dist": mh_d[sampled_indices],
            "total_cost_4d": total_cost_4d[sampled_indices],
        })
        all_pair_records.append(chunk_df)

    elapsed_ds = time.time() - t_ds
    print(f"[{idx:3d}/{len(datasets)}] {ds}: 正例 {ds_pos_count:,} 件, 負例 {ds_neg_count:,} 件 (Node Recall: {gt_recall_node*100:.1f}%) in {elapsed_ds:.1f}s")
    ds_summary_records.append({
        "dataset": ds,
        "gt_edges": len(gt_e_ds),
        "pairs_pos": ds_pos_count,
        "pairs_neg": ds_neg_count,
        "elapsed_sec": round(elapsed_ds, 2)
    })

# 3. 結合と Parquet 保存
print("\n[*] 全データフレーム結合中...")
df_all_pairs = pd.concat(all_pair_records, ignore_index=True)
print(f"  - 全ペア総行数: {len(df_all_pairs):,} 行")
print(f"  - 全正例数 (Label=1): {(df_all_pairs['label']==1).sum():,} 行")
print(f"  - 全負例数 (Label=0): {(df_all_pairs['label']==0).sum():,} 行")

print(f"[*] Parquet 保存中: {OUTPUT_PARQUET.name}")
df_all_pairs.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")
print(f"  - 保存完了: {OUTPUT_PARQUET.stat().st_size / (1024**2):.2f} MB")

df_sum = pd.DataFrame(ds_summary_records)
df_sum.to_csv(OUTPUT_EVIDENCE_CSV, index=False)
print(f"[*] エビデンス CSV 保存完了: {OUTPUT_EVIDENCE_CSV.name}")

print(f"\n総所要時間: {(time.time() - t_start)/60:.1f} 分")
print("=" * 80)
