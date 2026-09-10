# -*- coding: utf-8 -*-
"""
s5_002_generate_lgbm_pairs.py
代表 5 データセットの BlobDog 検出ノード (pred_nodes) と GT 正解データ (gt_nodes, gt_edges) を
Kaggle 公式仕様 (max_distance = 7.0 um) で空間照合 (TP/FP判定) し、
4D Mahalanobis 候補空間から「正例」と「騙されやすい難関負例 (Hard Negatives)」をサンプリングして
22次元のペア特徴量を算出し、Parquet 形式で保存するスクリプト。
"""
import glob
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# 1. パス環境の自動解決
BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development")
WORKING_DIR = BASE_DIR / "s5" / "github" / "working"
DATA_DIR = BASE_DIR / "s5" / "github" / "s5_analysys_data"

TARGET_5_DATASETS = [
    "44b6_0113de3b",  # 密・標準
    "44b6_0b24845f",  # 密・難関 (超低コントラスト)
    "44b6_74d0c52e",  # 密・組織深部
    "6bba_05b6850b",  # 疎・標準
    "6bba_085bf656",  # 疎・高コントラスト
]

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
MAX_SEARCH_RADIUS_UM = 7.0
MAX_NEG_PER_SOURCE = 4  # 1つの source ノードあたりの最大負例採用数 (不均衡比率 1:3〜1:5 調整)

print("=" * 80)
print(">>> s5_002: LightGBM 教師データ (細胞ペア特徴量) 生成開始")
print(f"[*] 対象データセット: {TARGET_5_DATASETS}")
print(f"[*] 探索半径: {MAX_SEARCH_RADIUS_UM} um, 物理スケール: (1.625, 0.40625, 0.40625)")
print("=" * 80)

t_start = time.time()

# 2. GT データのロード
gt_nodes_path = WORKING_DIR / "s3_gt_nodes.csv"
gt_edges_path = WORKING_DIR / "s3_gt_edges.csv"

print(f"[*] GT データロード中: {gt_nodes_path.name}, {gt_edges_path.name}")
df_gt_nodes_all = pd.read_csv(gt_nodes_path)
df_gt_edges_all = pd.read_csv(gt_edges_path)

df_gt_nodes_5ds = df_gt_nodes_all[df_gt_nodes_all["dataset"].isin(TARGET_5_DATASETS)].copy()
df_gt_edges_5ds = df_gt_edges_all[df_gt_edges_all["dataset"].isin(TARGET_5_DATASETS)].copy()
print(f"  - 代表 5 データセット GT ノード数: {len(df_gt_nodes_5ds):,} 件")
print(f"  - 代表 5 データセット GT エッジ数: {len(df_gt_edges_5ds):,} 本")

# 3. BlobDog 検出ノード (pred_nodes) のロード
pred_files = sorted(glob.glob(str(WORKING_DIR / "s3_01_detect_nodes_pred_blobdog_5dmahalanobis_*.csv")))
pred_list = []
for f in pred_files:
    df_chunk = pd.read_csv(f)
    sub = df_chunk[df_chunk["dataset"].isin(TARGET_5_DATASETS)]
    if not sub.empty:
        pred_list.append(sub)

df_pred_5ds = pd.concat(pred_list, ignore_index=True)
print(f"  - 代表 5 データセット BlobDog 検出ノード数: {len(df_pred_5ds):,} 件")

# 4. データセットごとに TP/FP マッチング ＆ ペア特徴量生成
all_pair_records = []
ds_summary_records = []

FEATURE_COLS_4D = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']

for ds in TARGET_5_DATASETS:
    t_ds = time.time()
    print(f"\n--- [{ds}] 処理開始 ---")
    
    gt_n_ds = df_gt_nodes_5ds[df_gt_nodes_5ds["dataset"] == ds]
    gt_e_ds = df_gt_edges_5ds[df_gt_edges_5ds["dataset"] == ds]
    pred_ds = df_pred_5ds[df_pred_5ds["dataset"] == ds].copy().reset_index(drop=True)
    
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
    print(f"  - GT ノードマッチング: {n_tp_nodes:,} / {len(gt_n_ds):,} 件 (Recall: {gt_recall_node*100:.1f}%)")
    
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
        
        # 4D Mahalanobis 特徴量距離
        fc = df_c[avail_feats].values.astype(np.float32)
        fn = df_n[avail_feats].values.astype(np.float32)
        try:
            feat_dist = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)
        except Exception:
            feat_dist = cdist(fc, fn, metric='cityblock')
            
        total_cost_4d = spatial_dist + feat_dist
        
        # 候補ペアの抽出
        curr_gt = df_c["gt_node_id"].values
        next_gt = df_n["gt_node_id"].values
        c_node_ids = df_c["node_id"].values
        n_node_ids = df_n["node_id"].values
        
        # 各 source ノード r について
        for r in range(len(df_c)):
            r_sp_dist = spatial_dist[r]
            cand_mask = (r_sp_dist <= MAX_SEARCH_RADIUS_UM)
            cand_cols = np.where(cand_mask)[0]
            if len(cand_cols) == 0:
                continue
                
            # 距離順位ソート
            sorted_cand_cols = cand_cols[np.argsort(r_sp_dist[cand_cols])]
            min_sp_dist = r_sp_dist[sorted_cand_cols[0]]
            
            # 正例ペアと負例ペアの分類
            pos_cols = []
            neg_cols = []
            
            g1 = curr_gt[r]
            for rank_idx, c in enumerate(sorted_cand_cols):
                g2 = next_gt[c]
                is_positive = (g1 != -1 and g2 != -1 and (g1, g2) in gt_edge_set)
                if is_positive:
                    pos_cols.append((c, rank_idx + 1))
                else:
                    neg_cols.append((c, rank_idx + 1))
                    
            # サンプリング: 正例は全件、負例は上位 K 件
            selected_pairs = []
            for c, rank in pos_cols:
                selected_pairs.append((c, rank, 1))
            for c, rank in neg_cols[:MAX_NEG_PER_SOURCE]:
                selected_pairs.append((c, rank, 0))
                
            # 特徴量レコードの作成
            for c, rank, label in selected_pairs:
                if label == 1:
                    ds_pos_count += 1
                else:
                    ds_neg_count += 1
                    
                sp_d = float(spatial_dist[r, c])
                sp_margin = float(sp_d - min_sp_dist)
                mh_d = float(feat_dist[r, c])
                cost_4d = float(total_cost_4d[r, c])
                
                # 空間座標の差分
                dz = float((df_n.iloc[c]['z'] - df_c.iloc[r]['z']) * SCALE_VEC[0])
                dy = float((df_n.iloc[c]['y'] - df_c.iloc[r]['y']) * SCALE_VEC[1])
                dx = float((df_n.iloc[c]['x'] - df_c.iloc[r]['x']) * SCALE_VEC[2])
                d_xy = float(np.sqrt(dy**2 + dx**2))
                
                # 特徴量の差分・比率
                int1 = float(df_c.iloc[r]['mean_intensity'])
                int2 = float(df_n.iloc[c]['mean_intensity'])
                snr1 = float(df_c.iloc[r]['snr'])
                snr2 = float(df_n.iloc[c]['snr'])
                rad1 = float(df_c.iloc[r]['estimated_radius_um'])
                rad2 = float(df_n.iloc[c]['estimated_radius_um'])
                vol1 = float(df_c.iloc[r]['volume_um3'])
                vol2 = float(df_n.iloc[c]['volume_um3'])
                zdep1 = float(df_c.iloc[r]['z_depth_ratio'])
                zdep2 = float(df_n.iloc[c]['z_depth_ratio'])
                dens1 = int(df_c.iloc[r]['local_density_r15'])
                dens2 = int(df_n.iloc[c]['local_density_r15'])
                
                record = {
                    "dataset": ds,
                    "t": int(t_curr),
                    "source_id": int(c_node_ids[r]),
                    "target_id": int(n_node_ids[c]),
                    "source_is_tp": bool(df_c.iloc[r]['is_tp']),
                    "target_is_tp": bool(df_n.iloc[c]['is_tp']),
                    "spatial_dist": sp_d,
                    "spatial_dist_xy": d_xy,
                    "delta_z_scaled": dz,
                    "delta_y_scaled": dy,
                    "delta_x_scaled": dx,
                    "abs_delta_z": abs(dz),
                    "spatial_rank": int(rank),
                    "spatial_margin": sp_margin,
                    "int_diff": abs(int1 - int2),
                    "int_ratio": int1 / (int2 + 1e-5),
                    "snr_diff": abs(snr1 - snr2),
                    "snr_ratio": snr1 / (snr2 + 1e-5),
                    "snr_min": min(snr1, snr2),
                    "radius_diff": abs(rad1 - rad2),
                    "radius_ratio": rad1 / (rad2 + 1e-5),
                    "volume_diff": abs(vol1 - vol2),
                    "volume_ratio": vol1 / (vol2 + 1e-5),
                    "z_depth_diff": abs(zdep1 - zdep2),
                    "density_source": dens1,
                    "density_target": dens2,
                    "density_diff": abs(dens1 - dens2),
                    "mahalanobis_dist": mh_d,
                    "total_cost_4d": cost_4d,
                    "label": int(label),
                }
                all_pair_records.append(record)
                
    elapsed_ds = time.time() - t_ds
    print(f"  - ペア生成完了: 正例={ds_pos_count:,} 件, 負例={ds_neg_count:,} 件 (比率 1:{ds_neg_count/(ds_pos_count+1e-5):.1f}) [{elapsed_ds:.2f}秒]")
    ds_summary_records.append({
        "dataset": ds,
        "pred_nodes": len(pred_ds),
        "gt_nodes": len(gt_n_ds),
        "gt_edges": len(gt_e_ds),
        "tp_nodes_matched": n_tp_nodes,
        "node_recall": round(gt_recall_node * 100, 2),
        "pairs_pos": ds_pos_count,
        "pairs_neg": ds_neg_count,
        "neg_to_pos_ratio": round(ds_neg_count / (ds_pos_count + 1e-5), 2),
        "total_pairs": ds_pos_count + ds_neg_count,
        "elapsed_sec": round(elapsed_ds, 2),
    })

# 5. DataFrame 化 & 保存
df_pairs = pd.DataFrame(all_pair_records)
print("\n" + "=" * 80)
print(f">>> 全 5 データセット ペアデータ生成完了: 総行数 {len(df_pairs):,} 行")
print(f"  - 正例 (Label = 1): {(df_pairs['label'] == 1).sum():,} 件 ({(df_pairs['label'] == 1).mean()*100:.2f}%)")
print(f"  - 負例 (Label = 0): {(df_pairs['label'] == 0).sum():,} 件 ({(df_pairs['label'] == 0).mean()*100:.2f}%)")
print(f"  - 全体負例比率: 1 : {((df_pairs['label'] == 0).sum() / (df_pairs['label'] == 1).sum()):.2f}")
print("=" * 80)

# Parquet で保存
parquet_path = DATA_DIR / "s5_002_train_pairs_5datasets.parquet"
df_pairs.to_parquet(parquet_path, index=False, engine="pyarrow")
print(f"[*] Parquet 保存完了: {parquet_path.name} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")

# 検証用サンプル CSV 保存 (先頭 3,000 行)
sample_csv_path = DATA_DIR / "s5_002_train_pairs_sample.csv"
df_pairs.head(3000).to_csv(sample_csv_path, index=False, encoding="utf-8")
print(f"[*] サンプル CSV 保存完了: {sample_csv_path.name}")

# エビデンスサマリー CSV 保存
df_summary = pd.DataFrame(ds_summary_records)
evidence_path = DATA_DIR / "s5_002_train_pairs_evidence.csv"
df_summary.to_csv(evidence_path, index=False, encoding="utf-8")
print(f"[*] 定量エビデンス保存完了: {evidence_path.name}")
print(df_summary.to_string(index=False))

total_time = time.time() - t_start
print(f"\n[*] 全処理完走時間: {total_time:.2f} 秒")
