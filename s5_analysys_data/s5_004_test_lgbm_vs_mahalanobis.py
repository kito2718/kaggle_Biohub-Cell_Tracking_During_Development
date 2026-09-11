# -*- coding: utf-8 -*-
"""
s5_004_test_lgbm_vs_mahalanobis.py
Step 4: 代表 5 データセットにおける LightGBM 追跡 vs 4D Mahalanobis 直接対決検証
- 閾値スウィープ: [0.005, 0.010, 0.020, 0.030, 0.050]
- 4D Mahalanobis (現行ベースライン) との直接比較
- 追跡エッジ数、刈取後ノード数、刈取後P/E比、真のエッジRecall、公式期待スコアを算出
- エビデンス CSV 出力: s5_004_lgbm_vs_mahalanobis_evidence.csv
"""

import glob
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# 1. パス設定
base_dir = r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github"
working_dir = os.path.join(base_dir, "working")
data_dir = os.path.join(base_dir, "s5_analysys_data")
model_path = os.path.join(data_dir, "s5_003_tracking_edge_lgbm.joblib")
output_evidence_csv = os.path.join(data_dir, "s5_004_lgbm_vs_mahalanobis_evidence.csv")

TARGET_5_DATASETS = [
    "44b6_0113de3b",  # 密・良好
    "44b6_0b24845f",  # 密・難関(超低コントラスト)
    "44b6_74d0c52e",  # 密・組織深部
    "6bba_05b6850b",  # 疎・標準
    "6bba_085bf656",  # 疎・高コントラスト
]

THRESHOLDS_TO_SWEEP = [0.005, 0.010, 0.020, 0.030, 0.050]
SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
MAX_SEARCH_RADIUS_UM = 7.0

print("=" * 90)
print(">>> s5_004: LightGBM 追跡 vs 4D Mahalanobis 直接対決 ＆ 閾値スウィープ検証")
print("=" * 90)

# 2. モデルロード
if not os.path.exists(model_path):
    raise FileNotFoundError(f"LightGBM モデルが見つかりません: {model_path}")

lgbm_model = joblib.load(model_path)
print(f"[*] LightGBM モデルロード成功: {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")

# 3. 検出ノードのロード
node_files = sorted(glob.glob(os.path.join(working_dir, "s3_01_detect_nodes_pred_blobdog_5dmahalanobis_*.csv")))
all_nodes_list = []
for f in node_files:
    df_chunk = pd.read_csv(f)
    sub_df = df_chunk[df_chunk['dataset'].isin(TARGET_5_DATASETS)]
    if not sub_df.empty:
        all_nodes_list.append(sub_df)

nodes_df_all = pd.concat(all_nodes_list, ignore_index=True)
print(f"[*] 代表 5 データセット 検出ノード数: {len(nodes_df_all):,} 件")

# GT データロード
gt_nodes_df = pd.read_csv(os.path.join(working_dir, "s3_gt_nodes.csv"))
gt_edges_df = pd.read_csv(os.path.join(working_dir, "s3_gt_edges.csv"))
gt_summary_df = pd.read_csv(os.path.join(working_dir, "s3_gt_summary.csv")).set_index("dataset")

# 4. トラッカー実装

class Mahalanobis4DTracker:
    def __init__(self, feature_cols=None, max_search_radius_um=7.0):
        self.feature_cols = feature_cols or ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']
        self.max_search_radius_um = max_search_radius_um

    def detect(self, nodes_df, scale=SCALE_VEC):
        edges = []
        frames = sorted(nodes_df['t'].unique())
        avail_feats = [c for c in self.feature_cols if c in nodes_df.columns]

        for i in range(len(frames) - 1):
            t_curr, t_next = frames[i], frames[i+1]
            if t_next != t_curr + 1:
                continue
            df_curr = nodes_df[nodes_df['t'] == t_curr]
            df_next = nodes_df[nodes_df['t'] == t_next]
            if df_curr.empty or df_next.empty:
                continue

            pos_curr = df_curr[['z', 'y', 'x']].values * scale
            pos_next = df_next[['z', 'y', 'x']].values * scale
            spatial_dist = cdist(pos_curr, pos_next)

            feats_curr = df_curr[avail_feats].values.astype(np.float32)
            feats_next = df_next[avail_feats].values.astype(np.float32)
            combined = np.vstack([feats_curr, feats_next])
            cov = np.cov(combined, rowvar=False)
            inv_cov = np.linalg.pinv(cov)

            try:
                feat_dist = cdist(feats_curr, feats_next, metric='mahalanobis', VI=inv_cov)
            except Exception:
                feat_dist = cdist(feats_curr, feats_next, metric='cityblock')

            total_cost = spatial_dist + feat_dist
            row_ind, col_ind = linear_sum_assignment(total_cost)

            curr_ids = df_curr['node_id'].values
            next_ids = df_next['node_id'].values

            for r, c in zip(row_ind, col_ind):
                if spatial_dist[r, c] <= self.max_search_radius_um:
                    edges.append({'source_id': int(curr_ids[r]), 'target_id': int(next_ids[c])})

        return pd.DataFrame(edges) if edges else pd.DataFrame(columns=['source_id', 'target_id'])


class LightGBMTracker:
    def __init__(self, model, threshold=0.02, max_search_radius_um=7.0):
        self.model = model
        self.threshold = threshold
        self.max_search_radius_um = max_search_radius_um
        self.feature_cols_4d = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']

    def detect(self, nodes_df, scale=SCALE_VEC):
        edges = []
        frames = sorted(nodes_df['t'].unique())
        avail_feats = [c for c in self.feature_cols_4d if c in nodes_df.columns]

        for i in range(len(frames) - 1):
            t_curr, t_next = frames[i], frames[i+1]
            if t_next != t_curr + 1:
                continue
            df_curr = nodes_df[nodes_df['t'] == t_curr].reset_index(drop=True)
            df_next = nodes_df[nodes_df['t'] == t_next].reset_index(drop=True)
            if df_curr.empty or df_next.empty:
                continue

            pos_curr = df_curr[['z', 'y', 'x']].values * scale
            pos_next = df_next[['z', 'y', 'x']].values * scale
            spatial_dist = cdist(pos_curr, pos_next)

            mask = (spatial_dist <= self.max_search_radius_um)
            r_idx, c_idx = np.where(mask)
            if len(r_idx) == 0:
                continue

            # 4D Mahalanobis
            fc = df_curr[avail_feats].values.astype(np.float32)
            fn = df_next[avail_feats].values.astype(np.float32)
            combined = np.vstack([fc, fn])
            cov = np.cov(combined, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            try:
                feat_dist = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)
            except Exception:
                feat_dist = cdist(fc, fn, metric='cityblock')

            # 23 特徴量のベクトル化計算
            sp_d = spatial_dist[r_idx, c_idx]
            mh_d = feat_dist[r_idx, c_idx]
            total_cost_4d = sp_d + mh_d

            dz = (df_next['z'].values[c_idx] - df_curr['z'].values[r_idx]) * scale[0]
            dy = (df_next['y'].values[c_idx] - df_curr['y'].values[r_idx]) * scale[1]
            dx = (df_next['x'].values[c_idx] - df_curr['x'].values[r_idx]) * scale[2]
            d_xy = np.sqrt(dy**2 + dx**2)

            int1 = df_curr['mean_intensity'].values[r_idx]
            int2 = df_next['mean_intensity'].values[c_idx]
            snr1 = df_curr['snr'].values[r_idx]
            snr2 = df_next['snr'].values[c_idx]
            rad1 = df_curr['estimated_radius_um'].values[r_idx]
            rad2 = df_next['estimated_radius_um'].values[c_idx]
            vol1 = df_curr['volume_um3'].values[r_idx]
            vol2 = df_next['volume_um3'].values[c_idx]
            zdep1 = df_curr['z_depth_ratio'].values[r_idx]
            zdep2 = df_next['z_depth_ratio'].values[c_idx]
            dens1 = df_curr['local_density_r15'].values[r_idx]
            dens2 = df_next['local_density_r15'].values[c_idx]

            # rank & margin
            min_sp_dist = np.min(spatial_dist, axis=1)
            sp_margin = sp_d - min_sp_dist[r_idx]

            ranks = np.zeros(len(r_idx), dtype=np.int32)
            for r in np.unique(r_idx):
                match_k = np.where(r_idx == r)[0]
                sorted_k = match_k[np.argsort(sp_d[match_k])]
                ranks[sorted_k] = np.arange(1, len(sorted_k) + 1)

            feat_df = pd.DataFrame({
                'spatial_dist': sp_d,
                'spatial_dist_xy': d_xy,
                'delta_z_scaled': dz,
                'delta_y_scaled': dy,
                'delta_x_scaled': dx,
                'abs_delta_z': np.abs(dz),
                'spatial_rank': ranks,
                'spatial_margin': sp_margin,
                'int_diff': np.abs(int1 - int2),
                'int_ratio': int1 / (int2 + 1e-5),
                'snr_diff': np.abs(snr1 - snr2),
                'snr_ratio': snr1 / (snr2 + 1e-5),
                'snr_min': np.minimum(snr1, snr2),
                'radius_diff': np.abs(rad1 - rad2),
                'radius_ratio': rad1 / (rad2 + 1e-5),
                'volume_diff': np.abs(vol1 - vol2),
                'volume_ratio': vol1 / (vol2 + 1e-5),
                'z_depth_diff': np.abs(zdep1 - zdep2),
                'density_source': dens1,
                'density_target': dens2,
                'density_diff': np.abs(dens1 - dens2),
                'mahalanobis_dist': mh_d,
                'total_cost_4d': total_cost_4d
            })

            probs = self.model.predict_proba(feat_df)[:, 1]

            # コスト行列構築
            cost_mat = np.full(spatial_dist.shape, 1e9, dtype=np.float32)
            valid_mask = (probs >= self.threshold)
            
            if np.sum(valid_mask) > 0:
                v_r = r_idx[valid_mask]
                v_c = c_idx[valid_mask]
                v_p = probs[valid_mask]
                cost_mat[v_r, v_c] = -np.log(v_p + 1e-6)

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            curr_ids = df_curr['node_id'].values
            next_ids = df_next['node_id'].values

            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e8:
                    edges.append({'source_id': int(curr_ids[r]), 'target_id': int(next_ids[c])})

        return pd.DataFrame(edges) if edges else pd.DataFrame(columns=['source_id', 'target_id'])


def filter_isolated_nodes(nodes_df, edges_df):
    if edges_df.empty:
        return pd.DataFrame(columns=nodes_df.columns)
    connected_nodes = set(edges_df['source_id']).union(set(edges_df['target_id']))
    return nodes_df[nodes_df['node_id'].isin(connected_nodes)].copy()


def evaluate_edge_recall_and_tp(ds, pred_nodes, pred_edges):
    gt_n_ds = gt_nodes_df[gt_nodes_df['dataset'] == ds]
    gt_e_ds = gt_edges_df[gt_edges_df['dataset'] == ds]
    if len(gt_e_ds) == 0:
        return 0, 0, 0.0

    gt_edge_set = set(zip(gt_e_ds['source_id'].astype(int), gt_e_ds['target_id'].astype(int)))
    
    # ノードマッチング
    pred_nodes_ds = pred_nodes.copy().reset_index(drop=True)
    pred_nodes_ds['gt_node_id'] = -1
    
    for t in sorted(pred_nodes_ds['t'].unique()):
        sub_p = pred_nodes_ds[pred_nodes_ds['t'] == t]
        sub_g = gt_n_ds[gt_n_ds['t'] == t]
        if sub_p.empty or sub_g.empty:
            continue
        pos_p = sub_p[['z', 'y', 'x']].values * SCALE_VEC
        pos_g = sub_g[['z', 'y', 'x']].values * SCALE_VEC
        dist_mat = cdist(pos_p, pos_g)
        r_ind, c_ind = linear_sum_assignment(dist_mat)
        p_indices = sub_p.index.values
        g_node_ids = sub_g['node_id'].values
        for r, c in zip(r_ind, c_ind):
            if dist_mat[r, c] <= MAX_SEARCH_RADIUS_UM:
                pred_nodes_ds.loc[p_indices[r], 'gt_node_id'] = int(g_node_ids[c])

    node_to_gt = dict(zip(pred_nodes_ds['node_id'], pred_nodes_ds['gt_node_id']))
    
    tp_count = 0
    for _, row in pred_edges.iterrows():
        g1 = node_to_gt.get(int(row['source_id']), -1)
        g2 = node_to_gt.get(int(row['target_id']), -1)
        if g1 != -1 and g2 != -1 and (g1, g2) in gt_edge_set:
            tp_count += 1
            
    total_gt = len(gt_e_ds)
    recall = tp_count / total_gt if total_gt > 0 else 0.0
    return tp_count, total_gt, recall


# 5. 実走直接対決テスト
results = []
tracker_4d = Mahalanobis4DTracker()

methods_to_test = [("4dmahalanobis", None)] + [("lgbm", th) for th in THRESHOLDS_TO_SWEEP]

for method_name, th in methods_to_test:
    method_label = f"lgbm_th_{th:.3f}" if th is not None else "4dmahalanobis"
    print(f"\n>> 手法検証中: {method_label}")
    t_start = time.time()
    
    tot_raw_nodes = 0
    tot_edges = 0
    tot_filtered_nodes = 0
    tot_tp = 0
    tot_gt = 0
    recalls = []
    pe_ratios = []
    scores = []
    
    for ds in TARGET_5_DATASETS:
        ds_nodes = nodes_df_all[nodes_df_all['dataset'] == ds].copy()
        est_nodes = gt_summary_df.loc[ds, 'estimated_number_of_nodes'] if ds in gt_summary_df.index else np.nan
        
        t0 = time.time()
        if method_name == "4dmahalanobis":
            pred_edges = tracker_4d.detect(ds_nodes)
        else:
            tracker_lgbm = LightGBMTracker(lgbm_model, threshold=th)
            pred_edges = tracker_lgbm.detect(ds_nodes)
            
        ds_elapsed = time.time() - t0
        
        # 孤立ノード刈取
        filtered_nodes = filter_isolated_nodes(ds_nodes, pred_edges)
        
        n_raw = len(ds_nodes)
        n_edges = len(pred_edges)
        n_filt = len(filtered_nodes)
        
        pe_ratio = (n_filt / est_nodes) if pd.notna(est_nodes) and est_nodes > 0 else np.nan
        
        # エッジ精度評価
        tp, gt_cnt, recall = evaluate_edge_recall_and_tp(ds, ds_nodes, pred_edges)
        
        # 公式期待スコア: Recall * (1 - 0.1 * max(0, pe - 1.0))
        penalty = 1.0 - 0.1 * max(0.0, pe_ratio - 1.0) if pd.notna(pe_ratio) else 1.0
        exp_score = recall * penalty
        
        tot_raw_nodes += n_raw
        tot_edges += n_edges
        tot_filtered_nodes += n_filt
        tot_tp += tp
        tot_gt += gt_cnt
        recalls.append(recall)
        pe_ratios.append(pe_ratio)
        scores.append(exp_score)
        
        results.append({
            "method": method_label,
            "dataset": ds,
            "raw_nodes": n_raw,
            "edges": n_edges,
            "filtered_nodes": n_filt,
            "est_nodes": int(est_nodes),
            "pe_ratio": pe_ratio,
            "tp_edges": tp,
            "gt_edges": gt_cnt,
            "edge_recall": recall,
            "penalty_factor": penalty,
            "expected_score": exp_score,
            "elapsed_sec": ds_elapsed
        })
        
    macro_recall = np.mean(recalls)
    mean_pe = np.mean(pe_ratios)
    mean_score = np.mean(scores)
    tot_elapsed = time.time() - t_start
    print(f"    - [{method_label}] エッジ数: {tot_edges:,} | 刈取後ノード: {tot_filtered_nodes:,} | 平均 P/E比: {mean_pe:.3f} | Macro Recall: {macro_recall*100:.2f}% (TP {tot_tp:,}/{tot_gt:,}) | 公式期待スコア: {mean_score:.4f} ({tot_elapsed:.1f}秒)")

df_results = pd.DataFrame(results)
df_results.to_csv(output_evidence_csv, index=False)
print(f"\n[*] 全検証完了! エビデンス CSV 保存: {output_evidence_csv}")
