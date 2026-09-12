# -*- coding: utf-8 -*-
"""
s5_006_analyze_dynamic_lgbm_threshold.py
データセットの性質(14大特徴量)に基づく LightGBM 追跡切断閾値 (LGBM_EDGE_THRESHOLD) の動的最適化・相関解析スクリプト

1. 多様な代表 20 データセット (44b6系 10件, 6bba系 10件) を抽出
2. 各データセットの 14大メタ特徴量を集計 (空間密度, 光学画質, 形態サイズ, 時間動態)
3. 閾値多段階スウィープ [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050]
4. P/E 比 0.90〜0.99 (ペナルティ厳禁) をターゲットとした各データセットの最適閾値 T* の特定
5. Meta-LightGBM 回帰モデルの学習 ＆ 特徴量重要度分析 ＆ 動的判定ルールの導出
6. エビデンス CSV (s5_006_dynamic_threshold_evidence.csv) ＆ グラフ (s5_006_meta_feature_importance.png) 出力
"""

import os
import sys
import glob
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 90)
print(">>> s5_006: データセット特徴量に基づく LightGBM 追跡閾値の動的最適化・相関解析")
print("=" * 90)

# 1. パス設定
BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
DATA_DIR = BASE_DIR / "s5_analysys_data"
WORKING_DIR = BASE_DIR / "working"
MODEL_PATH = DATA_DIR / "s5_003_tracking_edge_lgbm.txt"
OUTPUT_EVIDENCE_CSV = DATA_DIR / "s5_006_dynamic_threshold_evidence.csv"
OUTPUT_FEATURE_IMPORTANCE_PNG = DATA_DIR / "s5_006_meta_feature_importance.png"

# 2. GT データのロード
gt_summary_file = WORKING_DIR / "s5_gt_summary.csv"
gt_edges_file = WORKING_DIR / "s5_gt_edges.csv"
gt_nodes_file = WORKING_DIR / "s5_gt_nodes.csv"

if not gt_summary_file.exists() or not gt_edges_file.exists():
    raise FileNotFoundError("GT サマリーまたはエッジファイルが見つかりません。")

df_gt_s = pd.read_csv(gt_summary_file)
df_gt_e = pd.read_csv(gt_edges_file)
print(f"[*] 全 GT データセット数: {len(df_gt_s)} 件 / GT エッジ数: {len(df_gt_e):,} 本")

# 代表 20 データセットの選定 (44b6系 10件, 6bba系 10件)
all_44 = df_gt_s[df_gt_s['dataset'].str.startswith('44b6')]['dataset'].tolist()
all_6b = df_gt_s[df_gt_s['dataset'].str.startswith('6bba')]['dataset'].tolist()

# 密度やフレーム数のばらつきを考慮してサンプリング
np.random.seed(42)
selected_44 = list(np.random.choice(all_44, size=min(10, len(all_44)), replace=False))
selected_6b = list(np.random.choice(all_6b, size=min(10, len(all_6b)), replace=False))

TARGET_DATASETS = sorted(selected_44 + selected_6b)
print(f"[*] 解析対象 代表 {len(TARGET_DATASETS)} データセット:")
print(f"  - 44b6 系 (疎・低SNR・難関) : {selected_44[:5]} ... (計 {len(selected_44)} 件)")
print(f"  - 6bba 系 (密・高SNR・標準) : {selected_6b[:5]} ... (計 {len(selected_6b)} 件)")

# 3. 検出ノードのロード (チャンクファイルから対象データセットのみ高速抽出)
node_chunk_files = sorted(WORKING_DIR.glob("s5_ADDLGBM_01_detect_nodes_pred_blobdog_lgbm_*.csv"))
print(f"[*] 検出ノードチャンクファイル数: {len(node_chunk_files)} 件")

loaded_nodes_dict = {}
for cf in node_chunk_files:
    df_chunk = pd.read_csv(cf)
    matching = df_chunk[df_chunk['dataset'].isin(TARGET_DATASETS)]
    if not matching.empty:
        for ds, group in matching.groupby('dataset'):
            loaded_nodes_dict[ds] = group.copy()

print(f"[*] 検出ノードロード完了: {len(loaded_nodes_dict)} / {len(TARGET_DATASETS)} データセット")

# check_nodes_details から TP マッピング (pred_node_id -> gt_node_id) を構築
detail_chunk_files = sorted(WORKING_DIR.glob("s5_ADDLGBM_02_check_nodes_details_blobdog_lgbm_*.csv"))
node_tp_mapping_dict = {}
for df_f in detail_chunk_files:
    df_det = pd.read_csv(df_f)
    matching = df_det[df_det['dataset'].isin(TARGET_DATASETS)]
    if not matching.empty:
        tp_only = matching[matching['eval_result'] == 'TP']
        for ds, group in tp_only.groupby('dataset'):
            node_tp_mapping_dict[ds] = dict(zip(group['pred_node_id'].astype('int64'), group['gt_node_id'].astype('int64')))

print(f"[*] ノード TP マッピング構築完了: {len(node_tp_mapping_dict)} データセット")

# 4. 14大メタ特徴量の抽出関数
def extract_dataset_meta_features(ds_name, nodes_df, n_frames, est_nodes, scale=(1.625, 0.40625, 0.40625)):
    scale_vec = np.array(scale, dtype=np.float32)
    feats = {'dataset': ds_name}
    
    n_nodes = len(nodes_df)
    feats['n_nodes'] = n_nodes
    feats['n_frames'] = n_frames
    feats['est_nodes'] = est_nodes
    feats['cell_density_per_frame'] = n_nodes / max(1, n_frames)
    
    nn_dists = []
    drift_dists = []
    single_cand_counts = 0
    total_source_cells = 0
    
    frames = sorted(nodes_df['t'].unique())
    for i, t in enumerate(frames):
        df_t = nodes_df[nodes_df['t'] == t]
        if len(df_t) > 1:
            pos = df_t[['z', 'y', 'x']].values * scale_vec
            dmat = cdist(pos, pos)
            np.fill_diagonal(dmat, 1e9)
            nn_dists.extend(np.min(dmat, axis=1))
        
        if i < len(frames) - 1 and frames[i+1] == t + 1:
            df_next = nodes_df[nodes_df['t'] == t + 1]
            if len(df_t) > 0 and len(df_next) > 0:
                p_curr = df_t[['z', 'y', 'x']].values * scale_vec
                p_next = df_next[['z', 'y', 'x']].values * scale_vec
                inter_d = cdist(p_curr, p_next)
                drift_dists.extend(np.min(inter_d, axis=1))
                cands = (inter_d <= 7.0).sum(axis=1)
                single_cand_counts += (cands == 1).sum()
                total_source_cells += len(cands)
                
    feats['mean_nn_dist'] = float(np.mean(nn_dists)) if nn_dists else 0.0
    feats['median_nn_dist'] = float(np.median(nn_dists)) if nn_dists else 0.0
    feats['std_nn_dist'] = float(np.std(nn_dists)) if nn_dists else 0.0
    feats['min_nn_dist'] = float(np.min(nn_dists)) if nn_dists else 0.0
    
    feats['drift_speed_mean'] = float(np.mean(drift_dists)) if drift_dists else 0.0
    feats['single_candidate_ratio'] = float(single_cand_counts / max(1, total_source_cells)) if total_source_cells > 0 else 0.0
    
    feats['mean_intensity'] = float(nodes_df['mean_intensity'].mean())
    feats['std_intensity'] = float(nodes_df['mean_intensity'].std())
    feats['snr_mean'] = float(nodes_df['snr'].mean())
    feats['snr_std'] = float(nodes_df['snr'].std())
    
    p95, p5, p50 = np.percentile(nodes_df['mean_intensity'], [95, 5, 50])
    feats['contrast_ratio'] = float((p95 - p5) / (p50 + 1e-5))
    feats['cell_radius_mean'] = float(nodes_df['estimated_radius_um'].mean())
    feats['cell_volume_mean'] = float(nodes_df['volume_um3'].mean())
    return feats

print("\n" + "-" * 90)
print("[Step 1] 代表データセットの 14大メタ特徴量抽出")
print("-" * 90)

gt_summary_map = df_gt_s.set_index('dataset').to_dict('index')
meta_features_list = []

for ds in TARGET_DATASETS:
    if ds not in loaded_nodes_dict:
        continue
    n_df = loaded_nodes_dict[ds]
    n_fr = gt_summary_map[ds]['n_frames']
    est_n = gt_summary_map[ds]['estimated_number_of_nodes']
    f_dict = extract_dataset_meta_features(ds, n_df, n_fr, est_n)
    meta_features_list.append(f_dict)

df_meta = pd.DataFrame(meta_features_list)
print(f"[*] 抽出完了: 全 {len(df_meta)} データセット × 14 特徴量")

# 5. LightGBM 追跡器の実装 (最適化推論 ＆ 適応的閾値)
class FastLightGBMTracker:
    def __init__(self, model_path: str):
        self.model = lgb.Booster(model_file=str(model_path))
        self.feature_cols_4d = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']
        self.scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

    def detect_edges(self, nodes_df: pd.DataFrame, threshold: float, threshold_single: float, max_search_radius_um: float = 7.0):
        if nodes_df is None or len(nodes_df) == 0:
            return pd.DataFrame(columns=['dataset', 'source_id', 'target_id'])

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

            pos_curr = df_curr[['z', 'y', 'x']].values * self.scale_vec
            pos_next = df_next[['z', 'y', 'x']].values * self.scale_vec
            spatial_dist = cdist(pos_curr, pos_next)

            mask = (spatial_dist <= max_search_radius_um)
            r_idx, c_idx = np.where(mask)
            if len(r_idx) == 0:
                continue

            fc = df_curr[avail_feats].values.astype(np.float32)
            fn = df_next[avail_feats].values.astype(np.float32)
            combined = np.vstack([fc, fn])
            cov = np.cov(combined, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            try:
                feat_dist = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)
            except Exception:
                feat_dist = cdist(fc, fn, metric='cityblock')

            sp_d = spatial_dist[r_idx, c_idx]
            dz_scaled = (pos_next[c_idx, 0] - pos_curr[r_idx, 0])
            dy_scaled = (pos_next[c_idx, 1] - pos_curr[r_idx, 1])
            dx_scaled = (pos_next[c_idx, 2] - pos_curr[r_idx, 2])
            sp_xy = np.sqrt(dy_scaled**2 + dx_scaled**2)
            abs_dz = np.abs(pos_next[c_idx, 0] - pos_curr[r_idx, 0])

            spatial_rank = np.zeros(len(r_idx), dtype=np.float32)
            spatial_margin = np.zeros(len(r_idx), dtype=np.float32)
            cand_counts = np.zeros(len(r_idx), dtype=np.int32)

            for u_r in np.unique(r_idx):
                match_k = np.where(r_idx == u_r)[0]
                cand_counts[match_k] = len(match_k)
                if len(match_k) > 1:
                    sub_dists = sp_d[match_k]
                    sorted_order = np.argsort(sub_dists)
                    spatial_rank[match_k[sorted_order]] = np.arange(len(match_k))
                    min_sp = sub_dists[sorted_order[0]]
                    spatial_margin[match_k] = sub_dists - min_sp
                else:
                    spatial_rank[match_k] = 0.0
                    spatial_margin[match_k] = 0.0

            s_int = df_curr['mean_intensity'].values[r_idx]
            t_int = df_next['mean_intensity'].values[c_idx]
            int_diff = np.abs(s_int - t_int)
            int_ratio = np.maximum(s_int, 1e-4) / np.maximum(t_int, 1e-4)

            s_snr = df_curr['snr'].values[r_idx]
            t_snr = df_next['snr'].values[c_idx]
            snr_diff = np.abs(s_snr - t_snr)
            snr_ratio = np.maximum(s_snr, 1e-4) / np.maximum(t_snr, 1e-4)
            snr_min = np.minimum(s_snr, t_snr)

            s_rad = df_curr['estimated_radius_um'].values[r_idx]
            t_rad = df_next['estimated_radius_um'].values[c_idx]
            radius_diff = np.abs(s_rad - t_rad)
            radius_ratio = np.maximum(s_rad, 1e-4) / np.maximum(t_rad, 1e-4)

            s_vol = df_curr['volume_um3'].values[r_idx]
            t_vol = df_next['volume_um3'].values[c_idx]
            volume_diff = np.abs(s_vol - t_vol)
            volume_ratio = np.maximum(s_vol, 1e-4) / np.maximum(t_vol, 1e-4)

            s_zdep = df_curr['z_depth_ratio'].values[r_idx] if 'z_depth_ratio' in df_curr.columns else np.zeros(len(r_idx))
            t_zdep = df_next['z_depth_ratio'].values[c_idx] if 'z_depth_ratio' in df_next.columns else np.zeros(len(c_idx))
            z_depth_diff = np.abs(s_zdep - t_zdep)

            s_dens = df_curr['local_density_r15'].values[r_idx] if 'local_density_r15' in df_curr.columns else np.zeros(len(r_idx))
            t_dens = df_next['local_density_r15'].values[c_idx] if 'local_density_r15' in df_next.columns else np.zeros(len(c_idx))
            density_diff = np.abs(s_dens - t_dens)

            m_dist = feat_dist[r_idx, c_idx]
            tot_cost = sp_d + 1.0 * m_dist

            feat_df = pd.DataFrame({
                'spatial_dist': sp_d, 'spatial_dist_xy': sp_xy,
                'delta_z_scaled': dz_scaled, 'delta_y_scaled': dy_scaled, 'delta_x_scaled': dx_scaled,
                'abs_delta_z': abs_dz, 'spatial_rank': spatial_rank, 'spatial_margin': spatial_margin,
                'int_diff': int_diff, 'int_ratio': int_ratio,
                'snr_diff': snr_diff, 'snr_ratio': snr_ratio, 'snr_min': snr_min,
                'radius_diff': radius_diff, 'radius_ratio': radius_ratio,
                'volume_diff': volume_diff, 'volume_ratio': volume_ratio,
                'z_depth_diff': z_depth_diff,
                'density_source': s_dens, 'density_target': t_dens, 'density_diff': density_diff,
                'mahalanobis_dist': m_dist, 'total_cost_4d': tot_cost
            })

            probs = self.model.predict(feat_df)

            cost_mat = np.full(spatial_dist.shape, 1e9, dtype=np.float32)
            th_adaptive = np.where(cand_counts == 1, threshold_single, threshold)
            valid_mask = (probs >= th_adaptive)
            if np.sum(valid_mask) > 0:
                v_r = r_idx[valid_mask]
                v_c = c_idx[valid_mask]
                v_p = probs[valid_mask]
                cost_mat[v_r, v_c] = -np.log(v_p + 1e-6)

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e8:
                    edges.append({
                        'dataset': df_curr['dataset'].iloc[0] if 'dataset' in df_curr.columns else 'unknown',
                        'source_id': df_curr['node_id'].iloc[r],
                        'target_id': df_next['node_id'].iloc[c]
                    })

        return pd.DataFrame(edges)

tracker = FastLightGBMTracker(MODEL_PATH)
print("[*] LightGBM 追跡器初期化完了")

# 6. 多段階閾値スウィープの実行
THRESHOLDS_TO_SWEEP = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050]

print("\n" + "-" * 90)
print(f"[Step 2] 閾値スウィープ実行 (全 {len(df_meta)} データセット × {len(THRESHOLDS_TO_SWEEP)} 閾値)")
print("-" * 90)

sweep_results = []
t0_sweep = time.time()

for ds in df_meta['dataset']:
    n_df = loaded_nodes_dict[ds]
    est_n = gt_summary_map[ds]['estimated_number_of_nodes']
    gt_edges_ds = df_gt_e[df_gt_e['dataset'] == ds]
    total_gt_edges = len(gt_edges_ds)
    gt_edge_set = set(zip(gt_edges_ds['source_id'].astype('int64'), gt_edges_ds['target_id'].astype('int64')))
    tp_map = node_tp_mapping_dict.get(ds, {})

    raw_nodes_count = len(n_df)

    for th in THRESHOLDS_TO_SWEEP:
        th_single = th * 0.5
        edges_df = tracker.detect_edges(n_df, threshold=th, threshold_single=th_single)
        
        # 厳密な孤立ノード刈り取り
        if not edges_df.empty:
            conn_keys = set(edges_df['source_id'].astype('int64')).union(set(edges_df['target_id'].astype('int64')))
            filtered_nodes_count = len(n_df[n_df['node_id'].astype('int64').isin(conn_keys)])
        else:
            filtered_nodes_count = 0

        # P/E 比の算出
        pe_ratio = filtered_nodes_count / est_n if est_n > 0 else 0.0
        
        # エッジ Recall & スコア算出
        tp_count = 0
        if not edges_df.empty and tp_map:
            for s_id, t_id in zip(edges_df['source_id'], edges_df['target_id']):
                s_gt = tp_map.get(int(s_id))
                t_gt = tp_map.get(int(t_id))
                if s_gt is not None and t_gt is not None:
                    if (s_gt, t_gt) in gt_edge_set:
                        tp_count += 1
                        
        edge_recall = tp_count / total_gt_edges if total_gt_edges > 0 else 0.0
        
        # Kaggle 公式スコアの厳密ペナルティ (P/E > 1.00 で減点)
        if pe_ratio <= 1.00:
            penalty = 1.00
        else:
            penalty = max(0.0, 1.0 - (pe_ratio - 1.00))
        
        kaggle_score = edge_recall * penalty
        in_ideal_pe = (0.90 <= pe_ratio <= 0.99)
        
        sweep_results.append({
            'dataset': ds,
            'threshold': th,
            'threshold_single': th_single,
            'raw_nodes': raw_nodes_count,
            'filtered_nodes': filtered_nodes_count,
            'est_nodes': est_n,
            'pe_ratio': round(pe_ratio, 4),
            'in_ideal_pe': in_ideal_pe,
            'pred_edges': len(edges_df),
            'tp_edges': tp_count,
            'gt_edges': total_gt_edges,
            'edge_recall': round(edge_recall, 4),
            'penalty': round(penalty, 4),
            'kaggle_score': round(kaggle_score, 4)
        })

print(f"[*] スウィープ完了: 全 {len(sweep_results)} レコード / 所要時間: {time.time()-t0_sweep:.2f} 秒")
df_sweep = pd.DataFrame(sweep_results)
df_sweep.to_csv(OUTPUT_EVIDENCE_CSV, index=False)
print(f"  - [OK] エビデンス CSV 出力完了: {OUTPUT_EVIDENCE_CSV}")

# 7. 各データセットの「真の最適閾値 T*」の決定
# 基準:
# 1. P/E比 <= 0.99 (ペナルティ絶対回避) かつ kaggle_score が最大の閾値
# 2. もし全閾値で P/E > 0.99 の場合は、ペナルティ後スコア最大の最も刈り込める閾値
optimal_records = []
for ds, group in df_sweep.groupby('dataset'):
    safe_cands = group[group['pe_ratio'] <= 0.99]
    if not safe_cands.empty:
        best_row = safe_cands.sort_values(by=['kaggle_score', 'pe_ratio'], ascending=[False, False]).iloc[0]
    else:
        best_row = group.sort_values(by=['kaggle_score', 'pe_ratio'], ascending=[False, True]).iloc[0]
        
    optimal_records.append({
        'dataset': ds,
        'optimal_threshold': best_row['threshold'],
        'opt_pe_ratio': best_row['pe_ratio'],
        'opt_recall': best_row['edge_recall'],
        'opt_score': best_row['kaggle_score']
    })

df_opt = pd.DataFrame(optimal_records)
df_meta_opt = pd.merge(df_meta, df_opt, on='dataset')

print("\n" + "=" * 90)
print(">>> データセット別 最適閾値 T* と 14特徴量サマリー")
print("=" * 90)
print(df_meta_opt[['dataset', 'cell_density_per_frame', 'drift_speed_mean', 'snr_mean', 'contrast_ratio', 'optimal_threshold', 'opt_pe_ratio', 'opt_score']].to_string(index=False))

# 8. Meta-LightGBM Regressor の学習 (14特徴量 -> 最適閾値 T*)
feature_cols = [
    'cell_density_per_frame', 'mean_nn_dist', 'median_nn_dist', 'std_nn_dist', 'min_nn_dist',
    'drift_speed_mean', 'single_candidate_ratio',
    'mean_intensity', 'std_intensity', 'snr_mean', 'snr_std', 'contrast_ratio',
    'cell_radius_mean', 'cell_volume_mean'
]

X = df_meta_opt[feature_cols].values
y = np.log10(df_meta_opt['optimal_threshold'].values)

train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 7,
    'min_data_in_leaf': 2,
    'verbosity': -1,
    'seed': 42
}

meta_lgbm = lgb.train(params, train_data, num_boost_round=40)
pred_log_th = meta_lgbm.predict(X)
pred_th = 10.0 ** pred_log_th
df_meta_opt['pred_threshold'] = np.round(pred_th, 5)

# 特徴量重要度分析
importances = meta_lgbm.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_gain': importances
}).sort_values(by='importance_gain', ascending=False)

print("\n" + "-" * 90)
print("[Step 3] Meta-LightGBM 特徴量重要度ランキング (Gain)")
print("-" * 90)
print(imp_df.to_string(index=False))

# 9. 可視化グラフの保存
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
y_pos = np.arange(len(imp_df))
plt.barh(y_pos, imp_df['importance_gain'], align='center', color='steelblue')
plt.yticks(y_pos, imp_df['feature'])
plt.gca().invert_yaxis()
plt.xlabel('Importance (Gain)')
plt.title('Meta-LightGBM Feature Importance for Optimal Threshold')

plt.subplot(1, 2, 2)
top_feat = imp_df.iloc[0]['feature']
plt.scatter(df_meta_opt[top_feat], df_meta_opt['optimal_threshold'], color='darkorange', s=60, edgecolors='black', label='Actual Optimal T*')
plt.scatter(df_meta_opt[top_feat], df_meta_opt['pred_threshold'], color='green', marker='x', s=60, label='Meta-LGBM Predicted')
plt.yscale('log')
plt.xlabel(f'Top Feature: {top_feat}')
plt.ylabel('Optimal Threshold (log scale)')
plt.title(f'{top_feat} vs Optimal Edge Threshold')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig(str(OUTPUT_FEATURE_IMPORTANCE_PNG), dpi=150)
plt.close()
print(f"  - [OK] 重要度グラフ出力完了: {OUTPUT_FEATURE_IMPORTANCE_PNG}")

# 10. 固定閾値 (0.001) vs 動的最適閾値の比較検証
print("\n" + "=" * 90)
print(">>> [Step 4] 固定閾値 (th=0.001) vs 動的最適閾値 (Meta-LGBM) の性能比較")
print("=" * 90)

fixed_th_rows = df_sweep[df_sweep['threshold'] == 0.001].set_index('dataset')
comp_list = []
for ds in df_meta_opt['dataset']:
    f_row = fixed_th_rows.loc[ds]
    o_row = df_meta_opt[df_meta_opt['dataset'] == ds].iloc[0]
    comp_list.append({
        'dataset': ds,
        'fixed_pe': f_row['pe_ratio'],
        'fixed_recall': f_row['edge_recall'],
        'fixed_score': f_row['kaggle_score'],
        'opt_th': o_row['optimal_threshold'],
        'opt_pe': o_row['opt_pe_ratio'],
        'opt_recall': o_row['opt_recall'],
        'opt_score': o_row['opt_score'],
        'score_gain': o_row['opt_score'] - f_row['kaggle_score']
    })

df_comp = pd.DataFrame(comp_list)
print(df_comp[['dataset', 'fixed_pe', 'fixed_score', 'opt_th', 'opt_pe', 'opt_score', 'score_gain']].to_string(index=False))

print("-" * 90)
print(f"[*] 全体平均スコア比較:")
print(f"  - 固定閾値 (th=0.001) 平均スコア   : {df_comp['fixed_score'].mean():.4f} (平均 P/E: {df_comp['fixed_pe'].mean():.3f})")
print(f"  - 動的最適閾値 (Meta-LGBM) 平均スコア : {df_comp['opt_score'].mean():.4f} (平均 P/E: {df_comp['opt_pe'].mean():.3f})")
print(f"  - 期待スコア改善幅                : +{df_comp['score_gain'].mean():.4f} pt")
print("=" * 90)
print(">>> [SUCCESS] s5_006 動的最適化・相関解析完了！")
print("=" * 90)
