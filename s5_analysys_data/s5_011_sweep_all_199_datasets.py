# -*- coding: utf-8 -*-
"""
s5_011_sweep_all_199_datasets.py
全 199 データセットを対象とした 10 段階閾値スウィープ解析 ＆
Meta-LightGBM 学習用 正例・負例全数データ (1,990 レコード) 生成スクリプト。

24 コア CPU を活かしたマルチプロセス並列処理 (Parallel) により高速完走。
"""

import os
import sys
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
DATA_DIR = BASE_DIR / "s5_analysys_data"
WORKING_DIR = BASE_DIR / "working"
MODEL_PATH = DATA_DIR / "s5_003_tracking_edge_lgbm.txt"

OUTPUT_EVIDENCE_CSV = DATA_DIR / "s5_011_all199_threshold_evidence.csv"
OUTPUT_OPTIMAL_CSV = DATA_DIR / "s5_011_all199_optimal_thresholds.csv"
OUTPUT_IMPORTANCE_PNG = DATA_DIR / "s5_011_all199_meta_feature_importance.png"

print("=" * 90)
print(">>> s5_011: 全 199 データセット 10段階閾値スウィープ ＆ 正例・負例全数取得")
print("=" * 90)

# 1. GT データのロード
gt_summary_file = WORKING_DIR / "s5_gt_summary.csv"
gt_edges_file = WORKING_DIR / "s5_gt_edges.csv"

df_gt_s = pd.read_csv(gt_summary_file)
df_gt_e = pd.read_csv(gt_edges_file)
ALL_DATASETS = sorted(df_gt_s['dataset'].unique())
print(f"[*] 全 GT データセット数: {len(ALL_DATASETS)} 件 / GT エッジ数: {len(df_gt_e):,} 本")

# 2. 検出ノードのロード (全 199 データセット)
t0_load = time.time()
node_chunk_files = sorted(WORKING_DIR.glob("s5_ADDLGBM_01_detect_nodes_pred_blobdog_lgbm_*.csv"))
print(f"[*] 検出ノードチャンクファイル読み込み開始 ({len(node_chunk_files)} ファイル)...")

loaded_nodes_dict = {}
for cf in node_chunk_files:
    df_chunk = pd.read_csv(cf)
    for ds, group in df_chunk.groupby('dataset'):
        loaded_nodes_dict[ds] = group.copy()

print(f"[*] 検出ノードロード完了: {len(loaded_nodes_dict)} / {len(ALL_DATASETS)} データセット ({time.time()-t0_load:.2f} 秒)")

# TP マッピングのロード
detail_chunk_files = sorted(WORKING_DIR.glob("s5_ADDLGBM_02_check_nodes_details_blobdog_lgbm_*.csv"))
node_tp_mapping_dict = {}
for df_f in detail_chunk_files:
    df_det = pd.read_csv(df_f)
    tp_only = df_det[df_det['eval_result'] == 'TP']
    for ds, group in tp_only.groupby('dataset'):
        node_tp_mapping_dict[ds] = dict(zip(group['pred_node_id'].astype('int64'), group['gt_node_id'].astype('int64')))

print(f"[*] ノード TP マッピングロード完了: {len(node_tp_mapping_dict)} データセット")

# 3. 14大メタ特徴量の抽出
def extract_meta(ds_name, nodes_df, n_frames, est_nodes, scale=(1.625, 0.40625, 0.40625)):
    scale_vec = np.array(scale, dtype=np.float32)
    feats = {'dataset': ds_name}
    n_nodes = len(nodes_df)
    feats['n_nodes'] = n_nodes
    feats['n_frames'] = n_frames
    feats['est_nodes'] = est_nodes
    feats['density_per_frame'] = n_nodes / max(1, n_frames)
    
    nn_dists = []
    frames = sorted(nodes_df['t'].unique())[:3]
    for t in frames:
        df_t = nodes_df[nodes_df['t'] == t]
        if len(df_t) > 1:
            pos = df_t[['z', 'y', 'x']].values * scale_vec
            dmat = cdist(pos, pos)
            np.fill_diagonal(dmat, 1e9)
            nn_dists.extend(np.min(dmat, axis=1))
            
    feats['mean_nn_dist'] = float(np.mean(nn_dists)) if nn_dists else 10.0
    feats['median_nn_dist'] = float(np.median(nn_dists)) if nn_dists else 10.0
    feats['min_nn_dist'] = float(np.min(nn_dists)) if nn_dists else 1.0
    feats['std_nn_dist'] = float(np.std(nn_dists)) if nn_dists else 3.0
    
    feats['mean_intensity'] = float(nodes_df['mean_intensity'].mean()) if 'mean_intensity' in nodes_df.columns else 0.5
    feats['std_intensity'] = float(nodes_df['mean_intensity'].std()) if 'mean_intensity' in nodes_df.columns else 0.2
    feats['snr_mean'] = float(nodes_df['snr'].mean()) if 'snr' in nodes_df.columns else 2.5
    feats['snr_std'] = float(nodes_df['snr'].std()) if 'snr' in nodes_df.columns else 1.0
    p95, p5, p50 = np.percentile(nodes_df['mean_intensity'], [95, 5, 50])
    feats['contrast_ratio'] = float((p95 - p5) / (p50 + 1e-5))
    feats['cell_radius_mean'] = float(nodes_df['estimated_radius_um'].mean()) if 'estimated_radius_um' in nodes_df.columns else 2.35
    feats['cell_volume_mean'] = float(nodes_df['volume_um3'].mean()) if 'volume_um3' in nodes_df.columns else 50.0
    return feats

gt_s_map = df_gt_s.set_index('dataset').to_dict('index')
meta_list = []
for ds in ALL_DATASETS:
    if ds in loaded_nodes_dict:
        n_df = loaded_nodes_dict[ds]
        n_fr = gt_s_map[ds]['n_frames']
        est_n = gt_s_map[ds]['estimated_number_of_nodes']
        meta_list.append(extract_meta(ds, n_df, n_fr, est_n))

df_meta_all = pd.DataFrame(meta_list)
print(f"[*] 14大メタ特徴量抽出完了: 全 {len(df_meta_all)} データセット")

# 4. 単一データセットの 10段階閾値スウィープ関数
THRESHOLDS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050]

# GT エッジ辞書化 (高速参照用)
gt_edge_dict = {}
for ds, group in df_gt_e.groupby('dataset'):
    gt_edge_dict[ds] = set(zip(group['source_id'].astype('int64'), group['target_id'].astype('int64')))

def sweep_single_dataset(ds, nodes_df, est_n, gt_edges_set, tp_map, model_file):
    model = lgb.Booster(model_file=str(model_file))
    feature_cols_4d = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']
    scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
    
    frames = sorted(nodes_df['t'].unique())
    raw_nodes_count = len(nodes_df)
    total_gt_edges = len(gt_edges_set)
    
    # 候補ペアと特徴量を全フレームで事前計算
    frame_pairs = []
    for i in range(len(frames) - 1):
        t_curr, t_next = frames[i], frames[i+1]
        if t_next != t_curr + 1:
            continue
        df_curr = nodes_df[nodes_df['t'] == t_curr].reset_index(drop=True)
        df_next = nodes_df[nodes_df['t'] == t_next].reset_index(drop=True)
        if df_curr.empty or df_next.empty:
            continue
            
        pos_curr = df_curr[['z', 'y', 'x']].values * scale_vec
        pos_next = df_next[['z', 'y', 'x']].values * scale_vec
        sp_dist = cdist(pos_curr, pos_next)
        
        mask = (sp_dist <= 7.0)
        r_idx, c_idx = np.where(mask)
        if len(r_idx) == 0:
            continue
            
        fc = df_curr[feature_cols_4d].values.astype(np.float32)
        fn = df_next[feature_cols_4d].values.astype(np.float32)
        cov = np.cov(np.vstack([fc, fn]), rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        try:
            feat_dist = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)
        except Exception:
            feat_dist = cdist(fc, fn, metric='cityblock')
            
        sp_d = sp_dist[r_idx, c_idx]
        mh_d = feat_dist[r_idx, c_idx]
        dz = (pos_next[c_idx, 0] - pos_curr[r_idx, 0])
        dy = (pos_next[c_idx, 1] - pos_curr[r_idx, 1])
        dx = (pos_next[c_idx, 2] - pos_curr[r_idx, 2])
        d_xy = np.sqrt(dy**2 + dx**2)
        
        int1, int2 = df_curr['mean_intensity'].values[r_idx], df_next['mean_intensity'].values[c_idx]
        snr1, snr2 = df_curr['snr'].values[r_idx], df_next['snr'].values[c_idx]
        rad1, rad2 = df_curr['estimated_radius_um'].values[r_idx], df_next['estimated_radius_um'].values[c_idx]
        vol1, vol2 = df_curr['volume_um3'].values[r_idx], df_next['volume_um3'].values[c_idx]
        zdep1 = df_curr['z_depth_ratio'].values[r_idx] if 'z_depth_ratio' in df_curr.columns else np.zeros(len(r_idx))
        zdep2 = df_next['z_depth_ratio'].values[c_idx] if 'z_depth_ratio' in df_next.columns else np.zeros(len(c_idx))
        dens1 = df_curr['local_density_r15'].values[r_idx] if 'local_density_r15' in df_curr.columns else np.zeros(len(r_idx))
        dens2 = df_next['local_density_r15'].values[c_idx] if 'local_density_r15' in df_next.columns else np.zeros(len(c_idx))
        
        min_sp = np.min(sp_dist, axis=1)
        sp_margin = sp_d - min_sp[r_idx]
        
        ranks = np.zeros(len(r_idx), dtype=np.int32)
        cand_counts = np.zeros(len(r_idx), dtype=np.int32)
        for r in np.unique(r_idx):
            mk = np.where(r_idx == r)[0]
            cand_counts[mk] = len(mk)
            sorted_k = mk[np.argsort(sp_d[mk])]
            ranks[sorted_k] = np.arange(1, len(sorted_k) + 1)
            
        feat_df = pd.DataFrame({
            'spatial_dist': sp_d, 'spatial_dist_xy': d_xy,
            'delta_z_scaled': dz, 'delta_y_scaled': dy, 'delta_x_scaled': dx,
            'abs_delta_z': np.abs(dz), 'spatial_rank': ranks, 'spatial_margin': sp_margin,
            'int_diff': np.abs(int1 - int2), 'int_ratio': int1 / (int2 + 1e-5),
            'snr_diff': np.abs(snr1 - snr2), 'snr_ratio': snr1 / (snr2 + 1e-5), 'snr_min': np.minimum(snr1, snr2),
            'radius_diff': np.abs(rad1 - rad2), 'radius_ratio': rad1 / (rad2 + 1e-5),
            'volume_diff': np.abs(vol1 - vol2), 'volume_ratio': vol1 / (vol2 + 1e-5),
            'z_depth_diff': np.abs(zdep1 - zdep2),
            'density_source': dens1, 'density_target': dens2, 'density_diff': np.abs(dens1 - dens2),
            'mahalanobis_dist': mh_d, 'total_cost_4d': sp_d + mh_d
        })
        
        probs = model.predict(feat_df)
        curr_ids = df_curr['node_id'].values
        next_ids = df_next['node_id'].values
        
        frame_pairs.append({
            'sp_shape': sp_dist.shape,
            'r_idx': r_idx, 'c_idx': c_idx,
            'cand_counts': cand_counts,
            'probs': probs,
            'curr_ids': curr_ids, 'next_ids': next_ids
        })
        
    ds_records = []
    for th in THRESHOLDS:
        th_single = th * 0.5
        edges_all = []
        for fp in frame_pairs:
            cost_mat = np.full(fp['sp_shape'], 1e9, dtype=np.float32)
            th_adaptive = np.where(fp['cand_counts'] == 1, th_single, th)
            valid = (fp['probs'] >= th_adaptive)
            if np.sum(valid) > 0:
                cost_mat[fp['r_idx'][valid], fp['c_idx'][valid]] = -np.log(fp['probs'][valid] + 1e-6)
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e8:
                    edges_all.append((fp['curr_ids'][r], fp['next_ids'][c]))
                    
        # 孤立ノード刈り取り
        if edges_all:
            conn = set([s for s, t in edges_all]).union(set([t for s, t in edges_all]))
            filtered_nodes = len(nodes_df[nodes_df['node_id'].isin(conn)])
        else:
            filtered_nodes = 0
            
        pe_ratio = filtered_nodes / est_n if est_n > 0 else 0.0
        
        # エッジ Recall
        tp_count = 0
        for s, t in edges_all:
            s_gt = tp_map.get(int(s))
            t_gt = tp_map.get(int(t))
            if s_gt is not None and t_gt is not None:
                if (s_gt, t_gt) in gt_edges_set:
                    tp_count += 1
                    
        edge_recall = tp_count / total_gt_edges if total_gt_edges > 0 else 0.0
        penalty = 1.00 if pe_ratio <= 1.00 else max(0.0, 1.0 - (pe_ratio - 1.00))
        kaggle_score = edge_recall * penalty
        in_ideal_pe = (0.90 <= pe_ratio <= 0.99)
        
        ds_records.append({
            'dataset': ds,
            'threshold': th,
            'threshold_single': th_single,
            'raw_nodes': raw_nodes_count,
            'filtered_nodes': filtered_nodes,
            'est_nodes': est_n,
            'pe_ratio': round(pe_ratio, 4),
            'in_ideal_pe': in_ideal_pe,
            'is_positive': bool(in_ideal_pe and kaggle_score >= 0.50),  # Meta-LightGBM 用 正例フラグ
            'pred_edges': len(edges_all),
            'tp_edges': tp_count,
            'gt_edges': total_gt_edges,
            'edge_recall': round(edge_recall, 4),
            'penalty': round(penalty, 4),
            'kaggle_score': round(kaggle_score, 4)
        })
        
    return ds_records

print("\n" + "-" * 90)
print(f"[Step 2] 全 {len(ALL_DATASETS)} データセット × 10 段階閾値スウィープ並列実行 (24コア活用)...")
print("-" * 90)

t0_sweep = time.time()
results_nested = Parallel(n_jobs=8, verbose=5)(
    delayed(sweep_single_dataset)(
        ds=ds,
        nodes_df=loaded_nodes_dict[ds],
        est_n=gt_s_map[ds]['estimated_number_of_nodes'],
        gt_edges_set=gt_edge_dict.get(ds, set()),
        tp_map=node_tp_mapping_dict.get(ds, {}),
        model_file=MODEL_PATH
    ) for ds in ALL_DATASETS if ds in loaded_nodes_dict
)

all_records = [rec for sublist in results_nested for rec in sublist]
t_sweep = time.time() - t0_sweep
print(f"\n[*] 全スウィープ完了: 全 {len(all_records)} レコード生成 / 所要時間: {t_sweep:.2f} 秒 ({t_sweep/60:.2f} 分)")

df_evidence = pd.DataFrame(all_records)
df_evidence.to_csv(OUTPUT_EVIDENCE_CSV, index=False)
print(f"  - [OK] エビデンス CSV 出力完了: {OUTPUT_EVIDENCE_CSV} ({len(df_evidence)} 行)")

# 5. 各データセットの「真の最適閾値 T*」の選定
optimal_records = []
for ds, grp in df_evidence.groupby('dataset'):
    safe = grp[grp['pe_ratio'] <= 0.99]
    if not safe.empty:
        best = safe.sort_values(by=['kaggle_score', 'pe_ratio'], ascending=[False, False]).iloc[0]
    else:
        best = grp.sort_values(by=['kaggle_score', 'pe_ratio'], ascending=[False, True]).iloc[0]
        
    optimal_records.append({
        'dataset': ds,
        'optimal_threshold': best['threshold'],
        'opt_pe_ratio': best['pe_ratio'],
        'opt_recall': best['edge_recall'],
        'opt_score': best['kaggle_score']
    })

df_optimal = pd.DataFrame(optimal_records)
df_optimal.to_csv(OUTPUT_OPTIMAL_CSV, index=False)
print(f"  - [OK] 最適閾値 CSV 出力完了: {OUTPUT_OPTIMAL_CSV} ({len(df_optimal)} 件)")

# 6. Meta-LightGBM モデルの学習 (全 199 データセット)
df_meta_opt = pd.merge(df_meta_all, df_optimal, on='dataset')

feature_cols = [
    'density_per_frame', 'mean_nn_dist', 'median_nn_dist', 'min_nn_dist', 'std_nn_dist',
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
    'num_leaves': 15,
    'min_data_in_leaf': 5,
    'verbosity': -1,
    'seed': 42
}

meta_lgbm = lgb.train(params, train_data, num_boost_round=60)
importances = meta_lgbm.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_gain': importances
}).sort_values(by='importance_gain', ascending=False)

print("\n" + "=" * 90)
print(">>> [Step 3] 全 199 データセットに基づく Meta-LightGBM 特徴量重要度ランキング (Gain)")
print("=" * 90)
print(imp_df.to_string(index=False))

# 重要度グラフ出力
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(imp_df))
plt.barh(y_pos, imp_df['importance_gain'], align='center', color='teal')
plt.yticks(y_pos, imp_df['feature'])
plt.gca().invert_yaxis()
plt.xlabel('Importance (Gain)')
plt.title(f'Meta-LightGBM Feature Importance (All {len(ALL_DATASETS)} Datasets)')
plt.tight_layout()
plt.savefig(str(OUTPUT_IMPORTANCE_PNG), dpi=150)
plt.close()
print(f"  - [OK] 重要度グラフ出力完了: {OUTPUT_IMPORTANCE_PNG}")

print("\n" + "=" * 90)
print(f">>> [SUCCESS] 全 199 データセットの正例・負例データ (全 {len(df_evidence)} レコード) 生成完了！")
print("=" * 90)
