# -*- coding: utf-8 -*-
"""
s5_015_evaluate_dynamic_search_radius_199.py
全 199 データセットを対象とした Step 2 (動的探索半径 5.0〜8.5μm) vs 従来 (固定 7.0μm)
完全実走・定量比較評価スクリプト。

24 コア CPU マルチプロセス並列処理 (Parallel) により高速実行。
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
from joblib import Parallel, delayed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
DATA_DIR = BASE_DIR / "s5_analysys_data"
WORKING_DIR = BASE_DIR / "working"
EDGE_MODEL_PATH = DATA_DIR / "s5_003_tracking_edge_lgbm.txt"
TH_MODEL_PATH = DATA_DIR / "lightgbm_adaptive_th.txt"

OUTPUT_VALIDATION_CSV = DATA_DIR / "s5_015_dynamic_search_radius_validation.csv"

print("=" * 90)
print(">>> s5_015: 全 199 データセット Step 2 (動的探索半径) 定量比較評価")
print("=" * 90)

# 1. GT データのロード
gt_summary_file = WORKING_DIR / "s5_gt_summary.csv"
gt_edges_file = WORKING_DIR / "s5_gt_edges.csv"

df_gt_s = pd.read_csv(gt_summary_file)
df_gt_e = pd.read_csv(gt_edges_file)
ALL_DATASETS = sorted(df_gt_s['dataset'].unique())
print(f"[*] 全 GT データセット数: {len(ALL_DATASETS)} 件 / GT エッジ総数: {len(df_gt_e):,} 本")

# 2. 検出ノードのロード (全 199 データセット)
t0_load = time.time()
node_chunk_files = sorted(WORKING_DIR.glob("s5_002-2SPDUP(CCC)_FIXTRACKANDCULL_01_detect_nodes_pred_blobdog_lgbm_*.csv"))
if not node_chunk_files:
    node_chunk_files = sorted(WORKING_DIR.glob("s5_ADDLGBM_01_detect_nodes_pred_blobdog_lgbm_*.csv"))
print(f"[*] 検出ノードチャンクファイル読み込み開始 ({len(node_chunk_files)} ファイル)...")

loaded_nodes_dict = {}
for cf in node_chunk_files:
    df_chunk = pd.read_csv(cf)
    for ds, group in df_chunk.groupby('dataset'):
        loaded_nodes_dict[ds] = group.copy()

print(f"[*] 検出ノードロード完了: {len(loaded_nodes_dict)} / {len(ALL_DATASETS)} データセット ({time.time()-t0_load:.2f} 秒)")

# TP マッピングのロード
detail_chunk_files = sorted(WORKING_DIR.glob("s5_002-2SPDUP(CCC)_FIXTRACKANDCULL_02_check_nodes_details_blobdog_lgbm_*.csv"))
if not detail_chunk_files:
    detail_chunk_files = sorted(WORKING_DIR.glob("s5_ADDLGBM_02_check_nodes_details_blobdog_lgbm_*.csv"))
node_tp_mapping_dict = {}
for df_f in detail_chunk_files:
    df_det = pd.read_csv(df_f)
    tp_only = df_det[df_det['eval_result'] == 'TP']
    for ds, group in tp_only.groupby('dataset'):
        node_tp_mapping_dict[ds] = dict(zip(group['pred_node_id'].astype('int64'), group['gt_node_id'].astype('int64')))

print(f"[*] ノード TP マッピングロード完了: {len(node_tp_mapping_dict)} データセット")

# GT エッジ辞書
gt_edge_dict = {}
for ds, group in df_gt_e.groupby('dataset'):
    gt_edge_dict[ds] = set(zip(group['source_id'].astype('int64'), group['target_id'].astype('int64')))

gt_s_map = df_gt_s.set_index('dataset').to_dict('index')

# 動的半径推定関数
def predict_adaptive_radius(nodes_df, scale_v, default_r=7.0, min_r=5.0, max_r=8.5):
    if nodes_df is None or len(nodes_df) < 2:
        return float(default_r)
    frames = sorted(nodes_df['t'].unique())
    if len(frames) < 2:
        return float(default_r)
    inter_dists = []
    for i in range(min(3, len(frames) - 1)):
        t_c, t_n = frames[i], frames[i+1]
        if t_n != t_c + 1:
            continue
        p_c = nodes_df[nodes_df['t'] == t_c][['z', 'y', 'x']].values * scale_v
        p_n = nodes_df[nodes_df['t'] == t_n][['z', 'y', 'x']].values * scale_v
        if len(p_c) > 0 and len(p_n) > 0:
            dmat = cdist(p_c, p_n)
            inter_dists.extend(np.min(dmat, axis=1))
    if not inter_dists:
        return float(default_r)
    speed_med = float(np.median(inter_dists))
    if speed_med <= 1.8:
        r = min_r
    elif speed_med >= 2.8:
        r = min(max_r, 8.0 + (speed_med - 2.8) * 0.25)
    else:
        r = min_r + (speed_med - 1.8) / (2.8 - 1.8) * (8.0 - min_r)
    return float(np.clip(r, min_r, max_r)), speed_med

def run_tracking_single(nodes_df, max_r, booster_edge, th, th_single, scale_vec, feature_cols_4d):
    frames = sorted(nodes_df['t'].unique())
    avail_feats = [c for c in feature_cols_4d if c in nodes_df.columns]
    pred_edges = []
    n_cands = 0
    t0 = time.time()
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
        spatial_dist = cdist(pos_curr, pos_next)
        
        mask = (spatial_dist <= max_r)
        r_idx, c_idx = np.where(mask)
        if len(r_idx) == 0:
            continue
        n_cands += len(r_idx)
        
        fc = df_curr[avail_feats].values.astype(np.float32)
        fn = df_next[avail_feats].values.astype(np.float32)
        cov = np.cov(np.vstack([fc, fn]), rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        try:
            feat_dist = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)
        except Exception:
            feat_dist = cdist(fc, fn, metric='cityblock')
            
        sp_d = spatial_dist[r_idx, c_idx]
        mh_d = feat_dist[r_idx, c_idx]
        total_cost_4d = sp_d + mh_d
        
        dz = (df_next['z'].values[c_idx] - df_curr['z'].values[r_idx]) * scale_vec[0]
        dy = (df_next['y'].values[c_idx] - df_curr['y'].values[r_idx]) * scale_vec[1]
        dx = (df_next['x'].values[c_idx] - df_curr['x'].values[r_idx]) * scale_vec[2]
        d_xy = np.sqrt(dy**2 + dx**2)
        
        int1, int2 = df_curr['mean_intensity'].values[r_idx], df_next['mean_intensity'].values[c_idx]
        snr1, snr2 = df_curr['snr'].values[r_idx], df_next['snr'].values[c_idx]
        rad1, rad2 = df_curr['estimated_radius_um'].values[r_idx], df_next['estimated_radius_um'].values[c_idx]
        vol1, vol2 = df_curr['volume_um3'].values[r_idx], df_next['volume_um3'].values[c_idx]
        zdep1 = df_curr['z_depth_ratio'].values[r_idx] if 'z_depth_ratio' in df_curr.columns else np.zeros(len(r_idx))
        zdep2 = df_next['z_depth_ratio'].values[c_idx] if 'z_depth_ratio' in df_next.columns else np.zeros(len(c_idx))
        dens1 = df_curr['local_density_r15'].values[r_idx] if 'local_density_r15' in df_curr.columns else np.zeros(len(r_idx))
        dens2 = df_next['local_density_r15'].values[c_idx] if 'local_density_r15' in df_next.columns else np.zeros(len(c_idx))
        
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
        
        probs = booster_edge.predict(feat_df)
        
        cand_counts = np.zeros(len(r_idx), dtype=np.int32)
        for u_r in np.unique(r_idx):
            match_k = np.where(r_idx == u_r)[0]
            cand_counts[match_k] = len(match_k)
        th_adaptive = np.where(cand_counts == 1, th_single, th)
        
        cost_mat = np.full(spatial_dist.shape, 1e9, dtype=np.float32)
        valid_mask = (probs >= th_adaptive)
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
                pred_edges.append((int(curr_ids[r]), int(next_ids[c])))
                
    elapsed = time.time() - t0
    return pred_edges, n_cands, elapsed

def evaluate_single_dataset(ds):
    if ds not in loaded_nodes_dict or ds not in gt_s_map:
        return None
    nodes_df = loaded_nodes_dict[ds]
    gt_edges_set = gt_edge_dict.get(ds, set())
    tp_map = node_tp_mapping_dict.get(ds, {})
    est_nodes = gt_s_map[ds]['estimated_number_of_nodes']
    n_frames = gt_s_map[ds]['n_frames']
    scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
    feature_cols_4d = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']
    
    # Load boosters in thread/process
    booster_edge = lgb.Booster(model_file=str(EDGE_MODEL_PATH))
    th_booster = lgb.Booster(model_file=str(TH_MODEL_PATH))
    
    # 1. Predict adaptive threshold
    density = len(nodes_df) / max(1, n_frames)
    mean_int = float(nodes_df['mean_intensity'].mean())
    std_int = float(nodes_df['mean_intensity'].std())
    snr_m = float(nodes_df['snr'].mean())
    snr_s = float(nodes_df['snr'].std())
    rad_m = float(nodes_df['estimated_radius_um'].mean())
    vol_m = float(nodes_df['volume_um3'].mean())
    
    frames = sorted(nodes_df['t'].unique())[:3]
    nn_dists = []
    for t_f in frames:
        df_t = nodes_df[nodes_df['t'] == t_f]
        if len(df_t) > 1:
            pos = df_t[['z', 'y', 'x']].values * scale_vec
            dmat = cdist(pos, pos)
            np.fill_diagonal(dmat, 1e9)
            nn_dists.extend(np.min(dmat, axis=1))
    mean_nn = float(np.mean(nn_dists)) if nn_dists else 10.0
    med_nn = float(np.median(nn_dists)) if nn_dists else 10.0
    min_nn = float(np.min(nn_dists)) if nn_dists else 1.0
    std_nn = float(np.std(nn_dists)) if nn_dists else 3.0
    p95, p5, p50 = np.percentile(nodes_df['mean_intensity'], [95, 5, 50])
    contrast = float((p95 - p5) / (p50 + 1e-5))
    
    meta_vec = np.array([[
        std_int, std_nn, rad_m, contrast, mean_int, density,
        snr_s, vol_m, min_nn, mean_nn, med_nn, snr_m
    ]], dtype=np.float32)
    pred_log10 = float(np.asarray(th_booster.predict(meta_vec)).ravel()[0])
    th = float(np.clip(10 ** pred_log10, 0.0002, 0.050))
    th_single = th * 0.5
    
    # 2. Estimate adaptive radius
    adaptive_r, speed_med = predict_adaptive_radius(nodes_df, scale_vec)
    
    # Run Fixed 7.0um
    edges_fixed, cand_fixed, time_fixed = run_tracking_single(
        nodes_df, 7.0, booster_edge, th, th_single, scale_vec, feature_cols_4d
    )
    # Run Adaptive Radius
    edges_adapt, cand_adapt, time_adapt = run_tracking_single(
        nodes_df, adaptive_r, booster_edge, th, th_single, scale_vec, feature_cols_4d
    )
    
    # Eval Fixed
    tp_fixed = sum(1 for s_p, t_p in edges_fixed if s_p in tp_map and t_p in tp_map and (tp_map[s_p], tp_map[t_p]) in gt_edges_set)
    n_gt = len(gt_edges_set)
    rec_fixed = tp_fixed / max(1, n_gt)
    pe_fixed = len(edges_fixed) / max(1, est_nodes)
    pen_fixed = 1.0 if pe_fixed <= 1.0 else float(np.exp(-0.5 * ((pe_fixed - 1.0) / 0.15)**2))
    sc_fixed = pen_fixed * rec_fixed
    
    # Eval Adaptive
    tp_adapt = sum(1 for s_p, t_p in edges_adapt if s_p in tp_map and t_p in tp_map and (tp_map[s_p], tp_map[t_p]) in gt_edges_set)
    rec_adapt = tp_adapt / max(1, n_gt)
    pe_adapt = len(edges_adapt) / max(1, est_nodes)
    pen_adapt = 1.0 if pe_adapt <= 1.0 else float(np.exp(-0.5 * ((pe_adapt - 1.0) / 0.15)**2))
    sc_adapt = pen_adapt * rec_adapt
    
    return {
        'dataset': ds,
        'speed_med': speed_med,
        'adaptive_r': adaptive_r,
        'n_nodes': len(nodes_df),
        'est_nodes': est_nodes,
        'n_gt_edges': n_gt,
        'cand_fixed': cand_fixed,
        'cand_adapt': cand_adapt,
        'pred_fixed': len(edges_fixed),
        'pred_adapt': len(edges_adapt),
        'tp_fixed': tp_fixed,
        'tp_adapt': tp_adapt,
        'recall_fixed': rec_fixed,
        'recall_adapt': rec_adapt,
        'pe_fixed': pe_fixed,
        'pe_adapt': pe_adapt,
        'pen_fixed': pen_fixed,
        'pen_adapt': pen_adapt,
        'score_fixed': sc_fixed,
        'score_adapt': sc_adapt,
        'time_fixed': time_fixed,
        'time_adapt': time_adapt
    }

print(f"[*] 全 199 データセットの並列評価開始 (n_jobs=12)...")
t0_eval = time.time()
results = Parallel(n_jobs=12, verbose=5)(
    delayed(evaluate_single_dataset)(ds) for ds in ALL_DATASETS
)
results = [r for r in results if r is not None]
df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_VALIDATION_CSV, index=False)
print(f"[*] 全 199 データセット評価完了 ({time.time() - t0_eval:.2f} 秒) -> {OUTPUT_VALIDATION_CSV}")

# サマリー統計出力
mean_sc_fix = df_res['score_fixed'].mean()
mean_sc_ad = df_res['score_adapt'].mean()
mean_rec_fix = df_res['recall_fixed'].mean()
mean_rec_ad = df_res['recall_adapt'].mean()
mean_pe_fix = df_res['pe_fixed'].mean()
mean_pe_ad = df_res['pe_adapt'].mean()
mean_pen_fix = df_res['pen_fixed'].mean()
mean_pen_ad = df_res['pen_adapt'].mean()
pe_over_fix = (df_res['pe_fixed'] > 1.00).sum()
pe_over_ad = (df_res['pe_adapt'] > 1.00).sum()
tot_cand_fix = df_res['cand_fixed'].sum()
tot_cand_ad = df_res['cand_adapt'].sum()
tot_time_fix = df_res['time_fixed'].sum()
tot_time_ad = df_res['time_adapt'].sum()

print("\n" + "=" * 80)
print(">>> 全 199 データセット Step 2 定量比較検証サマリー <<<")
print("=" * 80)
print(f"指標                         固定 7.0um         動的適応半径 (5.0~8.5um)   差分 (Delta)")
print(f"--------------------------------------------------------------------------------")
print(f"総合 Kaggle スコア         : {mean_sc_fix:.4f}           {mean_sc_ad:.4f}                  {mean_sc_ad - mean_sc_fix:+.4f} pt ({((mean_sc_ad-mean_sc_fix)/mean_sc_fix)*100:+.2f}%)")
print(f"平均 Edge Recall           : {mean_rec_fix:.4f}           {mean_rec_ad:.4f}                  {mean_rec_ad - mean_rec_fix:+.4f} pt ({((mean_rec_ad-mean_rec_fix)/mean_rec_fix)*100:+.2f}%)")
print(f"平均 P/E 比                : {mean_pe_fix:.4f}           {mean_pe_ad:.4f}                  {mean_pe_ad - mean_pe_fix:+.4f} pt")
print(f"平均 ペナルティ係数        : {mean_pen_fix:.4f}           {mean_pen_ad:.4f}                  {mean_pen_ad - mean_pen_fix:+.4f} pt")
print(f"P/E > 1.00 超過件数        : {pe_over_fix}/199 ({pe_over_fix/199*100:.1f}%)    {pe_over_ad}/199 ({pe_over_ad/199*100:.1f}%)     {pe_over_ad - pe_over_fix:+d} 件")
print(f"総候補ペア数 (Candidates)  : {tot_cand_fix:,}     {tot_cand_ad:,}          {tot_cand_ad - tot_cand_fix:+,} ({(tot_cand_ad-tot_cand_fix)/tot_cand_fix*100:+.2f}%)")
print(f"総追跡所要時間             : {tot_time_fix:.1f} 秒         {tot_time_ad:.1f} 秒              {tot_time_ad - tot_time_fix:+.1f} 秒 ({(tot_time_ad-tot_time_fix)/tot_time_fix*100:+.2f}%)")
print("=" * 80)
