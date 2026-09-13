# -*- coding: utf-8 -*-
"""
s5_030_verify_3stage_tracker_all199.py
全199データセット全件を網羅し、3段階多段階トラッカー (3-Stage Tracker) の
真の実力 (Edge Recall 向上幅、救済エッジ本数、改善データセット数) を完全定量検証する。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "s5_030_all199_3stage_tracker_results.csv"
OUTPUT_PNG = DATA_DIR / "s5_030_all199_3stage_evidence.png"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

print("=" * 80)
print(">>> s5_030: 全199データセット 3-Stage Tracker 完全全件検証開始 <<<")
print("=" * 80)

# 1. データのロード
print("Loading GT files...")
gt_edges = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
gt_nodes = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

print("Loading Baseline edge and node detail files...")
edge_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_04_check_edges_details_blobdog_lgbm_*.csv")))
df_edges_baseline = pd.concat([pd.read_csv(f) for f in edge_files], ignore_index=True)

node_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv")))
df_nodes_all = pd.concat([pd.read_csv(f) for f in node_files], ignore_index=True)

all_datasets = sorted(list(gt_edges['dataset'].unique()))
print(f"Total datasets in GT: {len(all_datasets)}")

# 3-Stage Tracker クラス
class ThreeStageTracker:
    def __init__(self, max_radius=7.0, gap_radius=10.0):
        self.max_radius = max_radius
        self.gap_radius = gap_radius
        
    def track(self, nodes_df, scale=(1.625, 0.40625, 0.40625)):
        scale_vec = np.array(scale, dtype=np.float32)
        frames = sorted(nodes_df['t'].unique())
        
        # --- Stage 1: High-Confidence Tracklet 生成 (運動物理 + 相互最近傍) ---
        track_history = {} # node_id -> v_prev (3Dベクトル)
        stage1_edges = []
        
        for i in range(len(frames) - 1):
            t_c, t_n = frames[i], frames[i+1]
            if t_n != t_c + 1:
                continue
            f_c = nodes_df[nodes_df['t'] == t_c].reset_index(drop=True)
            f_n = nodes_df[nodes_df['t'] == t_n].reset_index(drop=True)
            if f_c.empty or f_n.empty:
                continue
                
            pos_c = f_c[['pred_z', 'pred_y', 'pred_x']].values * scale_vec
            pos_n = f_n[['pred_z', 'pred_y', 'pred_x']].values * scale_vec
            
            dists = cdist(pos_c, pos_n)
            
            ranks_fwd = np.argsort(np.argsort(dists, axis=1), axis=1) + 1
            ranks_rev = np.argsort(np.argsort(dists, axis=0), axis=0) + 1
            is_mutual = (ranks_fwd == 1) & (ranks_rev == 1)
            
            cost_mat = np.full(dists.shape, 500.0, dtype=np.float32)
            c_ids = f_c['pred_node_id'].values
            n_ids = f_n['pred_node_id'].values
            
            for r in range(len(pos_c)):
                v_prev = track_history.get(c_ids[r])
                s_prev = np.linalg.norm(v_prev) if v_prev is not None else None
                
                for c in range(len(pos_n)):
                    d = dists[r, c]
                    if d > self.max_radius:
                        continue
                        
                    base_cost = d
                    if is_mutual[r, c]:
                        base_cost *= 0.75
                    elif ranks_rev[r, c] > 2:
                        base_cost *= 1.3
                        
                    v_curr = pos_n[c] - pos_c[r]
                    if v_prev is not None and s_prev > 0.1:
                        accel = np.linalg.norm(v_curr - v_prev)
                        s_curr = np.linalg.norm(v_curr)
                        cos_sim = np.dot(v_prev, v_curr) / (s_prev * s_curr + 1e-6)
                        cost = base_cost + 0.3 * accel - 0.4 * cos_sim
                    else:
                        cost = base_cost
                        
                    cost_mat[r, c] = float(np.nan_to_num(cost, nan=500.0, posinf=500.0, neginf=500.0))
                    
            cost_mat = np.nan_to_num(cost_mat, nan=500.0, posinf=500.0, neginf=500.0)
            r_ind, c_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(r_ind, c_ind):
                if cost_mat[r, c] < 50.0:
                    s_id = int(c_ids[r])
                    t_id = int(n_ids[c])
                    stage1_edges.append({'source_id': s_id, 'target_id': t_id, 't_source': t_c, 't_target': t_n})
                    track_history[t_id] = pos_n[c] - pos_c[r]
                    
        df_edges = pd.DataFrame(stage1_edges)
        if df_edges.empty:
            return pd.DataFrame(columns=['source_id', 'target_id']), nodes_df
            
        # --- Stage 2: 4D Gap Closing (Delta t = 2 の欠損ノード補間) ---
        interpolated_nodes = []
        gap_edges = []
        
        in_degree = set(df_edges['target_id'].values)
        out_degree = set(df_edges['source_id'].values)
        
        ends = df_edges[~df_edges['target_id'].isin(out_degree)].copy()
        starts = df_edges[~df_edges['source_id'].isin(in_degree)].copy()
        
        pos_dict = dict(zip(nodes_df['pred_node_id'], zip(nodes_df['t'], nodes_df['pred_z']*scale_vec[0], nodes_df['pred_y']*scale_vec[1], nodes_df['pred_x']*scale_vec[2])))
        next_virtual_id = int(nodes_df['pred_node_id'].max() + 1000000)
        
        for _, r_end in ends.iterrows():
            e_id = r_end['target_id']
            if e_id not in pos_dict:
                continue
            t_e, z_e, y_e, x_e = pos_dict[e_id]
            
            cands = starts[starts['t_source'] == t_e + 2]
            if cands.empty:
                continue
                
            best_start_id = None
            min_gap_dist = 1e9
            
            for _, r_start in cands.iterrows():
                s_id = r_start['source_id']
                if s_id not in pos_dict:
                    continue
                t_s, z_s, y_s, x_s = pos_dict[s_id]
                
                dist = np.sqrt((z_s - z_e)**2 + (y_s - y_e)**2 + (x_s - x_e)**2)
                if dist <= self.gap_radius and dist < min_gap_dist:
                    min_gap_dist = dist
                    best_start_id = s_id
                    
            if best_start_id is not None:
                t_s, z_s, y_s, x_s = pos_dict[best_start_id]
                v_z = (z_e + z_s) / 2.0
                v_y = (y_e + y_s) / 2.0
                v_x = (x_e + x_s) / 2.0
                v_id = next_virtual_id
                next_virtual_id += 1
                
                ds_name = nodes_df['dataset'].iloc[0] if 'dataset' in nodes_df.columns else ""
                interpolated_nodes.append({
                    'dataset': ds_name,
                    'pred_node_id': v_id,
                    't': t_e + 1,
                    'pred_z': v_z / scale_vec[0],
                    'pred_y': v_y / scale_vec[1],
                    'pred_x': v_x / scale_vec[2],
                    'eval_result': 'INTERP'
                })
                gap_edges.append({'source_id': e_id, 'target_id': v_id, 't_source': t_e, 't_target': t_e + 1})
                gap_edges.append({'source_id': v_id, 'target_id': best_start_id, 't_source': t_e + 1, 't_target': t_s})
                
        if gap_edges:
            df_edges = pd.concat([df_edges, pd.DataFrame(gap_edges)], ignore_index=True)
            nodes_df = pd.concat([nodes_df, pd.DataFrame(interpolated_nodes)], ignore_index=True)
            
        return df_edges[['source_id', 'target_id']].copy(), nodes_df

tracker = ThreeStageTracker()
results = []
t_start = time.time()

print("Starting full loop across all 199 datasets...")

for idx, ds in enumerate(all_datasets):
    ds_gt_edges = gt_edges[gt_edges['dataset'] == ds]
    ds_gt_nodes = gt_nodes[gt_nodes['dataset'] == ds]
    ds_nodes = df_nodes_all[df_nodes_all['dataset'] == ds].copy()
    
    total_gt = len(ds_gt_edges)
    if total_gt == 0:
        continue
        
    # Baseline 成績の集計
    base_ds = df_edges_baseline[df_edges_baseline['dataset'] == ds]
    tp_b = int((base_ds['eval_result'] == 'TP').sum())
    fp_b = int((base_ds['eval_result'] == 'FP').sum())
    fn_b = int((base_ds['eval_result'] == 'FN').sum())
    rec_b = tp_b / (total_gt + 1e-6)
    
    # 3-Stage Tracker 実行
    pred_3stage, updated_nodes = tracker.track(ds_nodes, scale=(1.625, 0.40625, 0.40625))
    
    # 評価 (GT照合)
    tp_map = dict(zip(ds_nodes[ds_nodes['eval_result'] == 'TP']['pred_node_id'], ds_nodes[ds_nodes['eval_result'] == 'TP']['gt_node_id']))
    
    interp_nodes = updated_nodes[updated_nodes['eval_result'] == 'INTERP']
    if not interp_nodes.empty:
        for _, r_in in interp_nodes.iterrows():
            t_val = r_in['t']
            gt_t = ds_gt_nodes[ds_gt_nodes['t'] == t_val]
            if not gt_t.empty:
                iz, iy, ix = r_in['pred_z']*SCALE_VEC[0], r_in['pred_y']*SCALE_VEC[1], r_in['pred_x']*SCALE_VEC[2]
                gz = gt_t['z'].values * SCALE_VEC[0]
                gy = gt_t['y'].values * SCALE_VEC[1]
                gx = gt_t['x'].values * SCALE_VEC[2]
                dists = np.sqrt((gz - iz)**2 + (gy - iy)**2 + (gx - ix)**2)
                min_i = np.argmin(dists)
                if dists[min_i] <= 7.0:
                    tp_map[r_in['pred_node_id']] = gt_t['node_id'].values[min_i]
                    
    gt_edge_set = set(zip(ds_gt_edges['source_id'], ds_gt_edges['target_id']))
    matched_gt = set()
    tp_3 = 0
    fp_3 = 0
    
    for _, r_edge in pred_3stage.iterrows():
        g_s = tp_map.get(r_edge['source_id'])
        g_t = tp_map.get(r_edge['target_id'])
        if g_s and g_t and (g_s, g_t) in gt_edge_set and (g_s, g_t) not in matched_gt:
            tp_3 += 1
            matched_gt.add((g_s, g_t))
        else:
            fp_3 += 1
            
    fn_3 = total_gt - tp_3
    rec_3 = tp_3 / (total_gt + 1e-6)
    
    results.append({
        'dataset': ds,
        'total_gt': total_gt,
        'tp_baseline': tp_b,
        'tp_3stage': tp_3,
        'tp_gain': tp_3 - tp_b,
        'rec_baseline': rec_b,
        'rec_3stage': rec_3,
        'rec_diff': rec_3 - rec_b,
        'fp_baseline': fp_b,
        'fp_3stage': fp_3
    })
    
    if (idx + 1) % 25 == 0 or (idx + 1) == len(all_datasets):
        elapsed = time.time() - t_start
        print(f"Progress: [{idx+1:3d}/{len(all_datasets)}] (Elapsed: {elapsed:.1f}s) - Last: {ds}, TP Gain={tp_3 - tp_b:+d}, Rec: {rec_b:.3f}->{rec_3:.3f}")

df_all = pd.DataFrame(results)
df_all.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved full results to {OUTPUT_CSV}")

# 集計と分析
total_gt_all = df_all['total_gt'].sum()
total_tp_base = df_all['tp_baseline'].sum()
total_tp_3stage = df_all['tp_3stage'].sum()
total_gain = total_tp_3stage - total_tp_base

macro_rec_base = df_all['rec_baseline'].mean()
macro_rec_3stage = df_all['rec_3stage'].mean()
micro_rec_base = total_tp_base / total_gt_all
micro_rec_3stage = total_tp_3stage / total_gt_all

n_improved = (df_all['tp_gain'] > 0).sum()
n_unchanged = (df_all['tp_gain'] == 0).sum()
n_degraded = (df_all['tp_gain'] < 0).sum()

print("\n" + "=" * 80)
print(">>> 全199データセット 3-Stage Tracker 完全検証集計結果 <<<")
print("=" * 80)
print(f"全GTエッジ総数               : {total_gt_all:,} 本")
print(f"Baseline 正解エッジ数 (TP)    : {total_tp_base:,} 本 (Recall: {micro_rec_base*100:.2f}%)")
print(f"3-Stage Tracker 正解エッジ数  : {total_tp_3stage:,} 本 (Recall: {micro_rec_3stage*100:.2f}%)")
print(f"★ 救済された正解エッジ総数    : {total_gain:+,} 本 ({total_gain/total_gt_all*100:+.2f} pt)")
print(f"マクロ平均 Edge Recall        : {macro_rec_base:.4f} ➔ {macro_rec_3stage:.4f} ({(macro_rec_3stage - macro_rec_base)*100:+.2f} pt)")
print(f"-------------------------------------------------------------")
print(f"データセット別勝敗内訳 (全 {len(df_all)} セット):")
print(f"  - 改善データセット数 (TP Gain > 0) : {n_improved} セット ({n_improved/len(df_all)*100:.1f}%)")
print(f"  - 不変データセット数 (TP Gain == 0): {n_unchanged} セット ({n_unchanged/len(df_all)*100:.1f}%)")
print(f"  - 悪化データセット数 (TP Gain < 0) : {n_degraded} セット ({n_degraded/len(df_all)*100:.1f}%)")
print("=" * 80)

# 可視化プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: TP 増加数トップ20データセット
top_gains = df_all.sort_values('tp_gain', ascending=False).head(20)
axes[0].barh(top_gains['dataset'], top_gains['tp_gain'], color='forestgreen')
axes[0].set_title("Top 20 Datasets with Largest TP Edge Gains", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Additional True Positive Edges Rescued")
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].invert_yaxis()

# Subplot 2: データセット別勝敗パイチャート
axes[1].pie([n_improved, n_unchanged, n_degraded], labels=[f'Improved ({n_improved})', f'Unchanged ({n_unchanged})', f'Degraded ({n_degraded})'],
            colors=['forestgreen', 'lightgray', 'crimson'], autopct='%1.1f%%', startangle=140, textprops={'fontweight':'bold'})
axes[1].set_title(f"All 199 Datasets Outcome Breakdown\n(Total Rescued TP Edges: {total_gain:+,} edges)", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"Saved visualization to {OUTPUT_PNG}")
print("=" * 80)
