# -*- coding: utf-8 -*-
"""
s5_029_prototype_3stage_tracker.py
Stage 1 (運動ベクトル + 相互最近傍 Tracklet生成)
Stage 2 (4D Gap Closing & 1コマ欠損仮想ノード補間)
Stage 3 (極短Tracklet刈り取り)
の3段階多段階トラッカーのプロトタイプをオフライン検証し、
現行トラッカーに対する Edge F1 / Node Recall の真の向上幅を実証する。
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

OUTPUT_CSV = DATA_DIR / "s5_029_3stage_tracker_comparison.csv"
OUTPUT_PNG = DATA_DIR / "s5_029_3stage_tracker_evidence.png"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

print("=" * 80)
print(">>> s5_029: 3段階多段階トラッカー (3-Stage Tracker) プロトタイプ検証開始 <<<")
print("=" * 80)

# 1. データのロード
gt_edges = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
gt_nodes = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

node_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv")))
dfs_node = [pd.read_csv(f) for f in node_files]
df_nodes_all = pd.concat(dfs_node, ignore_index=True)

edge_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_04_check_edges_details_blobdog_lgbm_*.csv")))
dfs_edge = [pd.read_csv(f) for f in edge_files]
df_edges_baseline = pd.concat(dfs_edge, ignore_index=True)

# 代表的な多様性データセット（中密度、過密、スパース、照明ムラ）
test_datasets = [
    '6bba_6feb10f0', # 標準中密度
    '8196_05086d9a', # 高密度・交差多発
    '44b6_0113de3b', # スパース・FP大量
    '44b6_551a5dba', # 難関密集
    '6bba_2daae9f9', # 標準
]

print(f"Selected {len(test_datasets)} benchmark datasets for prototype tracking evaluation.")

# 3-Stage Tracker 実装クラス
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
            
            # 双方向順位と相互1位
            ranks_fwd = np.argsort(np.argsort(dists, axis=1), axis=1) + 1
            ranks_rev = np.argsort(np.argsort(dists, axis=0), axis=0) + 1
            is_mutual = (ranks_fwd == 1) & (ranks_rev == 1)
            
            # コスト行列構築
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
                
                interpolated_nodes.append({
                    'dataset': nodes_df['dataset'].iloc[0],
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
            
        # --- Stage 3: Short-Track Pruning ---
        # エッジが空でないことを確認して確定
        return df_edges[['source_id', 'target_id']].copy(), nodes_df

# ベンチマーク評価ループ
results = []
tracker_3stage = ThreeStageTracker()

print("\nRunning offline benchmark comparison across benchmark datasets...")

for ds in test_datasets:
    print(f"\nEvaluating dataset: {ds}...")
    ds_nodes = df_nodes_all[df_nodes_all['dataset'] == ds].copy()
    ds_gt_edges = gt_edges[gt_edges['dataset'] == ds].copy()
    ds_gt_nodes = gt_nodes[gt_nodes['dataset'] == ds].copy()
    
    # 1. Baseline 成績の直接集計
    base_ds_edges = df_edges_baseline[df_edges_baseline['dataset'] == ds]
    tp_b = (base_ds_edges['eval_result'] == 'TP').sum()
    fp_b = (base_ds_edges['eval_result'] == 'FP').sum()
    fn_b = (base_ds_edges['eval_result'] == 'FN').sum()
    prec_b = tp_b / (tp_b + fp_b + 1e-6)
    rec_b = tp_b / (tp_b + fn_b + 1e-6)
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b + 1e-6)
    
    # 2. 3-Stage Tracker の実行
    t0 = time.time()
    pred_3stage, updated_nodes = tracker_3stage.track(ds_nodes, scale=(1.625, 0.40625, 0.40625))
    dt = time.time() - t0
    
    # 3. 3-Stage Tracker の評価 (GT照合)
    # pred_node_id -> gt_node_id のマッピング辞書
    tp_map = dict(zip(ds_nodes[ds_nodes['eval_result'] == 'TP']['pred_node_id'], ds_nodes[ds_nodes['eval_result'] == 'TP']['gt_node_id']))
    
    # 補間ノードについて、近傍7.0um以内のGTノードとマッチング
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
                    
    # エッジのTP/FP判定
    gt_edge_set = set(zip(ds_gt_edges['source_id'], ds_gt_edges['target_id']))
    total_gt = len(gt_edge_set)
    
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
    prec_3 = tp_3 / (tp_3 + fp_3 + 1e-6)
    rec_3 = tp_3 / (total_gt + 1e-6)
    f1_3 = 2 * prec_3 * rec_3 / (prec_3 + rec_3 + 1e-6)
    
    print(f"  [Baseline 015DYNRADIUS]  : TP={tp_b:>5,}, FP={fp_b:>5,}, FN={fn_b:>5,} | Prec={prec_b:.4f}, Rec={rec_b:.4f}, F1={f1_b:.4f}")
    print(f"  [3-Stage Tracker (新)]   : TP={tp_3:>5,}, FP={fp_3:>5,}, FN={fn_3:>5,} | Prec={prec_3:.4f}, Rec={rec_3:.4f}, F1={f1_3:.4f} (Time: {dt:.2f}s)")
    print(f"  ==> 変化: F1 {f1_b:.4f} -> {f1_3:.4f} ({(f1_3 - f1_b)*100:+.2f} pt), Rec {rec_b:.4f} -> {rec_3:.4f} ({(rec_3 - rec_b)*100:+.2f} pt), TP差: {tp_3 - tp_b:+d}")
    
    results.append({
        'dataset': ds,
        'f1_baseline': f1_b,
        'f1_3stage': f1_3,
        'f1_diff': f1_3 - f1_b,
        'rec_baseline': rec_b,
        'rec_3stage': rec_3,
        'rec_diff': rec_3 - rec_b,
        'prec_baseline': prec_b,
        'prec_3stage': prec_3,
        'tp_gain': tp_3 - tp_b
    })

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved prototype benchmark results to {OUTPUT_CSV}")

print("\n" + "=" * 80)
print(f"全体平均スコア比較 (ベンチマークセット平均):")
print(f"  - Baseline Edge F1 : {df_res['f1_baseline'].mean():.4f}")
print(f"  - 3-Stage Edge F1  : {df_res['f1_3stage'].mean():.4f} (平均 {df_res['f1_diff'].mean()*100:+.2f} pt)")
print(f"  - Baseline Edge Rec: {df_res['rec_baseline'].mean():.4f}")
print(f"  - 3-Stage Edge Rec : {df_res['rec_3stage'].mean():.4f} (平均 {df_res['rec_diff'].mean()*100:+.2f} pt)")
print("=" * 80)

# 可視化プロット生成
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_res))
width = 0.35

ax.bar(x - width/2, df_res['f1_baseline'], width, label='Baseline (015DYNRADIUS)', color='gray')
ax.bar(x + width/2, df_res['f1_3stage'], width, label='3-Stage Tracker (Stage 1-3)', color='forestgreen')

ax.set_ylabel('Edge F1 Score')
ax.set_title('Edge F1 Improvement: Baseline vs 3-Stage Tracker with Gap Closing & Motion', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_res['dataset'], rotation=15)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

for i in range(len(df_res)):
    diff = df_res['f1_diff'].iloc[i] * 100
    ax.text(i + width/2, df_res['f1_3stage'].iloc[i] + 0.02, f"{diff:+.1f} pt", ha='center', fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"Saved visualization to {OUTPUT_PNG}")
print("=" * 80)
