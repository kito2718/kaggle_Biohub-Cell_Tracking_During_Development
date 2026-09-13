# -*- coding: utf-8 -*-
"""
s5_032_verify_all199_baseline_gap_closing.py
全199データセット全数に対して、
「現行 Baseline (LightGBM) のエッジ」に「Gap Closing (1コマ欠損補間)」を適用し、
悪化データセット数がゼロ (0敗) になるか、および全体の TP Gain を完全全件集計する。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"
SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

gt_edges = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
gt_nodes = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

edge_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_04_check_edges_details_blobdog_lgbm_*.csv")))
df_edges_base = pd.concat([pd.read_csv(f) for f in edge_files], ignore_index=True)

node_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv")))
df_nodes_all = pd.concat([pd.read_csv(f) for f in node_files], ignore_index=True)

all_datasets = sorted(list(gt_edges['dataset'].unique()))

results = []

for idx, ds in enumerate(all_datasets):
    ds_gt_e = gt_edges[gt_edges['dataset'] == ds]
    ds_gt_n = gt_nodes[gt_nodes['dataset'] == ds]
    ds_base_e = df_edges_base[df_edges_base['dataset'] == ds]
    ds_nodes = df_nodes_all[df_nodes_all['dataset'] == ds].copy()
    
    total_gt = len(ds_gt_e)
    if total_gt == 0:
        continue
        
    base_edges = ds_base_e[ds_base_e['eval_result'] != 'FN'][['source_id', 'target_id']].copy()
    pos_dict = dict(zip(ds_nodes['pred_node_id'], zip(ds_nodes['t'], ds_nodes['pred_z']*SCALE_VEC[0], ds_nodes['pred_y']*SCALE_VEC[1], ds_nodes['pred_x']*SCALE_VEC[2])))
    
    t_src = [pos_dict[s][0] if s in pos_dict else -1 for s in base_edges['source_id']]
    t_tgt = [pos_dict[t][0] if t in pos_dict else -1 for t in base_edges['target_id']]
    base_edges['t_source'] = t_src
    base_edges['t_target'] = t_tgt
    
    in_deg = set(base_edges['target_id'].values)
    out_deg = set(base_edges['source_id'].values)
    ends = base_edges[~base_edges['target_id'].isin(out_deg)]
    starts = base_edges[~base_edges['source_id'].isin(in_deg)]
    
    next_v_id = int(ds_nodes['pred_node_id'].max() + 1000000)
    gap_edges = []
    interpolated_nodes = []
    
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
            if dist <= 10.0 and dist < min_gap_dist:
                min_gap_dist = dist
                best_start_id = s_id
                
        if best_start_id is not None:
            t_s, z_s, y_s, x_s = pos_dict[best_start_id]
            v_z, v_y, v_x = (z_e + z_s)/2.0, (y_e + y_s)/2.0, (x_e + x_s)/2.0
            v_id = next_v_id
            next_v_id += 1
            interpolated_nodes.append({
                'dataset': ds, 'pred_node_id': v_id, 't': t_e + 1,
                'pred_z': v_z / SCALE_VEC[0], 'pred_y': v_y / SCALE_VEC[1], 'pred_x': v_x / SCALE_VEC[2],
                'eval_result': 'INTERP'
            })
            gap_edges.append({'source_id': e_id, 'target_id': v_id, 't_source': t_e, 't_target': t_e + 1})
            gap_edges.append({'source_id': v_id, 'target_id': best_start_id, 't_source': t_e + 1, 't_target': t_s})
            
    final_edges = pd.concat([base_edges, pd.DataFrame(gap_edges)], ignore_index=True) if gap_edges else base_edges
    
    tp_b = int((ds_base_e['eval_result'] == 'TP').sum())
    rec_b = tp_b / total_gt
    
    tp_map = dict(zip(ds_nodes[ds_nodes['eval_result'] == 'TP']['pred_node_id'], ds_nodes[ds_nodes['eval_result'] == 'TP']['gt_node_id']))
    if interpolated_nodes:
        for r_in in interpolated_nodes:
            t_val = r_in['t']
            gt_t = ds_gt_n[ds_gt_n['t'] == t_val]
            if not gt_t.empty:
                iz, iy, ix = r_in['pred_z']*SCALE_VEC[0], r_in['pred_y']*SCALE_VEC[1], r_in['pred_x']*SCALE_VEC[2]
                gz = gt_t['z'].values * SCALE_VEC[0]
                gy = gt_t['y'].values * SCALE_VEC[1]
                gx = gt_t['x'].values * SCALE_VEC[2]
                dists = np.sqrt((gz - iz)**2 + (gy - iy)**2 + (gx - ix)**2)
                min_i = np.argmin(dists)
                if dists[min_i] <= 7.0:
                    tp_map[r_in['pred_node_id']] = gt_t['node_id'].values[min_i]
                    
    gt_pairs = set(zip(ds_gt_e['source_id'], ds_gt_e['target_id']))
    matched_gt = set()
    tp_new = 0
    for _, r in final_edges.iterrows():
        g_s = tp_map.get(r['source_id'])
        g_t = tp_map.get(r['target_id'])
        if g_s and g_t and (g_s, g_t) in gt_pairs and (g_s, g_t) not in matched_gt:
            tp_new += 1
            matched_gt.add((g_s, g_t))
            
    rec_new = tp_new / total_gt
    gain = tp_new - tp_b
    results.append({
        'dataset': ds,
        'total_gt': total_gt,
        'tp_baseline': tp_b,
        'tp_new': tp_new,
        'tp_gain': gain,
        'rec_baseline': rec_b,
        'rec_new': rec_new
    })

df_res = pd.DataFrame(results)
out_csv = DATA_DIR / "s5_032_baseline_plus_gap_closing_all199.csv"
df_res.to_csv(out_csv, index=False)
