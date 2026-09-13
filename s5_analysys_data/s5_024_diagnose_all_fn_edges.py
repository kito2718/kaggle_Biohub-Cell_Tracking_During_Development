# -*- coding: utf-8 -*-
"""
s5_024_diagnose_all_fn_edges.py
全199データセットの全失点エッジ (FN: 36,502本) を1本ずつ照合し、
真の失点理由を完全定量化するエビデンス集計スクリプト。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"

OUTPUT_CSV = DATA_DIR / "s5_024_fn_edge_breakdown_all199.csv"

# Load node details files
node_detail_files = list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv"))
print(f"Loading {len(node_detail_files)} node detail files...")
dfs_node = [pd.read_csv(f) for f in node_detail_files]
df_node_details = pd.concat(dfs_node, ignore_index=True)
print(f"Total node detail rows: {len(df_node_details):,}")

# Build map: (dataset, gt_node_id) -> pred_node_id
tp_nodes = df_node_details[df_node_details['eval_result'] == 'TP']
gt_tp_set = set(zip(tp_nodes['dataset'], tp_nodes['gt_node_id']))
gt_to_pred = dict(zip(zip(tp_nodes['dataset'], tp_nodes['gt_node_id']), tp_nodes['pred_node_id']))

# Load edge details files
edge_detail_files = list(WORKING_DIR.glob("s5_015DYNRADIUS_04_check_edges_details_blobdog_lgbm_*.csv"))
print(f"Loading {len(edge_detail_files)} edge detail files...")
dfs_edge = [pd.read_csv(f) for f in edge_detail_files]
df_edge_details = pd.concat(dfs_edge, ignore_index=True)
print(f"Total edge detail rows: {len(df_edge_details):,}")

# Filter only FN edges
fn_edges = df_edge_details[df_edge_details['eval_result'] == 'FN'].copy()
print(f"Total FN edges to analyze: {len(fn_edges):,}")

# Load GT edges and GT nodes
gt_edges = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
gt_nodes = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv").set_index(['dataset', 'node_id'])

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

# Categorize each FN edge
reasons = []
distances = []

for idx, row in fn_edges.iterrows():
    ds = row['dataset']
    s_id = row['source_id']
    t_id = row['target_id']
    
    s_detected = (ds, s_id) in gt_tp_set
    t_detected = (ds, t_id) in gt_tp_set
    
    # Check GT distance
    dist = np.nan
    try:
        s_node = gt_nodes.loc[(ds, s_id)]
        t_node = gt_nodes.loc[(ds, t_id)]
        dz = (t_node['z'] - s_node['z']) * SCALE_VEC[0]
        dy = (t_node['y'] - s_node['y']) * SCALE_VEC[1]
        dx = (t_node['x'] - s_node['x']) * SCALE_VEC[2]
        dist = float(np.sqrt(dz**2 + dy**2 + dx**2))
    except Exception:
        pass
    distances.append(dist)
    
    if not s_detected and not t_detected:
        reasons.append("Both Nodes Missing (Node Detection Failure)")
    elif not s_detected:
        reasons.append("Source Node Missing (Node Detection Failure)")
    elif not t_detected:
        reasons.append("Target Node Missing (Node Detection Failure)")
    else:
        # Both nodes were detected! Why was edge not formed?
        if dist > 7.0:
            reasons.append("Distance > 7.0um (Spatial Search Radius Miss)")
        else:
            reasons.append("Both Nodes Detected <= 7.0um (Tracking Matching / Model Failure)")

fn_edges['fn_reason'] = reasons
fn_edges['gt_distance_um'] = distances

print("\n" + "=" * 80)
print(">>> 全199データセット FNエッジ（全失点エッジ）の真因内訳 <<<")
print("=" * 80)
counts = fn_edges['fn_reason'].value_counts()
for r, c in counts.items():
    print(f"  - {r:<65}: {c:>6,} 本 ({c / len(fn_edges) * 100:>5.1f}%)")

# Per-dataset summary
ds_summary = fn_edges.groupby(['dataset', 'fn_reason']).size().unstack(fill_value=0).reset_index()
ds_summary.to_csv(OUTPUT_CSV, index=False)
print(f"\nデータセット別内訳CSV保存完了: {OUTPUT_CSV}")
