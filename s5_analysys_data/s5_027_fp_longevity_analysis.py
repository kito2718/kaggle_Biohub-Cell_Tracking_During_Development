# -*- coding: utf-8 -*-
"""
s5_027_fp_longevity_analysis.py
FPノード (544万個) の寿命 (何フレーム連続して存在できるか) を検証し、
未アノテーションの本物細胞なのか、1〜2フレームで消滅する光学ノイズなのかを完全定量化する。
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"

OUTPUT_CSV = DATA_DIR / "s5_027_fp_longevity.csv"
OUTPUT_PNG = DATA_DIR / "s5_027_fp_longevity_evidence.png"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

print("=" * 80)
print(">>> s5_027: FPノードの寿命 (Tracklet Duration) 定量解析開始 <<<")
print("=" * 80)

# 代表的なスパース・中密度・高密度データセットを抽出して検証
# 44b6_0113de3b (スパース), 6bba_6feb10f0 (代表), 8196_05086d9a (高密度)
target_datasets = ['44b6_0113de3b', '6bba_6feb10f0', '8196_05086d9a']

node_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv")))
dfs = [pd.read_csv(f) for f in node_files]
df_nodes = pd.concat(dfs, ignore_index=True)

results = []

for ds in target_datasets:
    ds_nodes = df_nodes[df_nodes['dataset'] == ds].copy()
    if ds_nodes.empty:
        continue
        
    fp_nodes = ds_nodes[ds_nodes['eval_result'] == 'FP'].copy()
    tp_nodes = ds_nodes[ds_nodes['eval_result'] == 'TP'].copy()
    
    print(f"\n--- Dataset: {ds} ---")
    print(f"  TP nodes: {len(tp_nodes):,}, FP nodes: {len(fp_nodes):,}")
    
    # FPノード同士をフレーム間で Greedy Nearest Neighbor (半径 7.0um 以内) でリンクさせて Tracklet を形成
    frames = sorted(fp_nodes['t'].unique())
    
    # 簡単なトラッカー
    links = {} # curr_id -> next_id
    
    for i in range(len(frames) - 1):
        t_c = frames[i]
        t_n = frames[i+1]
        if t_n != t_c + 1:
            continue
            
        f_curr = fp_nodes[fp_nodes['t'] == t_c]
        f_next = fp_nodes[fp_nodes['t'] == t_n]
        if f_curr.empty or f_next.empty:
            continue
            
        coords_c = f_curr[['pred_z', 'pred_y', 'pred_x']].values * SCALE_VEC
        coords_n = f_next[['pred_z', 'pred_y', 'pred_x']].values * SCALE_VEC
        
        dists = cdist(coords_c, coords_n)
        ids_c = f_curr['pred_node_id'].values
        ids_n = f_next['pred_node_id'].values
        
        # Greedy matching
        used_n = set()
        for c_idx in range(len(ids_c)):
            best_n_idx = np.argmin(dists[c_idx])
            min_d = dists[c_idx, best_n_idx]
            if min_d <= 7.0 and best_n_idx not in used_n:
                used_n.add(best_n_idx)
                links[ids_c[c_idx]] = ids_n[best_n_idx]
                
    # Tracklet 長を算出
    target_ids = set(links.values())
    start_ids = [k for k in links.keys() if k not in target_ids]
    
    fp_track_lengths = []
    # リンクが一切できなかった完全孤立ノード (寿命 = 1)
    all_fp_ids = set(fp_nodes['pred_node_id'].values)
    linked_ids = set(links.keys()).union(target_ids)
    isolated_count = len(all_fp_ids - linked_ids)
    fp_track_lengths.extend([1] * isolated_count)
    
    for s_id in start_ids:
        l = 1
        curr = s_id
        while curr in links:
            curr = links[curr]
            l += 1
        fp_track_lengths.append(l)
        
    fp_track_lengths = np.array(fp_track_lengths)
    
    print(f"  FP Tracklets: Total={len(fp_track_lengths):,}")
    print(f"  FP Mean Length={np.mean(fp_track_lengths):.2f} frames, Median={np.median(fp_track_lengths):.0f} frames")
    print(f"  FP Length == 1 (単発・瞬時ノイズ): {(fp_track_lengths == 1).mean() * 100:.1f}%")
    print(f"  FP Length <= 2 (2フレーム以下で消滅): {(fp_track_lengths <= 2).mean() * 100:.1f}%")
    print(f"  FP Length >= 10 (10フレーム以上持続): {(fp_track_lengths >= 10).mean() * 100:.1f}%")
    
    for l in fp_track_lengths:
        results.append({'dataset': ds, 'type': 'FP_Noise', 'length': l})

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved longevity results to {OUTPUT_CSV}")

print("=" * 80)
