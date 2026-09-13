# -*- coding: utf-8 -*-
"""
s5_016_investigate_gt_mitosis_and_blinking.py
全 199 データセットの GT 正解データ (s3_gt_edges.csv, s3_gt_nodes.csv) を全数走査し、
1. 細胞分裂 (Mitosis: 親ノードの Out-degree >= 2) の発生頻度、割合、幾何学的距離、体積変化
2. 1フレーム欠落 (Blinking / Skip: Delta t >= 2 のエッジ) の有無と件数
3. 現行 015DYNRADIUS での分裂エッジの TP / FN 内訳
を客観的データとして定量集計するスクリプト。
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"

GT_NODES_PATH = WORKING_DIR / "s5_gt_nodes.csv"
GT_EDGES_PATH = WORKING_DIR / "s5_gt_edges.csv"
OUTPUT_CSV = DATA_DIR / "s5_016_gt_mitosis_blinking_stats.csv"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

print("=" * 80)
print(">>> s5_016: 全 199 データセット GT 細胞分裂 ＆ フレーム欠落 実態調査開始")
print("=" * 80)

t0 = time.time()
print(f"[*] GT データロード中: {GT_NODES_PATH.name}, {GT_EDGES_PATH.name}")
df_gt_nodes = pd.read_csv(GT_NODES_PATH)
df_gt_edges = pd.read_csv(GT_EDGES_PATH)

print(f"  - 総 GT ノード数: {len(df_gt_nodes):,} 件")
print(f"  - 総 GT エッジ数: {len(df_gt_edges):,} 本")
print(f"  - データセット数: {df_gt_edges['dataset'].nunique()} 件")

datasets = sorted(df_gt_edges['dataset'].unique())

# 現行 015DYNRADIUS のエッジ詳細ファイル
pred_edge_files = list(WORKING_DIR.glob("s5_015DYNRADIUS_04_check_edges_details_blobdog_lgbm_*.csv"))
df_edge_details = None
if pred_edge_files:
    print(f"[*] 015DYNRADIUS エッジ詳細照合用ファイル読込中 ({len(pred_edge_files)} 件)...")
    dfs = [pd.read_csv(f) for f in pred_edge_files]
    df_edge_details = pd.concat(dfs, ignore_index=True)
    print(f"  - 照合用詳細レコード: {len(df_edge_details):,} 行")

results = []
all_branch_distances = []
all_sister_distances = []

for ds in datasets:
    gn_ds = df_gt_nodes[df_gt_nodes['dataset'] == ds].set_index('node_id')
    ge_ds = df_gt_edges[df_gt_edges['dataset'] == ds].copy()
    
    total_edges = len(ge_ds)
    total_nodes = len(gn_ds)
    
    # 1. 時間差 Delta t の調査
    src_t = gn_ds.loc[ge_ds['source_id'], 't'].values
    tgt_t = gn_ds.loc[ge_ds['target_id'], 't'].values
    delta_t = tgt_t - src_t
    
    n_dt1 = int(np.sum(delta_t == 1))
    n_dt2 = int(np.sum(delta_t == 2))
    n_dt_gt2 = int(np.sum(delta_t > 2))
    n_dt_neg = int(np.sum(delta_t <= 0))
    
    # 2. 細胞分裂 (出次数 >= 2) の調査
    out_deg = ge_ds['source_id'].value_counts()
    branch_sources = out_deg[out_deg >= 2].index.tolist()
    n_branch_parents = len(branch_sources)
    
    branch_edges_mask = ge_ds['source_id'].isin(branch_sources)
    n_branch_edges = int(branch_edges_mask.sum())
    branch_edge_ratio = n_branch_edges / total_edges if total_edges > 0 else 0.0
    
    # 3. 分裂の幾何学的距離
    for p_id in branch_sources:
        daughters = ge_ds[ge_ds['source_id'] == p_id]['target_id'].values
        p_node = gn_ds.loc[p_id]
        p_pos = np.array([p_node['z'], p_node['y'], p_node['x']]) * SCALE_VEC
        
        d_positions = []
        for d_id in daughters:
            d_node = gn_ds.loc[d_id]
            d_pos = np.array([d_node['z'], d_node['y'], d_node['x']]) * SCALE_VEC
            dist_p_d = np.linalg.norm(d_pos - p_pos)
            all_branch_distances.append(dist_p_d)
            d_positions.append(d_pos)
            
        if len(d_positions) == 2:
            dist_sisters = np.linalg.norm(d_positions[0] - d_positions[1])
            all_sister_distances.append(dist_sisters)
            
    # 4. 015DYNRADIUS における分裂エッジの TP / FN 内訳
    n_branch_tp = 0
    n_branch_fn = 0
    if df_edge_details is not None:
        det_ds = df_edge_details[df_edge_details['dataset'] == ds]
        if not det_ds.empty and 'eval_result' in det_ds.columns:
            merged = ge_ds.merge(det_ds[['source_id', 'target_id', 'eval_result']], on=['source_id', 'target_id'], how='left')
            branch_sub = merged[merged['source_id'].isin(branch_sources)]
            n_branch_tp = int((branch_sub['eval_result'] == 'TP').sum())
            n_branch_fn = int((branch_sub['eval_result'] == 'FN').sum())
            
    results.append({
        'dataset': ds,
        'total_gt_nodes': total_nodes,
        'total_gt_edges': total_edges,
        'delta_t_1': n_dt1,
        'delta_t_2': n_dt2,
        'delta_t_gt2': n_dt_gt2,
        'branch_parents': n_branch_parents,
        'branch_edges': n_branch_edges,
        'branch_edge_ratio': round(branch_edge_ratio, 4),
        'branch_tp_015': n_branch_tp,
        'branch_fn_015': n_branch_fn
    })

df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)
print(f"\n[*] 調査結果 CSV 保存完了: {OUTPUT_CSV.name}")

print("\n" + "=" * 80)
print("【GT 全 199 データセット 生物学的実態調査 サマリー結果】")
print("=" * 80)

total_edges_all = df_res['total_gt_edges'].sum()
total_dt1_all   = df_res['delta_t_1'].sum()
total_dt2_all   = df_res['delta_t_2'].sum()
total_dt_gt2    = df_res['delta_t_gt2'].sum()
total_parents   = df_res['branch_parents'].sum()
total_branch_e  = df_res['branch_edges'].sum()

print(f"1. 時間差 Delta t の分布 (全 {total_edges_all:,} 本):")
print(f"   - Delta t = 1 (連続フレーム)   : {total_dt1_all:,} 本 ({total_dt1_all/total_edges_all*100:.2f}%)")
print(f"   - Delta t = 2 (1フレームスキップ): {total_dt2_all:,} 本 ({total_dt2_all/total_edges_all*100:.2f}%)")
print(f"   - Delta t > 2 (長距離スキップ)  : {total_dt_gt2:,} 本 ({total_dt_gt2/total_edges_all*100:.2f}%)")

print(f"\n2. 細胞分裂 (Mitosis: 出次数 >= 2):")
print(f"   - 分裂親ノード数               : {total_parents:,} 件 (平均 {total_parents/199:.1f} 件/セット)")
print(f"   - 分裂娘エッジ数               : {total_branch_e:,} 本 (全エッジの {total_branch_e/total_edges_all*100:.2f}%)")
print(f"   - 1セットあたり平均分裂エッジ比率: {df_res['branch_edge_ratio'].mean()*100:.2f}% (最大 {df_res['branch_edge_ratio'].max()*100:.2f}%, 最小 {df_res['branch_edge_ratio'].min()*100:.2f}%)")

if all_branch_distances:
    b_dist = pd.Series(all_branch_distances)
    print(f"\n3. 分裂時の距離特性 (親 ➔ 娘, n={len(b_dist):,}):")
    print(f"   - 平均距離: {b_dist.mean():.2f} um, 中央値: {b_dist.median():.2f} um, 95%タイル: {b_dist.quantile(0.95):.2f} μm")
    
if all_sister_distances:
    s_dist = pd.Series(all_sister_distances)
    print(f"\n4. 分裂娘細胞同士の距離特性 (娘 ➔ 娘, n={len(s_dist):,}):")
    print(f"   - 平均距離: {s_dist.mean():.2f} μm, 中央値: {s_dist.median():.2f} μm, 95%タイル: {s_dist.quantile(0.95):.2f} μm")

if df_edge_details is not None and df_res['branch_tp_015'].sum() > 0:
    tot_btp = df_res['branch_tp_015'].sum()
    tot_bfn = df_res['branch_fn_015'].sum()
    print(f"\n5. 現行 015DYNRADIUS における分裂エッジの捕捉状況:")
    print(f"   - 捕捉成功 (TP): {tot_btp:,} 本 ({tot_btp/(tot_btp+tot_bfn)*100:.1f}%)")
    print(f"   - 見落とし (FN): {tot_bfn:,} 本 ({tot_bfn/(tot_btp+tot_bfn)*100:.1f}%)")

print(f"\n総調査所要時間: {time.time() - t0:.2f} 秒")
print("=" * 80)
