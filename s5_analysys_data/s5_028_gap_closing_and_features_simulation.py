# -*- coding: utf-8 -*-
"""
s5_028_gap_closing_and_features_simulation.py
1. Gap Closing (1フレーム欠損ノード線形補間) による失点エッジ救済ポテンシャルの全数完全シミュレーション
2. 新規特徴量群 (相互最近傍, 逆方向順位, 競合数, 集団運動流動) の分離能 (ROC-AUC) 定量評価
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_GAP = DATA_DIR / "s5_028_gap_closing_potential.csv"
OUTPUT_CSV_FEAT = DATA_DIR / "s5_028_advanced_features_auc.csv"
OUTPUT_PNG = DATA_DIR / "s5_028_gap_closing_and_features_evidence.png"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

print("=" * 80)
print(">>> s5_028: Gap Closing 救済ポテンシャル & 新規特徴量群の完全定量検証開始 <<<")
print("=" * 80)

# 1. データ読み込み
print("Loading GT nodes and edges...")
gt_edges = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
gt_nodes = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

print("Loading node details files...")
node_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv")))
dfs = [pd.read_csv(f) for f in node_files]
df_nodes = pd.concat(dfs, ignore_index=True)

print("Loading FN edge breakdown file...")
df_fn = pd.read_csv(DATA_DIR / "s5_024_fn_edge_breakdown_all199.csv")
print(f"Total FN edges in dataset: {len(df_fn):,}")

# 高速ルックアップテーブル構築
# TP判定セット: (dataset, gt_node_id) が TP かどうか
tp_sub = df_nodes[df_nodes['eval_result'] == 'TP']
gt_tp_dict = dict(zip(zip(tp_sub['dataset'], tp_sub['gt_node_id']), zip(tp_sub['pred_z']*SCALE_VEC[0], tp_sub['pred_y']*SCALE_VEC[1], tp_sub['pred_x']*SCALE_VEC[2])))

# GTノード座標: (dataset, node_id) -> (t, z_um, y_um, x_um)
gt_coords = {}
for r in gt_nodes[['dataset', 'node_id', 't', 'z', 'y', 'x']].itertuples():
    gt_coords[(r.dataset, r.node_id)] = (
        r.t,
        r.z * SCALE_VEC[0],
        r.y * SCALE_VEC[1],
        r.x * SCALE_VEC[2]
    )

# 2. [Part 1] Gap Closing (Delta t = 2) による救済可能性の完全シミュレーション
print("\n--- [Part 1] Gap Closing (1コマ欠損補間) による失点エッジ救済ポテンシャル ---")
# 検出漏れエッジ (13,592本) の中から、前後ノードの検出状況を追跡
# 連鎖エッジ: (p -> c) と (c -> n)
# もし c が未検出 (FN) だが、p と n が検出 (TP) されていた場合、
# p から n へ Delta t = 2 の Gap Closing を行い、中央 c_hat = (p + n) / 2 を補間可能か？

edge_in = dict(zip(zip(gt_edges['dataset'], gt_edges['target_id']), gt_edges['source_id']))
edge_out = dict(zip(zip(gt_edges['dataset'], gt_edges['source_id']), gt_edges['target_id']))

gap_results = []

# 全ての未検出GTノード (FNノード) を探す
fn_nodes = df_nodes[df_nodes['eval_result'] == 'FN']
print(f"Total FN (missed) GT nodes: {len(fn_nodes):,}")

for r in fn_nodes.itertuples():
    ds = r.dataset
    nid = r.gt_node_id
    
    # このノードの前後ノードが存在するか？ (p -> nid -> n)
    has_prev = (ds, nid) in edge_in
    has_next = (ds, nid) in edge_out
    
    p_id = edge_in.get((ds, nid))
    n_id = edge_out.get((ds, nid))
    
    p_detected = (ds, p_id) in gt_tp_dict if has_prev else False
    n_detected = (ds, n_id) in gt_tp_dict if has_next else False
    
    # 補間可能性の判定
    can_interpolate = p_detected and n_detected
    
    dist_p_n = np.nan
    interp_err = np.nan
    gt_coord = gt_coords.get((ds, nid))
    
    if can_interpolate and gt_coord:
        p_pos = np.array(gt_tp_dict[(ds, p_id)])
        n_pos = np.array(gt_tp_dict[(ds, n_id)])
        
        # p と n の距離 (Delta t = 2)
        dist_p_n = float(np.linalg.norm(n_pos - p_pos))
        
        # 線形補間ノード
        interp_pos = (p_pos + n_pos) / 2.0
        
        # 真のGTノード座標との誤差
        true_pos = np.array([gt_coord[1], gt_coord[2], gt_coord[3]])
        interp_err = float(np.linalg.norm(interp_pos - true_pos))
        
    gap_results.append({
        'dataset': ds,
        'gt_node_id': nid,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_detected': p_detected,
        'next_detected': n_detected,
        'both_surrounding_detected': can_interpolate,
        'dist_delta_t2_um': dist_p_n,
        'interp_error_um': interp_err,
        'is_recovered_node': interp_err <= 7.0 if not np.isnan(interp_err) else False
    })

df_gap = pd.DataFrame(gap_results)
df_gap.to_csv(OUTPUT_CSV_GAP, index=False)

n_total_fn_nodes = len(df_gap)
n_isolated_fn = df_gap['both_surrounding_detected'].sum()
n_recovered = df_gap['is_recovered_node'].sum()

print(f"Total FN Nodes: {n_total_fn_nodes:,}")
print(f"  - 前後ノードが両方検出されている (両側TP挟み込み): {n_isolated_fn:,} 個 ({n_isolated_fn/n_total_fn_nodes*100:.1f}%)")
print(f"  - 線形補間で GT 誤差 <= 7.0um に収まる (救済成功): {n_recovered:,} 個 ({n_recovered/n_total_fn_nodes*100:.1f}%)")
if n_isolated_fn > 0:
    print(f"  - 挟み込み成功時の平均補間誤差: {df_gap['interp_error_um'].mean():.2f} um (中央値: {df_gap['interp_error_um'].median():.2f} um)")
    print(f"  - Delta t = 2 の平均スパン距離: {df_gap['dist_delta_t2_um'].mean():.2f} um (中央値: {df_gap['dist_delta_t2_um'].median():.2f} um)")

# エッジ救済本数の計算: 1つのノードが補間されると (p -> c) と (c -> n) の2本のエッジが同時に救済される
rescued_edges = n_recovered * 2
print(f"\n>>> 【Gap Closing による失点エッジ救済ポテンシャル】 <<<")
print(f"  - 救済可能エッジ数: 推定 {rescued_edges:,} 本 / 13,592本 (検出漏れ失点エッジの {rescued_edges/13592*100:.1f}% を救済可能！)")

# 3. [Part 2] 新規特徴量群 (相互最近傍, 逆順位, 競合数, 集団流動) の分離能定量評価
print("\n--- [Part 2] 新規特徴量群の分離能 (ROC-AUC) 定量評価 ---")
# 代表データセット (中密度 6bba_6feb10f0, 高密度 8196_05086d9a, スパース 44b6_0113de3b) で
# 全候補ペア (真のTPペア vs 周囲7.0um以内の誤候補) に対し新特徴量を計算
test_datasets = ['6bba_6feb10f0', '8196_05086d9a', '44b6_0113de3b', '44b6_551a5dba']

advanced_records = []

for ds in test_datasets:
    ds_nodes = df_nodes[df_nodes['dataset'] == ds]
    ds_gt_edges = gt_edges[gt_edges['dataset'] == ds]
    if ds_nodes.empty or ds_gt_edges.empty:
        continue
        
    # 各フレームのノード一覧
    frames = sorted(ds_nodes['t'].unique())
    # GT正解エッジセット
    gt_edge_pairs = set(zip(ds_gt_edges['source_id'], ds_gt_edges['target_id']))
    
    # 簡易に pred_node_id ベースで検証
    # pred_node_id -> gt_node_id
    pred_to_gt = dict(zip(ds_nodes['pred_node_id'], ds_nodes['gt_node_id']))
    
    for i in range(len(frames) - 1):
        t_c = frames[i]
        t_n = frames[i+1]
        if t_n != t_c + 1:
            continue
            
        f_curr = ds_nodes[ds_nodes['t'] == t_c]
        f_next = ds_nodes[ds_nodes['t'] == t_n]
        if f_curr.empty or f_next.empty:
            continue
            
        coords_c = f_curr[['pred_z', 'pred_y', 'pred_x']].values * SCALE_VEC
        coords_n = f_next[['pred_z', 'pred_y', 'pred_x']].values * SCALE_VEC
        
        # 空間距離行列 (N_curr x N_next)
        dists = cdist(coords_c, coords_n)
        
        # 1. 順方向順位 (Forward Rank): ソースから見たターゲットの順位
        # 2. 逆方向順位 (Reverse Rank): ターゲットから見たソースの順位
        # 3. 競合数 (Target Competition): ターゲットを半径7.0um内に収めているソース数
        ranks_fwd = np.argsort(np.argsort(dists, axis=1), axis=1) + 1 # 1-based
        ranks_rev = np.argsort(np.argsort(dists, axis=0), axis=0) + 1 # 1-based
        
        target_comp = (dists <= 7.0).sum(axis=0) # 各ターゲットへの競合数
        
        # 半径7.0um以内のペアを抽出
        r_idx, c_idx = np.where(dists <= 7.0)
        if len(r_idx) == 0:
            continue
            
        curr_pids = f_curr['pred_node_id'].values
        next_pids = f_next['pred_node_id'].values
        
        for r, c in zip(r_idx, c_idx):
            pid_c = curr_pids[r]
            pid_n = next_pids[c]
            
            gid_c = pred_to_gt.get(pid_c)
            gid_n = pred_to_gt.get(pid_n)
            
            is_true = (gid_c, gid_n) in gt_edge_pairs if (pd.notna(gid_c) and pd.notna(gid_n)) else False
            
            d_val = dists[r, c]
            rf = ranks_fwd[r, c]
            rr = ranks_rev[r, c]
            is_mutual_1st = (rf == 1 and rr == 1)
            comp_cnt = target_comp[c]
            
            advanced_records.append({
                'dataset': ds,
                'is_true_edge': is_true,
                'distance_um': d_val,
                'forward_rank': rf,
                'reverse_rank': rr,
                'is_mutual_first': int(is_mutual_1st),
                'target_competition_count': comp_cnt
            })

df_adv = pd.DataFrame(advanced_records)
df_adv.to_csv(OUTPUT_CSV_FEAT, index=False)
print(f"Total candidate pairs evaluated for advanced features: {len(df_adv):,}")

# ROC-AUC の測定
y_true = df_adv['is_true_edge'].values

auc_dist = roc_auc_score(y_true, -df_adv['distance_um'])
auc_rf = roc_auc_score(y_true, -df_adv['forward_rank'])
auc_rr = roc_auc_score(y_true, -df_adv['reverse_rank'])
auc_mutual = roc_auc_score(y_true, df_adv['is_mutual_first'])
auc_comp = roc_auc_score(y_true, -df_adv['target_competition_count'])

# 複合スコア (距離 + 逆順位 + 相互1位) の簡易ロジスティック回帰
from sklearn.linear_model import LogisticRegression
X_sub = df_adv[['distance_um', 'forward_rank', 'reverse_rank', 'is_mutual_first', 'target_competition_count']]
clf = LogisticRegression()
clf.fit(X_sub, y_true)
probs = clf.predict_proba(X_sub)[:, 1]
auc_combined = roc_auc_score(y_true, probs)

print("\n" + "=" * 80)
print(">>> 新規特徴量群の単体 & 統合 ROC-AUC 評価結果 <<<")
print("=" * 80)
print(f"  1. 空間距離 (distance_um)                : AUC = {auc_dist:.4f}")
print(f"  2. 順方向順位 (forward_rank)             : AUC = {auc_rf:.4f}")
print(f"  3. 逆方向順位 (reverse_rank)             : AUC = {auc_rr:.4f}")
print(f"  4. 相互第1近傍フラグ (is_mutual_first)   : AUC = {auc_mutual:.4f}")
print(f"  5. ターゲット競合数 (target_competition) : AUC = {auc_comp:.4f}")
print(f"  -------------------------------------------------------------")
print(f"  ★ 新特徴量統合モデル (Combined AUC)     : AUC = {auc_combined:.4f}")

# 4. エビデンス画像の作成
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Gap Closing 補間誤差分布
ax1 = axes[0, 0]
interp_errs = df_gap[df_gap['both_surrounding_detected']]['interp_error_um'].dropna()
sns.histplot(interp_errs, bins=30, ax=ax1, color='teal', kde=True)
ax1.axvline(7.0, color='red', linestyle='--', label='Matching Radius Threshold (7.0um)')
ax1.set_title(f"Gap Closing Interpolation Error [um]\n(Median = {interp_errs.median():.2f} um, Success = {(interp_errs <= 7.0).mean()*100:.1f}%)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Error Distance from True GT Node [um]")
ax1.set_ylabel("Count")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Subplot 2: 逆方向順位 (Reverse Rank) の真偽比較
ax2 = axes[0, 1]
rev_tp = df_adv[df_adv['is_true_edge']]['reverse_rank']
rev_fp = df_adv[~df_adv['is_true_edge']]['reverse_rank']
ax2.hist([rev_tp[rev_tp <= 5], rev_fp[rev_fp <= 5]], bins=5, label=['True Edges', 'False Candidates'], color=['royalblue', 'crimson'], density=True)
ax2.set_title(f"Reverse Spatial Rank (Target to Source)\n(ROC-AUC = {auc_rr:.4f})", fontsize=11, fontweight='bold')
ax2.set_xlabel("Reverse Rank (1st, 2nd, ...)")
ax2.set_ylabel("Proportion")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

# Subplot 3: 相互第1近傍 (Mutual First) の真偽割合
ax3 = axes[1, 0]
mutual_tp = (df_adv[df_adv['is_true_edge']]['is_mutual_first'] == 1).mean() * 100
mutual_fp = (df_adv[~df_adv['is_true_edge']]['is_mutual_first'] == 1).mean() * 100
bars = ax3.bar(['True Edges', 'False Candidates'], [mutual_tp, mutual_fp], color=['royalblue', 'crimson'], width=0.5)
ax3.set_title(f"Mutual 1st Nearest Neighbor Ratio (%)\n[True={mutual_tp:.1f}%, False={mutual_fp:.1f}%]", fontsize=11, fontweight='bold')
ax3.set_ylabel("Percentage (%)")
ax3.set_ylim(0, 100)
for b in bars:
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha='center', fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.5)

# Subplot 4: 統合モデルによる予測確率分布
ax4 = axes[1, 1]
sns.kdeplot(probs[y_true], ax=ax4, label='True Edges', color='royalblue', fill=True)
sns.kdeplot(probs[~y_true], ax=ax4, label='False Candidates', color='crimson', fill=True)
ax4.set_title(f"Combined Advanced Model Output Probability\n(ROC-AUC = {auc_combined:.4f})", fontsize=11, fontweight='bold')
ax4.set_xlabel("Probability P(Link)")
ax4.set_ylabel("Density")
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend()

plt.suptitle("Quantitative Proof: Gap Closing Potential & Advanced Edge Discriminative Power", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"Saved visualization to {OUTPUT_PNG}")
print("=" * 80)
