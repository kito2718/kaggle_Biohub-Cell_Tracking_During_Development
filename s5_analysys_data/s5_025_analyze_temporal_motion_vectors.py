# -*- coding: utf-8 -*-
"""
s5_025_analyze_temporal_motion_vectors.py
GTエッジおよび検出候補ペアにおける「時間的運動ベクトル (Temporal Motion Vectors)」の
分離能力を完全定量化するエビデンス集計・可視化スクリプト。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "s5_025_temporal_motion_analysis.csv"
OUTPUT_PNG = DATA_DIR / "s5_025_temporal_motion_evidence.png"

SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

print("=" * 80)
print(">>> s5_025: 時間的運動ベクトル (Temporal Motion Vectors) 詳細解析開始 <<<")
print("=" * 80)

# 1. GT データの読み込み
print("Loading GT edges and nodes...")
gt_edges = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
gt_nodes = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

# ノード座標テーブルを辞書化して超高速アクセス可能にする
# key: (dataset, node_id) -> (t, z_um, y_um, x_um)
node_dict = {}
for r in gt_nodes[['dataset', 'node_id', 't', 'z', 'y', 'x']].itertuples():
    node_dict[(r.dataset, r.node_id)] = (
        r.t,
        r.z * SCALE_VEC[0],
        r.y * SCALE_VEC[1],
        r.x * SCALE_VEC[2]
    )

print(f"Total GT nodes: {len(node_dict):,}, Total GT edges: {len(gt_edges):,}")

# 2. GT 軌跡 (3フレーム連続 t-1 -> t -> t+1) における運動ベクトルの抽出
# エッジテーブルから各ノードの入エッジ(parent)と出エッジ(child)をマッピング
# (dataset, target_id) -> source_id (incoming edge: t-1 -> t)
# (dataset, source_id) -> target_id (outgoing edge: t -> t+1)
incoming_map = {}
outgoing_map = {}

for r in gt_edges[['dataset', 'source_id', 'target_id']].itertuples():
    incoming_map[(r.dataset, r.target_id)] = r.source_id
    outgoing_map[(r.dataset, r.source_id)] = r.target_id

# 3フレーム連続ペア (t-1 -> t -> t+1) の収集
motion_records = []

for r in gt_edges[['dataset', 'source_id', 'target_id']].itertuples():
    ds = r.dataset
    s_id = r.source_id # t
    t_id = r.target_id # t+1
    
    # s_id に入ってくる直前エッジ (t-1 -> s_id) が存在するか？
    if (ds, s_id) in incoming_map:
        prev_id = incoming_map[(ds, s_id)] # t-1
        
        p_prev = node_dict.get((ds, prev_id))
        p_curr = node_dict.get((ds, s_id))
        p_next = node_dict.get((ds, t_id))
        
        if p_prev and p_curr and p_next:
            # 時間チェック (t-1, t, t+1)
            t_p, z_p, y_p, x_p = p_prev
            t_c, z_c, y_c, x_c = p_curr
            t_n, z_n, y_n, x_n = p_next
            
            if t_c == t_p + 1 and t_n == t_c + 1:
                # 速度ベクトル 1 (prev -> curr) [um]
                v_prev = np.array([z_c - z_p, y_c - y_p, x_c - x_p], dtype=np.float32)
                # 速度ベクトル 2 (curr -> next) [um]
                v_curr = np.array([z_n - z_c, y_n - y_c, x_n - x_c], dtype=np.float32)
                
                s_prev = np.linalg.norm(v_prev)
                s_curr = np.linalg.norm(v_curr)
                
                # 方向コサイン類似度
                dot = np.dot(v_prev, v_curr)
                cos_sim = dot / (s_prev * s_curr + 1e-6)
                
                # 速度比
                speed_ratio = s_curr / (s_prev + 1e-5)
                
                # 加速度ノルム
                accel_norm = np.linalg.norm(v_curr - v_prev)
                
                motion_records.append({
                    'dataset': ds,
                    'prev_id': prev_id,
                    'curr_id': s_id,
                    'next_id': t_id,
                    'edge_type': 'GT_True_Edge',
                    'speed_prev_um': s_prev,
                    'speed_curr_um': s_curr,
                    'cos_similarity': cos_sim,
                    'speed_ratio': speed_ratio,
                    'accel_norm_um': accel_norm
                })

df_gt_motion = pd.DataFrame(motion_records)
print(f"Total 3-frame GT trajectories analyzed: {len(df_gt_motion):,}")

# 3. 疑似ノイズ・誤接続エッジ (False Candidates) の運動ベクトル特性
# 各 (prev -> curr) に対して、半径 7.0um 以内にある「無関係な別ノード (ランダム・最近傍)」と結んだ場合の運動ベクトル
np.random.seed(42)
false_records = []

# 各データセット内のノードをフレーム別にグループ化
nodes_by_ds_t = {}
for (ds, n_id), (t, z, y, x) in node_dict.items():
    nodes_by_ds_t.setdefault((ds, t), []).append((n_id, z, y, x))

# 代表サンプル 25,000 件で False candidate を生成
sample_gt = df_gt_motion.sample(n=min(25000, len(df_gt_motion)), random_state=42)

for r in sample_gt.itertuples():
    ds = r.dataset
    s_id = r.curr_id
    true_next_id = r.next_id
    
    p_curr = node_dict[(ds, s_id)]
    t_c, z_c, y_c, x_c = p_curr
    v_prev = np.array([
        z_c - node_dict[(ds, r.prev_id)][1],
        y_c - node_dict[(ds, r.prev_id)][2],
        x_c - node_dict[(ds, r.prev_id)][3]
    ], dtype=np.float32)
    s_prev = np.linalg.norm(v_prev)
    
    # 次フレームの候補ノードたち
    candidates = nodes_by_ds_t.get((ds, t_c + 1), [])
    # 正解以外の候補で半径 7.0um 以内のものを探す
    false_cands = []
    for cand_id, z_n, y_n, x_n in candidates:
        if cand_id == true_next_id:
            continue
        dist = np.sqrt((z_n - z_c)**2 + (y_n - y_c)**2 + (x_n - x_c)**2)
        if dist <= 7.0:
            false_cands.append((cand_id, z_n, y_n, x_n, dist))
            
    if false_cands:
        # 最も近い誤候補、またはランダム誤候補を1つ選択
        false_cands.sort(key=lambda x: x[4])
        cand_id, z_n, y_n, x_n, dist = false_cands[0]
        
        v_false = np.array([z_n - z_c, y_n - y_c, x_n - x_c], dtype=np.float32)
        s_false = np.linalg.norm(v_false)
        cos_false = np.dot(v_prev, v_false) / (s_prev * s_false + 1e-6)
        ratio_false = s_false / (s_prev + 1e-5)
        accel_false = np.linalg.norm(v_false - v_prev)
        
        false_records.append({
            'dataset': ds,
            'prev_id': r.prev_id,
            'curr_id': s_id,
            'next_id': cand_id,
            'edge_type': 'False_Candidate_Edge',
            'speed_prev_um': s_prev,
            'speed_curr_um': s_false,
            'cos_similarity': cos_false,
            'speed_ratio': ratio_false,
            'accel_norm_um': accel_false
        })

df_false_motion = pd.DataFrame(false_records)
print(f"Total False candidate edges generated (within 7.0um): {len(df_false_motion):,}")

# 4. 統計比較と集計
df_combined = pd.concat([df_gt_motion, df_false_motion], ignore_index=True)
df_combined.to_csv(OUTPUT_CSV, index=False)
print(f"Saved analysis CSV to {OUTPUT_CSV}")

print("\n" + "=" * 80)
print(">>> 時間的運動ベクトル統計比較: 本物エッジ (GT) vs 誤接続候補 (False within 7um) <<<")
print("=" * 80)

for m_col in ['cos_similarity', 'speed_curr_um', 'speed_ratio', 'accel_norm_um']:
    gt_vals = df_gt_motion[m_col].dropna()
    fa_vals = df_false_motion[m_col].dropna()
    print(f"\n[Feature: {m_col}]")
    print(f"  - GT True Edge     : Mean={gt_vals.mean():.4f}, Median={gt_vals.median():.4f}, 25%={gt_vals.quantile(0.25):.4f}, 75%={gt_vals.quantile(0.75):.4f}")
    print(f"  - False Candidate  : Mean={fa_vals.mean():.4f}, Median={fa_vals.median():.4f}, 25%={fa_vals.quantile(0.25):.4f}, 75%={fa_vals.quantile(0.75):.4f}")
    
    if m_col == 'cos_similarity':
        gt_pos = (gt_vals > 0.0).mean() * 100
        fa_pos = (fa_vals > 0.0).mean() * 100
        print(f"  ==> 方向角が前進 (cos > 0): GT={gt_pos:.1f}%, False={fa_pos:.1f}%")
        gt_high = (gt_vals > 0.5).mean() * 100
        fa_high = (fa_vals > 0.5).mean() * 100
        print(f"  ==> 強い方向一貫性 (cos > 0.5): GT={gt_high:.1f}%, False={fa_high:.1f}%")

# 5. 可視化プロット生成
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: コサイン類似度分布
ax1 = axes[0, 0]
sns.kdeplot(df_gt_motion['cos_similarity'], ax=ax1, label='GT True Edges', color='royalblue', fill=True, bw_adjust=1.5)
sns.kdeplot(df_false_motion['cos_similarity'], ax=ax1, label='False Candidates (<=7um)', color='crimson', fill=True, bw_adjust=1.5)
ax1.set_title("Directional Cosine Similarity (cos theta)\n[v_prev vs v_curr]", fontsize=12, fontweight='bold')
ax1.set_xlabel("Cosine Similarity (-1: Reverse, +1: Forward)")
ax1.set_ylabel("Density")
ax1.set_xlim(-1.0, 1.0)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')

# Panel 2: 速度比分布 (対数スケール)
ax2 = axes[0, 1]
sns.kdeplot(np.log10(np.clip(df_gt_motion['speed_ratio'], 0.01, 100)), ax=ax2, label='GT True Edges', color='royalblue', fill=True)
sns.kdeplot(np.log10(np.clip(df_false_motion['speed_ratio'], 0.01, 100)), ax=ax2, label='False Candidates (<=7um)', color='crimson', fill=True)
ax2.set_title("Speed Ratio Consistency (log10[s_curr / s_prev])", fontsize=12, fontweight='bold')
ax2.set_xlabel("log10(Speed Ratio) [0 = Exact match]")
ax2.set_ylabel("Density")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right')

# Panel 3: 加速度ノルム分布
ax3 = axes[1, 0]
sns.kdeplot(df_gt_motion['accel_norm_um'], ax=ax3, label='GT True Edges', color='royalblue', fill=True, clip=(0, 10))
sns.kdeplot(df_false_motion['accel_norm_um'], ax=ax3, label='False Candidates (<=7um)', color='crimson', fill=True, clip=(0, 10))
ax3.set_title("Acceleration Norm (||v_curr - v_prev|| [um])", fontsize=12, fontweight='bold')
ax3.set_xlabel("Acceleration Norm [um]")
ax3.set_ylabel("Density")
ax3.set_xlim(0, 8.0)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right')

# Panel 4: コサイン類似度 vs 加速度ノルムの散布図 (サンプリング)
ax4 = axes[1, 1]
sample_gt_plot = df_gt_motion.sample(n=min(2000, len(df_gt_motion)), random_state=42)
sample_fa_plot = df_false_motion.sample(n=min(2000, len(df_false_motion)), random_state=42)
ax4.scatter(sample_fa_plot['cos_similarity'], sample_fa_plot['accel_norm_um'], color='crimson', alpha=0.3, s=15, label='False Candidates')
ax4.scatter(sample_gt_plot['cos_similarity'], sample_gt_plot['accel_norm_um'], color='royalblue', alpha=0.3, s=15, label='GT True Edges')
ax4.set_title("Bivariate Separation: Direction vs Acceleration", fontsize=12, fontweight='bold')
ax4.set_xlabel("Cosine Similarity")
ax4.set_ylabel("Acceleration Norm [um]")
ax4.set_xlim(-1.0, 1.0)
ax4.set_ylim(0, 10.0)
ax4.grid(True, linestyle='--', alpha=0.5)
ax4.legend(loc='upper left')

plt.suptitle("Quantitative Proof: Power of Temporal Motion Vectors in Distinguishing True vs False Cell Links", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()
print(f"Saved evidence visualization to {OUTPUT_PNG}")
print("=" * 80)
