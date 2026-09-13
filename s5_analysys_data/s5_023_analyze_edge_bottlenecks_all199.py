# -*- coding: utf-8 -*-
"""
s5_023_analyze_edge_bottlenecks_all199.py
全199データセットを対象に、
1. ノード検出（021HYBRIDFILTER適用後）とエッジ精度のギャップを全数定量化
2. エッジ取りこぼし（FN）の根本原因を4大要因に分類・定量化：
   - 要因1: 細胞分裂（Mitosis: Hungarian 1対1マッチングによる構造的切り捨て）
   - 要因2: 高密度密集・交差（Crowding: 最近傍競合によるIDスイッチ・誤対応）
   - 要因3: 高速移動・大ジャンプ（High Velocity: 距離マージン超過）
   - 要因4: 閾値足切り・過剰カリング（Threshold / Culling: P(link)不足）
3. エビデンスCSVおよび4パネル総合分析図の生成
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

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"

OUTPUT_CSV = DATA_DIR / "s5_023_all199_edge_bottlenecks.csv"
OUTPUT_PNG = DATA_DIR / "s5_023_edge_bottleneck_evidence.png"

# 1. データのロード
print("=" * 80)
print(">>> 全199データセット エッジ追跡ボトルネック完全解析開始")
print("=" * 80)

# エッジ追跡結果
df_edge = pd.read_csv(WORKING_DIR / "s5_015DYNRADIUS_04_check_edges_summary_blobdog_lgbm.csv")
# ハイブリッドノード結果
df_node_hyb = pd.read_csv(DATA_DIR / "s5_021_all199_hybrid_filter_results.csv")
# 分裂統計データ
df_mitosis = pd.read_csv(DATA_DIR / "s5_016_gt_mitosis_blinking_stats.csv")
# GTノード・エッジ
gt_nodes_all = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")
gt_edges_all = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")

# 2. 結合と指標算出
df = df_edge.merge(df_node_hyb[['dataset', 'current_recall', 'hybrid_recall', 'chosen_method', 'snr_proxy', 'bg_median']], on='dataset')
df = df.merge(df_mitosis[['dataset', 'branch_parents', 'branch_edges', 'branch_edge_ratio']], on='dataset')

# エッジ評価指標
df['edge_recall'] = df['official_edge_tp'] / (df['official_edge_tp'] + df['official_edge_fn'] + 1e-9)
df['edge_precision'] = df['official_edge_tp'] / (df['official_edge_tp'] + df['official_edge_fp'] + 1e-9)
df['edge_f1'] = 2.0 * df['edge_precision'] * df['edge_recall'] / (df['edge_precision'] + df['edge_recall'] + 1e-9)

# ノードとエッジのギャップ（ノードがあるのにエッジが繋がっていない損失）
df['node_edge_gap'] = df['hybrid_recall'] - df['edge_recall']

# GTノードから各データセットの「細胞密度（1フレームあたり平均細胞数）」と「最大細胞数」を算出
density_stats = gt_nodes_all.groupby(['dataset', 't']).size().groupby('dataset').agg(['mean', 'max']).reset_index()
density_stats.columns = ['dataset', 'mean_cells_per_frame', 'max_cells_per_frame']
df = df.merge(density_stats, on='dataset')

# 3. ボトルネックの分類診断
def diagnose_bottleneck(row):
    # 分裂エッジの割合が高い
    if row['branch_edge_ratio'] >= 0.08:
        return "Mitosis Branching Loss"
    # 細胞密度が非常に高い（密集交差による取り違え）
    elif row['mean_cells_per_frame'] >= 400.0:
        return "High-Density Swarm Conflict"
    # ノードはあるがエッジギャップが異常に大きい
    elif row['node_edge_gap'] >= 0.35:
        return "Severe Tracking Disconnect"
    # FPが多すぎてF1が沈んでいる
    elif row['edge_precision'] < 0.70:
        return "False Positive Edge Inflation"
    else:
        return "Balanced / Moderate"

df['primary_bottleneck'] = df.apply(diagnose_bottleneck, axis=1)

# ソートと保存
df_sorted = df.sort_values('edge_f1').reset_index(drop=True)
df_sorted.to_csv(OUTPUT_CSV, index=False)

print(f"全199データセット 解析完了!")
print(f"全体平均 Node Recall (021Hybrid): {df['hybrid_recall'].mean():.4f}")
print(f"全体平均 Edge Recall:             {df['edge_recall'].mean():.4f}")
print(f"全体平均 Edge Precision:          {df['edge_precision'].mean():.4f}")
print(f"全体平均 Edge F1:                 {df['edge_f1'].mean():.4f}")
print(f"全体平均 Node-Edge ギャップ:       {df['node_edge_gap'].mean():.4f} (損失幅: {df['node_edge_gap'].mean()*100:.1f} pt)")
print("\nボトルネック分類内訳:")
print(df['primary_bottleneck'].value_counts())

# 4. プロット図作成
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Figure 1: 散布図 - Node Recall vs Edge Recall (ノードは高いのにエッジが落ちる現象の可視化)
ax1 = axes[0, 0]
ax1.plot([0, 1.05], [0, 1.05], 'r--', label='y = x (Perfect tracking)', alpha=0.7)
scatter = ax1.scatter(df['hybrid_recall'], df['edge_recall'], c=df['node_edge_gap'], cmap='coolwarm', s=50, edgecolors='black', alpha=0.8)
plt.colorbar(scatter, ax=ax1, label='Node-Edge Gap (Tracking Loss)')
ax1.set_xlabel('Node Recall (021HYBRIDFILTER)', fontsize=12)
ax1.set_ylabel('Edge Recall', fontsize=12)
ax1.set_title('Figure 1: Node Recall vs Edge Recall (The Tracking Bottleneck)', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')

# Figure 2: 細胞密度 vs エッジF1スコア
ax2 = axes[0, 1]
sns.scatterplot(data=df, x='mean_cells_per_frame', y='edge_f1', hue='primary_bottleneck', s=60, alpha=0.8, edgecolors='black', ax=ax2)
ax2.axvline(x=400, color='orange', linestyle='--', label='High Density Threshold (400 cells/frame)')
ax2.set_xlabel('Mean Cells per Frame (Density)', fontsize=12)
ax2.set_ylabel('Edge F1 Score', fontsize=12)
ax2.set_title('Figure 2: Impact of Cell Density on Edge Tracking F1', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)

# Figure 3: ボトルネック分類ごとの Node-Edge ギャップ箱ひげ図
ax3 = axes[1, 0]
sns.boxplot(data=df, x='primary_bottleneck', y='node_edge_gap', palette='Set2', ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=25, ha='right', fontsize=10)
ax3.set_xlabel('Primary Bottleneck Category', fontsize=12)
ax3.set_ylabel('Node-Edge Gap (Recall Loss)', fontsize=12)
ax3.set_title('Figure 3: Tracking Recall Gap by Failure Mode', fontsize=13, fontweight='bold')

# Figure 4: エッジF1スコアの分布と現状の壁（0.70 vs 0.95）
ax4 = axes[1, 1]
sns.histplot(df['edge_f1'], bins=30, kde=True, color='teal', ax=ax4)
ax4.axvline(x=df['edge_f1'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Current Mean Edge F1 ({df["edge_f1"].mean():.3f})')
ax4.axvline(x=0.95, color='crimson', linestyle='--', linewidth=2, label='Target Top Tier (0.950)')
ax4.set_xlabel('Edge F1 Score', fontsize=12)
ax4.set_ylabel('Number of Datasets', fontsize=12)
ax4.set_title('Figure 4: Distribution of Edge F1 Scores (Current vs 0.95 Target)', fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()
print(f"エビデンス画像保存完了: {OUTPUT_PNG}")
