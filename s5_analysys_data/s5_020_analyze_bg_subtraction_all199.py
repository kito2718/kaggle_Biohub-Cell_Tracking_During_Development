# -*- coding: utf-8 -*-
"""
s5_020_analyze_bg_subtraction_all199.py
全199データセットを対象に、
1. 現行方式 vs 背景差分（Background Subtraction）の実測ノードRecall比較
2. 各データセットの光学特徴量の抽出
3. 確立した判定ルール (snr_proxy <= 12.0 or bg_median >= 80.0) が全199セットでも成立するか完全検証
4. エビデンスCSVおよび4パネル総合プロット図の生成
"""
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.feature import blob_dog
from scipy.ndimage import gaussian_filter
import warnings
import time

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

BASE_DIR = Path(r"d:/BizOwn/000_Biw2/51_googleantigravity/007_kaggle_Biohub-Cell_Tracking_During_Development")
TRAIN_ZARR_DIR = BASE_DIR / "s5/input/train"
WORKING_DIR = BASE_DIR / "s5/github/working"
DATA_DIR = BASE_DIR / "s5/github/s5_analysys_data"

OUTPUT_CSV = DATA_DIR / "s5_020_all199_bg_subtraction_results.csv"
OUTPUT_PNG = DATA_DIR / "s5_020_all199_bg_subtraction_evidence.png"

gt_nodes_all = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")
scale_vec = np.array([1.625, 0.40625, 0.40625])

zarr_files = sorted(list(TRAIN_ZARR_DIR.glob("*.zarr")))
all_datasets = [f.stem for f in zarr_files]

print("=" * 80)
print(f">>> 全 {len(all_datasets)} データセット 背景差分 ＆ 光学解析開始")
print("=" * 80)

SAMPLE_FRAMES = [0, 25, 50, 75]
results = []
t_start = time.time()

for idx, ds in enumerate(all_datasets, 1):
    zarr_path = TRAIN_ZARR_DIR / f"{ds}.zarr"
    if not zarr_path.exists():
        continue
    
    gt_ds = gt_nodes_all[gt_nodes_all["dataset"] == ds]
    z = zarr.open(str(zarr_path), mode="r")["0"]
    n_frames_avail = z.shape[0]
    frames_to_test = [f for f in SAMPLE_FRAMES if f < n_frames_avail]
    
    tp_cur_total = 0
    tp_new_total = 0
    gt_total = 0
    pred_cur_total = 0
    pred_new_total = 0
    
    # 代表フレーム t=0 の光学特徴量
    vol0 = np.array(z[0], dtype=np.float32)
    sub0 = vol0[::2, ::2, ::2]
    p01, p25, p50, p75, p995 = np.percentile(sub0, [1.0, 25.0, 50.0, 75.0, 99.5])
    bg_noise = float((p75 - p25) / 1.349)
    snr_proxy = float((p995 - p50) / (bg_noise + 1e-5))
    dyn_range = float(p995 - p01)
    fg_ratio = float((sub0 > (p50 + 3.0 * bg_noise)).mean())
    vol_max = float(vol0.max())
    max_to_med = float(vol_max / (p50 + 1e-5))
    bg_lowpass = gaussian_filter(sub0, sigma=(1.5, 4.0, 4.0))
    bg_gradient_std = float(bg_lowpass.std() / (p50 + 1e-5))
    
    for t in frames_to_test:
        vol = np.array(z[t], dtype=np.float32)
        gt_t = gt_ds[gt_ds["t"] == t]
        gt_total += len(gt_t)
        
        # 1. 現行方式
        p_low = np.percentile(vol, 1.0)
        p_high = np.percentile(vol, 99.5)
        vol_norm_cur = np.clip((vol - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0)
        blobs_cur = blob_dog(
            vol_norm_cur,
            min_sigma=(1.0, 2.0, 2.0),
            max_sigma=(2.5, 6.0, 6.0),
            sigma_ratio=1.6,
            threshold=0.035
        )
        pred_cur_total += len(blobs_cur)
        
        # 2. 新方式 (Background Subtraction + th=0.018)
        bg = gaussian_filter(vol, sigma=(2.0, 8.0, 8.0))
        vol_sub = np.maximum(0, vol - bg)
        p_sub_low = np.percentile(vol_sub, 1.0)
        p_sub_high = np.percentile(vol_sub, 99.5)
        vol_norm_new = np.clip((vol_sub - p_sub_low) / (p_sub_high - p_sub_low + 1e-5), 0.0, 1.0)
        blobs_new = blob_dog(
            vol_norm_new,
            min_sigma=(1.0, 2.0, 2.0),
            max_sigma=(2.5, 6.0, 6.0),
            sigma_ratio=1.6,
            threshold=0.018
        )
        pred_new_total += len(blobs_new)
        
        # Recall評価 (7.0 um マッチング)
        if len(gt_t) > 0:
            gt_pts = gt_t[["z", "y", "x"]].values * scale_vec
            if len(blobs_cur) > 0:
                cur_pts = blobs_cur[:, :3] * scale_vec
                for pt in gt_pts:
                    if np.linalg.norm(cur_pts - pt, axis=1).min() <= 7.0:
                        tp_cur_total += 1
            if len(blobs_new) > 0:
                new_pts = blobs_new[:, :3] * scale_vec
                for pt in gt_pts:
                    if np.linalg.norm(new_pts - pt, axis=1).min() <= 7.0:
                        tp_new_total += 1
                        
    recall_cur = tp_cur_total / (gt_total + 1e-9)
    recall_new = tp_new_total / (gt_total + 1e-9)
    diff = recall_new - recall_cur
    
    rec = {
        "dataset": ds,
        "gt_sampled_nodes": gt_total,
        "current_recall": recall_cur,
        "new_recall": recall_new,
        "recall_diff": diff,
        "avg_pred_current": pred_cur_total / len(frames_to_test),
        "avg_pred_new": pred_new_total / len(frames_to_test),
        "bg_median": p50,
        "bg_noise": bg_noise,
        "snr_proxy": snr_proxy,
        "dyn_range": dyn_range,
        "fg_ratio": fg_ratio,
        "vol_max": vol_max,
        "max_to_med": max_to_med,
        "bg_gradient_std": bg_gradient_std
    }
    results.append(rec)
    if idx % 10 == 0 or idx == len(all_datasets):
        print(f"[{idx:3d}/{len(all_datasets)}] {ds:15s} | Cur: {recall_cur:.3f} -> New: {recall_new:.3f} ({diff:+.3f})")

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
elapsed = time.time() - t_start
print("=" * 80)
print(f"全199データセット 解析完了! (所要時間: {elapsed:.1f}秒)")
print(f"全体平均Recall: {df['current_recall'].mean():.4f} -> {df['new_recall'].mean():.4f} ({df['recall_diff'].mean():+.4f})")
print(f"改善データセット数: {(df['recall_diff'] > 0).sum()} / {len(df)}")
print(f"不変データセット数: {(df['recall_diff'] == 0).sum()} / {len(df)}")
print(f"悪化データセット数: {(df['recall_diff'] < 0).sum()} / {len(df)}")
print("=" * 80)

# プロット図作成
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 散布図: 現行Recall vs 新方式Recall
ax1 = axes[0, 0]
ax1.plot([0, 1.05], [0, 1.05], 'r--', label='y = x (No change)', alpha=0.7)
scatter = ax1.scatter(df['current_recall'], df['new_recall'], c=df['recall_diff'], cmap='coolwarm', s=50, edgecolors='black', alpha=0.8)
plt.colorbar(scatter, ax=ax1, label='ΔRecall (Improvement)')
ax1.set_xlabel('Current Node Recall (Global Norm)', fontsize=12)
ax1.set_ylabel('New Node Recall (Background Subtraction)', fontsize=12)
ax1.set_title('Figure 1: Node Recall Improvement across All 199 Datasets', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')

# 2. 散布図: snr_proxy vs bg_median (判定ルールの境界可視化)
ax2 = axes[0, 1]
df['is_improved'] = df['recall_diff'] > 0
df['rule_triggered'] = (df['snr_proxy'] <= 12.0) | (df['bg_median'] >= 80.0)

for flag, color, label in [(True, '#2ecc71', 'Improved (ΔRecall > 0)'), (False, '#95a5a6', 'Unchanged / Slightly lower')]:
    sub_df = df[df['is_improved'] == flag]
    ax2.scatter(sub_df['snr_proxy'], sub_df['bg_median'], c=color, s=50, label=label, alpha=0.7, edgecolors='black')

ax2.axvline(x=12.0, color='blue', linestyle='--', label='snr_proxy <= 12.0')
ax2.axhline(y=80.0, color='purple', linestyle='--', label='bg_median >= 80.0')
ax2.set_xlabel('SNR Proxy (Signal-to-Noise Ratio)', fontsize=12)
ax2.set_ylabel('Background Median Intensity', fontsize=12)
ax2.set_title('Figure 2: Optical Space & Adaptive Rule Boundary (All 199)', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right')

# 3. ヒストグラム: Recall改善量 (ΔRecall) の分布
ax3 = axes[1, 0]
sns.histplot(df['recall_diff'], bins=30, kde=True, ax=ax3, color='royalblue')
ax3.axvline(x=0.0, color='red', linestyle='--', label='Zero Change')
ax3.set_xlabel('Recall Improvement (ΔRecall)', fontsize=12)
ax3.set_ylabel('Number of Datasets', fontsize=12)
ax3.set_title('Figure 3: Distribution of Recall Improvement (199 Datasets)', fontsize=13, fontweight='bold')
ax3.legend()

# 4. 箱ひげ図: 判定ルール適用グループ vs 非適用グループの Recall 比較
ax4 = axes[1, 1]
df['Group'] = np.where(df['rule_triggered'], 'Rule Triggered (Apply Bg-Sub)', 'Rule Skipped (Keep Original)')
df_melt = pd.melt(df, id_vars=['Group'], value_vars=['current_recall', 'new_recall'], var_name='Method', value_name='Recall')
df_melt['Method'] = df_melt['Method'].replace({'current_recall': 'Current', 'new_recall': 'With Bg-Sub'})
sns.boxplot(data=df_melt, x='Group', y='Recall', hue='Method', ax=ax4, palette='Set2')
ax4.set_title('Figure 4: Impact of Adaptive Rule on Recall (All 199)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Dataset Group', fontsize=12)
ax4.set_ylabel('Node Recall', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200)
plt.close()
print(f"Evidence plot saved to: {OUTPUT_PNG}")
