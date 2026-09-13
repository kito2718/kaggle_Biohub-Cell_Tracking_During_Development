# -*- coding: utf-8 -*-
"""
s5_018_simulate_bg_subtraction_all63.py
ノード検出率が平均未満（< 91.92%）の全63データセットを対象に、
1. 現行方式 (Global Norm + th_min=0.035)
2. 新方式 (Background Subtraction + 拡張適応閾値 th=0.018)
のノード検出Recall・検出ノード数を実画像から一括測定・比較するシミュレーション。
"""
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import blob_dog
from scipy.ndimage import gaussian_filter
import warnings
import time

warnings.filterwarnings('ignore')

BASE_DIR = Path(r"d:/BizOwn/000_Biw2/51_googleantigravity/007_kaggle_Biohub-Cell_Tracking_During_Development")
TRAIN_ZARR_DIR = BASE_DIR / "s5/input/train"
WORKING_DIR = BASE_DIR / "s5/github/working"
DATA_DIR = BASE_DIR / "s5/github/s5_analysys_data"
OUTPUT_CSV = DATA_DIR / "s5_018_bg_subtraction_simulation_results.csv"

# 対象63データセットの読み込み
below_df = pd.read_csv(DATA_DIR / "s5_017_below_average_datasets.csv")
target_datasets = below_df[below_df["below_mean_node"]]["dataset"].tolist()

print("=" * 80)
print(f">>> 全 {len(target_datasets)} データセット一括シミュレーション開始")
print("=" * 80)

gt_nodes_all = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")
scale_vec = np.array([1.625, 0.40625, 0.40625])

SAMPLE_FRAMES = [0, 25, 50, 75] # 時間軸で偏りがないよう4フレームを均等サンプリング

results = []
t_start = time.time()

for idx, ds in enumerate(target_datasets, 1):
    zarr_path = TRAIN_ZARR_DIR / f"{ds}.zarr"
    if not zarr_path.exists():
        print(f"[{idx:2d}/{len(target_datasets)}] Skip {ds}: Zarr not found")
        continue
    
    gt_ds = gt_nodes_all[gt_nodes_all["dataset"] == ds]
    z = zarr.open(str(zarr_path), mode="r")["0"]
    n_frames_avail = z.shape[0]
    frames_to_test = [f for f in SAMPLE_FRAMES if f < n_frames_avail]
    
    tp_current_total = 0
    tp_new_total = 0
    gt_eval_total = 0
    pred_count_current_total = 0
    pred_count_new_total = 0
    
    for t in frames_to_test:
        vol = np.array(z[t], dtype=np.float32)
        gt_t = gt_ds[gt_ds["t"] == t]
        gt_eval_total += len(gt_t)
        
        # --- 1. 現行方式 ---
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
        pred_count_current_total += len(blobs_cur)
        
        # --- 2. 新方式 (Background Subtraction + 適応閾値 0.018) ---
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
        pred_count_new_total += len(blobs_new)
        
        # Recall評価 (7.0 um マッチング)
        if len(gt_t) > 0:
            gt_pts = gt_t[["z", "y", "x"]].values * scale_vec
            
            # 現行
            if len(blobs_cur) > 0:
                cur_pts = blobs_cur[:, :3] * scale_vec
                for pt in gt_pts:
                    dists = np.linalg.norm(cur_pts - pt, axis=1)
                    if dists.min() <= 7.0:
                        tp_current_total += 1
                        
            # 新方式
            if len(blobs_new) > 0:
                new_pts = blobs_new[:, :3] * scale_vec
                for pt in gt_pts:
                    dists = np.linalg.norm(new_pts - pt, axis=1)
                    if dists.min() <= 7.0:
                        tp_new_total += 1
                        
    recall_cur = tp_current_total / (gt_eval_total + 1e-9)
    recall_new = tp_new_total / (gt_eval_total + 1e-9)
    diff = recall_new - recall_cur
    
    rec = {
        "dataset": ds,
        "gt_sampled_nodes": gt_eval_total,
        "current_recall": recall_cur,
        "new_recall": recall_new,
        "recall_diff": diff,
        "avg_pred_current": pred_count_current_total / len(frames_to_test),
        "avg_pred_new": pred_count_new_total / len(frames_to_test)
    }
    results.append(rec)
    print(f"[{idx:2d}/{len(target_datasets)}] {ds:15s} | Current: {recall_cur:.3f} -> New: {recall_new:.3f} ({diff:+.3f}) | Pred: {rec['avg_pred_current']:.0f} -> {rec['avg_pred_new']:.0f}")

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

elapsed = time.time() - t_start
print("=" * 80)
print(f"全63データセット シミュレーション完了! (所要時間: {elapsed:.1f}秒)")
print(f"平均Recall: {df_results['current_recall'].mean():.4f} -> {df_results['new_recall'].mean():.4f} ({df_results['recall_diff'].mean():+.4f})")
print(f"改善データセット数: {(df_results['recall_diff'] > 0).sum()} / {len(df_results)}")
print(f"悪化データセット数: {(df_results['recall_diff'] < 0).sum()} / {len(df_results)}")
print(f"結果保存先: {OUTPUT_CSV}")
print("=" * 80)
