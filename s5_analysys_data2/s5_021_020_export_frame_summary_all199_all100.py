"""
s5_021_020_export_frame_summary_all199_all100.py
=================================================
【021HYBRIDFILTER2: 全199データセット・全100フレーム完全全数詳細サマリー出力スクリプト】

目的:
  P/E比を引き下げるノード選別フィルターを検討するため、
  サンプリングは一切行わず、全199データセット × 全100フレーム（計19,900フレーム）を完全全数計算する。
  
  高速化:
    マシンの 24コア CPU から 16並列（Joblib）を投入して同時並列計算を行い、
    全数全フレームを走査しながら約18〜25分で完了する。

出力カラム:
  - dataset, t
  - estimated_number_of_nodes (メタデータ記録の真の推定細胞総数)
  - gt_nodes_frame, base_pred_nodes, base_tp, hyb_pred_nodes, hyb_tp, tp_diff, pred_diff
  - 単一フレーム光学・テクスチャ・形態特徴量:
    snr_proxy, bg_median, bg_noise, dyn_range, fg_ratio, bg_gradient_std, max_to_med,
    vol_mean, vol_std, cv_texture, sharpness_ratio, contrast_ratio

出力先:
  - s5/github/s5_analysys_data2/s5_021_frame_summary_all199_all100.csv (全19,900行)
"""

import os
import sys
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# 公式モジュールパス
sys.path.insert(0, str(Path("s1_local_env/src/kaggle_cell_tracking_competition/src").resolve()))
from tracking_cellmot.io import open_dataset

# -----------------------------------------------------------------------------
# 1. 定数・パス設定
# -----------------------------------------------------------------------------
GT_SUMMARY_PATH = Path("s5/github/working/s5_gt_summary.csv")
GT_NODES_PATH = Path("s5/github/working/s5_gt_nodes.csv")
BASELINE_PATTERN = "s5/github/working/s5_015DYNRADIUS_01_detect_nodes_pred_blobdog_lgbm_*.csv"
TRAIN_ZARR_DIR = Path("s5/input/train")
OUTPUT_FRAME_SUMMARY_CSV = Path("s5/github/s5_analysys_data2/s5_021_frame_summary_all199_all100.csv")

SCALE_Z = 1.625
SCALE_Y = 0.40625
SCALE_X = 0.40625
SCALE_VEC = np.array([SCALE_Z, SCALE_Y, SCALE_X], dtype=np.float32)
MATCH_THRESHOLD_UM = 7.0

# -----------------------------------------------------------------------------
# 2. 単一フレーム特徴量計測（光学・テクスチャ・形態）
# -----------------------------------------------------------------------------
def analyze_frame_optics_deep(frame: np.ndarray) -> dict:
    sub = frame[::2, ::2, ::2]
    p01, p25, p50, p75, p995 = np.percentile(sub, [1.0, 25.0, 50.0, 75.0, 99.5])
    bg_noise = float((p75 - p25) / 1.349)
    snr_proxy = float((p995 - p50) / (bg_noise + 1e-5))
    dyn_range = float(p995 - p01)
    fg_ratio = float((sub > (p50 + 3.0 * bg_noise)).mean())
    vol_max = float(sub.max())
    vol_mean = float(sub.mean())
    vol_std = float(sub.std())
    max_to_med = float(vol_max / (p50 + 1e-5))
    
    cv_texture = float(vol_std / (vol_mean + 1e-5))
    sharpness_ratio = float(vol_max / (vol_mean + 1e-5))
    contrast_ratio = float((vol_mean - p50) / (p50 + 1e-5))

    bg_lowpass = gaussian_filter(sub, sigma=(1.5, 4.0, 4.0))
    bg_gradient_std = float(bg_lowpass.std() / (p50 + 1e-5))

    return {
        "p01": float(p01),
        "p995": float(p995),
        "bg_median": float(p50),
        "bg_noise": float(bg_noise),
        "snr_proxy": float(snr_proxy),
        "dyn_range": float(dyn_range),
        "fg_ratio": float(fg_ratio),
        "vol_mean": float(vol_mean),
        "vol_std": float(vol_std),
        "cv_texture": float(cv_texture),
        "sharpness_ratio": float(sharpness_ratio),
        "contrast_ratio": float(contrast_ratio),
        "max_to_med": float(max_to_med),
        "bg_gradient_std": float(bg_gradient_std)
    }

def detect_frame_021(frame: np.ndarray, optics: dict) -> list[tuple[float, float, float]]:
    from skimage.feature import blob_dog
    snr_proxy = optics["snr_proxy"]
    bg_median = optics["bg_median"]
    bg_gradient_std = optics["bg_gradient_std"]
    max_to_med = optics["max_to_med"]

    is_triggered = (snr_proxy <= 12.0) or (bg_median >= 80.0)
    if not is_triggered:
        p_low = max(optics["p01"], optics["bg_median"] - 2.0 * optics["bg_noise"])
        p_high_pct = 99.8 if optics["fg_ratio"] < 0.05 else 99.3
        p_high = np.percentile(frame[::2, ::2, ::2], p_high_pct)
        th = float(np.clip(0.035 + 0.0020 * (snr_proxy - 2.0), 0.035, 0.068))
        target = frame
    else:
        if (bg_gradient_std >= 1.0) and (max_to_med >= 14.0):
            bg = gaussian_filter(frame, sigma=(2.0, 8.0, 8.0))
            vol_sub = np.maximum(0.0, frame.astype(np.float32, copy=False) - bg)
            p_low = np.percentile(vol_sub, 1.0)
            p_high = np.percentile(vol_sub, 99.5)
            th = 0.018
            target = vol_sub
        else:
            bg = gaussian_filter(frame, sigma=(4.0, 16.0, 16.0))
            vol_sub = np.maximum(0.0, frame.astype(np.float32, copy=False) - bg)
            p_low = np.percentile(vol_sub, 1.0)
            p_high = np.percentile(vol_sub, 99.5)
            th = 0.025
            target = vol_sub

    ov = 0.50 if optics["fg_ratio"] > 0.08 else 0.35
    if p_high > p_low:
        img_norm = np.clip((target.astype(np.float32, copy=False) - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0)
    else:
        img_norm = np.zeros_like(frame, dtype=np.float32)

    blobs = blob_dog(img_norm, min_sigma=1.5, max_sigma=3.5, sigma_ratio=1.6, threshold=th, overlap=ov)
    return [(float(b[0]), float(b[1]), float(b[2])) for b in blobs]

def match_nodes_frame(g_pts: np.ndarray, p_pts: np.ndarray) -> int:
    if len(g_pts) == 0 or len(p_pts) == 0:
        return 0
    dists = cdist(g_pts * SCALE_VEC, p_pts * SCALE_VEC)
    r_ind, c_ind = linear_sum_assignment(dists)
    tp = 0
    for r, c in zip(r_ind, c_ind):
        if dists[r, c] <= MATCH_THRESHOLD_UM:
            tp += 1
    return tp

# -----------------------------------------------------------------------------
# 3. 1データセット全100フレーム処理関数 (並列ワーカー用)
# -----------------------------------------------------------------------------
def process_single_dataset(ds_name: str, est_nodes_ds: int, gt_df: pd.DataFrame, base_df: pd.DataFrame) -> list[dict]:
    zarr_path = TRAIN_ZARR_DIR / f"{ds_name}.zarr"
    if not zarr_path.exists():
        return []

    ds = open_dataset(str(zarr_path), normalize=False, require_tracks=False, device="cpu")
    img_arr = getattr(ds, 'image', None)
    if hasattr(img_arr, 'numpy'):
        img_arr = img_arr.numpy()
    elif img_arr is not None:
        img_arr = np.array(img_arr)
    else:
        return []

    T = img_arr.shape[0]

    rows = []
    for t in range(T):
        frame = img_arr[t]
        optics = analyze_frame_optics_deep(frame)

        # GTノード
        gt_t = gt_df[gt_df['t'] == t]
        gt_cnt = len(gt_t)
        gt_coords = gt_t[['z', 'y', 'x']].values.astype(np.float32) if gt_cnt > 0 else np.empty((0, 3), dtype=np.float32)

        # ベースラインノード
        base_t = base_df[base_df['t'] == t]
        base_cnt = len(base_t)
        base_coords = base_t[['z', 'y', 'x']].values.astype(np.float32) if base_cnt > 0 else np.empty((0, 3), dtype=np.float32)
        base_tp = match_nodes_frame(gt_coords, base_coords)

        # 021 検出
        hyb_blobs = detect_frame_021(frame, optics)
        hyb_cnt = len(hyb_blobs)
        hyb_coords = np.array(hyb_blobs, dtype=np.float32) if hyb_cnt > 0 else np.empty((0, 3), dtype=np.float32)
        hyb_tp = match_nodes_frame(gt_coords, hyb_coords)

        rows.append({
            "dataset": ds_name,
            "t": t,
            "estimated_number_of_nodes": est_nodes_ds,
            "gt_nodes_frame": gt_cnt,
            "base_pred_nodes": base_cnt,
            "base_tp": base_tp,
            "hyb_pred_nodes": hyb_cnt,
            "hyb_tp": hyb_tp,
            "tp_diff": hyb_tp - base_tp,
            "pred_diff": hyb_cnt - base_cnt,
            # 光学・テクスチャ・形態特徴量列
            "snr_proxy": round(optics["snr_proxy"], 2),
            "bg_median": round(optics["bg_median"], 2),
            "bg_noise": round(optics["bg_noise"], 2),
            "dyn_range": round(optics["dyn_range"], 2),
            "fg_ratio": round(optics["fg_ratio"], 4),
            "bg_gradient_std": round(optics["bg_gradient_std"], 4),
            "max_to_med": round(optics["max_to_med"], 2),
            "vol_mean": round(optics["vol_mean"], 2),
            "vol_std": round(optics["vol_std"], 2),
            "cv_texture": round(optics["cv_texture"], 4),
            "sharpness_ratio": round(optics["sharpness_ratio"], 4),
            "contrast_ratio": round(optics["contrast_ratio"], 4)
        })
    return rows

# -----------------------------------------------------------------------------
# 4. メイン実行部
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("【021: 全199データセット全100フレーム詳細サマリー出力 (_all199_all100 完全全数版)】")
    print("=" * 80)

    # 1. メタデータ & GT読み込み
    print(f"Loading GT summary from: {GT_SUMMARY_PATH}")
    gt_sum_df = pd.read_csv(GT_SUMMARY_PATH)
    est_map = dict(zip(gt_sum_df['dataset'], gt_sum_df['estimated_number_of_nodes']))

    print(f"Loading GT nodes from: {GT_NODES_PATH}")
    gt_all = pd.read_csv(GT_NODES_PATH, usecols=['dataset', 't', 'z', 'y', 'x'])

    print(f"Loading baseline nodes from: {BASELINE_PATTERN}")
    base_files = sorted(glob.glob(BASELINE_PATTERN))
    base_dfs = [pd.read_csv(f, usecols=['dataset', 't', 'z', 'y', 'x']) for f in base_files]
    base_all = pd.concat(base_dfs, ignore_index=True)

    datasets = sorted(gt_sum_df['dataset'].unique())
    print(f"Total datasets to process: {len(datasets)} across 16 parallel workers...")
    print(f"Total frames to compute: {len(datasets) * 100:,} frames (No sampling, 100% full scan)")

    start_time = time.time()

    results = Parallel(n_jobs=16, backend="loky", verbose=10)(
        delayed(process_single_dataset)(
            ds_name,
            est_map[ds_name],
            gt_all[gt_all['dataset'] == ds_name],
            base_all[base_all['dataset'] == ds_name]
        ) for ds_name in datasets
    )

    all_rows = []
    for r in results:
        all_rows.extend(r)

    out_df = pd.DataFrame(all_rows)
    OUTPUT_FRAME_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FRAME_SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"Frame summary exported successfully: {len(out_df):,} rows (Expected: {len(datasets)*100})")
    print(f"Saved to: {OUTPUT_FRAME_SUMMARY_CSV}")
    print(f"Total execution time: {time.time() - start_time:.1f} seconds (approx {(time.time() - start_time)/60:.1f} min)")
    print("=" * 80)

if __name__ == "__main__":
    main()
