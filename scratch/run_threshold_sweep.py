import time
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from skimage.feature import blob_dog
from scipy.spatial import cKDTree

# 1. パスとデータセット設定
base_dir = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development")
train_dir = base_dir / "s1_local_env" / "input" / "train"
gt_csv = base_dir / "s3_results_integration_and_submission" / "working" / "s3_gt_nodes.csv"
gt_summary_csv = base_dir / "s3_results_integration_and_submission" / "working" / "s3_gt_summary.csv"

target_datasets = ['44b6_0b24845f', '44b6_74d0c52e', '6bba_05b6850b', '6bba_085bf656']
threshold_list = [0.12, 0.07, 0.04, 0.02, 0.01]

print(f"=== Starting BlobDog Threshold Sweep Experiment ===", flush=True)
print(f"Target Datasets: {target_datasets}", flush=True)
print(f"Thresholds: {threshold_list}", flush=True)

# GTデータのロード
gt_nodes_all = pd.read_csv(gt_csv)
gt_summary_all = pd.read_csv(gt_summary_csv)

scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
radius_threshold = 7.0

# 検出・評価関数 (check_nodes と完全互換)
def evaluate_matching(gt_sub_df, pred_sub_df):
    all_t = sorted(set(gt_sub_df['t'].unique()).union(set(pred_sub_df['t'].unique())))
    tp_total = 0
    fn_total = 0
    fp_total = 0
    distances = []

    for t_val in all_t:
        gt_t = gt_sub_df[gt_sub_df['t'] == t_val]
        pred_t = pred_sub_df[pred_sub_df['t'] == t_val]

        if gt_t.empty:
            fp_total += len(pred_t)
            continue
        if pred_t.empty:
            fn_total += len(gt_t)
            continue

        gt_coords = gt_t[['z', 'y', 'x']].values.astype(np.float32) * scale_vec
        pred_coords = pred_t[['z', 'y', 'x']].values.astype(np.float32) * scale_vec

        tree = cKDTree(pred_coords)
        dists, pred_indices = tree.query(gt_coords, distance_upper_bound=radius_threshold)

        used_preds = set()
        for d, pred_idx in zip(dists, pred_indices):
            if not np.isinf(d) and pred_idx < len(pred_coords) and pred_idx not in used_preds:
                tp_total += 1
                used_preds.add(pred_idx)
                distances.append(d)
            else:
                fn_total += 1
        fp_total += (len(pred_coords) - len(used_preds))

    total_gt = len(gt_sub_df)
    total_pred = len(pred_sub_df)
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_dist = float(np.mean(distances)) if distances else 0.0

    return {
        'tp': tp_total, 'fp': fp_total, 'fn': fn_total,
        'precision': prec, 'recall': rec, 'f1_score': f1,
        'mean_distance_um': mean_dist
    }

all_results = []
output_csv = base_dir / "scratch" / "blobdog_threshold_sweep_results.csv"

for ds_idx, ds in enumerate(target_datasets, 1):
    print(f"\n==================================================", flush=True)
    print(f"[{ds_idx}/{len(target_datasets)}] Processing Dataset: {ds}", flush=True)
    print(f"==================================================", flush=True)

    zarr_p = train_dir / f"{ds}.zarr"
    z = zarr.open(str(zarr_p), mode='r')
    arr = z['0'] if '0' in z else z[list(z.keys())[0]]
    n_frames = arr.shape[0]
    print(f"Image shape: {arr.shape} ({n_frames} frames)", flush=True)

    gt_sub = gt_nodes_all[gt_nodes_all['dataset'] == ds].copy()
    ds_summary = gt_summary_all[gt_summary_all['dataset'] == ds]
    est_nodes = int(ds_summary['estimated_number_of_nodes'].values[0]) if not ds_summary.empty else -1

    # 画像を全フレーム一度 float32 正規化してメモリに展開 (高速化)
    print("Pre-normalizing frames...", flush=True)
    t_start_load = time.time()
    frames_norm = []
    for t in range(n_frames):
        f = np.array(arr[t])
        p_low, p_high = np.percentile(f, (1.0, 99.5))
        if p_high > p_low:
            f_norm = np.clip((f.astype(np.float32, copy=False) - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            f_norm = np.zeros_like(f, dtype=np.float32)
        frames_norm.append(f_norm)
    print(f"Loaded and normalized {n_frames} frames in {time.time()-t_start_load:.2f}s", flush=True)

    for th in threshold_list:
        t_detect_start = time.time()
        detected_nodes = []
        node_id = 0
        for t in range(n_frames):
            f_norm = frames_norm[t]
            blobs = blob_dog(f_norm, min_sigma=2.0, max_sigma=5.0, sigma_ratio=1.6, threshold=th)
            for b in blobs:
                detected_nodes.append({
                    'node_id': node_id,
                    't': t,
                    'z': float(b[0]),
                    'y': float(b[1]),
                    'x': float(b[2])
                })
                node_id += 1

        pred_df = pd.DataFrame(detected_nodes)
        detect_sec = time.time() - t_detect_start

        # マッチング評価
        metrics = evaluate_matching(gt_sub, pred_df)
        pred_cnt = len(pred_df)
        ratio_est = pred_cnt / est_nodes if est_nodes > 0 else 0.0

        rec_dict = {
            'dataset': ds,
            'threshold': th,
            'total_pred': pred_cnt,
            'estimated_gt': est_nodes,
            'pred_ratio_to_est': round(ratio_est, 3),
            'gt_annotated': len(gt_sub),
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
            'recall': round(metrics['recall'], 4),
            'precision': round(metrics['precision'], 4),
            'f1_score': round(metrics['f1_score'], 4),
            'mean_dist_um': round(metrics['mean_distance_um'], 2),
            'detect_time_sec': round(detect_sec, 1)
        }
        all_results.append(rec_dict)

        print(f"  [th={th:5.3f}] Pred={pred_cnt:6d} (Est:{est_nodes:6d}, {ratio_est*100:5.1f}%) | "
              f"Recall={metrics['recall']:.4f} ({metrics['tp']}/{len(gt_sub)}), "
              f"Prec={metrics['precision']:.4f}, F1={metrics['f1_score']:.4f}, "
              f"Dist={metrics['mean_distance_um']:.2f}um | Time={detect_sec:.1f}s", flush=True)

        # 逐次保存
        pd.DataFrame(all_results).to_csv(output_csv, index=False)

print("\n=== Experiment Completed Successfully! ===", flush=True)
res_df = pd.DataFrame(all_results)
print(res_df.to_string(index=False))
