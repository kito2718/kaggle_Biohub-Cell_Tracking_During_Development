import os
import glob
import time
import zarr
import numpy as np
import pandas as pd
from skimage.feature import blob_dog
from scipy.spatial import cKDTree

scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
radius_threshold = 7.0

def analyze_frame_optics(frame: np.ndarray) -> dict:
    """Ultra-fast frame optics pre-analysis (~4ms)"""
    sub = frame[::2, ::2, ::2]
    p01, p25, p50, p75, p99, p995 = np.percentile(sub, [1.0, 25.0, 50.0, 75.0, 99.0, 99.5])
    bg_noise = (p75 - p25) / 1.349
    snr_proxy = (p995 - p50) / (bg_noise + 1e-5)
    dyn_range = p995 - p01
    fg_ratio = float((sub > (p50 + 3.0 * bg_noise)).mean())
    return {
        "p01": float(p01),
        "p995": float(p995),
        "bg_median": float(p50),
        "bg_noise": float(bg_noise),
        "snr_proxy": float(snr_proxy),
        "dyn_range": float(dyn_range),
        "fg_ratio": fg_ratio
    }

def eval_node_recall(gt_df, pred_df):
    if gt_df is None or gt_df.empty:
        return 0.0, 0, 0, []
    if pred_df is None or pred_df.empty:
        return 0.0, 0, len(gt_df), []
        
    total_gt = len(gt_df)
    tp_count = 0
    matched_distances = []
    
    common_t = set(gt_df['t'].unique()).intersection(set(pred_df['t'].unique()))
    for t_val in common_t:
        g_t = gt_df[gt_df['t'] == t_val]
        p_t = pred_df[pred_df['t'] == t_val]
        if g_t.empty or p_t.empty:
            continue
            
        gt_coords = g_t[['z', 'y', 'x']].values.astype(np.float32) * scale_vec
        pred_coords = p_t[['z', 'y', 'x']].values.astype(np.float32) * scale_vec
        
        gt_tree = cKDTree(gt_coords)
        pred_tree = cKDTree(pred_coords)
        
        sparse_mat = pred_tree.sparse_distance_matrix(gt_tree, max_distance=radius_threshold, output_type='dict')
        if sparse_mat:
            used_pred, used_gt = set(), set()
            sorted_pairs = sorted(sparse_mat.items(), key=lambda x: x[1])
            for (p_idx, g_idx), dist in sorted_pairs:
                if p_idx not in used_pred and g_idx not in used_gt:
                    used_pred.add(p_idx)
                    used_gt.add(g_idx)
                    tp_count += 1
                    matched_distances.append(dist)
                    
    recall = tp_count / total_gt if total_gt > 0 else 0.0
    return recall, tp_count, total_gt, matched_distances

def run_detection_pipeline(dataset, mode="baseline", test_frames=None):
    """
    mode: 'baseline' (固定パラメータ) or 'proposed' (動的プレ解析 + 動的パラメータ)
    """
    zarr_path = f"s1_local_env/input/train/{dataset}.zarr"
    z = zarr.open(zarr_path, mode="r")
    
    if test_frames is None:
        num_frames = z['0'].shape[0]
        test_frames = list(range(min(15, num_frames))) # First 15 frames for benchmark
        
    detected_nodes = []
    node_id = 0
    t0 = time.time()
    
    applied_params = []
    
    for t in test_frames:
        frame = z['0'][t]
        
        if mode == "baseline":
            # Baseline: 固定値
            p_low, p_high = np.percentile(frame, [1.0, 99.5])
            if p_high > p_low:
                img_norm = np.clip((frame.astype(np.float32) - p_low) / (p_high - p_low), 0.0, 1.0)
            else:
                img_norm = np.zeros_like(frame, dtype=np.float32)
                
            blobs = blob_dog(
                img_norm,
                min_sigma=2.0,
                max_sigma=5.0,
                sigma_ratio=1.6,
                threshold=0.038,
                overlap=0.5
            )
            applied_params.append({
                "t": t, "threshold": 0.038, "min_sigma": 2.0, "max_sigma": 5.0,
                "overlap": 0.5, "p_low": p_low, "p_high": p_high
            })
            
        elif mode == "proposed":
            # Proposed: 超軽量プレ解析 (4ms)
            optics = analyze_frame_optics(frame)
            
            # 1. 動的パーセンタイル
            p_low = max(optics['p01'], optics['bg_median'] - 2.0 * optics['bg_noise'])
            # 前景比率に応じた p_high
            p_high_pct = 99.8 if optics['fg_ratio'] < 0.05 else 99.4
            p_high = np.percentile(frame[::2, ::2, ::2], p_high_pct)
            
            # 2. 動的閾値 (SNR連動)
            dyn_threshold = float(np.clip(0.035 + 0.0015 * (optics['snr_proxy'] - 2.0), 0.035, 0.060))
            
            # 3. 異方性シグマ (Z:XY = 1:3:3)
            min_sigma = (1.0, 3.0, 3.0)
            max_sigma = (2.5, 7.5, 7.5)
            
            # 4. 動的 overlap
            dyn_overlap = 0.55 if optics['fg_ratio'] > 0.10 else 0.35
            
            # 正規化
            if p_high > p_low:
                img_norm = np.clip((frame.astype(np.float32) - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0)
            else:
                img_norm = np.zeros_like(frame, dtype=np.float32)
                
            blobs = blob_dog(
                img_norm,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sigma_ratio=1.6,
                threshold=dyn_threshold,
                overlap=dyn_overlap
            )
            applied_params.append({
                "t": t, "threshold": dyn_threshold, "min_sigma": min_sigma, "max_sigma": max_sigma,
                "overlap": dyn_overlap, "p_low": p_low, "p_high": p_high,
                "snr_proxy": optics['snr_proxy'], "fg_ratio": optics['fg_ratio']
            })
            
        for b in blobs:
            z_c, y_c, x_c = b[:3]
            detected_nodes.append({
                'node_id': node_id,
                't': int(t),
                'z': float(z_c),
                'y': float(y_c),
                'x': float(x_c)
            })
            node_id += 1
            
    elapsed = time.time() - t0
    df_pred = pd.DataFrame(detected_nodes)
    return df_pred, elapsed, applied_params

def main():
    print("=== Verification: Proposed Dynamic Architecture vs Baseline ===")
    
    test_datasets = [
        ("44b6_0b24845f", "44b6", "超高密度・低コントラスト (課題データセット)"),
        ("44b6_0113de3b", "44b6", "高密度・良好"),
        ("6bba_05b6850b", "6bba", "中密度・標準"),
        ("6bba_085bf656", "6bba", "疎・高コントラスト (過剰検出多め)"),
    ]
    
    df_gt_nodes = pd.read_csv("s3_results_integration_and_submission/working/s3_gt_nodes.csv")
    df_gt_summary = pd.read_csv("s3_results_integration_and_submission/working/s3_gt_summary.csv").set_index("dataset")
    
    results = []
    
    for ds_name, series, desc in test_datasets:
        print(f"\n--- Testing Dataset: {ds_name} ({series}: {desc}) ---")
        gt_ds = df_gt_nodes[df_gt_nodes['dataset'] == ds_name]
        gt_frames = sorted(gt_ds['t'].unique())
        
        # Select common frames for fair evaluation (up to 15 frames)
        eval_frames = gt_frames[:15] if len(gt_frames) >= 10 else list(range(15))
        gt_eval = gt_ds[gt_ds['t'].isin(eval_frames)]
        
        est_total = int(df_gt_summary.loc[ds_name, "estimated_number_of_nodes"])
        n_frames_total = int(df_gt_summary.loc[ds_name, "n_frames"])
        est_frame_slice = est_total * (len(eval_frames) / n_frames_total)
        
        # 1. Baseline
        pred_base, time_base, params_base = run_detection_pipeline(ds_name, mode="baseline", test_frames=eval_frames)
        rec_base, tp_base, gt_tot_base, dists_base = eval_node_recall(gt_eval, pred_base)
        pe_base = len(pred_base) / est_frame_slice if est_frame_slice > 0 else 0.0
        mean_dist_base = np.mean(dists_base) if dists_base else 0.0
        
        # 2. Proposed Dynamic
        pred_prop, time_prop, params_prop = run_detection_pipeline(ds_name, mode="proposed", test_frames=eval_frames)
        rec_prop, tp_prop, gt_tot_prop, dists_prop = eval_node_recall(gt_eval, pred_prop)
        pe_prop = len(pred_prop) / est_frame_slice if est_frame_slice > 0 else 0.0
        mean_dist_prop = np.mean(dists_prop) if dists_prop else 0.0
        
        # Avg applied threshold in proposed
        avg_th = np.mean([p['threshold'] for p in params_prop])
        avg_snr = np.mean([p['snr_proxy'] for p in params_prop])
        
        print(f"  [Baseline] Recall: {rec_base*100:.1f}% ({tp_base}/{gt_tot_base}) | P/E: {pe_base:.2f} ({len(pred_base)} nodes) | Dist: {mean_dist_base:.2f}um | Time: {time_base:.2f}s")
        print(f"  [Proposed] Recall: {rec_prop*100:.1f}% ({tp_prop}/{gt_tot_prop}) | P/E: {pe_prop:.2f} ({len(pred_prop)} nodes) | Dist: {mean_dist_prop:.2f}um | Time: {time_prop:.2f}s | Avg TH: {avg_th:.4f} (SNR: {avg_snr:.1f})")
        
        results.append({
            "dataset": ds_name,
            "series": series,
            "description": desc,
            "eval_frames": len(eval_frames),
            "gt_nodes": len(gt_eval),
            
            # Baseline
            "base_nodes": len(pred_base),
            "base_pe_ratio": round(pe_base, 3),
            "base_recall": round(rec_base, 4),
            "base_mean_dist_um": round(mean_dist_base, 3),
            "base_time_sec": round(time_base, 2),
            
            # Proposed
            "prop_nodes": len(pred_prop),
            "prop_pe_ratio": round(pe_prop, 3),
            "prop_recall": round(rec_prop, 4),
            "prop_mean_dist_um": round(mean_dist_prop, 3),
            "prop_time_sec": round(time_prop, 2),
            "prop_avg_threshold": round(avg_th, 4),
            "prop_avg_snr": round(avg_snr, 2),
            
            # Comparison
            "pe_improvement": round(pe_base - pe_prop, 3),
            "recall_diff": round(rec_prop - rec_base, 4)
        })
        
    df_res = pd.DataFrame(results)
    out_csv = "s3_results_integration_and_submission/__履歴__/s4_dynamic_parameters_validation_results.csv"
    out_xlsx = "s3_results_integration_and_submission/__履歴__/s4_dynamic_parameters_validation_results.xlsx"
    df_res.to_csv(out_csv, index=False)
    df_res.to_excel(out_xlsx, index=False)
    print(f"\n[DONE] Saved validation results to:\n  - {out_csv}\n  - {out_xlsx}")

if __name__ == "__main__":
    main()
