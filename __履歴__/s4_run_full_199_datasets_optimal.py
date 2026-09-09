import os
import glob
import time
import zarr
import numpy as np
import pandas as pd
from skimage.feature import blob_dog
from scipy.spatial import cKDTree
from multiprocessing import Pool

scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
radius_threshold = 7.0

def analyze_frame_optics(frame: np.ndarray) -> dict:
    sub = frame[::2, ::2, ::2]
    p01, p25, p50, p75, p995 = np.percentile(sub, [1.0, 25.0, 50.0, 75.0, 99.5])
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

def process_single_dataset(args):
    ds_name, est_nodes, n_frames, gt_rows = args
    t_start = time.time()
    zarr_path = f"s1_local_env/input/train/{ds_name}.zarr"
    z = zarr.open(store=zarr_path, mode="r")
    
    total_frames = z['0'].shape[0]
    detected_nodes = []
    applied_thresholds = []
    
    for t in range(total_frames):
        frame = z['0'][t]
        optics = analyze_frame_optics(frame)
        
        # 1. Dynamic percentiles
        p_low = max(optics['p01'], optics['bg_median'] - 2.0 * optics['bg_noise'])
        p_high_pct = 99.8 if optics['fg_ratio'] < 0.05 else 99.3
        p_high = np.percentile(frame[::2, ::2, ::2], p_high_pct)
        
        # 2. Optimal dynamic threshold
        th = float(np.clip(0.035 + 0.0020 * (optics['snr_proxy'] - 2.0), 0.035, 0.068))
        applied_thresholds.append(th)
        
        # 3. Anisotropic sigmas
        min_sig = (1.0, 2.0, 2.0)
        max_sig = (2.5, 6.0, 6.0)
        
        # 4. Dynamic overlap
        ov = 0.50 if optics['fg_ratio'] > 0.08 else 0.35
        
        # Normalization
        if p_high > p_low:
            img_norm = np.clip((frame.astype(np.float32) - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0)
        else:
            img_norm = np.zeros_like(frame, dtype=np.float32)
            
        blobs = blob_dog(
            img_norm,
            min_sigma=min_sig,
            max_sigma=max_sig,
            sigma_ratio=1.6,
            threshold=th,
            overlap=ov
        )
        
        for b in blobs:
            detected_nodes.append({
                't': int(t),
                'z': float(b[0]),
                'y': float(b[1]),
                'x': float(b[2])
            })
            
    elapsed = time.time() - t_start
    df_pred = pd.DataFrame(detected_nodes)
    
    # Evaluate recall against GT
    df_gt = pd.DataFrame(gt_rows) if gt_rows else pd.DataFrame()
    rec, tp, gt_tot, dists = eval_node_recall(df_gt, df_pred)
    
    pe_ratio = len(df_pred) / est_nodes if est_nodes > 0 else 0.0
    mean_dist = float(np.mean(dists)) if dists else 0.0
    avg_th = float(np.mean(applied_thresholds)) if applied_thresholds else 0.0
    
    res = {
        "dataset": ds_name,
        "series": ds_name.split("_")[0],
        "n_frames": total_frames,
        "estimated_nodes": est_nodes,
        "pred_nodes": len(df_pred),
        "pe_ratio": round(pe_ratio, 4),
        "recall": round(rec, 4),
        "tp_count": tp,
        "gt_total": gt_tot,
        "mean_dist_um": round(mean_dist, 3),
        "avg_threshold": round(avg_th, 4),
        "elapsed_sec": round(elapsed, 2)
    }
    return res

def main():
    print("=== Full-scale 199 Datasets Optimal Detection & Evaluation ===")
    t_global_start = time.time()
    
    df_gt_nodes = pd.read_csv("s3_results_integration_and_submission/working/s3_gt_nodes.csv")
    df_gt_summary = pd.read_csv("s3_results_integration_and_submission/working/s3_gt_summary.csv").set_index("dataset")
    
    all_zarrs = sorted(glob.glob("s1_local_env/input/train/*.zarr"))
    all_ds_names = [os.path.basename(p).replace(".zarr", "") for p in all_zarrs]
    print(f"Total datasets to process: {len(all_ds_names)}")
    
    # Prepare task arguments
    tasks = []
    for ds in all_ds_names:
        est_nodes = int(df_gt_summary.loc[ds, "estimated_number_of_nodes"]) if ds in df_gt_summary.index else 0
        n_frames = int(df_gt_summary.loc[ds, "n_frames"]) if ds in df_gt_summary.index else 100
        gt_ds = df_gt_nodes[df_gt_nodes['dataset'] == ds]
        gt_rows = gt_ds[['t', 'z', 'y', 'x']].to_dict(orient="records") if not gt_ds.empty else []
        tasks.append((ds, est_nodes, n_frames, gt_rows))
        
    out_progress = "s3_results_integration_and_submission/working/s4_full_199_eval_progress.csv"
    out_final_csv = "s3_results_integration_and_submission/__履歴__/s4_optimal_parameters_all199_summary.csv"
    out_final_xlsx = "s3_results_integration_and_submission/__履歴__/s4_optimal_parameters_all199_summary.xlsx"
    
    results = []
    # Use Pool with chunksize=1 for incremental tracking
    with Pool(16) as pool:
        for idx, res in enumerate(pool.imap_unordered(process_single_dataset, tasks), 1):
            results.append(res)
            # Save incremental progress
            pd.DataFrame(results).to_csv(out_progress, index=False)
            
            elapsed_min = (time.time() - t_global_start) / 60.0
            avg_sec_per_ds = (time.time() - t_global_start) / idx
            eta_min = (len(tasks) - idx) * avg_sec_per_ds / 60.0
            
            print(f"[{idx}/{len(tasks)}] {res['dataset']:15s} ({res['series']}) | "
                  f"Recall: {res['recall']*100:5.1f}% ({res['tp_count']}/{res['gt_total']}) | "
                  f"P/E: {res['pe_ratio']:5.2f} ({res['pred_nodes']}/{res['estimated_nodes']}) | "
                  f"TH: {res['avg_threshold']:.4f} | Time: {res['elapsed_sec']:.1f}s | "
                  f"Total: {elapsed_min:.1f}m (ETA: {eta_min:.1f}m)", flush=True)
                  
    df_final = pd.DataFrame(results).sort_values("dataset")
    df_final.to_csv(out_final_csv, index=False)
    df_final.to_excel(out_final_xlsx, index=False)
    
    total_time_min = (time.time() - t_global_start) / 60.0
    print(f"\n=== ALL 199 DATASETS PROCESSED IN {total_time_min:.2f} MINUTES ===")
    print(f"Macro-mean Recall: {df_final['recall'].mean()*100:.2f}%")
    print(f"Macro-mean P/E ratio: {df_final['pe_ratio'].mean():.3f}")
    print(f"Total Predicted Nodes: {df_final['pred_nodes'].sum():,}")
    print(f"Total Estimated Nodes: {df_final['estimated_nodes'].sum():,}")
    print(f"Micro P/E ratio: {df_final['pred_nodes'].sum() / df_final['estimated_nodes'].sum():.3f}")
    print(f"Saved results to:\n  - {out_final_csv}\n  - {out_final_xlsx}")

if __name__ == '__main__':
    main()
