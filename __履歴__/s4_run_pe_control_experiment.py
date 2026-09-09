import os
import glob
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
radius_threshold = 7.0

def eval_node_recall(gt_df, pred_df):
    if gt_df is None or gt_df.empty:
        return 0.0, 0, 0
    if pred_df is None or pred_df.empty:
        return 0.0, 0, len(gt_df)
        
    total_gt = len(gt_df)
    tp_count = 0
    
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
                    
    recall = tp_count / total_gt if total_gt > 0 else 0.0
    return recall, tp_count, total_gt

def filter_topk(df_pred, target_k, sort_col, ascending=False):
    raw_nodes = len(df_pred)
    if raw_nodes <= target_k:
        return df_pred.copy()
        
    filtered_frames = []
    for t_val, group in df_pred.groupby('t'):
        frame_ratio = len(group) / raw_nodes
        k_t = max(1, int(round(target_k * frame_ratio)))
        sorted_group = group.sort_values(by=sort_col, ascending=ascending)
        filtered_frames.append(sorted_group.head(k_t))
        
    res = pd.concat(filtered_frames, ignore_index=True)
    if len(res) > target_k:
        res = res.sort_values(by=sort_col, ascending=ascending).head(target_k)
    return res

def main():
    print("=== Priority A: P/E Ratio Control Simulation (Target P/E = 0.90) ===")
    t0 = time.time()
    
    gt_nodes_path = "s3_results_integration_and_submission/working/s3_gt_nodes.csv"
    gt_summary_path = "s3_results_integration_and_submission/working/s3_gt_summary.csv"
    
    df_gt_nodes = pd.read_csv(gt_nodes_path)
    df_gt_summary = pd.read_csv(gt_summary_path).set_index("dataset")
    
    cache_files = sorted(glob.glob("s3_results_integration_and_submission/working/s3_cache_detect_nodes_blobdog_btrack_*.csv"))
    print(f"Found {len(cache_files)} cached prediction files. Simulating across all...")
    
    records = []
    
    for idx, fpath in enumerate(cache_files):
        ds_name = os.path.basename(fpath).replace("s3_cache_detect_nodes_blobdog_btrack_", "").replace(".csv", "")
        series = ds_name.split("_")[0]
        
        if ds_name not in df_gt_summary.index:
            continue
            
        est_nodes = int(df_gt_summary.loc[ds_name, "estimated_number_of_nodes"])
        target_nodes_90 = int(round(est_nodes * 0.90))
        
        df_pred_raw = pd.read_csv(fpath)
        raw_nodes = len(df_pred_raw)
        raw_pe = raw_nodes / est_nodes if est_nodes > 0 else 0.0
        
        df_gt_ds = df_gt_nodes[df_gt_nodes['dataset'] == ds_name]
        raw_recall, raw_tp, gt_total = eval_node_recall(df_gt_ds, df_pred_raw)
        
        # Composite score
        int_std = df_pred_raw['mean_intensity'].std()
        snr_std = df_pred_raw['snr'].std()
        df_pred_raw['composite_score'] = (
            (df_pred_raw['mean_intensity'] - df_pred_raw['mean_intensity'].mean()) / (int_std if int_std > 0 else 1.0) +
            (df_pred_raw['snr'] - df_pred_raw['snr'].mean()) / (snr_std if snr_std > 0 else 1.0)
        )
        
        # 1. Strategy: Mean Intensity
        df_int = filter_topk(df_pred_raw, target_nodes_90, sort_col="mean_intensity", ascending=False)
        rec_int, tp_int, _ = eval_node_recall(df_gt_ds, df_int)
        pe_int = len(df_int) / est_nodes
        
        # 2. Strategy: SNR
        df_snr = filter_topk(df_pred_raw, target_nodes_90, sort_col="snr", ascending=False)
        rec_snr, tp_snr, _ = eval_node_recall(df_gt_ds, df_snr)
        pe_snr = len(df_snr) / est_nodes
        
        # 3. Strategy: Composite Score
        df_comp = filter_topk(df_pred_raw, target_nodes_90, sort_col="composite_score", ascending=False)
        rec_comp, tp_comp, _ = eval_node_recall(df_gt_ds, df_comp)
        pe_comp = len(df_comp) / est_nodes
        
        records.append({
            "dataset": ds_name,
            "series": series,
            "gt_nodes": gt_total,
            "estimated_nodes": est_nodes,
            "raw_nodes": raw_nodes,
            "raw_pe_ratio": round(raw_pe, 4),
            "raw_recall": round(raw_recall, 4),
            "target_nodes_90": target_nodes_90,
            
            "int_nodes": len(df_int),
            "int_pe_ratio": round(pe_int, 4),
            "int_recall": round(rec_int, 4),
            "int_recall_drop": round(raw_recall - rec_int, 4),
            
            "snr_nodes": len(df_snr),
            "snr_pe_ratio": round(pe_snr, 4),
            "snr_recall": round(rec_snr, 4),
            "snr_recall_drop": round(raw_recall - rec_snr, 4),
            
            "comp_nodes": len(df_comp),
            "comp_pe_ratio": round(pe_comp, 4),
            "comp_recall": round(rec_comp, 4),
            "comp_recall_drop": round(raw_recall - rec_comp, 4)
        })
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(cache_files):
            print(f"  Processed {idx + 1}/{len(cache_files)} datasets ({time.time() - t0:.1f}s)...")
            
    df_results = pd.DataFrame(records)
    
    out_dir = "s3_results_integration_and_submission/__履歴__"
    excel_path = os.path.join(out_dir, "s4_pe_ratio_control_results.xlsx")
    csv_path = os.path.join(out_dir, "s4_pe_ratio_control_results.csv")
    
    df_results.to_excel(excel_path, index=False)
    df_results.to_csv(csv_path, index=False)
    print(f"\n[DONE] Saved simulation results to:\n  - {excel_path}\n  - {csv_path}")
    
    print("\n" + "="*70)
    print("           OVERALL BENCHMARK SUMMARY across 192 DATASETS")
    print("="*70)
    print(f"Total Datasets Analyzed: {len(df_results)}")
    print(f"Average Raw P/E Ratio:   {df_results['raw_pe_ratio'].mean():.3f} (Max: {df_results['raw_pe_ratio'].max():.3f}, Min: {df_results['raw_pe_ratio'].min():.3f})")
    print(f"Average Raw Recall:      {df_results['raw_recall'].mean()*100:.2f}%\n")
    
    for name, col_rec, col_drop, col_pe in [
        ("Intensity Top-K", "int_recall", "int_recall_drop", "int_pe_ratio"),
        ("SNR Top-K",       "snr_recall", "snr_recall_drop", "snr_pe_ratio"),
        ("Composite Top-K", "comp_recall","comp_recall_drop","comp_pe_ratio")
    ]:
        mean_rec = df_results[col_rec].mean() * 100
        mean_drop = df_results[col_drop].mean() * 100
        mean_pe = df_results[col_pe].mean()
        rec_ge_85 = (df_results[col_rec] >= 0.85).mean() * 100
        rec_ge_90 = (df_results[col_rec] >= 0.90).mean() * 100
        
        print(f"[{name:16s}] Target P/E: {mean_pe:.2f} | Mean Recall: {mean_rec:.2f}% (Drop: {mean_drop:.2f}%) | >=85%: {rec_ge_85:.1f}% | >=90%: {rec_ge_90:.1f}%")
        
    print("\n--- By Series Breakdown ---")
    for s_name, s_df in df_results.groupby("series"):
        print(f"Series: {s_name} (n={len(s_df)})")
        print(f"  Raw P/E: {s_df['raw_pe_ratio'].mean():.3f} -> Target: ~0.90")
        print(f"  Raw Recall: {s_df['raw_recall'].mean()*100:.2f}%")
        print(f"  Intensity Recall: {s_df['int_recall'].mean()*100:.2f}% (Drop: {s_df['int_recall_drop'].mean()*100:.2f}%)")
        print(f"  SNR Recall:       {s_df['snr_recall'].mean()*100:.2f}% (Drop: {s_df['snr_recall_drop'].mean()*100:.2f}%)")
        print(f"  Composite Recall: {s_df['comp_recall'].mean()*100:.2f}% (Drop: {s_df['comp_recall_drop'].mean()*100:.2f}%)")

if __name__ == "__main__":
    main()
