import os
import glob
import time
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Step 1: Extract Optical Features across 199 Datasets ===")
    t0 = time.time()
    
    # 1. Load ground truth summary and current performance summary
    gt_summary_path = "s3_results_integration_and_submission/working/s3_gt_summary.csv"
    nodes_summary_path = "s3_results_integration_and_submission/working/old_blobdog/s3_02_check_nodes_summary_blobdog_5dmahalanobis_THRESHOLD=0.038.xlsx"
    
    df_gt = pd.read_csv(gt_summary_path)
    df_perf = pd.read_excel(nodes_summary_path)
    
    # Clean df_perf column names
    df_perf = df_perf.rename(columns={
        "total_pred_nodes/estimated_number_of_nodes": "pe_ratio"
    })
    
    zarr_dir = "s1_local_env/input/train"
    zarr_files = sorted(glob.glob(os.path.join(zarr_dir, "*.zarr")))
    print(f"Found {len(zarr_files)} zarr files.")
    
    records = []
    for i, zarr_path in enumerate(zarr_files):
        ds_name = os.path.basename(zarr_path).replace(".zarr", "")
        series = ds_name.split("_")[0] # '44b6' or '6bba'
        
        try:
            z = zarr.open(zarr_path, mode="r")
            # Sample middle frame (e.g. frame 25 or 50) for representative optical properties
            arr = z["0"][25] # shape: (64, 256, 256)
            
            # Fast percentiles
            p01, p10, p25, p50, p75, p90, p99, p995 = np.percentile(
                arr, [1.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.0, 99.5]
            )
            
            # Noise estimation: robust std of background via IQR (p75 - p25) / 1.349
            bg_noise = (p75 - p25) / 1.349
            # Dynamic range & contrast ratio
            dyn_range = p995 - p01
            # SNR proxy: difference between bright spots and background divided by noise
            snr_proxy = (p995 - p50) / (bg_noise + 1e-5)
            # Relative contrast: ratio of signal to background median
            contrast_ratio = p995 / (p50 + 1e-5)
            
            records.append({
                "dataset": ds_name,
                "series": series,
                "bg_median": float(p50),
                "bg_noise": float(bg_noise),
                "dyn_range": float(dyn_range),
                "snr_proxy": float(snr_proxy),
                "contrast_ratio": float(contrast_ratio),
                "p01": float(p01),
                "p995": float(p995)
            })
        except Exception as e:
            print(f"Error reading {ds_name}: {e}")
            
        if (i + 1) % 50 == 0 or (i + 1) == len(zarr_files):
            print(f"  Processed {i + 1}/{len(zarr_files)} datasets ({time.time() - t0:.1f}s)...")
            
    df_opt = pd.DataFrame(records)
    
    # Merge with GT info and current detection performance
    df_merged = df_opt.merge(
        df_gt[["dataset", "estimated_number_of_nodes", "n_frames"]], on="dataset", how="left"
    )
    df_merged["est_per_frame"] = df_merged["estimated_number_of_nodes"] / df_merged["n_frames"]
    
    df_merged = df_merged.merge(
        df_perf[["dataset", "total_pred_nodes", "pe_ratio", "recall", "precision", "f1_score"]],
        on="dataset",
        how="left"
    )
    
    out_csv = "s3_results_integration_and_submission/working/dataset_optical_profiles.csv"
    df_merged.to_csv(out_csv, index=False)
    print(f"Saved optical profiles to {out_csv} ({len(df_merged)} rows). Total time: {time.time() - t0:.1f}s")
    
    # === Step 2: Correlation Analysis & Plotting ===
    print("\n=== Step 2: Correlation Analysis & Visualization ===")
    
    # Select key numerical features
    num_cols = ["bg_median", "bg_noise", "dyn_range", "snr_proxy", "contrast_ratio", "est_per_frame", "pe_ratio", "recall"]
    corr = df_merged[num_cols].corr()
    print("\n--- Correlation with pe_ratio (P/E比) and recall ---")
    print(corr[["pe_ratio", "recall"]].to_string())
    
    # Plotting
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Biohub Cell Tracking: Optical Features vs Detection Performance (P/E Ratio & Recall)", fontsize=15)
    
    # Palette by series
    palette = {"44b6": "#1f77b4", "6bba": "#ff7f0e"}
    
    # 1. Histogram of P/E Ratio
    ax = axes[0, 0]
    sns.histplot(data=df_merged, x="pe_ratio", hue="series", bins=25, kde=True, ax=ax, palette=palette)
    ax.axvline(1.0, color="red", linestyle="--", label="Target Max (1.0)")
    ax.axvline(0.9, color="green", linestyle="--", label="Goal (0.90)")
    ax.set_title("Histogram: P/E Ratio Distribution")
    ax.set_xlabel("P/E Ratio (Pred / Est)")
    ax.legend()
    
    # 2. Histogram of Recall
    ax = axes[0, 1]
    sns.histplot(data=df_merged, x="recall", hue="series", bins=25, kde=True, ax=ax, palette=palette)
    ax.axvline(0.9, color="green", linestyle="--", label="Recall Goal (0.90)")
    ax.set_title("Histogram: Cell Recall Distribution")
    ax.set_xlabel("Cell Recall")
    ax.legend()
    
    # 3. Correlation Heatmap
    ax = axes[0, 2]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, square=True)
    ax.set_title("Correlation Heatmap")
    
    # 4. Scatter: SNR vs P/E Ratio
    ax = axes[1, 0]
    sns.scatterplot(data=df_merged, x="snr_proxy", y="pe_ratio", hue="series", ax=ax, palette=palette, s=50, alpha=0.8)
    ax.axhline(0.9, color="green", linestyle="--")
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_title("SNR Proxy vs P/E Ratio")
    ax.set_xlabel("SNR Proxy ((p99.5 - p50) / bg_noise)")
    ax.set_ylabel("P/E Ratio")
    
    # 5. Scatter: Background Noise vs P/E Ratio
    ax = axes[1, 1]
    sns.scatterplot(data=df_merged, x="bg_noise", y="pe_ratio", hue="series", ax=ax, palette=palette, s=50, alpha=0.8)
    ax.axhline(0.9, color="green", linestyle="--")
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_title("Background Noise vs P/E Ratio")
    ax.set_xlabel("Background Noise (IQR / 1.349)")
    ax.set_ylabel("P/E Ratio")
    
    # 6. Scatter: Estimated Cells Per Frame vs P/E Ratio
    ax = axes[1, 2]
    sns.scatterplot(data=df_merged, x="est_per_frame", y="pe_ratio", hue="series", ax=ax, palette=palette, s=50, alpha=0.8)
    ax.axhline(0.9, color="green", linestyle="--")
    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_title("Est Cells Per Frame vs P/E Ratio")
    ax.set_xlabel("Est Cells / Frame")
    ax.set_ylabel("P/E Ratio")
    
    plt.tight_layout()
    plot_path = "s3_results_integration_and_submission/working/dataset_optical_analysis.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
