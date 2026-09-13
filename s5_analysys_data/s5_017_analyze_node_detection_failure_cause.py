# -*- coding: utf-8 -*-
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(r"d:/BizOwn/000_Biw2/51_googleantigravity/007_kaggle_Biohub-Cell_Tracking_During_Development")
TRAIN_ZARR_DIR = BASE_DIR / "s5/input/train"
WORKING_DIR = BASE_DIR / "s5/github/working"
OUTPUT_CSV = BASE_DIR / "s5/github/s5_analysys_data/s5_017_node_failure_analysis.csv"

summary_df = pd.read_csv(WORKING_DIR / "s5_015DYNRADIUS_02_check_nodes_summary_blobdog_lgbm.csv")
worst10 = summary_df.sort_values("recall").head(10)["dataset"].tolist()
best10 = summary_df.sort_values("recall", ascending=False).head(10)["dataset"].tolist()

gt_nodes_all = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

pred_files = list(WORKING_DIR.glob("s5_015DYNRADIUS_01_detect_nodes_pred_blobdog_lgbm_*.csv"))
print(f"Loading pred node files: {len(pred_files)} files...")
pred_dfs = [pd.read_csv(f) for f in pred_files]
pred_nodes_all = pd.concat(pred_dfs, ignore_index=True)
print(f"Total pred nodes loaded: {len(pred_nodes_all):,}")

records = []
targets = [("Worst", ds) for ds in worst10] + [("Best", ds) for ds in best10]

for group, ds in targets:
    zarr_path = TRAIN_ZARR_DIR / f"{ds}.zarr"
    if not zarr_path.exists():
        print(f"Skip {ds}: Zarr not found")
        continue
    
    ds_summary = summary_df[summary_df["dataset"] == ds].iloc[0]
    node_recall = ds_summary["recall"]
    node_prec = ds_summary["precision"]
    gt_total = ds_summary["total_gt_nodes"]
    pred_total = ds_summary["total_pred_nodes"]
    
    z = zarr.open(str(zarr_path), mode="r")["0"]
    vol0 = np.array(z[0], dtype=np.float32)
    
    vol_min = float(vol0.min())
    vol_max = float(vol0.max())
    vol_mean = float(vol0.mean())
    vol_med = float(np.median(vol0))
    vol_p95 = float(np.percentile(vol0, 95))
    vol_p99 = float(np.percentile(vol0, 99))
    vol_p99_5 = float(np.percentile(vol0, 99.5))
    
    gt_ds_t0 = gt_nodes_all[(gt_nodes_all["dataset"] == ds) & (gt_nodes_all["t"] == 0)]
    n_gt_t0 = len(gt_ds_t0)
    
    pred_ds_t0 = pred_nodes_all[(pred_nodes_all["dataset"] == ds) & (pred_nodes_all["t"] == 0)]
    n_pred_t0 = len(pred_ds_t0)
    
    if n_pred_t0 > 0:
        z_border = (pred_ds_t0["z"] <= 1) | (pred_ds_t0["z"] >= vol0.shape[0] - 2)
        y_border = (pred_ds_t0["y"] <= 3) | (pred_ds_t0["y"] >= vol0.shape[1] - 4)
        x_border = (pred_ds_t0["x"] <= 3) | (pred_ds_t0["x"] >= vol0.shape[2] - 4)
        border_pred_ratio = float((z_border | y_border | x_border).mean())
        pred_x_mean = float(pred_ds_t0["x"].mean())
        pred_y_mean = float(pred_ds_t0["y"].mean())
        pred_z_mean = float(pred_ds_t0["z"].mean())
    else:
        border_pred_ratio = 0.0
        pred_x_mean, pred_y_mean, pred_z_mean = 0, 0, 0
        
    gt_center_vals = []
    gt_local_maxs = []
    gt_contrasts = []
    if n_gt_t0 > 0:
        for _, row in gt_ds_t0.iterrows():
            zz = max(0, min(vol0.shape[0]-1, int(round(row["z"]))))
            yy = max(0, min(vol0.shape[1]-1, int(round(row["y"]))))
            xx = max(0, min(vol0.shape[2]-1, int(round(row["x"]))))
            c_val = vol0[zz, yy, xx]
            gt_center_vals.append(c_val)
            
            z1, z2 = max(0, zz-1), min(vol0.shape[0], zz+2)
            y1, y2 = max(0, yy-2), min(vol0.shape[1], yy+3)
            x1, x2 = max(0, xx-2), min(vol0.shape[2], xx+3)
            box = vol0[z1:z2, y1:y2, x1:x2]
            l_max = box.max()
            gt_local_maxs.append(l_max)
            gt_contrasts.append(l_max - vol_med)
            
        gt_val_mean = float(np.mean(gt_center_vals))
        gt_val_med = float(np.median(gt_center_vals))
        gt_contrast_med = float(np.median(gt_contrasts))
        gt_to_max_ratio = gt_val_med / (vol_max + 1e-6)
        
        scale_vec = np.array([1.625, 0.40625, 0.40625])
        pts = gt_ds_t0[["z", "y", "x"]].values * scale_vec
        if len(pts) > 1:
            from scipy.spatial.distance import pdist, squareform
            dists = squareform(pdist(pts))
            np.fill_diagonal(dists, np.inf)
            mean_nn_dist = float(dists.min(axis=1).mean())
        else:
            mean_nn_dist = np.nan
    else:
        gt_val_mean, gt_val_med, gt_contrast_med = np.nan, np.nan, np.nan
        gt_to_max_ratio = np.nan
        mean_nn_dist = np.nan

    p_low = np.percentile(vol0, 1.0)
    p_high = np.percentile(vol0, 99.5)
    vol_norm = np.clip((vol0 - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)
    
    if n_gt_t0 > 0:
        gt_norm_vals = []
        for _, row in gt_ds_t0.iterrows():
            zz = max(0, min(vol0.shape[0]-1, int(round(row["z"]))))
            yy = max(0, min(vol0.shape[1]-1, int(round(row["y"]))))
            xx = max(0, min(vol0.shape[2]-1, int(round(row["x"]))))
            gt_norm_vals.append(vol_norm[zz, yy, xx])
        gt_norm_med = float(np.median(gt_norm_vals))
    else:
        gt_norm_med = np.nan

    rec = {
        "Group": group,
        "Dataset": ds,
        "NodeRecall": node_recall,
        "NodePrec": node_prec,
        "GT_Nodes": gt_total,
        "Pred_Nodes": pred_total,
        "Ratio_Pred_GT": pred_total / (gt_total + 1e-6),
        "Border_Pred_Ratio": border_pred_ratio,
        "Pred_X_Mean": pred_x_mean,
        "Vol_Min": vol_min,
        "Vol_Med": vol_med,
        "Vol_P99_5": vol_p99_5,
        "Vol_Max": vol_max,
        "Max_to_Med_Ratio": vol_max / (vol_med + 1e-6),
        "GT_Val_Med": gt_val_med,
        "GT_Norm_Med": gt_norm_med,
        "GT_to_Max_Ratio": gt_to_max_ratio,
        "Mean_NN_Dist_um": mean_nn_dist
    }
    records.append(rec)
    print(f"[{group:5s}] {ds}: Recall={node_recall:.3f}, BorderPred={border_pred_ratio:.1%}, GT_NormMed={gt_norm_med:.3f}, Max/Med={rec['Max_to_Med_Ratio']:.1f}")

df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False)
print("Analysis finished. Saved to:", OUTPUT_CSV)
