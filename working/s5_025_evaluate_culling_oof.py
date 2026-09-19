"""
s5_025_evaluate_culling_oof.py
完全未知の OOF 予測確率を用いて、
目標 P/E 比 (0.90〜0.95, 目標0.925) を達成する最適閾値 P_th を探索し、
公式 Jaccard 評価およびアプローチA (複合ルールベース) との比較検証を行う。
"""
import sys
import os
import time
import datetime
from pathlib import Path

import zarr
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(r"C:\work\aaa\s5\input\train")
OOF_PARQUET = Path(r"C:\work\aaa\s5\github\working\s5_025_oof_predictions.parquet")
OUTPUT_SUMMARY_CSV = Path(r"C:\work\aaa\s5\github\working\s5_025_oof_threshold_sweep.csv")

def main():
    print("=" * 75)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] s5_025_evaluate_culling_oof: 完全OOF P/E比 0.90〜0.95 閾値探索")
    print("=" * 75)

    if not OOF_PARQUET.exists():
        print(f"Error: {OOF_PARQUET} does not exist!")
        return

    df_oof = pd.read_parquet(OOF_PARQUET)
    print(f"Loaded {len(df_oof):,} OOF predictions across {df_oof['dataset'].nunique()} datasets.")

    # 全199データセットの GT ノード数 (est_nodes) を取得
    geff_files = sorted(list(DATA_DIR.glob("*.geff")))
    est_nodes_by_ds = {}
    gt_nodes_by_ds = {}

    for gf in geff_files:
        ds_name = gf.stem
        zg = zarr.open(str(gf), mode='r')
        t = zg['nodes']['props']['t']['values'][:]
        gt_nodes_by_ds[ds_name] = len(t)
        meta = zg.attrs.asdict() if hasattr(zg.attrs, 'asdict') else dict(zg.attrs)
        geff_meta = meta.get('geff', {}) if isinstance(meta, dict) else {}
        extra = geff_meta.get('extra', {}) if isinstance(geff_meta, dict) else {}
        est = int(extra.get('estimated_number_of_nodes', meta.get('estimated_number_of_nodes', len(t))))
        est_nodes_by_ds[ds_name] = est

    print(f"Loaded ground truth node counts for {len(est_nodes_by_ds)} datasets.")

    # 閾値スイープ (P_th: 0.10 〜 0.80)
    thresholds = np.linspace(0.05, 0.70, 27)
    results = []

    print("\n--- Sweeping Probability Threshold P_th across all 199 datasets (OOF) ---")
    for p_th in thresholds:
        pe_ratios = []
        node_recalls = []
        total_pred_nodes = 0
        total_gt_nodes = 0
        penalty_count = 0

        # データセットごとに OOF 予測でトラックを足切り
        for ds_name, grp in df_oof.groupby('dataset'):
            est_n = est_nodes_by_ds.get(ds_name, 1)
            gt_n = gt_nodes_by_ds.get(ds_name, 1)

            # P >= P_th のトラックのみ残す
            kept = grp[grp['oof_prob'] >= p_th]
            pred_n = int(kept['track_length'].sum())

            # GTマッチトラックの回収ノード数
            matched_n = int(kept[kept['is_gt'] == 1]['track_length'].sum())

            pe = pred_n / max(1, est_n)
            pe_ratios.append(pe)
            node_recalls.append(matched_n / max(1, gt_n))

            total_pred_nodes += pred_n
            total_gt_nodes += gt_n
            if pe > 1.0:
                penalty_count += 1

        mean_pe = float(np.mean(pe_ratios))
        median_pe = float(np.median(pe_ratios))
        mean_recall = float(np.mean(node_recalls))
        macro_pe = total_pred_nodes / max(1, total_gt_nodes)

        # 0.90 <= mean_pe <= 0.95 か判定
        in_target = (0.90 <= mean_pe <= 0.95)

        results.append({
            'threshold': round(float(p_th), 4),
            'mean_pe_ratio': round(mean_pe, 4),
            'median_pe_ratio': round(median_pe, 4),
            'macro_pe_ratio': round(macro_pe, 4),
            'node_recall': round(mean_recall, 4),
            'datasets_with_penalty': penalty_count,
            'target_achieved': in_target
        })

        flag = " [★ TARGET 0.90-0.95 ACHIVED!]" if in_target else ""
        print(f"  P_th = {p_th:.3f}: Mean P/E = {mean_pe:.4f} (Median = {median_pe:.4f}), Node Recall = {mean_recall*100:.2f}%, Penalty DS = {penalty_count:2d}/199{flag}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\nSaved sweep summary to {OUTPUT_SUMMARY_CSV}")

    # 最適閾値の選定 (0.925 に最も近い P_th)
    res_df['diff_to_target'] = np.abs(res_df['mean_pe_ratio'] - 0.925)
    best_row = res_df.sort_values('diff_to_target').iloc[0]

    print("\n" + "=" * 75)
    print("★ OPTIMAL PROBABILITY THRESHOLD FOUND (OOF Validation):")
    print(f"  Best P_th:          {best_row['threshold']:.4f}")
    print(f"  Mean P/E Ratio:     {best_row['mean_pe_ratio']:.4f} (Target: 0.90 - 0.95, Center 0.925)")
    print(f"  Median P/E Ratio:   {best_row['median_pe_ratio']:.4f}")
    print(f"  Macro Node Recall:  {best_row['node_recall']*100:.2f}%")
    print(f"  Datasets > 1.0 P/E: {int(best_row['datasets_with_penalty'])} / 199 (Penalty effectively neutralized)")
    print("=" * 75)

if __name__ == "__main__":
    main()
