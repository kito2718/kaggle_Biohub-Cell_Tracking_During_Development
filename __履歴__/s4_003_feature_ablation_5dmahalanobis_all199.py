"""
s4_003_feature_ablation_5dmahalanobis_all199.py
全199データセットを対象とした「5D Mahalanobis 5大特徴量アブレーション検証」
1. ベースライン (5D全部入り, w=1.0)
2. 空間距離のみ (0D / Spatial only, w=0.0)
3. 1特徴量除外 (Leave-One-Out: -1D) x 5パターン
4. 1特徴量のみ追加 (Single Feature: +1D) x 5パターン
"""
import glob
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Maru Gothic Pro', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    start_total = time.time()
    print("=" * 75)
    print(">>> s4_003: 全199データセット 5D Mahalanobis 特徴量アブレーション検証")
    print("=" * 75)

    working_dir = "s3_results_integration_and_submission/working"
    history_dir = None
    for d in os.listdir("s3_results_integration_and_submission"):
        if d.startswith("__"):
            history_dir = os.path.join("s3_results_integration_and_submission", d)
            break
    if history_dir is None:
        history_dir = "s3_results_integration_and_submission/__履歴__"
        os.makedirs(history_dir, exist_ok=True)
    print(f"[*] 出力先履歴ディレクトリ: {history_dir}")

    # 1. データ読み込み
    t0 = time.time()
    node_files = sorted(glob.glob(os.path.join(working_dir, "s3_01_detect_nodes_pred_blobdog_5dmahalanobis_*.csv")))
    print(f"[*] ノードファイル {len(node_files)} 個を読み込み中...")
    all_nodes = pd.concat([pd.read_csv(f) for f in node_files], ignore_index=True)
    print(f"  -> ノード総数: {len(all_nodes):,} 件 ({all_nodes['dataset'].nunique()} データセット) [{time.time()-t0:.2f}s]")

    t0 = time.time()
    nc_files = sorted(glob.glob(os.path.join(working_dir, "s3_02_check_nodes_details_blobdog_5dmahalanobis_*.csv")))
    print(f"[*] ノード照合ファイル {len(nc_files)} 個を読み込み中...")
    dfs_nc = [pd.read_csv(f, usecols=['dataset', 'eval_result', 'gt_node_id', 'pred_node_id']) for f in nc_files]
    all_nc = pd.concat(dfs_nc, ignore_index=True)
    tp_nc = all_nc[all_nc['eval_result'] == 'TP']
    print(f"  -> TPノード総数: {len(tp_nc):,} 件 [{time.time()-t0:.2f}s]")

    gt_edges = pd.read_csv(os.path.join(working_dir, "s3_gt_edges.csv"))
    print(f"[*] GTエッジ総数: {len(gt_edges):,} 件 ({gt_edges['dataset'].nunique()} データセット)")

    # 2. アブレーションパターンの定義
    all_5_features = ['mean_intensity', 'snr', 'z_depth_ratio', 'estimated_radius_um', 'volume_um3']
    
    ablation_configs = [
        # (設定名, 特徴量リスト, 説明)
        ("Baseline_5D", all_5_features, "5大特徴量すべて (現状)"),
        ("Spatial_Only", [], "空間距離のみ (特徴量なし)"),
        
        # Leave-One-Out (1特徴量を除外)
        ("No_mean_intensity", [f for f in all_5_features if f != 'mean_intensity'], "輝度(mean_intensity)を除外"),
        ("No_snr", [f for f in all_5_features if f != 'snr'], "SNR(snr)を除外"),
        ("No_z_depth_ratio", [f for f in all_5_features if f != 'z_depth_ratio'], "Z深度比率(z_depth_ratio)を除外"),
        ("No_estimated_radius_um", [f for f in all_5_features if f != 'estimated_radius_um'], "推定半径(estimated_radius_um)を除外"),
        ("No_volume_um3", [f for f in all_5_features if f != 'volume_um3'], "体積(volume_um3)を除外"),

        # Single Feature (1特徴量のみ追加)
        ("Only_mean_intensity", ['mean_intensity'], "空間 + 輝度のみ"),
        ("Only_snr", ['snr'], "空間 + SNRのみ"),
        ("Only_z_depth_ratio", ['z_depth_ratio'], "空間 + Z深度比率のみ"),
        ("Only_estimated_radius_um", ['estimated_radius_um'], "空間 + 推定半径のみ"),
        ("Only_volume_um3", ['volume_um3'], "空間 + 体積のみ"),
    ]

    # 厳格ルール: 物理異方性スケールを適用
    scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
    max_search_radius_um = 7.0
    feature_weight = 1.0  # 最適重み

    unique_datasets = sorted(all_nodes['dataset'].unique())
    n_datasets = len(unique_datasets)
    print(f"\n[*] 全 {n_datasets} データセットに対して {len(ablation_configs)} 種類のアブレーション検証を開始します...")

    results = {cfg[0]: [] for cfg in ablation_configs}

    t_ablation_start = time.time()
    for idx, ds in enumerate(unique_datasets):
        ds_nodes = all_nodes[all_nodes['dataset'] == ds].sort_values('t')
        ds_gt = gt_edges[gt_edges['dataset'] == ds]
        ds_tp = tp_nc[tp_nc['dataset'] == ds]
        gt_to_pred = dict(zip(ds_tp['gt_node_id'], ds_tp['pred_node_id']))
        n_gt_edges = len(ds_gt)

        frames = sorted(ds_nodes['t'].unique())
        # フレームごとのノードデータと空間距離・各特徴量差分を準備
        frame_pairs = []
        for f_i in range(len(frames) - 1):
            t_curr, t_next = frames[f_i], frames[f_i+1]
            if t_next != t_curr + 1:
                continue
            df_c = ds_nodes[ds_nodes['t'] == t_curr]
            df_n = ds_nodes[ds_nodes['t'] == t_next]
            if df_c.empty or df_n.empty:
                continue
            pos_c = df_c[['z', 'y', 'x']].values * scale_vec
            pos_n = df_n[['z', 'y', 'x']].values * scale_vec
            s_dist = cdist(pos_c, pos_n)
            frame_pairs.append((df_c, df_n, s_dist))

        # 各アブレーション設定でマッチング
        for cfg_name, feat_cols, _ in ablation_configs:
            edges = []
            for df_c, df_n, s_dist in frame_pairs:
                cost = 1.0 * s_dist
                if len(feat_cols) > 0:
                    feats_c = df_c[feat_cols].values.astype(np.float32)
                    feats_n = df_n[feat_cols].values.astype(np.float32)
                    if len(feat_cols) == 1:
                        # 1次元特徴量の場合は標準化絶対値距離
                        comb = np.concatenate([feats_c, feats_n])
                        std = np.std(comb) if np.std(comb) > 1e-6 else 1.0
                        f_dist = cdist(feats_c, feats_n, metric='euclidean') / std
                    else:
                        comb = np.vstack([feats_c, feats_n])
                        cov = np.cov(comb, rowvar=False)
                        inv_cov = np.linalg.pinv(cov)
                        try:
                            f_dist = cdist(feats_c, feats_n, metric='mahalanobis', VI=inv_cov)
                        except Exception:
                            f_dist = cdist(feats_c, feats_n, metric='cityblock')
                    cost += feature_weight * f_dist

                r_ind, c_ind = linear_sum_assignment(cost)
                c_ids = df_c['node_id'].values
                n_ids = df_n['node_id'].values
                for r, c in zip(r_ind, c_ind):
                    if s_dist[r, c] <= max_search_radius_um:
                        edges.append((int(c_ids[r]), int(n_ids[c])))

            p_set = set(edges)
            tp_count = 0
            if n_gt_edges > 0:
                for _, row in ds_gt.iterrows():
                    g_src = row['source_id']
                    g_tgt = row['target_id']
                    p_src = gt_to_pred.get(g_src, g_src)
                    p_tgt = gt_to_pred.get(g_tgt, g_tgt)
                    if (p_src, p_tgt) in p_set:
                        tp_count += 1
                rec = tp_count / n_gt_edges
            else:
                rec = 0.0

            n_pred = len(edges)
            prec = tp_count / n_pred if n_pred > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            results[cfg_name].append({
                'dataset': ds,
                'series': ds.split('_')[0],
                'gt_edges': n_gt_edges,
                'pred_edges': n_pred,
                'tp': tp_count,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

        if (idx + 1) % 20 == 0 or (idx + 1) == n_datasets:
            elapsed = time.time() - t_ablation_start
            print(f"  [{idx+1:3d}/{n_datasets:3d}] ({elapsed:.1f}s) {ds} 完了")

    print(f"\n[*] 全199データセットのアブレーション検証が完了しました ({time.time()-t_ablation_start:.2f}s)")

    # 3. 集計と保存
    summary_rows = []
    full_dfs = []
    for cfg_name, feat_cols, desc in ablation_configs:
        df_cfg = pd.DataFrame(results[cfg_name])
        df_cfg['config'] = cfg_name
        full_dfs.append(df_cfg)

        rec_44b6 = df_cfg[df_cfg['series'] == '44b6']['recall'].mean()
        rec_6bba = df_cfg[df_cfg['series'] == '6bba']['recall'].mean()
        macro_rec = df_cfg['recall'].mean()
        total_tp = df_cfg['tp'].sum()
        total_gt = df_cfg['gt_edges'].sum()
        micro_rec = total_tp / total_gt if total_gt > 0 else 0.0

        summary_rows.append({
            'config': cfg_name,
            'description': desc,
            'num_features': len(feat_cols),
            'features_used': ', '.join(feat_cols) if feat_cols else '(None)',
            'macro_recall': macro_rec,
            'recall_44b6_dense': rec_44b6,
            'recall_6bba_sparse': rec_6bba,
            'micro_recall': micro_rec,
            'macro_precision': df_cfg['precision'].mean(),
            'macro_f1': df_cfg['f1'].mean(),
            'total_tp': total_tp,
            'total_gt': total_gt
        })

    summary_df = pd.DataFrame(summary_rows)
    all_details_df = pd.concat(full_dfs, ignore_index=True)

    # ベースラインとの差分 (Impact / Delta) を計算
    base_macro = summary_df.loc[summary_df['config'] == 'Baseline_5D', 'macro_recall'].values[0]
    base_tp = summary_df.loc[summary_df['config'] == 'Baseline_5D', 'total_tp'].values[0]
    summary_df['delta_recall'] = (summary_df['macro_recall'] - base_macro) * 100.0
    summary_df['delta_tp'] = summary_df['total_tp'] - base_tp

    csv_path = os.path.join(history_dir, "s4_003_feature_ablation_all199.csv")
    xlsx_path = os.path.join(history_dir, "s4_003_feature_ablation_all199.xlsx")
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Ablation_Summary', index=False)
        all_details_df.to_excel(writer, sheet_name='Details_All199', index=False)
    print(f"[+] 保存完了: {csv_path}")
    print(f"[+] 保存完了: {xlsx_path}")

    # 4. 可視化プロット図の生成 (300 DPI)
    print("\n[*] アブレーション比較プロット図を生成中...")

    # (1) Leave-One-Out (寄与度・重要度分析) プロット
    loo_configs = [cfg for cfg in summary_rows if cfg['config'].startswith('No_') or cfg['config'] == 'Baseline_5D']
    df_loo = pd.DataFrame(loo_configs)
    df_loo['delta'] = (df_loo['macro_recall'] - base_macro) * 100.0

    plt.figure(figsize=(11, 6), dpi=300)
    # ベースラインからの低下幅 = その特徴量の重要度 (低下が大きいほど重要)
    loo_only = df_loo[df_loo['config'] != 'Baseline_5D'].sort_values('delta')
    feature_names = [c.replace('No_', '') for c in loo_only['config']]
    y_pos = np.arange(len(feature_names))

    colors = ['#d62728' if d < 0 else '#2ca02c' for d in loo_only['delta']]
    bars = plt.barh(y_pos, loo_only['delta'], color=colors, edgecolor='black', height=0.6)

    plt.axvline(0, color='black', linewidth=1.2)
    plt.yticks(y_pos, feature_names, fontsize=11, fontweight='bold')
    plt.xlabel("Edge Recall 変化幅 (ポイント: 5Dベースラインとの差分)", fontsize=12)
    plt.title("5D Mahalanobis: 特徴量アブレーション (Leave-One-Out 除外時のRecall変化)", fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6, axis='x')

    for bar, d, cfg in zip(bars, loo_only['delta'], loo_only['config']):
        x_val = bar.get_width()
        txt = f"{d:+.2f}% (Recall: {summary_df.loc[summary_df['config']==cfg, 'macro_recall'].values[0]*100:.2f}%)"
        if x_val < 0:
            plt.text(x_val - 0.02, bar.get_y() + bar.get_height()/2, txt, va='center', ha='right', fontsize=10, fontweight='bold')
        else:
            plt.text(x_val + 0.02, bar.get_y() + bar.get_height()/2, txt, va='center', ha='left', fontsize=10, fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join(history_dir, "s4_003_feature_ablation_leave_one_out.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[+] プロット保存完了: {p1}")

    # (2) 全構成の Recall 比較棒グラフ (All Configurations)
    plt.figure(figsize=(13, 7), dpi=300)
    order_cfg = ['Spatial_Only'] + [f"Only_{f}" for f in all_5_features] + [f"No_{f}" for f in all_5_features] + ['Baseline_5D']
    summary_plot = summary_df.set_index('config').loc[order_cfg].reset_index()

    x_idx = np.arange(len(summary_plot))
    bar_colors = ['#7f7f7f'] + ['#1f77b4']*5 + ['#ff7f0e']*5 + ['#2ca02c']
    bars = plt.bar(x_idx, summary_plot['macro_recall'] * 100, color=bar_colors, edgecolor='black', width=0.7)

    plt.ylim(64.5, 68.5)
    plt.axhline(base_macro * 100, color='#2ca02c', linestyle='--', linewidth=2.0, label=f'5D Baseline ({base_macro*100:.2f}%)')
    spatial_rec = summary_plot.loc[summary_plot['config'] == 'Spatial_Only', 'macro_recall'].values[0] * 100
    plt.axhline(spatial_rec, color='#7f7f7f', linestyle=':', linewidth=1.8, label=f'Spatial Only ({spatial_rec:.2f}%)')

    plt.xticks(x_idx, summary_plot['config'], rotation=45, ha='right', fontsize=10)
    plt.ylabel("Macro Mean Edge Recall (%)", fontsize=12)
    plt.title("5D Mahalanobis: 特徴量アブレーション全12構成比較 (全199データセット)", fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend(fontsize=11, loc='upper left')

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p2 = os.path.join(history_dir, "s4_003_feature_ablation_all_configs.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[+] プロット保存完了: {p2}")

    # 5. サマリ出力
    print("\n" + "=" * 75)
    print(">>> 5D Mahalanobis 特徴量アブレーション結果サマリ (全199データセット)")
    print("=" * 75)
    display_cols = ['config', 'num_features', 'macro_recall', 'recall_44b6_dense', 'recall_6bba_sparse', 'delta_recall', 'delta_tp']
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 75)
    print(f"総所要時間: {time.time()-start_total:.2f} 秒")

if __name__ == '__main__':
    main()
