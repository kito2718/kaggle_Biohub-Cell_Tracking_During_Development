"""
s4_002_optimize_5dmahalanobis_all199.py
全199データセットを対象とした「5D Mahalanobis 特徴量重み最適化」および「孤立ノード刈り取りによるP/E比制御シミュレーション」
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

# Set Japanese font / clean style
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Maru Gothic Pro', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    start_total = time.time()
    print("=" * 70)
    print(">>> s4_002: 全199データセット 5D Mahalanobis 重み最適化 & 孤立ノード検証")
    print("=" * 70)

    # 1. データの読み込み
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

    # ノードデータの読み込み
    t0 = time.time()
    node_files = sorted(glob.glob(os.path.join(working_dir, "s3_01_detect_nodes_pred_blobdog_5dmahalanobis_*.csv")))
    print(f"[*] ノードファイル {len(node_files)} 個を読み込み中...")
    all_nodes = pd.concat([pd.read_csv(f) for f in node_files], ignore_index=True)
    print(f"  -> ノード総数: {len(all_nodes):,} 件 ({all_nodes['dataset'].nunique()} データセット) [{time.time()-t0:.2f}s]")

    # TP ノード対応表の読み込み
    t0 = time.time()
    nc_files = sorted(glob.glob(os.path.join(working_dir, "s3_02_check_nodes_details_blobdog_5dmahalanobis_*.csv")))
    print(f"[*] ノード照合ファイル {len(nc_files)} 個を読み込み中...")
    dfs_nc = [pd.read_csv(f, usecols=['dataset', 'eval_result', 'gt_node_id', 'pred_node_id']) for f in nc_files]
    all_nc = pd.concat(dfs_nc, ignore_index=True)
    tp_nc = all_nc[all_nc['eval_result'] == 'TP']
    print(f"  -> TPノード総数: {len(tp_nc):,} 件 [{time.time()-t0:.2f}s]")

    # GT エッジデータの読み込み
    gt_edges = pd.read_csv(os.path.join(working_dir, "s3_gt_edges.csv"))
    print(f"[*] GTエッジ総数: {len(gt_edges):,} 件 ({gt_edges['dataset'].nunique()} データセット)")

    # GT サマリデータの読み込み (estimated_number_of_nodes)
    gt_summary = pd.read_csv(os.path.join(working_dir, "s3_gt_summary.csv")).set_index("dataset")

    # 2. 全199データセットに対する重みスイープ [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    feature_cols = ['mean_intensity', 'snr', 'z_depth_ratio', 'estimated_radius_um', 'volume_um3']
    scale_vec = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

    unique_datasets = sorted(all_nodes['dataset'].unique())
    n_datasets = len(unique_datasets)
    print(f"\n[*] 全 {n_datasets} データセットに対して重みスイープを開始します...")

    weight_results = {w: [] for w in weights}
    filtering_results = []
    track_lengths_all = []

    # 各データセットごとに処理
    t_sweep_start = time.time()
    for idx, ds in enumerate(unique_datasets):
        ds_t0 = time.time()
        ds_nodes = all_nodes[all_nodes['dataset'] == ds].sort_values('t')
        ds_gt = gt_edges[gt_edges['dataset'] == ds]
        ds_tp = tp_nc[tp_nc['dataset'] == ds]
        gt_to_pred = dict(zip(ds_tp['gt_node_id'], ds_tp['pred_node_id']))
        n_gt_edges = len(ds_gt)

        # フレーム間距離行列の事前計算
        frames = sorted(ds_nodes['t'].unique())
        precomputed = []
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

            feats_c = df_c[feature_cols].values.astype(np.float32)
            feats_n = df_n[feature_cols].values.astype(np.float32)
            comb = np.vstack([feats_c, feats_n])
            cov = np.cov(comb, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            try:
                f_dist = cdist(feats_c, feats_n, metric='mahalanobis', VI=inv_cov)
            except Exception:
                f_dist = cdist(feats_c, feats_n, metric='cityblock')
            precomputed.append((df_c['node_id'].values, df_n['node_id'].values, s_dist, f_dist))

        # 各重みでマッチング
        best_edges_for_ds = None
        for fw in weights:
            edges = []
            for c_ids, n_ids, s_dist, f_dist in precomputed:
                cost = 1.0 * s_dist + fw * f_dist
                r_ind, c_ind = linear_sum_assignment(cost)
                for r, c in zip(r_ind, c_ind):
                    if s_dist[r, c] <= 7.0:
                        edges.append((int(c_ids[r]), int(n_ids[c])))

            # 評価
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

            weight_results[fw].append({
                'dataset': ds,
                'weight': fw,
                'gt_edges': n_gt_edges,
                'pred_edges': n_pred,
                'tp': tp_count,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

            if fw == 0.5:
                best_edges_for_ds = edges

        # 3. 孤立ノード刈り取り検証 (w=0.5 のトラッキング結果に基づく)
        total_detected_nodes = len(ds_nodes)
        edge_node_ids = set()
        in_degree = {}
        out_degree = {}
        for s, t_n in best_edges_for_ds:
            edge_node_ids.add(s)
            edge_node_ids.add(t_n)
            out_degree[s] = out_degree.get(s, 0) + 1
            in_degree[t_n] = in_degree.get(t_n, 0) + 1

        active_nodes_count = len(ds_nodes[ds_nodes['node_id'].isin(edge_node_ids)])
        isolated_nodes_count = total_detected_nodes - active_nodes_count
        isolated_ratio = isolated_nodes_count / total_detected_nodes if total_detected_nodes > 0 else 0.0

        gt_est = gt_summary.loc[ds, 'estimated_number_of_nodes'] if ds in gt_summary.index else np.nan
        pe_before = total_detected_nodes / gt_est if pd.notna(gt_est) and gt_est > 0 else np.nan
        pe_after = active_nodes_count / gt_est if pd.notna(gt_est) and gt_est > 0 else np.nan

        filtering_results.append({
            'dataset': ds,
            'gt_estimated_nodes': gt_est,
            'nodes_before_filtering': total_detected_nodes,
            'nodes_after_filtering': active_nodes_count,
            'isolated_nodes_removed': isolated_nodes_count,
            'isolated_removal_pct': isolated_ratio * 100.0,
            'pe_ratio_before': pe_before,
            'pe_ratio_after': pe_after,
            'edge_count': len(best_edges_for_ds)
        })

        # トラック長のサンプリング (グラフ連結成分長)
        # 次のノードへのポインタ
        next_ptr = {s: t_n for s, t_n in best_edges_for_ds}
        start_nodes = [s for s in edge_node_ids if s not in in_degree]
        for s in start_nodes:
            curr = s
            length = 1
            while curr in next_ptr:
                curr = next_ptr[curr]
                length += 1
            track_lengths_all.append(length)
        # 孤立ノード (length = 1)
        track_lengths_all.extend([1] * isolated_nodes_count)

        if (idx + 1) % 20 == 0 or (idx + 1) == n_datasets:
            elapsed = time.time() - t_sweep_start
            print(f"  [{idx+1:3d}/{n_datasets:3d}] ({elapsed:.1f}s) {ds} 完了 (nodes: {total_detected_nodes:,}, isolated: {isolated_nodes_count:,} [{isolated_ratio*100:.1f}%])")

    print(f"\n[*] 全199データセットの探索・計算が完了しました ({time.time()-t_sweep_start:.2f}s)")

    # 4. 結果の集計と保存
    # (1) 重み最適化結果
    all_weight_dfs = []
    summary_rows = []
    for fw in weights:
        df_w = pd.DataFrame(weight_results[fw])
        all_weight_dfs.append(df_w)
        summary_rows.append({
            'weight': fw,
            'macro_mean_recall': df_w['recall'].mean(),
            'macro_mean_precision': df_w['precision'].mean(),
            'macro_mean_f1': df_w['f1'].mean(),
            'total_tp': df_w['tp'].sum(),
            'total_pred_edges': df_w['pred_edges'].sum(),
            'total_gt_edges': df_w['gt_edges'].sum(),
            'micro_recall': df_w['tp'].sum() / df_w['gt_edges'].sum() if df_w['gt_edges'].sum() > 0 else 0.0
        })

    full_weight_df = pd.concat(all_weight_dfs, ignore_index=True)
    summary_weight_df = pd.DataFrame(summary_rows)

    csv_weight_path = os.path.join(history_dir, "s4_002_edge_weight_optimization_all199.csv")
    xlsx_weight_path = os.path.join(history_dir, "s4_002_edge_weight_optimization_all199.xlsx")
    full_weight_df.to_csv(csv_weight_path, index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(xlsx_weight_path, engine='openpyxl') as writer:
        summary_weight_df.to_excel(writer, sheet_name='Summary', index=False)
        full_weight_df.to_excel(writer, sheet_name='Details_All199', index=False)
    print(f"[+] 保存完了: {csv_weight_path}")
    print(f"[+] 保存完了: {xlsx_weight_path}")

    # (2) 孤立ノード刈り取り結果
    df_filter = pd.DataFrame(filtering_results)
    csv_filter_path = os.path.join(history_dir, "s4_002_isolated_node_filtering_all199.csv")
    xlsx_filter_path = os.path.join(history_dir, "s4_002_isolated_node_filtering_all199.xlsx")
    df_filter.to_csv(csv_filter_path, index=False, encoding='utf-8-sig')
    df_filter.to_excel(xlsx_filter_path, index=False, engine='openpyxl')
    print(f"[+] 保存完了: {csv_filter_path}")
    print(f"[+] 保存完了: {xlsx_filter_path}")

    # 5. プロット図の生成 (3点, 300 DPI)
    print("\n[*] プロット図を生成中...")

    # 図1: 重みスイープ曲線 (s4_002_edge_weight_sweep_curve.png)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(summary_weight_df['weight'], summary_weight_df['macro_mean_recall'] * 100,
             marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='Macro Mean Recall (%)')
    plt.plot(summary_weight_df['weight'], summary_weight_df['micro_recall'] * 100,
             marker='s', color='#2ca02c', linewidth=2.0, linestyle='--', markersize=7, label='Micro Recall (Total TP / Total GT) (%)')
    
    # 最適点のアノテーション
    best_idx = summary_weight_df['macro_mean_recall'].idxmax()
    best_w = summary_weight_df.loc[best_idx, 'weight']
    best_r = summary_weight_df.loc[best_idx, 'macro_mean_recall'] * 100
    plt.scatter([best_w], [best_r], color='#d62728', s=150, zorder=5)
    plt.annotate(f"Optimal Weight: {best_w}\nRecall: {best_r:.2f}%",
                 xy=(best_w, best_r), xytext=(best_w + 0.05, best_r - 0.5),
                 arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=11, fontweight='bold', color='#d62728',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1))

    plt.title("5D Mahalanobis: 特徴量重み vs トラッキングEdge Recall (全199データセット)", fontsize=14, pad=12)
    plt.xlabel("特徴量重み (feature_weight, spatial_weight=1.0)", fontsize=12)
    plt.ylabel("Edge Recall (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plot1_path = os.path.join(history_dir, "s4_002_edge_weight_sweep_curve.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] プロット保存完了: {plot1_path}")

    # 図2: 孤立ノード刈り取り前後の P/E比 度数分布 (s4_002_pe_ratio_before_after_filtering.png)
    plt.figure(figsize=(11, 6), dpi=300)
    bins = np.linspace(0.6, 2.2, 33)
    plt.hist(df_filter['pe_ratio_before'], bins=bins, alpha=0.55, color='#d62728',
             label=f"刈り取り前 (Before) [平均 P/E: {df_filter['pe_ratio_before'].mean():.3f}]", edgecolor='black')
    plt.hist(df_filter['pe_ratio_after'], bins=bins, alpha=0.65, color='#2ca02c',
             label=f"孤立ノード刈り取り後 (After) [平均 P/E: {df_filter['pe_ratio_after'].mean():.3f}]", edgecolor='black')

    plt.axvline(0.90, color='#1f77b4', linestyle='--', linewidth=2.5, label='目標ターゲット P/E = 0.90')
    plt.axvline(df_filter['pe_ratio_before'].mean(), color='#d62728', linestyle=':', linewidth=1.8)
    plt.axvline(df_filter['pe_ratio_after'].mean(), color='#2ca02c', linestyle=':', linewidth=1.8)

    plt.title("トラッキング後処理 (孤立ノード刈り取り) 前後の P/E比 度数分布 (全199データセット)", fontsize=14, pad=12)
    plt.xlabel("P/E比 (予測ノード数 / 推定GTノード数)", fontsize=12)
    plt.ylabel("データセット数 (頻度)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    plot2_path = os.path.join(history_dir, "s4_002_pe_ratio_before_after_filtering.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] プロット保存完了: {plot2_path}")

    # 図3: トラック長 (Track Length) の度数分布 (s4_002_track_length_distribution.png)
    plt.figure(figsize=(10, 6), dpi=300)
    max_len = 30
    hist_lens = [min(l, max_len) for l in track_lengths_all]
    counts, edges_arr = np.histogram(hist_lens, bins=range(1, max_len + 2))

    colors = ['#d62728'] + ['#1f77b4'] * (len(counts) - 1)
    bars = plt.bar(edges_arr[:-1], counts, color=colors, edgecolor='black', width=0.8)

    # 孤立ノード比率のアノテーション
    isolated_total = counts[0]
    total_tracks = len(hist_lens)
    iso_pct = isolated_total / total_tracks * 100
    plt.annotate(f"長さ1の孤立ノード (除外対象)\n件数: {isolated_total:,} ({iso_pct:.1f}%)\n※時間的一貫性のない微小ノイズ",
                 xy=(1, isolated_total), xytext=(3, isolated_total * 0.8),
                 arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=11, fontweight='bold', color='#d62728',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728", lw=1))

    plt.title("全軌跡長 (Track Length) の度数分布", fontsize=14, pad=12)
    plt.xlabel("軌跡長 (フレーム連続追跡数, 30以上は集約)", fontsize=12)
    plt.ylabel("軌跡件数 (対数スケール)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot3_path = os.path.join(history_dir, "s4_002_track_length_distribution.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] プロット保存完了: {plot3_path}")

    # 6. サマリ出力
    print("\n" + "=" * 70)
    print(">>> 5D Mahalanobis 重み最適化サマリ (全199データセット)")
    print("=" * 70)
    print(summary_weight_df.to_string(index=False))

    print("\n" + "=" * 70)
    print(">>> 孤立ノード刈り取り (P/E比制御) サマリ (全199データセット)")
    print("=" * 70)
    print(f"平均 P/E比 (刈り取り前): {df_filter['pe_ratio_before'].mean():.4f}")
    print(f"平均 P/E比 (刈り取り後): {df_filter['pe_ratio_after'].mean():.4f}")
    print(f"平均 孤立ノード除外率  : {df_filter['isolated_removal_pct'].mean():.2f}%")
    print(f"P/E <= 0.95 達成データセット数: {(df_filter['pe_ratio_after'] <= 0.95).sum()} / 199")
    print(f"P/E <= 1.00 達成データセット数: {(df_filter['pe_ratio_after'] <= 1.00).sum()} / 199")
    print(f"総所要時間: {time.time()-start_total:.2f} 秒")
    print("=" * 70)

if __name__ == '__main__':
    main()
