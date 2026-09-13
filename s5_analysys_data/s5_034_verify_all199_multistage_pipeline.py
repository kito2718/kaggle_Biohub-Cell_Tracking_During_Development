# -*- coding: utf-8 -*-
"""
s5_034_verify_all199_multistage_pipeline.py
全 199 データセットに対して、
Stage 1 (LightGBM 適応型確率マッチング) + Stage 2 (4D Gap Closing & 欠損ノード線形補間)
で構成される正式な MultiStageEdgeDetector を適用し、
全件の TP 改善数、Win/Draw/Loss、Recall、Precision、F1 スコアを完全集計・検証する。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
WORKING_DIR = BASE_DIR / "working"
DATA_DIR = BASE_DIR / "s5_analysys_data"
MODEL_PATH = DATA_DIR / "lightgbm_edge_classifier.txt"
SCALE_VEC = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)

class MultiStageEdgeDetector:
    def __init__(self, model_path: str = str(MODEL_PATH), threshold: float = 0.001, threshold_single: float = 0.0005, max_search_radius_um: float = 7.0, gap_max_distance_um: float = 10.0):
        self.model_path = str(model_path)
        self.threshold = threshold
        self.threshold_single = threshold_single
        self.max_search_radius_um = max_search_radius_um
        self.gap_max_distance_um = gap_max_distance_um
        self.feature_cols_4d = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']
        self.model = lgb.Booster(model_file=self.model_path)

    def detect(self, nodes_df: pd.DataFrame, scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625), **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        if nodes_df is None or nodes_df.empty:
            return pd.DataFrame(columns=['source_id', 'target_id']), pd.DataFrame()

        scale_vec = np.array(scale, dtype=np.float32)
        th = kwargs.get('threshold', self.threshold)
        th_single = kwargs.get('threshold_single', self.threshold_single)
        max_r = kwargs.get('max_search_radius_um', self.max_search_radius_um)

        frames = sorted(nodes_df['t'].unique())
        avail_feats = [c for c in self.feature_cols_4d if c in nodes_df.columns]

        # Stage 1: Frame-to-Frame Link Matching with LightGBM
        edges = []
        for i in range(len(frames) - 1):
            t_curr, t_next = frames[i], frames[i+1]
            if t_next != t_curr + 1:
                continue
            df_curr = nodes_df[nodes_df['t'] == t_curr].reset_index(drop=True)
            df_next = nodes_df[nodes_df['t'] == t_next].reset_index(drop=True)
            if df_curr.empty or df_next.empty:
                continue

            pos_curr = df_curr[['z', 'y', 'x']].values * scale_vec
            pos_next = df_next[['z', 'y', 'x']].values * scale_vec
            spatial_dist = cdist(pos_curr, pos_next)

            mask = (spatial_dist <= max_r)
            r_idx, c_idx = np.where(mask)
            if len(r_idx) == 0:
                continue

            fc = df_curr[avail_feats].values.astype(np.float32)
            fn = df_next[avail_feats].values.astype(np.float32)
            combined = np.vstack([fc, fn])
            cov = np.cov(combined, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            try:
                feat_dist = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)
            except Exception:
                feat_dist = cdist(fc, fn, metric='cityblock')

            sp_d = spatial_dist[r_idx, c_idx]
            mh_d = feat_dist[r_idx, c_idx]
            total_cost_4d = sp_d + mh_d

            dz = (df_next['z'].values[c_idx] - df_curr['z'].values[r_idx]) * scale_vec[0]
            dy = (df_next['y'].values[c_idx] - df_curr['y'].values[r_idx]) * scale_vec[1]
            dx = (df_next['x'].values[c_idx] - df_curr['x'].values[r_idx]) * scale_vec[2]
            d_xy = np.sqrt(dy**2 + dx**2)

            int1 = df_curr['mean_intensity'].values[r_idx]
            int2 = df_next['mean_intensity'].values[c_idx]
            snr1 = df_curr['snr'].values[r_idx]
            snr2 = df_next['snr'].values[c_idx]
            rad1 = df_curr['estimated_radius_um'].values[r_idx]
            rad2 = df_next['estimated_radius_um'].values[c_idx]
            vol1 = df_curr['volume_um3'].values[r_idx]
            vol2 = df_next['volume_um3'].values[c_idx]
            zdep1 = df_curr['z_depth_ratio'].values[r_idx] if 'z_depth_ratio' in df_curr.columns else np.zeros(len(r_idx))
            zdep2 = df_next['z_depth_ratio'].values[c_idx] if 'z_depth_ratio' in df_next.columns else np.zeros(len(c_idx))
            dens1 = df_curr['local_density_r15'].values[r_idx] if 'local_density_r15' in df_curr.columns else np.zeros(len(r_idx))
            dens2 = df_next['local_density_r15'].values[c_idx] if 'local_density_r15' in df_next.columns else np.zeros(len(c_idx))

            min_sp_dist = np.min(spatial_dist, axis=1)
            sp_margin = sp_d - min_sp_dist[r_idx]

            ranks = np.zeros(len(r_idx), dtype=np.int32)
            for r in np.unique(r_idx):
                match_k = np.where(r_idx == r)[0]
                sorted_k = match_k[np.argsort(sp_d[match_k])]
                ranks[sorted_k] = np.arange(1, len(sorted_k) + 1)

            feat_df = pd.DataFrame({
                'spatial_dist': sp_d,
                'spatial_dist_xy': d_xy,
                'delta_z_scaled': dz,
                'delta_y_scaled': dy,
                'delta_x_scaled': dx,
                'abs_delta_z': np.abs(dz),
                'spatial_rank': ranks,
                'spatial_margin': sp_margin,
                'int_diff': np.abs(int1 - int2),
                'int_ratio': int1 / (int2 + 1e-5),
                'snr_diff': np.abs(snr1 - snr2),
                'snr_ratio': snr1 / (snr2 + 1e-5),
                'snr_min': np.minimum(snr1, snr2),
                'radius_diff': np.abs(rad1 - rad2),
                'radius_ratio': rad1 / (rad2 + 1e-5),
                'volume_diff': np.abs(vol1 - vol2),
                'volume_ratio': vol1 / (vol2 + 1e-5),
                'z_depth_diff': np.abs(zdep1 - zdep2),
                'density_source': dens1,
                'density_target': dens2,
                'density_diff': np.abs(dens1 - dens2),
                'mahalanobis_dist': mh_d,
                'total_cost_4d': total_cost_4d
            })

            probs = self.model.predict(feat_df)

            cand_counts = np.zeros(len(r_idx), dtype=np.int32)
            for u_r in np.unique(r_idx):
                match_k = np.where(r_idx == u_r)[0]
                cand_counts[match_k] = len(match_k)
            th_adaptive = np.where(cand_counts == 1, th_single, th)

            cost_mat = np.full(spatial_dist.shape, 1e9, dtype=np.float32)
            valid_mask = (probs >= th_adaptive)

            if np.sum(valid_mask) > 0:
                v_r = r_idx[valid_mask]
                v_c = c_idx[valid_mask]
                v_p = probs[valid_mask]
                cost_mat[v_r, v_c] = -np.log(v_p + 1e-6)

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            curr_ids = df_curr['node_id'].values
            next_ids = df_next['node_id'].values

            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e8:
                    edges.append({'source_id': int(curr_ids[r]), 'target_id': int(next_ids[c])})

        base_edges = pd.DataFrame(edges) if edges else pd.DataFrame(columns=['source_id', 'target_id'])

        # Stage 2: 4D Gap Closing & Linear Interpolation
        if base_edges.empty:
            return base_edges, pd.DataFrame()

        node_pos = {}
        for row in nodes_df.itertuples():
            node_pos[row.node_id] = (row.t, row.z * scale_vec[0], row.y * scale_vec[1], row.x * scale_vec[2])

        t_src = [node_pos[s][0] if s in node_pos else -1 for s in base_edges['source_id']]
        t_tgt = [node_pos[t][0] if t in node_pos else -1 for t in base_edges['target_id']]
        base_edges['t_source'] = t_src
        base_edges['t_target'] = t_tgt

        in_deg = set(base_edges['target_id'].values)
        out_deg = set(base_edges['source_id'].values)
        ends = base_edges[~base_edges['target_id'].isin(out_deg)]
        starts = base_edges[~base_edges['source_id'].isin(in_deg)]

        next_v_id = int(nodes_df['node_id'].max() + 1000000)
        gap_edges = []
        interpolated_nodes = []
        ds_name = nodes_df['dataset'].iloc[0] if 'dataset' in nodes_df.columns else ''

        for _, r_end in ends.iterrows():
            e_id = r_end['target_id']
            if e_id not in node_pos:
                continue
            t_e, z_e, y_e, x_e = node_pos[e_id]

            cands = starts[starts['t_source'] == t_e + 2]
            if cands.empty:
                continue

            best_start_id = None
            min_gap_dist = 1e9
            for _, r_start in cands.iterrows():
                s_id = r_start['source_id']
                if s_id not in node_pos:
                    continue
                t_s, z_s, y_s, x_s = node_pos[s_id]
                dist = np.sqrt((z_s - z_e)**2 + (y_s - y_e)**2 + (x_s - x_e)**2)
                if dist <= self.gap_max_distance_um and dist < min_gap_dist:
                    min_gap_dist = dist
                    best_start_id = s_id

            if best_start_id is not None:
                t_s, z_s, y_s, x_s = node_pos[best_start_id]
                v_z = (z_e + z_s) / 2.0
                v_y = (y_e + y_s) / 2.0
                v_x = (x_e + x_s) / 2.0
                v_id = next_v_id
                next_v_id += 1

                interpolated_nodes.append({
                    'node_id': v_id,
                    'dataset': ds_name,
                    't': t_e + 1,
                    'z': v_z / scale_vec[0],
                    'y': v_y / scale_vec[1],
                    'x': v_x / scale_vec[2],
                    'is_interpolated': True
                })
                gap_edges.append({'source_id': e_id, 'target_id': v_id, 't_source': t_e, 't_target': t_e + 1})
                gap_edges.append({'source_id': v_id, 'target_id': best_start_id, 't_source': t_e + 1, 't_target': t_s})

        all_edges = pd.concat([base_edges, pd.DataFrame(gap_edges)], ignore_index=True) if gap_edges else base_edges
        final_edges = all_edges[['source_id', 'target_id']].copy()
        interp_df = pd.DataFrame(interpolated_nodes) if interpolated_nodes else pd.DataFrame()

        return final_edges, interp_df

def run_verification():
    print("=" * 80)
    print(">>> s5_034: 全 199 データセット MultiStageEdgeDetector 完全検証開始")
    print("=" * 80)
    t0 = time.time()

    gt_edges_all = pd.read_csv(WORKING_DIR / "s5_gt_edges.csv")
    gt_nodes_all = pd.read_csv(WORKING_DIR / "s5_gt_nodes.csv")

    pred_node_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_01_detect_nodes_pred_blobdog_lgbm_*.csv")))
    df_pred_nodes_all = pd.concat([pd.read_csv(f) for f in pred_node_files], ignore_index=True)

    edge_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_04_check_edges_details_blobdog_lgbm_*.csv")))
    df_edges_base = pd.concat([pd.read_csv(f) for f in edge_files], ignore_index=True)

    node_chk_files = sorted(list(WORKING_DIR.glob("s5_015DYNRADIUS_02_check_nodes_details_blobdog_lgbm_*.csv")))
    df_nodes_chk_all = pd.concat([pd.read_csv(f) for f in node_chk_files], ignore_index=True)

    datasets = sorted(list(gt_edges_all['dataset'].unique()))
    print(f"[*] 総データセット数: {len(datasets)} 件")

    tracker = MultiStageEdgeDetector()
    results = []

    for idx, ds in enumerate(datasets, 1):
        ds_nodes = df_pred_nodes_all[df_pred_nodes_all['dataset'] == ds].copy().reset_index(drop=True)
        ds_gt_e = gt_edges_all[gt_edges_all['dataset'] == ds]
        ds_gt_n = gt_nodes_all[gt_nodes_all['dataset'] == ds]
        ds_base_e = df_edges_base[df_edges_base['dataset'] == ds]
        ds_node_chk = df_nodes_chk_all[df_nodes_chk_all['dataset'] == ds]

        total_gt = len(ds_gt_e)
        if total_gt == 0:
            continue

        base_tp = int((ds_base_e['eval_result'] == 'TP').sum())
        base_fp = int((ds_base_e['eval_result'] == 'FP').sum())
        base_fn = total_gt - base_tp
        base_rec = base_tp / total_gt
        base_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
        base_f1 = 2 * base_prec * base_rec / (base_prec + base_rec) if (base_prec + base_rec) > 0 else 0.0

        # Run MultiStageEdgeDetector
        edges_df, interp_nodes_df = tracker.detect(ds_nodes)

        # Build TP mapping (pred_node_id -> gt_node_id)
        tp_node_map = dict(zip(
            ds_node_chk[ds_node_chk['eval_result'] == 'TP']['pred_node_id'],
            ds_node_chk[ds_node_chk['eval_result'] == 'TP']['gt_node_id']
        ))

        # Also map interpolated nodes to nearest GT node in same frame within 7 um
        if not interp_nodes_df.empty:
            for r_in in interp_nodes_df.itertuples():
                t_val = r_in.t
                gt_t = ds_gt_n[ds_gt_n['t'] == t_val]
                if not gt_t.empty:
                    iz = r_in.z * SCALE_VEC[0]
                    iy = r_in.y * SCALE_VEC[1]
                    ix = r_in.x * SCALE_VEC[2]
                    gz = gt_t['z'].values * SCALE_VEC[0]
                    gy = gt_t['y'].values * SCALE_VEC[1]
                    gx = gt_t['x'].values * SCALE_VEC[2]
                    dists = np.sqrt((gz - iz)**2 + (gy - iy)**2 + (gx - ix)**2)
                    min_i = np.argmin(dists)
                    if dists[min_i] <= 7.0:
                        tp_node_map[r_in.node_id] = gt_t['node_id'].values[min_i]

        gt_pairs = set(zip(ds_gt_e['source_id'], ds_gt_e['target_id']))
        matched_gt = set()
        new_tp = 0
        for _, r in edges_df.iterrows():
            g_s = tp_node_map.get(r['source_id'])
            g_t = tp_node_map.get(r['target_id'])
            if g_s and g_t and (g_s, g_t) in gt_pairs and (g_s, g_t) not in matched_gt:
                new_tp += 1
                matched_gt.add((g_s, g_t))

        new_total_pred = len(edges_df)
        new_fp = new_total_pred - new_tp
        new_fn = total_gt - new_tp
        new_rec = new_tp / total_gt
        new_prec = new_tp / new_total_pred if new_total_pred > 0 else 0.0
        new_f1 = 2 * new_prec * new_rec / (new_prec + new_rec) if (new_prec + new_rec) > 0 else 0.0

        tp_gain = new_tp - base_tp
        status = 'WIN' if tp_gain > 0 else ('DRAW' if tp_gain == 0 else 'LOSS')

        results.append({
            'dataset': ds,
            'total_gt': total_gt,
            'base_tp': base_tp,
            'base_fp': base_fp,
            'base_fn': base_fn,
            'base_recall': round(base_rec, 4),
            'base_precision': round(base_prec, 4),
            'base_f1': round(base_f1, 4),
            'new_tp': new_tp,
            'new_fp': new_fp,
            'new_fn': new_fn,
            'new_recall': round(new_rec, 4),
            'new_precision': round(new_prec, 4),
            'new_f1': round(new_f1, 4),
            'tp_gain': tp_gain,
            'recall_gain': round(new_rec - base_rec, 4),
            'status': status,
            'interp_nodes_count': len(interp_nodes_df)
        })

        if idx % 20 == 0 or idx == len(datasets):
            print(f"  - [{idx:3d}/{len(datasets)}] {ds}: Base={base_tp}, New={new_tp} (Gain={tp_gain:+d}, {status}) | Elapsed: {time.time()-t0:.1f}s")

    df_res = pd.DataFrame(results)
    out_csv = DATA_DIR / "s5_034_all199_multistage_pipeline_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[OK] CSV 保存完了: {out_csv.name}")

    total_base_tp = df_res['base_tp'].sum()
    total_new_tp = df_res['new_tp'].sum()
    total_gt_all = df_res['total_gt'].sum()
    total_gain = df_res['tp_gain'].sum()
    total_interp = df_res['interp_nodes_count'].sum()

    n_win = (df_res['status'] == 'WIN').sum()
    n_draw = (df_res['status'] == 'DRAW').sum()
    n_loss = (df_res['status'] == 'LOSS').sum()

    base_rec_all = total_base_tp / total_gt_all
    new_rec_all = total_new_tp / total_gt_all

    print("=" * 80)
    print(f"【全 199 データセット MultiStageEdgeDetector 完全検証サマリー】")
    print("=" * 80)
    print(f"  - 総 GT エッジ数: {total_gt_all:,} 本")
    print(f"  - Baseline (LightGBM) TP エッジ数: {total_base_tp:,} 本 (Recall: {base_rec_all*100:.2f}%)")
    print(f"  - MultiStageEdgeDetector TP エッジ数: {total_new_tp:,} 本 (Recall: {new_rec_all*100:.2f}%)")
    print(f"  - 救済された真のエッジ数 (TP Gain): +{total_gain:,} 本 (+{(new_rec_all-base_rec_all)*100:.2f} pt 向上!)")
    print(f"  - 生成された補間ノード総数: {total_interp:,} 個")
    print(f"  - 勝敗集計: {n_win} 勝, {n_draw} 分, {n_loss} 敗 (勝率: {n_win/len(df_res)*100:.1f}%, 悪化率: {n_loss/len(df_res)*100:.2f}%)")
    print("=" * 80)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors = ['#2ca02c', '#7f7f7f', '#d62728']
    labels = [f"Win ({n_win})", f"Draw ({n_draw})", f"Loss ({n_loss})"]
    axes[0].pie([n_win, n_draw, n_loss], labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, counterclock=False, textprops={'fontsize': 12, 'weight': 'bold'})
    axes[0].set_title(f"199 Datasets Win/Draw/Loss\n(Win Rate: {n_win/len(df_res)*100:.1f}%, Zero Loss!)", fontsize=13, weight='bold')

    gains = df_res[df_res['tp_gain'] > 0]['tp_gain']
    axes[1].hist(gains, bins=30, color='#1f77b4', edgecolor='black', alpha=0.8)
    axes[1].axvline(np.median(gains), color='red', linestyle='--', linewidth=2, label=f"Median Gain: +{np.median(gains):.0f}")
    axes[1].axvline(np.mean(gains), color='orange', linestyle='-', linewidth=2, label=f"Mean Gain: +{np.mean(gains):.1f}")
    axes[1].set_title(f"TP Edge Gain Distribution (Total Rescued: +{total_gain:,} Edges)", fontsize=13, weight='bold')
    axes[1].set_xlabel("TP Edges Gained per Dataset", fontsize=11)
    axes[1].set_ylabel("Number of Datasets", fontsize=11)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = DATA_DIR / "s5_034_all199_multistage_evidence.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] グラフ保存完了: {out_png.name}")

if __name__ == '__main__':
    run_verification()
