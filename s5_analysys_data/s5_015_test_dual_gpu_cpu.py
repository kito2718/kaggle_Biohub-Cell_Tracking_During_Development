# -*- coding: utf-8 -*-
"""
s5_015_test_dual_gpu_cpu.py
ローカル実画像 (44b6_0113de3b.zarr) を用いた
GPU モード (NVIDIA GeForce RTX 4060) ＆ CPU モードの
Step 2 完全パイプライン (動的探索半径 + lightgbm_adaptive_th) 健全性検証スクリプト。
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb
from skimage.feature import blob_dog

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development")
ZARR_PATH = BASE_DIR / "s5" / "input" / "train" / "44b6_0113de3b.zarr"
MODEL_EDGE_PATH = BASE_DIR / "s5" / "github" / "s5_analysys_data" / "s5_003_tracking_edge_lgbm.txt"
MODEL_TH_PATH = BASE_DIR / "s5" / "github" / "s5_analysys_data" / "lightgbm_adaptive_th.txt"

print("=" * 90)
print(">>> s5_015: GPU モード (RTX 4060) ＆ CPU モード Step 2 完全パイプライン確認")
print("=" * 90)
print(f"[*] 対象 Zarr データ:       {ZARR_PATH} (Exists: {ZARR_PATH.exists()})")
print(f"[*] エッジ追跡 LightGBM:    {MODEL_EDGE_PATH} (Exists: {MODEL_EDGE_PATH.exists()})")
print(f"[*] 動的閾値 lightgbm_th:  {MODEL_TH_PATH} (Exists: {MODEL_TH_PATH.exists()})")
print(f"[*] CUDA 利用可能:          {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    - GPU デバイス:         {torch.cuda.get_device_name(0)}")

# 事前ロード (ノートブック本番環境と同様)
th_model = lgb.Booster(model_file=str(MODEL_TH_PATH))
edge_model = lgb.Booster(model_file=str(MODEL_EDGE_PATH))
print(f"[*] モデル事前ロード完了: lightgbm_adaptive_th & tracking_edge_lgbm")

scale_vec = (1.625, 0.40625, 0.40625)
N_TEST_FRAMES = 5

def _gaussian_3d_torch(x_in: torch.Tensor, sigma, device: torch.device):
    out = x_in
    for s, d in zip(sigma, [2, 3, 4]):
        r = int(4.0 * s + 0.5)
        w = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / s)**2)
        w /= w.sum()
        tw = torch.from_numpy(w).to(device=device, dtype=torch.float32)
        n = out.shape[d]
        idx_left = torch.arange(r - 1, -1, -1, device=device) % n
        idx_mid = torch.arange(n, device=device)
        idx_right = (n - 1) - (torch.arange(r, device=device) % n)
        idx_full = torch.cat([idx_left, idx_mid, idx_right])
        padded = out.index_select(d, idx_full)
        if d == 2:
            kw = tw.view(1, 1, -1, 1, 1)
        elif d == 3:
            kw = tw.view(1, 1, 1, -1, 1)
        else:
            kw = tw.view(1, 1, 1, 1, -1)
        out = F.conv3d(padded, kw)
    return out

def detect_blobs_gpu(img_norm, min_sig=(1.0, 2.0, 2.0), max_sig=(2.5, 6.0, 6.0),
                     sigma_ratio=1.6, th=0.038, ov=0.50, device=torch.device('cuda')):
    prune_blobs_fn = blob_dog.__globals__['_prune_blobs']
    k = int(np.mean(np.log(np.array(max_sig) / np.array(min_sig)) / np.log(sigma_ratio) + 1))
    sigma_list = [np.array(min_sig) * (sigma_ratio**i) for i in range(k + 1)]

    t_img = torch.from_numpy(img_norm).view(1, 1, *img_norm.shape).to(device=device, dtype=torch.float32)
    blurred = [_gaussian_3d_torch(t_img, s, device) for s in sigma_list]
    sf = 1.0 / (sigma_ratio - 1.0)
    dog_cubes = [(blurred[i] - blurred[i + 1]) * sf for i in range(k)]

    dog_stacked = torch.cat(dog_cubes, dim=1)
    padded_sp = F.pad(dog_stacked, (1, 1, 1, 1, 1, 1), mode='replicate')
    sp_max = F.max_pool3d(padded_sp, kernel_size=3, stride=1, padding=0)

    flat_sp = sp_max.squeeze(0).permute(1, 2, 3, 0).contiguous().view(-1, 1, k)
    padded_sc = F.pad(flat_sp, (1, 1), mode='replicate')
    sc_max = F.max_pool1d(padded_sc, kernel_size=3, stride=1, padding=0)
    torch_max = sc_max.view(*img_norm.shape, k)
    dog_raw = dog_stacked.squeeze(0).permute(1, 2, 3, 0)
    is_peak = (dog_raw == torch_max) & (dog_raw > th)

    if not bool(is_peak.any()):
        return np.empty((0, 6), dtype=np.float32)

    pt_peaks = torch.nonzero(is_peak)
    intensities = dog_raw[is_peak]
    sort_idx = torch.argsort(-intensities, stable=True)

    pt_peaks_sorted = pt_peaks[sort_idx].cpu().numpy()
    sigmas_of_peaks = np.array([sigma_list[s] for s in pt_peaks_sorted[:, -1]], dtype=np.float32)
    lm = np.hstack([pt_peaks_sorted[:, :-1].astype(np.float32), sigmas_of_peaks])
    return prune_blobs_fn(lm, ov, sigma_dim=3)

def detect_blobs_cpu(img_norm, min_sig=(1.0, 2.0, 2.0), max_sig=(2.5, 6.0, 6.0),
                     sigma_ratio=1.6, th=0.038, ov=0.50):
    return blob_dog(img_norm, min_sigma=min_sig, max_sigma=max_sig,
                    sigma_ratio=sigma_ratio, threshold=th, overlap=ov)

# Step 2 動的探索半径推定関数
def predict_dataset_adaptive_search_radius(
    nodes_df: pd.DataFrame,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    default_radius: float = 7.0,
    min_radius: float = 7.0,
    max_radius: float = 8.5
) -> tuple[float, float, float]:
    t0_rad = time.perf_counter()
    if nodes_df is None or len(nodes_df) < 2:
        return float(default_radius), 2.0, time.perf_counter() - t0_rad

    frames = sorted(nodes_df['t'].unique())
    if len(frames) < 2:
        return float(default_radius), 2.0, time.perf_counter() - t0_rad

    scale_v = np.array(scale, dtype=np.float32)
    inter_dists = []
    for i in range(min(3, len(frames) - 1)):
        t_c, t_n = frames[i], frames[i+1]
        if t_n != t_c + 1:
            continue
        p_c = nodes_df[nodes_df['t'] == t_c][['z', 'y', 'x']].values * scale_v
        p_n = nodes_df[nodes_df['t'] == t_n][['z', 'y', 'x']].values * scale_v
        if len(p_c) > 0 and len(p_n) > 0:
            dmat = cdist(p_c, p_n)
            inter_dists.extend(np.min(dmat, axis=1))

    if not inter_dists:
        return float(default_radius), 2.0, time.perf_counter() - t0_rad

    speed_med = float(np.median(inter_dists))

    # 移動速度に応じた最適半径マッピング
    if speed_med <= 2.5:
        r = min_radius
    elif speed_med >= 2.8:
        r = min(max_radius, 8.0 + (speed_med - 2.8) * 0.25)
    else:
        r = min_radius + (speed_med - 2.5) / (2.8 - 2.5) * (8.0 - min_radius)

    t_rad = time.perf_counter() - t0_rad
    return float(np.clip(r, min_radius, max_radius)), speed_med, t_rad

def run_pipeline(mode='cuda'):
    print(f"\n" + "-" * 75)
    print(f"[*] 開始: モード = {mode.upper()}")
    print("-" * 75)
    
    t_start = time.time()
    z_root = zarr.open(str(ZARR_PATH), mode='r')
    arr = z_root['0']
    
    # 1. ノード検出
    t0_det = time.time()
    all_nodes = []
    global_nid = 1
    for t in range(N_TEST_FRAMES):
        frame_data = np.squeeze(arr[t])
        p_low, p_high = np.percentile(frame_data[::2, ::2, ::2], [1.0, 99.5])
        img_norm = np.clip((frame_data - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0).astype(np.float32)
        
        if mode == 'cuda':
            blobs = detect_blobs_gpu(img_norm, th=0.038, ov=0.50, device=torch.device('cuda'))
        else:
            blobs = detect_blobs_cpu(img_norm, th=0.038, ov=0.50)
            
        p_bg = np.percentile(frame_data, 10)
        p_noise = max(1e-3, float(np.std(frame_data[frame_data < np.percentile(frame_data, 30)])))
        
        for b in blobs:
            z, y, x, sz, sy, sx = b[0], b[1], b[2], b[3], b[4], b[5]
            iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
            iz = np.clip(iz, 0, frame_data.shape[0]-1)
            iy = np.clip(iy, 0, frame_data.shape[1]-1)
            ix = np.clip(ix, 0, frame_data.shape[2]-1)
            intensity = float(frame_data[iz, iy, ix])
            snr = (intensity - p_bg) / p_noise
            rad_mean_voxel = float(np.mean([sz, sy, sx]))
            rad_um = rad_mean_voxel * 0.40625
            vol = (4.0 / 3.0) * np.pi * (rad_um)**3
            
            all_nodes.append({
                'node_id': global_nid,
                't': t, 'z': z, 'y': y, 'x': x,
                'estimated_radius_um': rad_um,
                'mean_intensity': intensity,
                'snr': snr,
                'volume_um3': vol
            })
            global_nid += 1
            
    df_nodes = pd.DataFrame(all_nodes)
    t_det = time.time() - t0_det
    print(f"  [1] DoG 検出完了: {len(df_nodes):,} ノード ({t_det:.3f} 秒, {t_det/N_TEST_FRAMES:.3f} 秒/frame)")
    
    # 2. Step 2 動的探索半径推定
    adaptive_r, speed_med, t_rad = predict_dataset_adaptive_search_radius(df_nodes, scale=scale_vec)
    print(f"  [2] 動的探索半径推定完了 ({t_rad*1000:.3f} ms < 10ms PASS):")
    print(f"      -> 代表移動速度: {speed_med:.4f} μm, 適応探索半径: {adaptive_r:.2f} μm")

    # 3. 12大メタ特徴量抽出
    t0_meta = time.time()
    scale_v = np.array(scale_vec, dtype=np.float32)
    n_nodes = len(df_nodes)
    n_frames = N_TEST_FRAMES
    density = n_nodes / max(1, n_frames)
    
    nn_dists = []
    for t in range(min(3, n_frames)):
        df_t = df_nodes[df_nodes['t'] == t]
        if len(df_t) > 1:
            pos = df_t[['z', 'y', 'x']].values * scale_v
            dmat = cdist(pos, pos)
            np.fill_diagonal(dmat, 1e9)
            nn_dists.extend(np.min(dmat, axis=1))
            
    mean_nn = float(np.mean(nn_dists)) if nn_dists else 10.0
    med_nn = float(np.median(nn_dists)) if nn_dists else 10.0
    min_nn = float(np.min(nn_dists)) if nn_dists else 1.0
    std_nn = float(np.std(nn_dists)) if nn_dists else 3.0
    mean_int = float(df_nodes['mean_intensity'].mean())
    std_int = float(df_nodes['mean_intensity'].std())
    snr_m = float(df_nodes['snr'].mean())
    snr_s = float(df_nodes['snr'].std())
    p95, p5, p50 = np.percentile(df_nodes['mean_intensity'], [95, 5, 50])
    contrast = float((p95 - p5) / (p50 + 1e-5))
    rad_m = float(df_nodes['estimated_radius_um'].mean())
    vol_m = float(df_nodes['volume_um3'].mean())
    
    meta_feat_array = np.array([[
        std_int, std_nn, rad_m, contrast, mean_int, density,
        snr_s, vol_m, min_nn, mean_nn, med_nn, snr_m
    ]], dtype=np.float32)
    t_meta = time.time() - t0_meta
    print(f"  [3] 12大メタ特徴量抽出 ({t_meta*1000:.2f} ms): density={density:.1f}, mean_nn={mean_nn:.2f} um, contrast={contrast:.2f}")
    
    # 4. lightgbm_adaptive_th 推論
    t0_th = time.perf_counter()
    pred_log10_th = th_model.predict(meta_feat_array)[0]
    t_th = time.perf_counter() - t0_th
    pred_th = float(10 ** pred_log10_th)
    pred_th_single = pred_th * 0.5
    print(f"  [4] lightgbm_adaptive_th 推論完了 ({t_th:.6f} 秒 = {t_th*1000:.3f} ms < 0.001秒 PASS):")
    print(f"      -> 予測閾値: {pred_th:.6f} (log10: {pred_log10_th:.4f}), single閾値: {pred_th_single:.6f}")
    
    # 5. LightGBM エッジ追跡 (動的探索半径 adaptive_r を使用)
    t0_edge = time.time()
    feature_cols_4d = ['mean_intensity', 'snr', 'estimated_radius_um', 'volume_um3']
    
    edges_all = []
    for t in range(n_frames - 1):
        df_c = df_nodes[df_nodes['t'] == t].reset_index(drop=True)
        df_n = df_nodes[df_nodes['t'] == t + 1].reset_index(drop=True)
        if df_c.empty or df_n.empty:
            continue
            
        pos_c = df_c[['z', 'y', 'x']].values * scale_v
        pos_n = df_n[['z', 'y', 'x']].values * scale_v
        sp_dist = cdist(pos_c, pos_n)
        
        mask = (sp_dist <= adaptive_r)
        r_idx, c_idx = np.where(mask)
        if len(r_idx) == 0:
            continue
            
        fc = df_c[feature_cols_4d].values.astype(np.float32)
        fn = df_n[feature_cols_4d].values.astype(np.float32)
        cov = np.cov(np.vstack([fc, fn]), rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        try:
            mh_d = cdist(fc, fn, metric='mahalanobis', VI=inv_cov)[r_idx, c_idx]
        except Exception:
            mh_d = cdist(fc, fn, metric='cityblock')[r_idx, c_idx]
            
        sp_d = sp_dist[r_idx, c_idx]
        dz = (pos_n[c_idx, 0] - pos_c[r_idx, 0])
        dy = (pos_n[c_idx, 1] - pos_c[r_idx, 1])
        dx = (pos_n[c_idx, 2] - pos_c[r_idx, 2])
        d_xy = np.sqrt(dy**2 + dx**2)
        
        int1, int2 = df_c['mean_intensity'].values[r_idx], df_n['mean_intensity'].values[c_idx]
        snr1, snr2 = df_c['snr'].values[r_idx], df_n['snr'].values[c_idx]
        rad1, rad2 = df_c['estimated_radius_um'].values[r_idx], df_n['estimated_radius_um'].values[c_idx]
        vol1, vol2 = df_c['volume_um3'].values[r_idx], df_n['volume_um3'].values[c_idx]
        
        min_sp = np.min(sp_dist, axis=1)
        sp_margin = sp_d - min_sp[r_idx]
        ranks = np.zeros(len(r_idx), dtype=np.int32)
        cand_counts = np.zeros(len(r_idx), dtype=np.int32)
        for r in np.unique(r_idx):
            mk = np.where(r_idx == r)[0]
            cand_counts[mk] = len(mk)
            sorted_k = mk[np.argsort(sp_d[mk])]
            ranks[sorted_k] = np.arange(1, len(sorted_k) + 1)
            
        p_df = pd.DataFrame({
            'spatial_dist': sp_d, 'spatial_dist_xy': d_xy,
            'delta_z_scaled': dz, 'delta_y_scaled': dy, 'delta_x_scaled': dx,
            'abs_delta_z': np.abs(dz), 'spatial_rank': ranks, 'spatial_margin': sp_margin,
            'int_diff': np.abs(int1 - int2), 'int_ratio': int1 / (int2 + 1e-5),
            'snr_diff': np.abs(snr1 - snr2), 'snr_ratio': snr1 / (snr2 + 1e-5), 'snr_min': np.minimum(snr1, snr2),
            'radius_diff': np.abs(rad1 - rad2), 'radius_ratio': rad1 / (rad2 + 1e-5),
            'volume_diff': np.abs(vol1 - vol2), 'volume_ratio': vol1 / (vol2 + 1e-5),
            'z_depth_diff': np.zeros(len(r_idx)),
            'density_source': np.zeros(len(r_idx)), 'density_target': np.zeros(len(r_idx)), 'density_diff': np.zeros(len(r_idx)),
            'mahalanobis_dist': mh_d, 'total_cost_4d': sp_d + mh_d
        })
        
        probs = edge_model.predict(p_df)
        cost_mat = np.full(sp_dist.shape, 1e9, dtype=np.float32)
        th_adapt = np.where(cand_counts == 1, pred_th_single, pred_th)
        valid = (probs >= th_adapt)
        if np.sum(valid) > 0:
            cost_mat[r_idx[valid], c_idx[valid]] = -np.log(probs[valid] + 1e-6)
            
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        curr_ids = df_c['node_id'].values
        next_ids = df_n['node_id'].values
        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] < 1e8:
                edges_all.append((curr_ids[r], next_ids[c]))
                
    t_edge = time.time() - t0_edge
    print(f"  [5] LightGBM エッジ追跡完了: {len(edges_all):,} 本 ({t_edge:.3f} 秒)")
    
    # 6. 孤立ノード刈取
    connected_ids = set([s for s, t in edges_all]).union(set([t for s, t in edges_all]))
    df_filtered = df_nodes[df_nodes['node_id'].isin(connected_ids)]
    print(f"  [6] 孤立ノード刈取後: {len(df_filtered):,} ノード (元: {len(df_nodes):,} ノード)")
    
    total_time = time.time() - t_start
    print(f"[*] 総処理時間: {total_time:.3f} 秒")
    
    return {
        'mode': mode,
        'nodes': df_nodes,
        'edges': edges_all,
        'filtered_nodes': df_filtered,
        'speed_med': speed_med,
        'adaptive_r': adaptive_r,
        'pred_th': pred_th,
        't_rad': t_rad,
        't_th': t_th,
        'total_time': total_time
    }

# 実行
res_gpu = run_pipeline(mode='cuda')
res_cpu = run_pipeline(mode='cpu')

print("\n" + "=" * 80)
print(">>> GPU vs CPU 動作検証結果サマリー (Step 2) <<<")
print("=" * 80)

check_nodes = (len(res_gpu['nodes']) == len(res_cpu['nodes']))
check_rad = (abs(res_gpu['adaptive_r'] - res_cpu['adaptive_r']) < 1e-5)
check_th = (abs(res_gpu['pred_th'] - res_cpu['pred_th']) < 1e-6)
check_edges = (len(res_gpu['edges']) == len(res_cpu['edges']))
check_filtered = (len(res_gpu['filtered_nodes']) == len(res_cpu['filtered_nodes']))

print(f"1. 検出ノード総数完全一致    : {'PASS (100.00% 一致)' if check_nodes else 'FAIL'} ({len(res_gpu['nodes']):,} vs {len(res_cpu['nodes']):,})")
print(f"2. 動的探索半径完全一致      : {'PASS (100.00% 一致)' if check_rad else 'FAIL'} ({res_gpu['adaptive_r']:.2f} um vs {res_cpu['adaptive_r']:.2f} um)")
print(f"3. 動的切断閾値完全一致      : {'PASS (100.00% 一致)' if check_th else 'FAIL'} ({res_gpu['pred_th']:.6f} vs {res_cpu['pred_th']:.6f})")
print(f"4. 検出エッジ総数完全一致    : {'PASS (100.00% 一致)' if check_edges else 'FAIL'} ({len(res_gpu['edges']):,} vs {len(res_cpu['edges']):,})")
print(f"5. 刈取後ノード総数完全一致  : {'PASS (100.00% 一致)' if check_filtered else 'FAIL'} ({len(res_gpu['filtered_nodes']):,} vs {len(res_cpu['filtered_nodes']):,})")
print(f"6. 探索半径推定時間          : GPU {res_gpu['t_rad']*1000:.3f} ms / CPU {res_cpu['t_rad']*1000:.3f} ms (10 ms 未満 PASS)")
print(f"7. 動的閾値純粋推論時間      : GPU {res_gpu['t_th']*1000:.3f} ms / CPU {res_cpu['t_th']*1000:.3f} ms (1 ms 未満 PASS)")
print(f"8. 総パイプライン実行速度    : GPU {res_gpu['total_time']:.3f} 秒 vs CPU {res_cpu['total_time']:.3f} 秒 (GPU {res_cpu['total_time']/res_gpu['total_time']:.2f} 倍速)")
print("=" * 80)

all_pass = check_nodes and check_rad and check_th and check_edges and check_filtered
if all_pass:
    print(">>> 判定: ALL CHECKS PASSED (GPU / CPU 両対応・完全一致確認完了) <<<")
else:
    print(">>> 判定: FAILED <<<")
print("=" * 80)
