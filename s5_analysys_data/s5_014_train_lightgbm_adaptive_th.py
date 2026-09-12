# -*- coding: utf-8 -*-
"""
s5_014_train_lightgbm_adaptive_th.py
全 199 データセットの 12大メタ特徴量から
Kaggle 公式スコア向上 と P/E 比安全マージン制御 を両立するパレート最適動的閾値 T* を予測する
LightGBM 回帰モデル `lightgbm_adaptive_th` の 5-Fold 交差検証学習 ＆ 正式モデル出力スクリプト。

- 入力: 12大メタ特徴量 (Gain上位12特徴量)
- 教師: パレート効用関数 Utility = kaggle_score - 0.20 * max(0.0, pe_ratio - 0.99) を最大化する最適閾値 T*
- 評価: 5-Fold K-Fold CV による OOF 予測性能 ＆ 全199データセット実測 Kaggle スコア検証
- 出力:
    - モデルファイル: s5_analysys_data/lightgbm_adaptive_th.txt
    - メタ特徴量一覧: s5_analysys_data/s5_014_all199_meta_features.csv
    - 詳細検証エビデンス: s5_analysys_data/s5_014_lightgbm_adaptive_th_validation.csv
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import lightgbm as lgb
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
DATA_DIR = BASE_DIR / "s5_analysys_data"
WORKING_DIR = BASE_DIR / "working"

EVIDENCE_CSV = DATA_DIR / "s5_011_all199_threshold_evidence.csv"
OUTPUT_MODEL_TXT = DATA_DIR / "lightgbm_adaptive_th.txt"
OUTPUT_META_CSV = DATA_DIR / "s5_014_all199_meta_features.csv"
OUTPUT_VAL_CSV = DATA_DIR / "s5_014_lightgbm_adaptive_th_validation.csv"
OUTPUT_PRED_PLOT = DATA_DIR / "s5_014_predicted_vs_optimal_thresholds.png"

print("=" * 90)
print(">>> s5_014: `lightgbm_adaptive_th` モデル学習 ＆ 5-Fold CV (パレート最適版)")
print("=" * 90)

# 1. データロード
print("[*] 過去解析データおよび GT サマリーのロード中...")
df_evidence = pd.read_csv(EVIDENCE_CSV)
df_gt_s = pd.read_csv(WORKING_DIR / "s5_gt_summary.csv")
ALL_DATASETS = sorted(df_gt_s['dataset'].unique())

# パレート効用関数 (スコア向上と P/E 比過剰抑制の両立)
PE_CAP = 0.99
PENALTY_WEIGHT = 0.20

optimal_list = []
for ds, grp in df_evidence.groupby('dataset'):
    grp_c = grp.copy()
    excess = np.maximum(0.0, grp_c['pe_ratio'] - PE_CAP)
    grp_c['utility'] = grp_c['kaggle_score'] - PENALTY_WEIGHT * excess
    best = grp_c.sort_values(by=['utility', 'kaggle_score', 'pe_ratio'], ascending=[False, False, False]).iloc[0]
    optimal_list.append({
        'dataset': ds,
        'optimal_threshold': best['threshold'],
        'opt_pe_ratio': best['pe_ratio'],
        'opt_recall': best['edge_recall'],
        'opt_score': best['kaggle_score']
    })
df_optimal = pd.DataFrame(optimal_list)
print(f"[*] 全 {len(df_optimal)} データセットのパレート最適閾値選定完了 (理論上限スコア: {df_optimal['opt_score'].mean():.4f}, 平均P/E: {df_optimal['opt_pe_ratio'].mean():.4f})")

# 2. 12大メタ特徴量のロード
if OUTPUT_META_CSV.exists():
    print(f"[*] 既存の 12大メタ特徴量ファイルをロード: {OUTPUT_META_CSV}")
    df_meta_all = pd.read_csv(OUTPUT_META_CSV)
else:
    raise FileNotFoundError(f"{OUTPUT_META_CSV} が見つかりません。")

# 3. 特徴量とターゲットの準備
FEATURE_COLS = [
    'std_intensity', 'std_nn_dist', 'cell_radius_mean', 'contrast_ratio',
    'mean_intensity', 'density_per_frame', 'snr_std', 'cell_volume_mean',
    'min_nn_dist', 'mean_nn_dist', 'median_nn_dist', 'snr_mean'
]

df_train = pd.merge(df_meta_all, df_optimal[['dataset', 'optimal_threshold']], on='dataset')
X = df_train[FEATURE_COLS].values
y_true_th = df_train['optimal_threshold'].values
y_log = np.log10(y_true_th)

# 4. 5-Fold Cross Validation による OOF 性能評価
print("\n" + "-" * 90)
print("[*] 5-Fold Cross Validation による汎化性能評価開始...")
print("-" * 90)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred_log = np.zeros(len(df_train))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 10,
    'min_data_in_leaf': 5,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'verbosity': -1,
    'seed': 42
}

fold_rmses = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
    X_tr, y_tr = X[train_idx], y_log[train_idx]
    X_val, y_val = X[val_idx], y_log[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data, feature_name=FEATURE_COLS)
    
    bst = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    val_pred = bst.predict(X_val)
    oof_pred_log[val_idx] = val_pred
    rmse_fold = np.sqrt(np.mean((val_pred - y_val)**2))
    fold_rmses.append(rmse_fold)
    print(f"  - Fold {fold}: RMSE = {rmse_fold:.4f} (best iteration: {bst.best_iteration})")

oof_rmse = np.sqrt(np.mean((oof_pred_log - y_log)**2))
r_val, p_val = pearsonr(oof_pred_log, y_log)
print(f"\n[+] 5-Fold CV 全体 OOF 評価:")
print(f"    - OOF RMSE (log10 space): {oof_rmse:.4f}")
print(f"    - Pearson 相関係数 r:     {r_val:.4f} (p = {p_val:.2e})")

# 5. 全 199 データセットの本番学習 ＆ モデル出力
print("\n" + "-" * 90)
print(f"[*] 全 {len(df_train)} データセットによる本番モデル学習 ＆ ファイル出力中...")
print("-" * 90)

full_data = lgb.Dataset(X, label=y_log, feature_name=FEATURE_COLS)
final_model = lgb.train(lgb_params, full_data, num_boost_round=70)
final_model.save_model(str(OUTPUT_MODEL_TXT))
print(f"  - [OK] モデル出力完了: {OUTPUT_MODEL_TXT} (サイズ: {os.path.getsize(OUTPUT_MODEL_TXT):,} bytes)")

# 6. 全 199 データセットにおける実測 Before (固定 0.0010) vs After (予測閾値) の定量評価
print("\n" + "-" * 90)
print("[*] 全 199 データセットにおける Before vs After 実測スコア定量検証中...")
print("-" * 90)

pred_th_continuous = 10 ** oof_pred_log
AVAILABLE_THRESHOLDS = np.array([0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050])

val_rows = []
for i, row in df_train.iterrows():
    ds = row['dataset']
    p_th = pred_th_continuous[i]
    nearest_idx = np.argmin(np.abs(AVAILABLE_THRESHOLDS - p_th))
    applied_th = AVAILABLE_THRESHOLDS[nearest_idx]
    
    ds_evi = df_evidence[df_evidence['dataset'] == ds]
    rec_fixed = ds_evi[ds_evi['threshold'] == 0.0010].iloc[0]
    opt_th = row['optimal_threshold']
    rec_opt = ds_evi[ds_evi['threshold'] == opt_th].iloc[0]
    rec_pred = ds_evi[ds_evi['threshold'] == applied_th].iloc[0]
    
    val_rows.append({
        'dataset': ds,
        'fixed_th': 0.0010,
        'fixed_pe': rec_fixed['pe_ratio'],
        'fixed_recall': rec_fixed['edge_recall'],
        'fixed_penalty': rec_fixed['penalty'],
        'fixed_score': rec_fixed['kaggle_score'],
        
        'optimal_th': opt_th,
        'opt_pe': rec_opt['pe_ratio'],
        'opt_recall': rec_opt['edge_recall'],
        'opt_penalty': rec_opt['penalty'],
        'opt_score': rec_opt['kaggle_score'],
        
        'pred_raw_th': round(float(p_th), 6),
        'pred_applied_th': applied_th,
        'pred_pe': rec_pred['pe_ratio'],
        'pred_recall': rec_pred['edge_recall'],
        'pred_penalty': rec_pred['penalty'],
        'pred_score': rec_pred['kaggle_score']
    })

df_val = pd.DataFrame(val_rows)
df_val.to_csv(OUTPUT_VAL_CSV, index=False)
print(f"  - [OK] 検証エビデンス CSV 出力完了: {OUTPUT_VAL_CSV}")

# 散布図プロット
plt.figure(figsize=(7, 6))
plt.scatter(df_val['optimal_th'], df_val['pred_raw_th'], alpha=0.7, color='teal', edgecolors='k')
lims = [0.0001, 0.06]
plt.plot(lims, lims, '--', color='red', label='Ideal: y = x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ground Truth Optimal Threshold T* (Pareto Optimal)')
plt.ylabel('lightgbm_adaptive_th Predicted Threshold')
plt.title(f'lightgbm_adaptive_th Pareto OOF Predictions (r = {r_val:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(str(OUTPUT_PRED_PLOT), dpi=150)
plt.close()
print(f"  - [OK] 予測精度散布図出力完了: {OUTPUT_PRED_PLOT}")

# 7. 集計結果の表示
n_all = len(df_val)
over_pe_fixed = (df_val['fixed_pe'] > 1.00).sum()
over_pe_pred = (df_val['pred_pe'] > 1.00).sum()
mean_pe_fixed = df_val['fixed_pe'].mean()
mean_pe_pred = df_val['pred_pe'].mean()
mean_rec_fixed = df_val['fixed_recall'].mean()
mean_rec_pred = df_val['pred_recall'].mean()
mean_pen_fixed = df_val['fixed_penalty'].mean()
mean_pen_pred = df_val['pred_penalty'].mean()
score_fixed = df_val['fixed_score'].mean()
score_pred = df_val['pred_score'].mean()
score_opt = df_val['opt_score'].mean()

print("\n" + "=" * 90)
print(f">>> [検証結果] 全 {n_all} データセットにおける Before (固定) vs After (lightgbm_adaptive_th パレート最適) 定量比較")
print("=" * 90)
print(f"  項目                         | 固定閾値 (th=0.0010) | lightgbm_adaptive_th (パレート最適) | 理論最適 (T*) | 改善効果")
print(f"  -----------------------------+----------------------+-------------------------------------+---------------+------------------------")
print(f"  総合 Kaggle スコア           | {score_fixed:20.4f} | {score_pred:35.4f} | {score_opt:13.4f} | {score_pred - score_fixed:+.4f} pt 向上！")
print(f"  平均 Edge Recall             | {mean_rec_fixed:20.4f} | {mean_rec_pred:35.4f} | {df_val['opt_recall'].mean():13.4f} | {mean_rec_pred - mean_rec_fixed:+.4f} pt 向上！")
print(f"  平均 P/E 比                  | {mean_pe_fixed:20.4f} | {mean_pe_pred:35.4f} | {df_val['opt_pe'].mean():13.4f} | {mean_pe_pred - mean_pe_fixed:+.4f} pt 改善！")
print(f"  P/E > 1.00 超過件数          | {over_pe_fixed:3d} / {n_all} ({over_pe_fixed/n_all*100:4.1f}%) | {over_pe_pred:3d} / {n_all} ({over_pe_pred/n_all*100:4.1f}%)               | {(df_val['opt_pe']>1.0).sum():3d} / {n_all}   | {over_pe_fixed - over_pe_pred:+3d} 件削減！")
print(f"  平均 ペナルティ係数          | {mean_pen_fixed:20.4f} | {mean_pen_pred:35.4f} | {df_val['opt_penalty'].mean():13.4f} | {mean_pen_pred - mean_pen_fixed:+.4f} pt 改善！")
print("=" * 90)
print("[*] lightgbm_adaptive_th パレート最適学習・検証プロセス正常終了！")
