# -*- coding: utf-8 -*-
"""
s5_002_test_pair_dataset.py
生成された LightGBM 教師データ (s5_002_train_pairs_5datasets.parquet) の
品質・整合性・特徴量分離度を検証するスクリプト。
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development")
DATA_DIR = BASE_DIR / "s5" / "github" / "s5_analysys_data"
PARQUET_PATH = DATA_DIR / "s5_002_train_pairs_5datasets.parquet"

print("=" * 80)
print(f">>> s5_002: 教師データ品質 & 特徴量分離度 検証開始")
print(f"[*] 対象ファイル: {PARQUET_PATH.name}")
print("=" * 80)

t0 = time.time()
df = pd.read_parquet(PARQUET_PATH)
print(f"[*] データ読み込み完了: {len(df):,} 行, {len(df.columns)} 列 ({time.time()-t0:.2f}秒)")

# 1. 基本統計
pos_mask = (df["label"] == 1)
neg_mask = (df["label"] == 0)
n_pos = pos_mask.sum()
n_neg = neg_mask.sum()

print(f"\n[1] クラス内訳:")
print(f"  - 正例 (Label = 1: 真の追跡エッジ) : {n_pos:,} 件 ({n_pos/len(df)*100:.2f}%)")
print(f"  - 負例 (Label = 0: 偽結合・ノイズ) : {n_neg:,} 件 ({n_neg/len(df)*100:.2f}%)")
print(f"  - 全体不均衡比率                    : 1 : {n_neg/n_pos:.2f}")

# 2. 欠損値・無限大チェック
null_counts = df.isnull().sum()
has_null = null_counts.sum() > 0
print(f"\n[2] 欠損値チェック (NaN/null):")
if not has_null:
    print(f"  - PASS [OK]: 全 {len(df.columns)} 列で欠損値 0 件")
else:
    print(f"  - FAIL [NG]: 欠損値検出 -> {null_counts[null_counts > 0].to_dict()}")

# 数値列の inf チェック
num_cols = df.select_dtypes(include=[np.number]).columns
has_inf = np.isinf(df[num_cols].values).any()
print(f"\n[3] 無限大チェック (inf/-inf):")
if not has_inf:
    print(f"  - PASS [OK]: 全数値列で inf 0 件")
else:
    print(f"  - FAIL [NG]: 無限大値を検出")

# 3. データセット別内訳
print(f"\n[4] データセット別ペア件数 (GroupKFold の準備):")
ds_group = df.groupby("dataset")["label"].agg(["count", lambda x: (x == 1).sum(), lambda x: (x == 0).sum()])
ds_group.columns = ["total_pairs", "pos_pairs", "neg_pairs"]
ds_group["pos_ratio_%"] = (ds_group["pos_pairs"] / ds_group["total_pairs"] * 100).round(2)
print(ds_group.to_string())

# 4. 主要特徴量の正例 vs 負例の分離度検証
key_features = [
    "spatial_dist", "spatial_margin", "spatial_rank",
    "int_diff", "int_ratio", "snr_diff", "snr_ratio", "snr_min",
    "radius_diff", "volume_diff", "mahalanobis_dist", "total_cost_4d"
]

print(f"\n[5] 主要特徴量の正例 vs 負例の分離度 (平均値比較):")
print(f"{'特徴量名':20s} | {'正例 (平均)':12s} | {'負例 (平均)':12s} | {'差異の向き / 解釈':30s}")
print("-" * 80)

feature_separation_report = []
for feat in key_features:
    if feat not in df.columns:
        continue
    pos_mean = df.loc[pos_mask, feat].mean()
    neg_mean = df.loc[neg_mask, feat].mean()
    
    if feat in ["spatial_dist", "mahalanobis_dist", "total_cost_4d", "spatial_rank"]:
        interp = "正例の方が大幅に小さい (高分離)" if pos_mean < neg_mean else "逆転"
    elif feat in ["snr_min"]:
        interp = "正例の方が高SNR (偽ノード排除)" if pos_mean > neg_mean else "逆転"
    else:
        interp = "正例の方が差分小 / 安定" if pos_mean < neg_mean else "負例の方が差分大"
        
    print(f"{feat:20s} | {pos_mean:12.4f} | {neg_mean:12.4f} | {interp}")
    feature_separation_report.append({
        "feature": feat,
        "pos_mean": round(pos_mean, 4),
        "neg_mean": round(neg_mean, 4),
        "interpretation": interp
    })

# 5. 総合判定
all_ok = (not has_null) and (not has_inf) and (n_pos > 1800) and (n_neg > 50000)
print("\n" + "=" * 80)
if all_ok:
    print(">>> [SUCCESS] 教師データ品質検証 ALL PASS: LightGBM 学習準備完了！")
else:
    print(">>> [FAILED] 検証項目に不一致が存在します。")
print("=" * 80)
