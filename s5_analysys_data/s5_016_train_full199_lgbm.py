# -*- coding: utf-8 -*-
"""
s5_016_train_full199_lgbm.py
全 199 データセットのペア特徴量 (s5_016_train_pairs_all199.parquet) を用いて、
5-Fold GroupKFold (データセット単位の完全分離) による LightGBM エッジ分類モデルの学習を実行し、
正式な `lightgbm_th0.005.txt` を生成するスクリプト。
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github")
DATA_DIR = BASE_DIR / "s5_analysys_data"

INPUT_PARQUET = DATA_DIR / "s5_016_train_pairs_all199.parquet"
OUTPUT_MODEL_TXT = DATA_DIR / "s5_016_tracking_edge_lgbm.txt"
CANONICAL_MODEL_TXT = DATA_DIR / "lightgbm_th=0.005.txt"
OUTPUT_EVIDENCE_CSV = DATA_DIR / "s5_016_train_full199_evidence.csv"
OUTPUT_IMPORTANCE_CSV = DATA_DIR / "s5_016_feature_importance.csv"

def train():
    print("=" * 80)
    print(">>> s5_016: 全 199 データセット LightGBM 本格再学習開始")
    print("=" * 80)
    t0 = time.time()
    
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"教師データが存在しません: {INPUT_PARQUET}")
        
    print(f"[*] 教師データ読込中: {INPUT_PARQUET.name}")
    df = pd.read_parquet(INPUT_PARQUET)
    
    exclude_cols = ['dataset', 't', 'source_id', 'target_id', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['label'].values
    groups = df['dataset'].values
    
    print(f"  - 総データ行数: {len(df):,} 行")
    print(f"  - 特徴量数: {len(feature_cols)} 列: {feature_cols}")
    print(f"  - 正例 (Label=1): {np.sum(y == 1):,} 行, 負例 (Label=0): {np.sum(y == 0):,} 行")
    print(f"  - 負例比率: 1 : {np.sum(y == 0) / np.sum(y == 1):.2f}")
    print(f"  - データセット数: {len(np.unique(groups))} 件")
    
    # 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 400,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    fold_records = []
    feature_importances = np.zeros(len(feature_cols))
    
    print("\n[*] 5-Fold GroupKFold (データセット完全分離) 交差検証開始...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        t_fold = time.time()
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        val_ds_count = len(np.unique(groups[val_idx]))
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        pr_auc = average_precision_score(y_val, preds)
        loss = log_loss(y_val, preds)
        
        # 閾値ごとの Recall
        rec_001 = np.sum((preds >= 0.001) & (y_val == 1)) / np.sum(y_val == 1)
        rec_005 = np.sum((preds >= 0.005) & (y_val == 1)) / np.sum(y_val == 1)
        
        elapsed_f = time.time() - t_fold
        print(f"  Fold {fold} (検証セット {val_ds_count} 件): AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, Loss={loss:.4f} | Recall@0.001={rec_001:.4f}, Recall@0.005={rec_005:.4f} in {elapsed_f:.1f}s")
        
        fold_records.append({
            'fold': fold,
            'val_datasets': val_ds_count,
            'auc': round(auc, 4),
            'pr_auc': round(pr_auc, 4),
            'log_loss': round(loss, 4),
            'recall_th001': round(rec_001, 4),
            'recall_th005': round(rec_005, 4),
            'best_iteration': model.best_iteration_
        })
        feature_importances += model.booster_.feature_importance(importance_type='gain') / 5.0
        
    df_folds = pd.DataFrame(fold_records)
    print("\n=== 5-Fold 平均検証結果 ===")
    print(df_folds.mean(numeric_only=True))
    df_folds.to_csv(OUTPUT_EVIDENCE_CSV, index=False)
    
    # 全データによる最終モデル学習
    print("\n[*] 全 199 データセット全数を用いた最終本番モデル学習中...")
    avg_best_iter = int(df_folds['best_iteration'].mean())
    final_params = dict(params)
    final_params['n_estimators'] = min(450, max(200, int(avg_best_iter * 1.05)))
    
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X, y)
    
    print(f"[*] モデル保存中: {OUTPUT_MODEL_TXT.name}")
    final_model.booster_.save_model(str(OUTPUT_MODEL_TXT))
    final_model.booster_.save_model(str(CANONICAL_MODEL_TXT))
    print(f"  - 正式モデル保存完了: {CANONICAL_MODEL_TXT}")
    
    # 特徴量重要度の保存
    df_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance_gain': feature_importances
    }).sort_values('importance_gain', ascending=False)
    df_imp.to_csv(OUTPUT_IMPORTANCE_CSV, index=False)
    print(f"[*] 特徴量重要度保存完了: {OUTPUT_IMPORTANCE_CSV.name}")
    print("\n上位 10 特徴量 (Gain):")
    for _, r in df_imp.head(10).iterrows():
        print(f"  {r['feature']:<20}: {r['importance_gain']:.2f}")
        
    print(f"\n総学習所要時間: {(time.time() - t0)/60:.1f} 分")
    print("=" * 80)

if __name__ == '__main__':
    train()
