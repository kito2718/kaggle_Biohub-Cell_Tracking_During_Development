"""
s5_025_train_lightgbm.py
全199データセットのGT教師データ (約68万トラック) に対し、
5-Fold GroupKFold (group = dataset) で完全未知の OOF 予測確率を学習・推論し、
モデルファイルおよび OOF 予測結果を保存する。
"""
import sys
import os
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

TRAIN_PARQUET = Path(r"C:\work\aaa\s5\github\working\s5_025_track_training_data.parquet")
OUTPUT_MODEL_TXT = Path(r"C:\work\aaa\s5\github\working\s5_025_lightgbm_track_model.txt")
OUTPUT_OOF_PARQUET = Path(r"C:\work\aaa\s5\github\working\s5_025_oof_predictions.parquet")

FEATURES = [
    'track_length', 'straightness_ratio', 'net_velocity_um',
    'mean_step_um', 'max_step_um', 'total_disp_um', 'path_length_um',
    'confinement_ratio', 'trajectory_aspect_ratio', 'step_velocity_cv',
    'directed_motion_index',
    'mean_raw_intensity', 'min_raw_intensity', 'max_raw_intensity',
    'mean_local_snr', 'min_local_snr', 'max_local_snr',
    'mean_dog_response', 'mean_sigma_z', 'mean_sigma_xy'
]

LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 600,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

def main():
    print("=" * 75)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] s5_025_train_lightgbm: 5-Fold GroupKFold 学習・OOF推論 開始")
    print("=" * 75)

    if not TRAIN_PARQUET.exists():
        print(f"Error: {TRAIN_PARQUET} does not exist!")
        return

    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Loaded {len(df):,} tracks across {df['dataset'].nunique()} datasets.")
    print(f"  Positive (is_gt=1): {df['is_gt'].sum():,} ({df['is_gt'].mean()*100:.2f}%)")
    print(f"  Negative (is_gt=0): {(~df['is_gt'].astype(bool)).sum():,} ({(1-df['is_gt'].mean())*100:.2f}%)")

    X = df[FEATURES]
    y = df['is_gt'].values
    groups = df['dataset'].values

    # 5-Fold GroupKFold (データセット単位で完全に未知Foldに分離)
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df), dtype=np.float32)
    models = []
    fold_aucs = []
    fold_prs = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        val_ds_count = df.iloc[val_idx]['dataset'].nunique()

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        models.append(model)

        auc = roc_auc_score(y_val, val_preds)
        pr = average_precision_score(y_val, val_preds)
        fold_aucs.append(auc)
        fold_prs.append(pr)
        print(f"  Fold {fold}: Validation Datasets = {val_ds_count}, Val Tracks = {len(val_idx):,}, ROC-AUC = {auc:.4f}, PR-AUC = {pr:.4f}")

    total_auc = roc_auc_score(y, oof_preds)
    total_pr = average_precision_score(y, oof_preds)
    print("\n" + "=" * 75)
    print(f"5-Fold GroupKFold OOF Total ROC-AUC: {total_auc:.4f} (Mean Fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f})")
    print(f"5-Fold GroupKFold OOF Total PR-AUC:  {total_pr:.4f} (Mean Fold PR:  {np.mean(fold_prs):.4f} ± {np.std(fold_prs):.4f})")
    print("=" * 75)

    # 特徴量重要度 (Gain)
    importance_df = pd.DataFrame({
        'feature': FEATURES,
        'importance_gain': np.mean([m.booster_.feature_importance(importance_type='gain') for m in models], axis=0),
        'importance_split': np.mean([m.booster_.feature_importance(importance_type='split') for m in models], axis=0),
    }).sort_values('importance_gain', ascending=False).reset_index(drop=True)

    print("\nTop 15 Feature Importances (Gain):")
    print(importance_df.head(15).to_string(index=False))

    # OOF 予測結果を保存
    df_oof = df[['dataset', 'is_gt', 'track_length', 'total_disp_um']].copy()
    df_oof['oof_prob'] = oof_preds
    df_oof.to_parquet(OUTPUT_OOF_PARQUET, index=False)
    print(f"\nSaved OOF predictions to {OUTPUT_OOF_PARQUET}")

    # 全データ再学習モデル (本番推論用)
    print("\nTraining final model on 100% of data for production inference...")
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.set_params(n_estimators=int(np.mean([m.best_iteration_ for m in models])))
    final_model.fit(X, y)

    # モデル保存 (テキストツリー)
    final_model.booster_.save_model(str(OUTPUT_MODEL_TXT))
    print(f"Saved production LightGBM model to {OUTPUT_MODEL_TXT} ({os.path.getsize(OUTPUT_MODEL_TXT)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
