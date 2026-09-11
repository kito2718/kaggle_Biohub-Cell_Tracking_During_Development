"""
s5_003_train_lgbm_model.py
Step 3: LightGBM 分類モデルの学習、Optuna ハイパーパラメータ探索、評価エビデンス出力
- 5-Fold GroupKFold (データセット単位の完全分離)
- Optuna によるハイパーパラメータ探索 (PR-AUC 最適化)
- 特徴量重要度 (Gain / Split) の算出とプロット
- 最適確率閾値 (Threshold) 解析 (偽結合切断率 vs 真エッジ維持率)
- モデル保存: s5_003_tracking_edge_lgbm.joblib / .txt
"""

import os
import sys
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github\s5_analysys_data"
INPUT_PARQUET = os.path.join(BASE_DIR, "s5_002_train_pairs_5datasets.parquet")
OUTPUT_MODEL_JOBLIB = os.path.join(BASE_DIR, "s5_003_tracking_edge_lgbm.joblib")
OUTPUT_MODEL_TXT = os.path.join(BASE_DIR, "s5_003_tracking_edge_lgbm.txt")
OUTPUT_IMPORTANCE_PNG = os.path.join(BASE_DIR, "s5_003_lgbm_feature_importance.png")
OUTPUT_ROC_PR_PNG = os.path.join(BASE_DIR, "s5_003_lgbm_roc_pr_curve.png")
OUTPUT_EVIDENCE_CSV = os.path.join(BASE_DIR, "s5_003_train_evidence.csv")
OUTPUT_THRESHOLD_CSV = os.path.join(BASE_DIR, "s5_003_threshold_analysis.csv")

def load_data():
    print(f">> [1/6] 教師データ読込中: {INPUT_PARQUET}")
    t0 = time.time()
    df = pd.read_parquet(INPUT_PARQUET)
    
    exclude_cols = ['dataset', 't', 'source_id', 'target_id', 'source_is_tp', 'target_is_tp', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['label'].values
    groups = df['dataset'].values
    
    print(f"    - データ形状: {df.shape}")
    print(f"    - 特徴量数: {len(feature_cols)} 列")
    print(f"    - 正例(Label=1): {np.sum(y == 1):,} 件, 負例(Label=0): {np.sum(y == 0):,} 件")
    print(f"    - 負例比率: 1 : {np.sum(y == 0) / np.sum(y == 1):.2f}")
    print(f"    - データセット数 (グループ): {len(np.unique(groups))} ({list(np.unique(groups))})")
    print(f"    - 読込完了所要時間: {time.time() - t0:.2f} 秒\n")
    
    return df, X, y, groups, feature_cols

def run_optuna_tuning(X, y, groups, feature_cols, n_trials=25):
    print(f">> [2/6] Optuna ハイパーパラメータ探索開始 ({n_trials} トライアル)...")
    t0 = time.time()
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'random_state': 42,
            'n_estimators': 300,
            'verbose': -1,
            'n_jobs': -1,
        }
        
        pr_aucs = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            val_preds = model.predict_proba(X_va)[:, 1]
            pr_auc = average_precision_score(y_va, val_preds)
            pr_aucs.append(pr_auc)
            
        return np.mean(pr_aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"    - Optuna 探索完了 (所要時間: {time.time() - t0:.2f} 秒)")
    print(f"    - 最良 PR-AUC (OOF平均): {study.best_value:.4f}")
    print(f"    - 最適パラメータ:")
    for k, v in study.best_params.items():
        print(f"        {k}: {v}")
    print()
    return study.best_params

def evaluate_5fold_cv(X, y, groups, feature_cols, best_params):
    print(">> [3/6] 5-Fold GroupKFold 交差検証の厳密評価実行...")
    t0 = time.time()
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    
    oof_preds = np.zeros(len(y))
    fold_evidences = []
    feature_importances_gain = np.zeros(len(feature_cols))
    feature_importances_split = np.zeros(len(feature_cols))
    models = []
    
    base_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': 600,
        'verbose': -1,
        'n_jobs': -1,
    }
    base_params.update(best_params)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        val_dataset_name = groups[val_idx][0]
        
        model = lgb.LGBMClassifier(**base_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs
        models.append(model)
        
        roc_auc = roc_auc_score(y_va, val_probs)
        pr_auc = average_precision_score(y_va, val_probs)
        loss = log_loss(y_va, val_probs)
        
        feature_importances_gain += model.booster_.feature_importance(importance_type='gain') / 5.0
        feature_importances_split += model.booster_.feature_importance(importance_type='split') / 5.0
        
        pos_count = np.sum(y_va == 1)
        neg_count = np.sum(y_va == 0)
        
        fold_evidences.append({
            'fold': fold,
            'val_dataset': val_dataset_name,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'log_loss': loss,
            'best_iteration': model.best_iteration_,
        })
        
        print(f"    Fold {fold} [Val: {val_dataset_name} (正例{pos_count}件, 負例{neg_count}件)]: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, LogLoss={loss:.4f} (Trees: {model.best_iteration_})")
        
    overall_roc_auc = roc_auc_score(y, oof_preds)
    overall_pr_auc = average_precision_score(y, oof_preds)
    overall_loss = log_loss(y, oof_preds)
    
    print(f"\n    >> 【全体 OOF 統合評価】")
    print(f"        - Overall ROC-AUC : {overall_roc_auc:.4f}")
    print(f"        - Overall PR-AUC  : {overall_pr_auc:.4f}")
    print(f"        - Overall LogLoss : {overall_loss:.4f}")
    print(f"        - 評価所要時間    : {time.time() - t0:.2f} 秒\n")
    
    return oof_preds, fold_evidences, feature_importances_gain, feature_importances_split, base_params

def analyze_thresholds(y, oof_preds):
    print(">> [4/6] 確率閾値 (Threshold) トレードオフ解析実行中...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    records = []
    
    total_pos = np.sum(y == 1)
    total_neg = np.sum(y == 0)
    
    print(f"    {'Threshold':<10} | {'Recall(正例維持)':<15} | {'Neg Cut%(偽結合切断)':<18} | {'Precision':<12} | {'F1-Score':<10}")
    print("    " + "-" * 75)
    
    for th in thresholds:
        pred_bin = (oof_preds >= th).astype(int)
        tp = np.sum((pred_bin == 1) & (y == 1))
        fp = np.sum((pred_bin == 1) & (y == 0))
        tn = np.sum((pred_bin == 0) & (y == 0))
        fn = np.sum((pred_bin == 0) & (y == 1))
        
        recall = tp / total_pos if total_pos > 0 else 0.0
        neg_cut_rate = tn / total_neg if total_neg > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        records.append({
            'threshold': th,
            'recall_pos': recall,
            'neg_cut_rate': neg_cut_rate,
            'precision': precision,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        })
        print(f"    {th:<10.2f} | {recall*100:<14.2f}% | {neg_cut_rate*100:<17.2f}% | {precision*100:<11.2f}% | {f1:<10.4f}")
        
    df_th = pd.DataFrame(records)
    df_th.to_csv(OUTPUT_THRESHOLD_CSV, index=False)
    print(f"    - 閾値解析データ保存完了: {OUTPUT_THRESHOLD_CSV}\n")
    return df_th

def plot_and_export_evidence(feature_cols, importances_gain, importances_split, y, oof_preds, fold_evidences):
    print(">> [5/6] 特徴量重要度 ＆ 評価曲線プロット生成中...")
    
    df_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance_gain': importances_gain,
        'importance_split': importances_split,
    }).sort_values('importance_gain', ascending=True)
    
    plt.figure(figsize=(10, 8), dpi=150)
    plt.barh(df_imp['feature'], df_imp['importance_gain'], color='#1f77b4', edgecolor='black', alpha=0.85)
    plt.title("LightGBM Feature Importance (Mean Gain across 5 Folds)", fontsize=13, fontweight='bold')
    plt.xlabel("Importance (Gain)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMPORTANCE_PNG)
    plt.close()
    print(f"    - 特徴量重要度プロット保存: {OUTPUT_IMPORTANCE_PNG}")
    
    fpr, tpr, _ = roc_curve(y, oof_preds)
    precision_curve, recall_curve, _ = precision_recall_curve(y, oof_preds)
    roc_auc = roc_auc_score(y, oof_preds)
    pr_auc = average_precision_score(y, oof_preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    axes[0].plot(fpr, tpr, color='#2ca02c', lw=2, label=f'OOF ROC (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recall_curve, precision_curve, color='#d62728', lw=2, label=f'OOF PR (AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Positive)', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Precision-Recall Curve (PR)', fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_PR_PNG)
    plt.close()
    print(f"    - ROC/PR 評価曲線プロット保存: {OUTPUT_ROC_PR_PNG}")
    
    df_ev = pd.DataFrame(fold_evidences)
    df_ev.to_csv(OUTPUT_EVIDENCE_CSV, index=False)
    print(f"    - Fold別評価エビデンス CSV 保存: {OUTPUT_EVIDENCE_CSV}\n")

def train_and_save_final_model(X, y, base_params):
    print(">> [6/6] 全量データセットによる最終単一モデルの学習と保存...")
    t0 = time.time()
    
    final_model = lgb.LGBMClassifier(**base_params)
    final_model.fit(X, y)
    
    joblib.dump(final_model, OUTPUT_MODEL_JOBLIB)
    final_model.booster_.save_model(OUTPUT_MODEL_TXT)
    
    file_size_mb = os.path.getsize(OUTPUT_MODEL_JOBLIB) / (1024 * 1024)
    print(f"    - 最終モデル保存完了: {OUTPUT_MODEL_JOBLIB} ({file_size_mb:.2f} MB)")
    print(f"    - テキスト形式保存完了: {OUTPUT_MODEL_TXT}")
    print(f"    - 学習・保存所要時間: {time.time() - t0:.2f} 秒\n")

def main():
    print("=" * 80)
    print(" s5_003: LightGBM トラッキングモデル学習 & 5-Fold GroupKFold 最適化パイプライン")
    print("=" * 80)
    t_start = time.time()
    
    df, X, y, groups, feature_cols = load_data()
    best_params = run_optuna_tuning(X, y, groups, feature_cols, n_trials=25)
    oof_preds, fold_evidences, imp_gain, imp_split, final_params = evaluate_5fold_cv(X, y, groups, feature_cols, best_params)
    df_th = analyze_thresholds(y, oof_preds)
    plot_and_export_evidence(feature_cols, imp_gain, imp_split, y, oof_preds, fold_evidences)
    train_and_save_final_model(X, y, final_params)
    
    print("=" * 80)
    print(f" ALL COMPLETED SUCCESSFUL (総所要時間: {time.time() - t_start:.2f} 秒)")
    print("=" * 80)

if __name__ == "__main__":
    main()
