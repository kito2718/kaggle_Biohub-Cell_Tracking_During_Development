# 単一事前学習済み LightGBM によるエッジ追跡 (トラッキング) 最適化計画

## 概要

本計画は、全199データセット網羅の5大特徴量アブレーション検証(s4_003)で得られた知見(「SNRが最重要」「Z深度比率はノイズ」「4D Mahalanobis で Recall 67.46% 達成」)を発展させ、**「全199データセットで一度だけ事前学習した単一の LightGBM 分類モデルを全データセットで流用する」** 高精度かつ超高速なトラッキングパイプラインを構築・検証するものです。

---

## なぜ LightGBM なのか？ (他手法との比較)

1. **現行 4D Mahalanobis (線形距離) の限界を突破**:
   - マハラノビス距離は、特徴量の共分散に基づく正規化ユークリッド距離です。
   - しかし現実の細胞追跡では、「空間距離が非常に近い($< 1.5\,\mu\text{m}$)なら多少の輝度変化を許容」「空間距離が離れている($> 4.0\,\mu\text{m}$)なら極めて高いSNR/輝度の一致を要求」といった**非線形な条件付き判断**が求められます。
   - 決定木アンサンブルである LightGBM は、この非線形な境界線を自然に学習できます。
2. **既存ベースライン(4D Mahalanobis)のスコアを理論的に下回らない**:
   - 物理空間距離だけでなく、**「4D Mahalanobis 距離そのもの」を LightGBM の入力特徴量として組み込みます**。
   - これにより、モデルは最悪でも 4D Mahalanobis と同等の精度を維持し、追加の特徴量相互作用によって精度を上積みできます。
3. **推論速度とKaggle実行制約の完全クリア**:
   - GNN(グラフニューラルネットワーク)や深層学習モデルは推論にGPUと時間を要し、19,417フレームの全量推論ではタイムアウトのリスクがあります。
   - LightGBM は CPU 上で1フレームあたり数ミリ秒で推論可能であり、メモリ消費も極小です。

---

## 提案する実装フェーズ

### フェーズ 1: 教師データ(ペアワイズ特徴量)の作成
- **スクリプト**: `s3_results_integration_and_submission/__履歴__/s4_004_create_lgbm_tracking_dataset.py`
- **データセット構成**:
  - **正例 (Label = 1)**: 全199データセットの真のGTエッジ (128,883組)。
  - **負例 (Label = 0)**: フレーム $t$ の細胞 $i$ に対し、次フレーム $t+1$ の探索半径 $d \le 7.0\,\mu\text{m}$ (物理スケール `(1.625, 0.40625, 0.40625)`) 内に存在する候補細胞 $j$ のうち、GTエッジ以外の近傍誤結合候補(Hard Negatives)。
  - **サンプリング比率**: 各ノードの距離最近傍 3〜5 件をサンプリング(正例1に対し負例3〜5程度、合計約50万行の学習データ)。
- **抽出特徴量 (約18次元)**:
  1. 3D空間幾何: $\Delta z, \Delta y, \Delta x$, 3Dユークリッド距離 $d_{3D}$, XY平面距離 $d_{xy}$, 異方性考慮 $|\Delta z|$
  2. 光学・形態の比率と絶対差分:
     - SNR比率 $\text{snr}_1 / (\text{snr}_2 + 10^{-5})$, SNR差分 $|\text{snr}_1 - \text{snr}_2|$
     - 輝度比率 $\text{int}_1 / (\text{int}_2 + 10^{-5})$, 輝度差分 $|\text{int}_1 - \text{int}_2|$
     - 体積比率 $\text{vol}_1 / (\text{vol}_2 + 10^{-5})$, 体積差分 $|\text{vol}_1 - \text{vol}_2|$
     - 推定半径比率 $r_1 / (r_2 + 10^{-5})$, 半径差分 $|r_1 - r_2|$
  3. ベースライン指標:
     - 4D Mahalanobis 距離 (最良ベンチマークスコア)
  4. 近傍ランキング特徴量:
     - 送信側ノード $i$ から見た距離順位 (1位, 2位, 3位...)
     - 受信側ノード $j$ から見た逆方向距離順位
- **出力ファイル**: `s3_results_integration_and_submission/__履歴__/s4_004_lgbm_train_pairs.parquet` (約20MB〜30MB)

---

### フェーズ 2: 単一モデルのオフライン事前学習
- **スクリプト**: `s3_results_integration_and_submission/__履歴__/s4_004_train_lgbm_tracking_model.py`
- **検証スキーム**:
  - **5-Fold GroupKFold**: データセットIDをグループ単位とし、同一データセットのペアが学習と検証に跨がらない完全リークフリー構成。
  - **評価指標**: Validation AUC, LogLoss, Binary Accuracy。
- **モデルハイパーパラメータ**:
  - `objective='binary'`, `metric='auc'`, `learning_rate=0.05`, `num_leaves=31`, `n_estimators=1000`, `early_stopping_rounds=50`
- **出力成果物**:
  - 学習済みモデル: `s3_results_integration_and_submission/__履歴__/s4_004_tracking_edge_lgbm.joblib`
  - 特徴量重要度プロット: `s4_004_lgbm_feature_importance.png`
  - ROC / PR 曲線プロット: `s4_004_lgbm_roc_pr_curve.png`

---

### フェーズ 3: 全199データセット推論 & ハンガリアンマッチング検証
- **スクリプト**: `s3_results_integration_and_submission/__履歴__/s4_004_evaluate_lgbm_tracking_all199.py`
- **推論・割り当てアルゴリズム**:
  1. フレーム間の候補ペア ($d \le 7.0\,\mu\text{m}$) を抽出。
  2. 事前学習モデル `tracking_edge_lgbm.joblib` で接続確率 $P(\text{link})$ を一括推論。
  3. コスト行列 $C_{ij} = -\log(P(\text{link}) + 10^{-9})$ (または $1 - P(\text{link})$) を構築。
  4. scipy `linear_sum_assignment` (ハンガリアン法) で最適マッチング。
  5. 孤立ノード刈り取り(Track Length $\ge 2$)を適用。
- **評価・比較指標**:
  - 全199データセット平均 Macro Recall
  - `44b6` 系列 (組織密, n=71) Recall
  - `6bba` 系列 (組織疎, n=128) Recall
  - 総真陽性エッジ数 (TP本数)
  - 4D Mahalanobis (67.46%) / 5D Mahalanobis (67.40%) / 純粋空間距離 (65.85%) との直接比較。
  - 1フレームあたりの推論処理時間 (ms)。
- **出力成果物**:
  - `s4_004_lgbm_tracking_evaluation_all199.csv` / `.xlsx`
  - `s4_004_lgbm_vs_mahalanobis_comparison.png` (比較棒グラフ)

---

### フェーズ 4: 総合レポート (`s4_999_pe_ratio_control_report.md`) への統合
- 第14章として「単一事前学習済み LightGBM モデルによるエッジ追跡検証結果 (s4_004)」を追記。
- 第15章「まとめ」を最終更新。

---

## ユーザー確認事項 (User Review Required)

> [!IMPORTANT]
> - **物理スケール厳守**: すべての距離計算・特徴量抽出において、`AGENTS.md` 規則通り `scale=(1.625, 0.40625, 0.40625)` を徹底適用します。
> - **事前学習済み単一モデルの採用**: 提出用ノートブック(`s3_100_results_integration_and_submission.ipynb`)には、学習コードではなく「学習済み重みファイル(`s4_004_tracking_edge_lgbm.joblib`)をロードして推論するコード」のみを配置するため、提出時の実行時間が数分伸びる程度に収まります。
> - **命名規則**: すべての成果物にプリフィックス `s4_004_` を付与します。

## 検証手順 (Verification Plan)
1. 教師データ作成スクリプトを実行し、正例12.8万本・負例約40万本のペアデータが正常に生成されることを確認。
2. 5-Fold GroupKFold による LightGBM 学習を実行し、Validation AUC が 0.90 以上に達することを確認。
3. 全199データセットでハンガリアンマッチングを実行し、Macro Recall が現行ベスト(67.46%)を上回り、TPエッジ数が増加しているかを定量確認。
