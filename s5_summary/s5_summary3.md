# s5_summary3.md: 022MULTISTAGE_TRACKER2（同一細胞判定器の特徴量EDA）調査データ全数抽出ノートブック設計・実装記録

## 1. エグゼクティブサマリー

本ドキュメントは、021/022のスコア急落（0.701 ➔ 0.592/0.582）の根本原因解明および反省記録（`s5_summary2.md`）に基づき、新たに立ち上げられた**「022MULTISTAGE_TRACKER2（同一細胞判定器の特徴量EDA）」専門スレッド**における初期調査データ全数抽出の設計・実装成果を克明に記録したものである。

ローカル環境のストレージI/O帯域の制約を解決し、クラウド上の計算資源（Kaggle Notebook）を活用して高速・安全に全数抽出を実行するため、標準パイプライン形式を踏襲した **`s5/github/working/s5_022_try_and_error.ipynb`** を新規設計・生成した。

---

## 2. 本専門スレッドのミッションと確定ルール

1. **021と022の完全分離**:
   - 021（ノード検出単体・P/E比適正化）と 022（エッジ追跡単体・同一細胞判定器）を完全に分離する。
   - 021の背景差分ノイズは1ミリも混入させず、**100%信用できるクリーンなGTテーブル（`s5_gt_nodes.csv`, `s5_gt_edges.csv`）のみを入力データ**とする。
2. **モデル学習の完全凍結と実測データ検証（全数EDA）への専念**:
   - モデル学習（LightGBM等）やエッジ予測・提出ファイル生成は一切行わず、純粋な「同一細胞ペアと多角的不変量特徴量の全数抽出・EDA調査」に専念する。
3. **エビデンス至上主義と4大監査の義務付け**:
   - 憶測による判断を完全排除し、全199データセット・全100フレームの完全貫通測定を実施する。
   - コードを読まずに出力ログの数行だけで動作の完全性を証明する「4大監査機能（入力整合性、プレビュー表示、全数貫通確認、数学的保存則アサーション）」を組み込む。
4. **不要処理・不要変数の完全削除**:
   - 021に関する機能（背景差分ハイブリッド等）、ノード検出、エッジ追跡、提出生成、および不要変数は無効化ではなく**完全削除**する。

---

## 3. 調査対象ペアの厳密定義（母集団）

全199データセット・全100フレームのGTデータから、始点ノード $A(t)$ と終点候補ノード $B(t+\Delta t)$（$\Delta t \in \{2, 3\}$）のペアを以下の基準で全数抽出する：

1. **正例ペア（同一細胞の真の欠損ペア, `label = 1`）**:
   - GTグラフ（有向エッジ連鎖）において、中間ノードが検出漏れ（瞬き・局所ボケ）したと想定されるペア。
   - $\Delta t = 2$（1コマ欠損）: $u(t) \to v(t+1) \to w(t+2)$ における $(u, w)$ ペア。
   - $\Delta t = 3$（2コマ欠損）: $u(t) \to \dots \to z(t+3)$ における $(u, z)$ ペア。
2. **難関負例ペア（至近10μm以内の他人細胞ペア, `label = 0`）**:
   - 従来のGap Closingが無条件結合して誤接続（公式FP）を招いていた最大の原因である「至近距離の他人」を厳密に抽出。
   - 始点 $A(t)$ に対し、時刻 $t+\Delta t$（$\Delta t \in \{2, 3\}$）に存在するすべてのGTノードの中から、同一トラック（祖先・子孫）に属さない真の他人細胞であり、かつ **3D実空間物理距離（Z: 1.625μm, Y: 0.40625μm, X: 0.40625μm）が 10.0μm 以内にあるペア**。

---

## 4. 多角的不変量特徴量の完全網羅設計（全7領域・40項目以上）

照明変動・経時退色・焦点ボケに頑健な、同一細胞判定のための客観的特徴量を網羅的に設計した：

### (1) ペア基本属性
- `dataset`: データセット名
- `source_id`, `target_id`: 始点・終点ノードID
- `t_source`, `t_target`: 始点・終点フレーム時刻
- `dt`: 時間差（$\Delta t \in \{2, 3\}$）
- `label`: 正例 `1` / 難関負例 `0`

### (2) 空間幾何・実空間変位
- `dist_3d`: 3D物理ユークリッド距離（$\mu m$）
- `dist_xy`: XY平面距離（$\mu m$）
- `dist_z`: Z軸絶対距離（$\mu m$）
- `dx`, `dy`, `dz`: 各軸の実空間変位（$\mu m$）

### (3) 双方向時系列運動物理（物理法則の連続性・慣性）
- `v_prev`: 始点 $A$ の直前移動速度ノルム $\|P(t) - P(t-1)\|$（$\mu m/\text{frame}$）
- `v_gap`: ギャップ平均移動速度ノルム $\text{dist\_3d} / \Delta t$（$\mu m/\text{frame}$）
- `v_next`: 終点 $B$ の直後移動速度ノルム $\|P(t+\Delta t+1) - P(t+\Delta t)\|$（$\mu m/\text{frame}$）
- `v_ratio_gap_prev`, `v_ratio_next_gap`, `v_ratio_next_prev`: 速度変化比3種
- `accel_fwd`, `accel_bwd`: 順方向・逆方向の推定加速度ノルム
- `cos_gap_prev`: 進行方向コサイン類似度 $\cos(\vec{v}_{prev}, \vec{v}_{gap})$
- `cos_next_gap`: 未来方向コサイン類似度 $\cos(\vec{v}_{gap}, \vec{v}_{next})$
- `cos_next_prev`: 前後軌跡コサイン類似度 $\cos(\vec{v}_{prev}, \vec{v}_{next})$
- **双方向推測航法残差 (Dead Reckoning Error)**:
  - `dead_reckoning_fwd`: 順方向予測着弾点 $\hat{P}_B = P_A + \vec{v}_{prev} \cdot \Delta t$ と $P_B$ の残差距離（$\mu m$）
  - `dead_reckoning_bwd`: 逆方向予測出発点 $\hat{P}_A = P_B - \vec{v}_{next} \cdot \Delta t$ と $P_A$ の残差距離（$\mu m$）
  - `dead_reckoning_mean`: 順逆推測航法残差の平均値（$\mu m$）

### (4) 空間トポロジー・近傍競合コンテキスト
- `comp_count_10um`, `comp_count_15um`: 周囲10μm/15μm以内の候補ノード総数（混雑度）
- `fwd_rank`, `bwd_rank`: 始点から見た終点、および終点から見た始点の最近傍ランク（1位, 2位...）
- `is_mnn`: 双方向相互最近傍フラグ（互いに1位同士なら1, 他は0）
- `margin_fwd`, `margin_bwd`: 第1位候補と第2位候補の距離マージン（$\mu m$）
- `local_density_ratio`, `local_density_diff`: 局所密度の比率および差

### (5) 形態保存性・幾何アスペクト・分裂前兆シグナル
- `radius_ratio`, `radius_diff`: 推定細胞半径の比率および差
- `vol_ratio`: 体積比 $V_B / (V_A + 1e-5)$（$\approx 1.0$ で同一移動、$\approx 0.5$ で分裂娘細胞）
- `vol_diff_rel`: 相対体積変化率 $|V_A - V_B| / \max(V_A, V_B)$
- `mitosis_score`: 生物学的分裂シグナルスコア $|V_B / V_A - 0.5|$
- `mitosis_flag`: 分裂候補フラグ（$0.35 \le vol\_ratio \le 0.65$）
- `z_depth_ratio_diff`: Z深度比率の変化量

### (6) 光学・退色不変量
- `mean_intensity_ratio`: 平均輝度比
- `mean_intensity_diff_rel`: 相対輝度変化率
- `intensity_rank_diff`: 同一フレーム内輝度パーセンタイル順位の差（退色や照明ムラに不変）
- `snr_ratio`, `snr_diff`: SNR比率および差

---

## 5. ノートブック構成（`s5_022_try_and_error.ipynb`）

`s5_001_try_and_error.ipynb` の流儀を踏襲しつつ、今回のデータ取得タスクに特化した全12セル構成：

```text
Cell 0 [Markdown] : タイトル & 目的
Cell 1 [Markdown] : 処理フロー & 4大監査仕様解説
Cell 2 [Code]     : オフライン パッケージインストール (polars, pyarrow, scipy 最小限)
Cell 3 [Code]     : パラメータ設定 (TARGET_DATASETS, dt=[2,3], 10um, 物理スケール, GitHub連携)
Cell 4 [Code]     : 環境構築・シード固定 (setup_environment)
Cell 5 [Code]     : 共通ユーティリティ (get_or_init_git_repo, push_to_github)
Cell 6 [Code]     : 【監査1】必須GT入力データ完全性検証 (check_environment)
Cell 7 [Code]     : クリーンGTデータ読み込み & メモリ展開 (load_gt_data)
Cell 8 [Code]     : 【主処理】正例・難関負例全数抽出 & 40種特徴量算出 (extract_same_cell_features)
Cell 9 [Code]     : 【監査2・3・4】4大監査 & サニティチェック (check_extracted_features)
Cell 10 [Code]    : Parquet/CSV保存 & GitHub自動プッシュ (save_and_push_results)
Cell 11 [Code]    : メイン関数 (main 一気通貫エントリポイント)
```

---

## 6. 実装上の重要監査機能と検証エビデンス

### (1) `check_environment()` による入力完全性アサーション
- Kaggle Dataset、Gitクローン先、ローカル作業ディレクトリを自動探索。
- `s5_gt_nodes.csv`（全199データセット完備・13.3万行・必須12カラム）および `s5_gt_edges.csv`（必須4カラム）が100%揃っているかを自動検証。不足があれば例外スローして停止。

### (2) `check_extracted_features()` による4大監査
1. **入力・出力整合性アサーション**: サマリーデータセット数とGT入力件数の完全一致。
2. **人間可読プレビュー表示**: 正例・負例の先頭抜粋を標準出力へ表示し、目視確認可能化。
3. **全数貫通確認**: 全199データセット・全フレームの走査完了と正負例件数サマリーの表示。
4. **数学的保存則 & 数値安定性アサーション**:
   - 特徴量テーブル内の `NaN` / `Inf` ゼロ検証。
   - 抽出正例総数がGTグラフ経路数の理論値と完全一致することの検証。

### (3) コード検証エビデンス
- **構文検証**: Python AST（抽象構文木）パースにより、全コードセルが文法エラーなし（`[ALL CODE CELLS PASSED SYNTAX CHECK!]`）であることを確認済み。
- **パス解決テスト**: ローカルおよびKaggle環境における入力ファイル自動検出ロジックが正常動作することを確認済み。

---

## 7. 出力成果物とGitHub同期

- **スクリプトファイル**: `s5/github/working/s5_022_try_and_error.ipynb`
- **生成データ出力先**:
  - `s5/github/working/s5_022_gt_pairs_features_all199.parquet`（全特徴量テーブル、高圧縮・高速）
  - `s5/github/working/s5_022_gt_pairs_summary_all199.csv`（データセット別サマリー）
- **GitHub自動同期**: ノートブック実行完了時に `push_to_github()` により `origin/main` ブランチへ自動コミット＆プッシュ。

お役に立てれば幸いです。
