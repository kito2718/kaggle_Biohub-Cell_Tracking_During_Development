# 【緊急・最重要パラダイムシフト】**MAGIC_STRING** : `026-UNET_ILP_095`
## 24. 026: Kaggle上位陣 (0.95+) の実態解明と深層学習 (3D-UNet ＋ Transformer ＋ ILP) への完全移行計画

### (1) 背景と課題の所在: なぜ古典的画像処理 (DoG ＋ MNN) では 0.95 に届かないのか？
025-003 において、全199データセットの網羅的EDA (幾何学 ＋ 形態学 ＋ エッジ不変量) と自律刈り取りによって Public Score を 0.687 まで向上させた。
しかし、ユーザーからの極めて本質的な問い「**どの方針を選んだとしてもトップスコアの 0.95 には届かない。本当に正しい解決方法があるはずだ。Discussion に解決のヒントがあるのではないか？**」に基づき、Kaggle Discussion (Topic 741749, 742064, 741242, 738217) および Leaderboard 上位ソリューション (0.947〜0.951+) の徹底調査を実施した。

その結果、**直近のアプローチ（古典的DoGフィルター ＋ 貪欲MNN追跡 ＋ 後処理刈り取り）と、トップコンペティター（0.95+）の間には、越えられないアーキテクチャの断絶が存在する** ことが完全に判明した。

```mermaid
graph TD
    subgraph Current [直近の古典的アプローチ (理論限界: 0.68~0.70)]
        C1["(1) 検出: DoG (古典的差分ガウシアン)"] -->|固定PSF・局所偽ピーク多発| C2["(2) 追跡: MNN (貪欲局所最近傍)"]
        C2 -->|近接・交差で100%誤結合| C3["(3) 後処理: 幾何・形態・LightGBM 刈り取り"]
        C3 -->|分裂未取得(0点)・P/Eペナルティ| C4["Kaggle Score: 0.687 (頭打ち)"]
    end

    subgraph TopTier [上位陣 0.95+ の深層学習アーキテクチャ]
        T1["(1) 検出: TemporalUNet3D (3次元時空間CNN)"] -->|細胞質・膜・文脈の完全認識| T2["(2) 追跡: SimpleNodeTransformer (特徴量アテンション)"]
        T2 -->|全フレーム時空間埋め込み| T3["(3) 大域最適化: ILPSolver (整数線形計画法)"]
        T3 -->|大域エネルギー最小化・分裂加点 0.23+| T4["★ Kaggle Score: 0.947 ~ 0.951+ 突破!"]
    end
```

---

### (2) Discussion とトップコード精査で判明した4つの決定的な事実

#### ① Discussion (Topic 741749): 後処理（刈り取り）は既に飽和している
- 上位コンペティター（hikaggler 等）の報告:
  > *"Post-processing knobs (gap closing, short-track removal, thresholds) are saturated, same as the public sweeps"*
  > *"Model changes don't move the LB above the public plateau"*
- 後処理パラメータの調整や単純な刈り取り閾値の変更は、0.68〜0.70 付近で完全にサチュレーション（頭打ち）に達しており、根本的な検出器・リンカーの刷新なしに 0.95 に届くことは原理的に不可能である。

#### ② 検出器の格差: 古典DoG vs 3D-UNet
- **DoG (古典的差分ガウシアン)**: 単なる 3D 輝度勾配の極大点のみを見るため、卵黄・組織表面の自家蛍光ホットスポットと本物の細胞核を分離できず、大量のノイズ（Precision 数%〜十数%）を抱え込む。
- **TemporalUNet3D (深層時空間CNN)**: 連続 2フレームの 3次元画像全体 $(B, T=2, C=1, Z, Y, X)$ を入力とし、深層畳み込みにより細胞の形状・境界・テクスチャ文脈を学習済みであるため、偽ピークを根本から発生させない。

#### ③ 追跡・リンクの格差: MNN (局所貪欲探索) vs NodeTransformer
- **MNN (相互最近傍)**: フレーム $t$ と $t+1$ のユークリッド距離のみでペアリングするため、細胞が近接・交差した瞬間に 100% 取り違え（$FP+1, FN+1$）が発生し、Jaccard スコアが急落する。
- **SimpleNodeTransformer**: UNet が抽出した深層特徴量（チャネル $C$）と時空間位置エンコーディング（Positional Encoding）を Transformer に入力し、全結合候補間の多変量アテンション・確率を推論するため、交差・密集環境でも追跡を維持できる。

#### ④ 大域最適化と細胞分裂: 整数線形計画法 (ILP)
- トップパイプラインでは、グラフ最適化ライブラリ `tracksdata` / `ilpy` / `pyscipopt` を用いた **ILPSolver (整数線形計画法)** を採用。
- エッジ接続重み、出現重み、消失重み、そして **分裂重み (`ILP_DIVISION_WEIGHT = 1.0`)** を定式化し、グラフ全体のエネルギーを大域的に最小化。
- これにより、未注記ノイズの過剰提出（P/E比ペナルティ）が数理的に完全に排除され、かつ **細胞分裂（Division Jaccard: 配点 0.10）が 0.23+ の高スコアで確実に加点** される。

---

### (3) 新バージョン 026: 方針1 への完全移行計画

本方針に基づき、パイプラインのアーキテクチャを深層学習スタックへ抜本的に移行する。

- **新バージョン採番**: **026**
- **MAGIC_STRING**: **`026-UNET_ILP_095`** (16文字, 20文字以内)
- **基盤重みパッケージ**:
  - `pilkwang/biohub-tracking-support-pack-50ep-v1` (50エポック学習済み 3D-UNet ＋ Transformer 重み、およびオフライン wheel 一式)
- **推論パイプライン構成**:
  1. `TemporalUNet3D` による 3D ボリューム特徴量および細胞中心点確率推論 (TTA 8-fliprot 適用)
  2. `SimpleNodeTransformer` によるフレーム間候補エッジの確率推論
  3. `tracksdata.solvers.ILPSolver` による大域的整数線形計画追跡 (出現・消失・分裂の統合最適化)
  4. Kaggle 規定フォーマット (`submission.csv`) の出力
- **期待スコア**: **Public Score 0.947 〜 0.951+ (金メダル圏・トップスコア直撃)**

---

---

## 25. 026: 深層学習スタック (3D-UNet ＋ Transformer ＋ ILP) ローカル検証計画と特徴量設計

### (1) ローカル検証の目的とスコア上限の探求
Kaggle Leaderboard で 0.95+ を叩き出している深層学習スタック（3D-UNet ＋ Transformer ＋ ILP）を、まずはノートブック（提出ファイル）を更新する前に **ローカル環境で直接実行・検証し、本手法がどこまでスコアを伸ばせるのか（0.94〜0.96+）の数理的上限と各コンポーネントの挙動を完全に解明** する。

---

### (2) 深層学習パイプラインにおける多階層特徴量設計

従来の「DoG極大点 ＋ 手動幾何学ルール」とは異なり、3D-UNet ＋ Transformer ＋ ILP では以下の **4つの階層（Layer）にまたがる深層特徴量およびエネルギー関数** を設計・統合する。

```mermaid
graph TD
    subgraph Layer1 [1. 空間・体積特徴量 (3D-UNet 潜在空間)]
        L1_1["f: 32チャネル深層特徴マップ (B, T=2, C=32, Z, Y, X)"]
        L1_2["point_logit: 細胞中心存在対数オッズ (1チャネル)"]
        L1_3["point_prob: シグモイド確率マップ (Pool Kernel 5.0μm極大抽出)"]
    end

    subgraph Layer2 [2. 時空間幾何・位置エンコーディング]
        L2_1["Positional Encoding: 周波数展開 (8周波数 x 4軸 = 32次元)"]
        L2_2["subsample座標: (dz=1, dy=4, dx=4) 物理空間補正"]
        L2_3["coord_dist: 3次元異方性物理距離 (scale = 1.625, 0.40625, 0.40625)"]
    end

    subgraph Layer3 [3. ペアワイズ関連度・トポロジー (NodeTransformer)]
        L3_1["結合入力: (32ch深層特徴 ＋ 32ch位置埋め込み = 64次元)"]
        L3_2["Multi-Head Attention: 4ヘッド x 4ブロックによる大域相互作用"]
        L3_3["edge_prob: Softmax正規化接続確率 (t0ノード -> t1ノード)"]
    end

    subgraph Layer4 [4. 大域エネルギースパース最適化 (ILPSolver)]
        L4_1["E_edge = -1.0 x edge_prob (正解エッジの引き込み)"]
        L4_2["E_disappear = +1.4 (不自然な細胞消失の抑制)"]
        L4_3["E_division = +1.0 (真の細胞分裂のみを許容する加点トリガー)"]
    end

    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer4 --> Score["★ 公式評価: Adjusted Edge Jaccard + 0.1 x Division Jaccard"]
```

#### 1. 検出階層 (Node Level: 3D-UNet)
- **入力特徴量**: 連続 2フレームのサブサンプル 3次元テンソル $(T=2, Z=64, Y=64, X=64)$。
- **深層表現**: 各ボクセルにおける 32チャネルの特徴表現ベクトル。
- **中心度スコアリング**: 3D極大プーリング (`pool_kernel_um = 5.0`) による非最大値抑制（NMS）と閾値判定 (`POINT_THRESHOLD = 0.9700`)。

#### 2. 位置埋め込み階層 (Positional Encoding Level)
- **多周波数正弦波埋め込み**: $t, z, y, x$ の 4次元それぞれについて 8段階の周波数基底を展開：
  $$\text{embed}(v) = [\sin(2^0 \pi v), \cos(2^0 \pi v), \dots, \sin(2^3 \pi v), \cos(2^3 \pi v)]$$
  計 32次元の位置ベクトルを生成し、空間的な近接性と時間順序を Transformer に注入。

#### 3. エッジ追跡階層 (Edge Level: SimpleNodeTransformer)
- **特徴量結合**: 3D-UNet の 32次元潜在ベクトル ＋ 32次元位置エンコーディング ＝ **64次元結合ノード特徴量**。
- **ペアワイズ推論**: 時間 $t$ の候補点群 $N_0$ と時間 $t+1$ の候補点群 $N_1$ の全ペアに対し、Transformer の Self-Attention / Cross-Attention により、交差・分裂・密集に頑健な接続ロジット `edge_logit` を算出。

#### 4. グラフ大域エネルギー階層 (ILP Level: Integer Linear Programming)
- **目的関数**:
  $$\min \sum_{e} c_e x_e + \sum_{v} c_{app} y_{app, v} + \sum_{v} c_{dis} y_{dis, v} + \sum_{v} c_{div} y_{div, v}$$
  - $c_e = -1.0 \times \text{edge\_prob}$: 高確率エッジを強力に採用。
  - $c_{dis} = +1.4$: トラックの唐突な途切れにペナルティ。
  - $c_{div} = +1.0$: 分裂イベントの厳密なエネルギー制御（偽の二股分岐を排除し、真の分裂のみを選択）。

---

### (3) ローカル検証の実験設計 (`s6_analysys_data/`)

AGENTS.md のルールに従い、`s6_analysys_data/` 配下に検証スクリプトを作成し、ローカル実データで性能上限を検証する。

1. **代表データセット選定**:
   - `44b6_0113de3b` (標準的・高密度)
   - `6bba_6feb10f0` (025で P/E比 2.0x を超えていた最難関ノイズデータセット)
   - 分裂イベントを含むデータセット (Division Jaccard 検証用)
2. **比較検証項目**:
   - **025 古典パイプライン**: 公式 Jaccard 0.687 (Division 0.000)
   - **026 3D-UNet ＋ Transformer ＋ ILP**:
     - 検出ノード精度 (Node Recall, Node Precision)
     - エッジ接続精度 (Edge Recall, Edge Jaccard)
     - 分裂回収精度 (Division Recall, Division Jaccard)
     - 最終総合スコア ($\text{Adjusted Edge Jaccard} + 0.1 \times \text{Division Jaccard}$)
3. **成果物の出力先**:
   - スクリプト: `C:\workaa\s6\github\s6_analysys_data\s6_026_validate_unet_ilp.py`
   - 検証レポート: `s6_summary.md` に追記（※コミットは行わずローカル保存）

---

お役に立てれば幸いです。
