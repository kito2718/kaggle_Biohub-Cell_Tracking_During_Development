# s7_analysys_data: 0.867 → 0.97+ スコアギャップ調査エビデンス集

## 1. 調査背景と問題設定
- **現在地**: `026-UNET_ILP_095` から `029-FACTS_LGB_ADAPTIVE_099` までの Public LB スコアは **0.867** で完全固定。
- **課題**: Top層 (0.940〜0.975+) との間に +0.07〜+0.10 の隔絶が存在。
- **目的**: 歴史的背景、Kaggle Discussion (トピック 741749, 742064, 741242, 738217 等)、最新文献 (Nature Methods 2025, bioRxiv 2026, arXiv 2026)、GitHub コードベースの多角精査による、スコアブレイクスルーの技術的エビデンス集積。

---

## 2. 調査エビデンス詳細

### エビデンス A: 提供ベースラインの構造的限界 (点検出パラダイムの終焉)
- 提供コード (`TemporalUNet3D` + `SimpleNodeTransformer` + `ILPSolver`) は「細胞中心点のみを抽出して距離/外観で結ぶ」点検出 (Point-based Detection) パラダイム。
- **欠陥 1: 境界・体積・形態情報の完全欠落**
  - 点検出では、細胞の体積、扁平率、主軸方向、境界ボクセル勾配が破棄される。
  - 細胞分裂直前・直後の形状歪みや長軸配向を捉えられず、分裂判定 (`division_jaccard`) が破綻。
- **欠陥 2: 近接密集・低コントラスト領域での分離不能**
  - 解像度 (1.625μm 等方) に対し、近接する2つの核が同一の極大点として統合されてしまう。
- **欠陥 3: ノード数調整のサチュレーション**
  - 028/029 で実証された通り、閾値微調整や GBDT によるノード数適応では、難所胚で +0.03 程度救出できても、Public LB (容易な胚) では 0.867 から動かない。

### エビデンス B: Priority 1 - FOCUS-3D + HOCT パイプライン (SOTA 3D Instance Segmentation + Edge Transformer)
1. **FOCUS-3D (bioRxiv 2026.08, yu-lab-vt/FOCUS-3D)**:
   - 3次元蛍光顕微鏡画像に特化した最先端の Volumetric Instance Segmentation 基盤モデル。
   - 5.1TBの自己教師学習、2000万個のシルバーラベル、46万個のエキスパートアノテーションで訓練済み。
   - ゼブラフィッシュ胚を含む多種生物種・複数顕微鏡モダリティに高い汎化性能。
   - 点ではなく「3D領域マスク (Instance Mask)」を直接出力するため、細胞の体積・重心・主軸・輝度プロファイルが完全に得られる。
2. **HOCT (Higher-Order Cell Tracking Transformer, arXiv:2607.11754, royerlab/hoct)**:
   - Loïc Royer Lab (本コンペ主催者研究室) が 2026年7月に発表した次世代セル追跡アーキテクチャ。
   - **Edge-Centric Formulation**: 従来のノード埋め込みではなく、「エッジ(候補リンク)同士のアテンション」により、細胞分裂時の枝分かれ競合を直接解く。
   - **3D Geometric Prior**: 3次元幾何学的事前分布をアテンションバイアスとして注入。
   - Cell Tracking Challenge および細菌分裂ベンチマークで SOTA を達成。
   - 深い画像エンコーダを必要とせず、推論・微調整が高速。

### エビデンス C: Priority 2 - 199胚による UNet / Linker の再学習・ファインチューニング
- 提供重み (`edge_predictor_best.pth` 等) は少数のエポックで学習されたチェックポイントであり、収束していない可能性が高い。
- 全199胚の Zarr + GEFF データを活用し、以下の2段階ファインチューニングを行うエビデンス:
  1. `TemporalUNet3D`: 199胚の細胞中心ヒートマップで損失 (Focal Loss / Dice Loss) を追加計算し、検出 Recall/Precision を向上。
  2. `SimpleNodeTransformer`: 199胚の正解エッジ (GT Edges) を教師データとして、クロスエントロピー損失でエッジスコアを再学習。

### エビデンス D: Priority 3 - EMA 速度予測型 Gap Closing (Velocity-Projected Relinking)
- 静的なユークリッド距離による Gap Closing は FP を急増させ、028 で有害と判定された。
- しかし、Top層の Discussion では **「指数移動平均 (EMA) 3D速度ベクトルによる前方外挿」** が有効と報告されている。
  - $v_t = \alpha v_{t-1} + (1-\alpha) (x_t - x_{t-1})$
  - 消失フレーム $t+\Delta t$ における予測位置: $\hat{x}_{t+\Delta t} = x_t + v_t \times \Delta t$
  - 予測位置と候補ノード間の距離ゲートを適用することで、誤結合 (FP) を抑制しつつ高速移動細胞の分断トラックを正確に縫合可能。

### エビデンス E: Priority 4 - 多重スケール DoG + Hungarian アルゴリズム (ルールベースの底力)
- 機械学習モデルの過学習・ドメインシフトに対する強力な防壁。
- 3次元差分ガウシアン (DoG) を複数のスケール ($\sigma_1, \sigma_2$) で適用し、極大値を統合。
- フレーム間マッチングを `scipy.optimize.linear_sum_assignment` (Hungarian アルゴリズム) で解き、大域的距離最小化を行う。
- コンペ初期〜中期において、過学習に苦しむ深層モデルを退けて Gold Zone (0.87〜0.93) を記録した堅牢な代替解。
