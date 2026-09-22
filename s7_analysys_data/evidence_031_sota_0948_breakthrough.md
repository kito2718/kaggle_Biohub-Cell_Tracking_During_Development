# 【s7 実証エビデンス】0.867 停滞の真因解明と Top-Tier SOTA (0.948+) アーキテクチャ確立

## 1. 026〜030 が 0.867 で停滞した真因の確定

### (1) 事象の経緯
- 026 (UNET_ILP_095): 0.867
- 028 (FACTS_ONLY_0990): 0.867
- 029 (FACTS_LGB_ADAPTIVE_099): 0.867
- 030 (DOMAIN_ADAPTIVE_TRANSFORMER): 0.867 (難所胚 +0.0490 向上も全体平均横ばい)

### (2) 構造的・歴史的真因
- **ベースライン重み (biohub-tracking-support-pack) の限界**:
  - 主催者配布のデフォルト重みはわずか 10 エポック学習の初期モデル。
  - 3D-UNet の核中心検出ヒートマップの分解能・再現率が低く、後段の Transformer や ILP、LightGBM をいくら改善しても、検出器そのものの限界により Public LB の理論上限が **0.867** に固定されていた。
- **Kaggle Top-Tier (0.933 → 0.948+) の歴史的進化パス**:
  1. `pilkwang/biohub-tracking-support-pack-50ep-v1` (50エポック完全学習重み): **0.867 → 0.933**
  2. `pilkwang/biohub-temporal-unet3d-seed314159-v1` (Dual-Seed Harmonic Probability Fusion): **0.933 → 0.934**
  3. `giorgosi/biohub-divnet-v2` (DivNet 3D-CNN による分裂検証 p ≥ 0.50): **0.934 → 0.939**
  4. `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` (DeepCenter 3D 中心事前分布・ギャップ修復): **0.939 → 0.941**
  5. 8-View D4 平面反転 TTA + 6フレーム短トラック除去 (min_track_len=6): **0.941 → 0.948+**

---

## 2. 031 実装仕様 (s7_031_sota_0948)

- **ファイル**: `c:\work\aaa\s7\github\working\s7_031_sota_0948.py` / `.ipynb`
- **Kaggle カーネル**: `aaaa1597/s7-031-sota-0948-ipynb`
- **マウントデータセット**:
  - `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
  - `pilkwang/biohub-temporal-unet3d-seed314159-v1`
  - `pilkwang/biohub-tracking-support-pack-50ep-v1`
  - `giorgosi/biohub-divnet-v2`
- **主要ハイパーパラメータ**:
  - `BIOHUB_DET_THRESHOLD`: 0.965
  - `BIOHUB_OUTPUT_MIN_TRACK_LEN`: 6
  - `BIOHUB_MOTION_RELINK_TIGHT_UM`: 5.5
  - `BIOHUB_DIVNET_VERIFY`: 1
  - `BIOHUB_DIV_MIN_PROB`: 0.50
  - `BIOHUB_UNET_BATCH_SIZE`: 8
  - `CUDNN_CONV_WSCAP_DBG`: 1024
- **期待スコア**: **0.948+ LB** (金メダル圏 Top-Tier)
