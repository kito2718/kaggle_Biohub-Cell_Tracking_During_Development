# S1 (土台の再構築) ローカル実行環境 (s1_local_env)

作業ディレクトリのルート:
`d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\`

```
├── s1_local_env/                                <- ★ S1(土台の再構築)専用のPython仮想環境および実験空間
│   ├── Scripts/python.exe                       <- VS Code / Jupyterで指定するPythonインタープリタ
│   ├── data/                                    <- ダウンロード済みの全データ（81.4 GB）
│   │    ├── train/
│   │    └── test/
│   ├── src/                                     <- S1コアモジュール群
│   │    ├── detect/                             <- 3D細胞検出 (StarDist 3D / Cellpose 3D)
│   │    │    └── stardist_detector.py
│   │    ├── track/                              <- 3D時空間追跡 (btrack / ILP大域最適化)
│   │    │    └── btrack_tracker.py
│   │    ├── utils/                              <- ★ 前処理・後処理・可視化ユーティリティ
│   │    │    ├── visualize.py                   <- 3D可視化 & エラー解析 (GT/予測重ね合わせ) [Priority 1]
│   │    │    ├── anisotropy.py                  <- Z軸異方性 (Anisotropy) スケール補正 [Priority 1]
│   │    │    ├── contrast.py                    <- パーセンタイル正規化 [Priority 1] & 3D Top-Hat [Priority 3]
│   │    │    └── post_filtering.py             <- 1フレーム抜けの幾何補間 & FPエッジカット [Priority 3]
│   │    └── kaggle_cell_tracking_competition/   <- 主催者の公式評価コード (CV測定用)
│   │
│   └── notebooks/
│       ├── s1_01_stardist_btrack_pipeline.ipynb <- ★【メイン】S1統合パイプライン & Full CV計測
│       └── s1_02_tuning_and_ablation.ipynb      <- アブレーション実験・チューニング用
```

## 🗺️ 開発アプローチ（Priority 1 〜 Priority 5）
- **【Priority 1: 最優先開発基盤】**: 3D可視化・エラー解析 (`visualize.py`) ＋ ランクS基本前処理 (`contrast.py`, `anisotropy.py`)
- **【Priority 2: S1 土台の再構築 (CV 0.45 ➔ 0.90+)】**: 3D StarDist 検出器 ＋ btrack (ベイズ推論 + ILP) 大域トラッキング ＋ 基本特徴量
- **【Priority 3: S2 精密後処理 & 構造強化 (CV 0.90 ➔ 0.96+)】**: 1フレーム抜けの幾何線形補間 ＋ 非合理エッジカット ＋ Top-Hat アブレーション
- **【Priority 4: 機械的スコア削り出し】**: 3D TTA ＋ アンサンブル (WBF 3D) ＋ Optuna ハイパーパラメータ自動最適化
- **【Priority 5: S3 頂点突破 & 本番提出 (CV 0.96 ➔ 0.982+)】**: Dice/IoU Loss ＋ 擬似ラベル (Self-Training) ＋ 3Dパッチ並列化推論

## 実験目的
- **S1 (土台の再構築)**: 3D StarDist による高密度細胞ノード検出と、btrack (ベイズ推論 + ILP) による大域移動・細胞分裂 (Division) トラッキングを統合。
- **目標スコア**: CV 0.4578 ➔ **0.90+**
