# S1 (基底ジャンプ) ローカル実行環境 (s1_local_env)

作業ディレクトリのルート:
`d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\`

```
├── s1_local_env/                                <- ★ S1(基底ジャンプ)専用のPython仮想環境および実験空間
│   ├── Scripts/python.exe                       <- VS Code / Jupyterで指定するPythonインタープリタ
│   ├── data/                                    <- ダウンロード済みの全データ（81.4 GB）
│   │    ├── train/
│   │    └── test/
│   ├── src/                                     <- S1コアモジュール群
│   │    ├── detect/                             <- 3D細胞検出 (StarDist 3D / Cellpose 3D)
│   │    │    └── stardist_detector.py
│   │    ├── track/                              <- 3D時空間追跡 (btrack / ILP大域最適化)
│   │    │    └── btrack_tracker.py
│   │    ├── utils/                              <- 前処理・後処理 (Z軸異方性補正, ノイズフィルタ)
│   │    │    ├── anisotropy.py
│   │    │    └── post_filtering.py
│   │    └── kaggle_cell_tracking_competition/   <- 主催者の公式評価コード (CV測定用)
│   │
│   └── notebooks/
│       ├── s1_01_stardist_btrack_pipeline.ipynb <- ★【メイン】S1統合パイプライン & Full CV計測
│       └── s1_02_tuning_and_ablation.ipynb      <- パラメータチューニング・実験用
```

## 実験目的
- **S1 (基底ジャンプ)**: 3D StarDist による高密度細胞ノード検出と、btrack (ベイズ推論 + ILP) による大域移動・細胞分裂 (Division) トラッキングを統合。
- **目標スコア**: CV 0.4578 ➔ **0.90+**
