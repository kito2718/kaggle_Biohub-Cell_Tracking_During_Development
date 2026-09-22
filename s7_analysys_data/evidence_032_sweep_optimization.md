# 【s7 実証エビデンス】第 1 枠：ポスプロ・スイープ最適化版 (032) の実装

## 1. 最適化方針とパラメータ選定エビデンス

Kaggle 0.947〜0.948+ の検証スイープ (`score_validator_config`) の合意結果に基づき、以下の 6 大ポスプロ・ノブを最適化:

1. **`MOTION_RELINK_TIGHT_UM = 5.0` (5.5 → 5.0μm)**:
   - 運動再結合の許容半径を 5.0μm に引き締め、高密度領域での誤結合 FP を大幅削減。
2. **`MOTION_RELINK_RELAXED_UM = 9.0` (10.0 → 9.0μm)**:
   - 長距離の飛び移り誤結合を遮断 (`relaxed9` 検証候補)。
3. **`GAP_CLOSE_UM = 4.5` (5.0 → 4.5μm)**:
   - 1フレーム消失時の過剰な長距離結合を抑制 (`gap45` 検証候補)。
4. **`GAP2_MAX_STEP_UM = 4.0` (4.4 → 4.0μm)**:
   - 2フレーム消失時のステップ移動距離上限を厳格化 (`gap2step40` 検証候補)。
5. **`GAP_CLOSE_REUSE_UM = 2.8` (3.2 → 2.8μm)**:
   - 既存孤立ノードの再利用距離を安全側に制約 (`reuse28` 検証候補)。
6. **`BIOHUB_DIV_MIN_PROB = 0.52` (0.50 → 0.52)**:
   - DivNet 3D-CNN による分裂承認の閾値を引き上げ、Jaccard ペナルティとなる偽分裂枝を撃墜。

---

## 2. 032 実装構成

- **スクリプト**: `c:\work\aaa\s7\github\working\s7_032_sweep_optimized.py`
- **ノートブック**: `c:\work\aaa\s7\github\working\s7_032_sweep_optimized.ipynb`
- **Kaggle カーネル**: `aaaa1597/s7-032-sweep-optimized-ipynb`
- **投入日時**: 2026-09-22 09:22 JST
- **状態**: `KernelWorkerStatus.RUNNING`
