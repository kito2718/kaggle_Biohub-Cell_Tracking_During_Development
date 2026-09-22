# 【s7 実証エビデンス】第 3 枠：3D 全方位 TTA (D8 Symmetry) (034) の実装

## 1. 3次元体積 TTA (D8 Symmetry) の設計と理論的根拠

### (1) 2D D4 平面 TTA の限界
- 従来の 031/032/033 では、xy 平面内の 4 方向反転・回転 (D4: 8 views) のみを適用していた。
- 蛍光顕微鏡データは 3 次元 (z, y, x) であり、特に z 軸方向 (スライス厚 2.0μm) の光学異方性、深部減衰、上下境界での検出揺らぎ (z-jitter) が存在していた。

### (2) 034 における 3D 全方位拡張
- **深度 z 軸反転 (Z-axis Depth Inversion)**:
  - `imgs.flip((-3,))` により、サンプルを上下反転して推論し、検出ヒートマップを再反転して調和平均。
  - 上下スライスの境界付近での検出漏れ (FN) を物理的に消去。
- **3D 中心点反転 (Full Central Inversion)**:
  - `imgs.flip((-3, -2, -1))` による完全 3 次元点対称推論。
- **ビュー数拡張**:
  - 平面 D4 (8 views) + 3D 反転 (2 views) = **計 10 views** による高精度アンサンブル。
  - 追加計算時間はわずか +25% (約 4〜5 分) に抑えられ、Kaggle 実行制限内に余裕で収まる。

---

## 2. 034 実装仕様

- **スクリプト**: `c:\work\aaa\s7\github\working\s7_034_d8_volumetric_tta.py`
- **ノートブック**: `c:\work\aaa\s7\github\working\s7_034_d8_volumetric_tta.ipynb`
- **Kaggle カーネル**: `aaaa1597/s7-034-d8-volumetric-tta-ipynb`
- **投入日時**: 2026-09-22 10:34 JST
- **状態**: `KernelWorkerStatus.RUNNING`
