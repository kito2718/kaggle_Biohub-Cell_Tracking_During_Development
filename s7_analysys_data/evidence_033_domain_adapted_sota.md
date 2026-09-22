# 【s7 実証エビデンス】第 2 枠：ドメイン適応 24 胚 Transformer 融合モデル (033) の実装

## 1. アーキテクチャ設計と融合エビデンス

### (1) 課題の克服
- 030 では、24 胚の正解エッジで学習した「ドメイン適応 Transformer」が最悪難所胚で +0.0490 の飛躍をもたらしたものの、検出器が 10 エポック初期重みだったため Public LB の上限に阻まれていた。
- 031/032 では、50 エポック完全学習 UNet3D により検出基盤が 0.933+ へ跳ね上がったが、エッジ予測器は初期の汎用モデルのままであった。

### (2) 033 における完全融合
- **検出基盤**: `pilkwang/biohub-tracking-support-pack-50ep-v1` (50エポック UNet3D) を 100% 保持し、未検出核や縮退核を解消。
- **エッジ予測ヘッド**: `aaaa1597/biohub-domain-adapted-weights` (24 胚ドメイン適応済みの 62 重み) を動的注入 (`strict=False`)。
- **副モデル結合**: `pilkwang/biohub-temporal-unet3d-seed314159-v1` との Dual-Seed 調和確率結合を維持。
- **ポスプロ**: 032 で特定したスイープ黄金比率 (`tight50`, `gap45`, `relaxed9`, `gap2step40`, `reuse28`, `div052`) を継承。

---

## 2. 033 実装仕様

- **スクリプト**: `c:\work\aaa\s7\github\working\s7_033_domain_adapted_sota.py`
- **ノートブック**: `c:\work\aaa\s7\github\working\s7_033_domain_adapted_sota.ipynb`
- **Kaggle カーネル**: `aaaa1597/s7-033-domain-adapted-sota-ipynb`
- **投入日時**: 2026-09-22 10:23 JST
- **状態**: `KernelWorkerStatus.RUNNING`
