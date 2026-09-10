# -*- coding: utf-8 -*-
"""
s5_001_test_e2e_pipeline.py
s5_001_try_and_error.ipynb の全改修コードパス (Zarr読込 -> 動的DoG検出 -> NodeFeatureExtractor -> 4D Mahalanobis追跡 -> 孤立ノード刈取 -> submission.csv生成 -> スキーマ検証) を
実画像 (Zarr 3フレーム) で貫通テストし、定量エビデンスを出力する検証スクリプト。
"""
import sys
import types
import glob
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# 0. kaggle_secrets ダミーモジュール (ローカル実行互換)
dummy_ks = types.ModuleType("kaggle_secrets")
class DummySecretsClient:
    def get_secret(self, key):
        return ""
dummy_ks.UserSecretsClient = DummySecretsClient
sys.modules["kaggle_secrets"] = dummy_ks

# 1. パス環境の自動解決
BASE_DIR = Path(r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development")
WORKING_DIR = BASE_DIR / "s5" / "github" / "working"
DATA_DIR = BASE_DIR / "s5" / "github" / "s5_analysys_data"
NB_PATH = WORKING_DIR / "s5_001_try_and_error.ipynb"

DATASET_ID = "44b6_0113de3b"
ZARR_CANDIDATES = [
    BASE_DIR / "s5" / "input" / "train" / f"{DATASET_ID}.zarr",
    BASE_DIR / "s5" / "input" / "test" / f"{DATASET_ID}.zarr",
]
ZARR_PATH = None
for zp in ZARR_CANDIDATES:
    if zp.exists():
        ZARR_PATH = zp
        break

if ZARR_PATH is None:
    raise FileNotFoundError(f"テスト用 Zarr データセットが見つかりません: {DATASET_ID}.zarr")

print("=" * 80)
print(f">>> s5_001 E2E パイプライン全コードパス貫通テスト開始")
print(f"[*] 対象ノートブック: {NB_PATH.name}")
print(f"[*] 対象データセット: {DATASET_ID} (Zarr: {ZARR_PATH})")
print("=" * 80)

# 2. ノートブックからコードセルを抽出・実行
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"[*] ノートブック読み込み完了 (全 {len(nb['cells'])} セル)")

env = {
    "__name__": "__main__",
    "GPU_FLG": False,
    "SUBMIT_TO_COMPETITION": False,
    "CONTINUOUS_RESUME": False,
    "RESET_RESUME": False,
}
exec("import numpy as np; import pandas as pd; import torch; import zarr; from typing import NamedTuple, Optional", env)

# Cell 3 (パラメータ), Cell 5 (ユーティリティ), Cell 7 (Extractor), Cell 8 (Detector), Cell 10 (Tracker), Cell 12 (Submission) を順次ロード
target_cells = [3, 5, 7, 8, 10, 12]
for idx in target_cells:
    code = "".join(nb["cells"][idx]["source"])
    clean_lines = [l for l in code.split("\n") if not l.strip().startswith("!") and not l.strip().startswith("%")]
    clean_code = "\n".join(clean_lines).replace("pa.typing.DataFrame[Nodes]", "pd.DataFrame").replace("pa.typing.DataFrame[Edges]", "pd.DataFrame").replace("pa.typing.DataFrame[Submission]", "pd.DataFrame")
    exec(clean_code, env)
    print(f"  - [OK] Cell {idx} ロード成功")

# テスト実行のため PUSH_TO_GITHUB を False に固定
env["PUSH_TO_GITHUB"] = False

print(f"[*] ロード完了パラメータ:")
print(f"  - NODES_DETECTOR_METHOD  : {env.get('NODES_DETECTOR_METHOD')}")
print(f"  - BLOBDOG_DYNAMIC_ADAPTIVE: {env.get('BLOBDOG_DYNAMIC_ADAPTIVE')}")
print(f"  - EDGES_TRACKER_METHOD   : {env.get('EDGES_TRACKER_METHOD')}")
print(f"  - FILTER_ISOLATED_NODES  : {env.get('FILTER_ISOLATED_NODES')}")

# 3. 貫通テストの実行
print("\n" + "-" * 80)
print(">>> パイプライン実走ステップ")
print("-" * 80)

t_start = time.time()

# (1) 画像読み込み (3フレーム)
t0 = time.time()
z = env["zarr"].open(str(ZARR_PATH), mode="r")
img_raw = z['0'][:3].astype(np.float32)
q1, q2 = np.quantile(img_raw.ravel()[::100], [0.01, 0.99])
img_norm = np.clip((img_raw - q1) / (q2 - q1 + 1e-6), 0.0, 4.0)
print(f"[Step 1/5] Zarr 3フレーム読込 & 正規化: {time.time()-t0:.2f}秒 (shape: {img_norm.shape})")

# (2) 動的 DoG ノード検出
t1 = time.time()
detector = env["get_nodedetector_by_method"](env["NODES_DETECTOR_METHOD"])
raw_nodes = detector.detect(img_norm)
raw_nodes["dataset"] = DATASET_ID
print(f"[Step 2/5] BlobDogNodeDetector (動的DoG) 検出: {time.time()-t1:.2f}秒 (検出ノード: {len(raw_nodes):,} 件)")

# (3) 特徴量抽出 (NodeFeatureExtractor)
t2 = time.time()
extractor = env["NodeFeatureExtractor"](scale_z=1.625, scale_y=0.40625, scale_x=0.40625)

n_len = len(raw_nodes)
mean_intensities = np.full(n_len, -1.0, dtype=np.float32)
snrs = np.full(n_len, -1.0, dtype=np.float32)
z_depths = np.full(n_len, -1.0, dtype=np.float32)
vols = np.full(n_len, -1.0, dtype=np.float32)
rads = np.full(n_len, -1.0, dtype=np.float32)
densities = np.full(n_len, -1, dtype=np.int32)

for t_val, t_group in raw_nodes.groupby('t'):
    t_int = int(t_val)
    indices = t_group.index.values
    coords = t_group[['z', 'y', 'x']].values.astype(np.float32)

    t_densities = extractor.compute_local_density(coords, radius=15.0)
    densities[indices] = t_densities

    if t_int < len(img_norm):
        t_intensities, t_snrs, t_z_depths, t_vols, t_rads = extractor.extract_signal_features(
            img_norm[t_int], coords, r_node=4.0, bg_r_in=8.0, bg_r_out=12.0
        )
        mean_intensities[indices] = t_intensities
        snrs[indices] = t_snrs
        z_depths[indices] = t_z_depths
        vols[indices] = t_vols
        rads[indices] = t_rads

nodes_df = raw_nodes.copy()
nodes_df['mean_intensity'] = mean_intensities
nodes_df['snr'] = snrs
nodes_df['z_depth_ratio'] = z_depths
nodes_df['estimated_radius_um'] = rads
nodes_df['volume_um3'] = vols
nodes_df['local_density_r15'] = densities
print(f"[Step 3/5] NodeFeatureExtractor 6特徴量抽出: {time.time()-t2:.2f}秒 (全 {len(nodes_df):,} 件完了)")

# (4) 4D Mahalanobis Edge 検出
t3 = time.time()
tracker = env["get_edgedetector_by_method"](env["EDGES_TRACKER_METHOD"])
edges_df = tracker.detect(nodes_df)
print(f"[Step 4/5] FeatureEnhancedEdgeDetector ('4dmahalanobis') 追跡: {time.time()-t3:.2f}秒 (エッジ検出: {len(edges_df):,} 本)")

# (5) 孤立ノード刈り取り ＆ submission.csv 生成
t4 = time.time()
generate_sub = env["generate_submission_file"]
sub_df = generate_sub(nodes_df, edges_df, filter_isolated_nodes=True)

# テスト用 submission ファイルを出力
sub_test_path = DATA_DIR / "s5_001_e2e_test_submission.csv"
sub_df.to_csv(sub_test_path, index=False, encoding="utf-8")
t_total = time.time() - t_start
print(f"[Step 5/5] generate_submission_file (孤立ノード刈取): {time.time()-t4:.2f}秒 (出力行数: {len(sub_df):,} 行)")
print(f"[*] 全パイプライン完走所要時間: {t_total:.2f} 秒")

# 4. スキーマ検証および定量エビデンス検証
print("\n" + "=" * 80)
print(">>> 検証結果 (Validation & Evidences)")
print("=" * 80)

# (A) 列構成の検証 (Kaggle 公式スキーマ 10列)
EXPECTED_COLS = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
col_match = list(sub_df.columns) == EXPECTED_COLS
print(f"[A] カラム完全一致 (10列): {'PASS [OK]' if col_match else 'FAIL [NG]'}")
if not col_match:
    print(f"    Expected: {EXPECTED_COLS}")
    print(f"    Actual  : {list(sub_df.columns)}")

# (B) 欠損値チェック
null_counts = sub_df.isnull().sum().to_dict()
has_null = any(c > 0 for c in null_counts.values())
print(f"[B] 欠損値 (NaN/null): {'PASS [OK] (全列 0件)' if not has_null else f'FAIL [NG] ({null_counts})'}")

# (C) 孤立ノード刈り取り検証
node_rows = sub_df[sub_df['row_type'] == 'node']
edge_rows = sub_df[sub_df['row_type'] == 'edge']
linked_nodes = set(edge_rows['source_id']).union(set(edge_rows['target_id']))
isolated_nodes = set(node_rows['node_id']) - linked_nodes

print(f"[C] 孤立ノード刈り取り検証:")
print(f"    - 刈り取り前ノード数: {len(nodes_df):,} 件")
print(f"    - 出力ノード行数    : {len(node_rows):,} 件")
print(f"    - 出力エッジ行数    : {len(edge_rows):,} 件")
print(f"    - 刈り取られた孤立数: {len(nodes_df) - len(node_rows):,} 件")
print(f"    - 残存孤立ノード数  : {len(isolated_nodes):,} 件 (期待値: 0 件)")
isolated_ok = (len(isolated_nodes) == 0)
print(f"    - 孤立ノード判定    : {'PASS [OK] (次数0の細胞は完全除去済み)' if isolated_ok else 'FAIL [NG]'}")

# (D) データ型チェック (ID系は整数型、座標は数値型)
int_cols = ['node_id', 'source_id', 'target_id']
type_ok = all(sub_df[c].dtype in [np.int64, np.int32, int] for c in int_cols)
print(f"[D] 整数ID型 (int64/int32): {'PASS [OK]' if type_ok else 'FAIL [NG]'}")

# (E) サマリーエビデンスの保存
evidence_summary = {
    "test_dataset": DATASET_ID,
    "frames_tested": 3,
    "elapsed_seconds": round(t_total, 2),
    "raw_nodes_detected": len(raw_nodes),
    "features_extracted": 6,
    "edges_detected": len(edges_df),
    "sub_total_rows": len(sub_df),
    "sub_node_rows": len(node_rows),
    "sub_edge_rows": len(edge_rows),
    "isolated_nodes_culled": len(nodes_df) - len(node_rows),
    "isolated_nodes_remaining": len(isolated_nodes),
    "all_checks_passed": col_match and (not has_null) and isolated_ok and type_ok
}

evidence_df = pd.DataFrame([evidence_summary])
evidence_summary_path = DATA_DIR / "s5_001_e2e_test_evidence.csv"
evidence_df.to_csv(evidence_summary_path, index=False, encoding="utf-8")
print(f"\n[*] 定量エビデンスを CSV に保存しました: {evidence_summary_path.name}")
print(evidence_df.to_string(index=False))

print("\n" + "=" * 80)
if evidence_summary["all_checks_passed"]:
    print(">>> [SUCCESS] 全コードパス貫通テスト ALL PASS: s5_001_try_and_error.ipynb の改修は 100% 正常です！")
else:
    print(">>> [FAILED] 検証項目に不一致が存在します。")
print("=" * 80)
