# -*- coding: utf-8 -*-
"""
s5_001_test_notebook_step1.py
s5_001_try_and_error.ipynb から get_edgedetector_by_method('4dmahalanobis') をロードして
代表 5 データセットで実走テストし、定量エビデンスを CSV に保存するスクリプト。
"""
import glob
import json
import os
import sys
import time
import numpy as np
import pandas as pd

# 1. パス設定
base_dir = r"d:\BizOwn\000_Biw2\51_googleantigravity\007_kaggle_Biohub-Cell_Tracking_During_Development\s5\github"
working_dir = os.path.join(base_dir, "working")
data_dir = os.path.join(base_dir, "s5_analysys_data")
nb_path = os.path.join(working_dir, "s5_001_try_and_error.ipynb")

evidence_csv_path = os.path.join(data_dir, "s5_001_step1_test_evidence_5datasets.csv")

TARGET_5_DATASETS = [
    "44b6_0113de3b",  # 密・良好
    "44b6_0b24845f",  # 密・難関(超低コントラスト)
    "44b6_74d0c52e",  # 密・組織深部
    "6bba_05b6850b",  # 疎・標準
    "6bba_085bf656",  # 疎・高コントラスト
]

print("=" * 80)
print(">>> s5_001: Step 1 get_edgedetector_by_method('4dmahalanobis') テスト開始")
print("=" * 80)

# 2. ノートブックから detect_edges と generate_submission を動的探索してロード
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

edges_code = ""
sub_code = ""

for cell in nb["cells"]:
    code = "".join(cell["source"])
    if "def get_edgedetector_by_method" in code or "def detect_edges" in code:
        edges_code = code
    elif "def generate_submission_file" in code:
        sub_code = code

if not edges_code or not sub_code:
    raise RuntimeError("ノートブック内から detect_edges または generate_submission_file が見つかりませんでした。")

env = {
    "BaseEdgeDetector": object,
    "EdgeItemDict": dict,
    "EDGES_TRACKER_METHOD": "4dmahalanobis",
    "PUSH_TO_GITHUB": False,
    "pa": None
}
exec("import numpy as np; import pandas as pd", env)

class DummyPA:
    class typing:
        DataFrame = dict
DummyPA.DataFrameModel = object
env["pa"] = DummyPA

exec(edges_code.replace("pa.typing.DataFrame[Nodes]", "pd.DataFrame").replace("pa.typing.DataFrame[Edges]", "pd.DataFrame"), env)
exec(sub_code.replace("pa.typing.DataFrame[Nodes]", "pd.DataFrame").replace("pa.typing.DataFrame[Edges]", "pd.DataFrame").replace("pa.typing.DataFrame[Submission]", "pd.DataFrame"), env)

get_edge_factory = env["get_edgedetector_by_method"]
generate_submission_file = env["generate_submission_file"]

tracker = get_edge_factory("4dmahalanobis")
print(f"[*] get_edgedetector_by_method('4dmahalanobis') 取得成功: {type(tracker).__name__}")
print(f"  - 特徴量: {tracker.feature_cols}")
print(f"  - 重み: feature_weight={tracker.feature_weight}, spatial_weight={tracker.spatial_weight}")

# 3. 検出ノードのロード
node_files = sorted(glob.glob(os.path.join(working_dir, "s3_01_detect_nodes_pred_blobdog_5dmahalanobis_*.csv")))
all_nodes_list = []
for f in node_files:
    df_chunk = pd.read_csv(f)
    sub_df = df_chunk[df_chunk['dataset'].isin(TARGET_5_DATASETS)]
    if not sub_df.empty:
        all_nodes_list.append(sub_df)

nodes_df_all = pd.concat(all_nodes_list, ignore_index=True)
print(f"[*] 代表 5 データセット 検出ノード数: {len(nodes_df_all):,} 件")

gt_summary_path = os.path.join(working_dir, "s3_gt_summary.csv")
gt_summary = pd.read_csv(gt_summary_path).set_index("dataset") if os.path.exists(gt_summary_path) else pd.DataFrame()

# 4. 代表 5 データセットでの実走テスト
evidence_records = []
print("\n[*] 5 データセットトラッキング ＆ 刈取テスト開始:")

OFFICIAL_COLS = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']

for ds in TARGET_5_DATASETS:
    t0 = time.time()
    ds_nodes = nodes_df_all[nodes_df_all['dataset'] == ds].copy()
    
    # トラッカーでエッジ検出
    edges_df = tracker.detect(ds_nodes, scale=(1.625, 0.40625, 0.40625))
    edges_df['dataset'] = ds
    
    # 刈取なし
    sub_raw = generate_submission_file(ds_nodes, edges_df, filter_isolated_nodes=False)
    n_raw = len(sub_raw[sub_raw['row_type'] == 'node'])
    
    # 刈取あり (本番設定)
    sub_filtered = generate_submission_file(ds_nodes, edges_df, filter_isolated_nodes=True)
    n_filtered = len(sub_filtered[sub_filtered['row_type'] == 'node'])
    
    n_removed = n_raw - n_filtered
    removal_pct = (n_removed / n_raw * 100.0) if n_raw > 0 else 0.0
    
    est_nodes = gt_summary.loc[ds, 'estimated_number_of_nodes'] if ds in gt_summary.index else np.nan
    pe_raw = (n_raw / est_nodes) if pd.notna(est_nodes) and est_nodes > 0 else np.nan
    pe_filtered = (n_filtered / est_nodes) if pd.notna(est_nodes) and est_nodes > 0 else np.nan
    
    # フォーマット検証
    is_cols_ok = list(sub_filtered.columns) == OFFICIAL_COLS
    is_null_ok = sub_filtered.isnull().sum().sum() == 0
    is_type_ok = all(sub_filtered[c].dtype == np.int64 for c in ['id', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id'])
    val_status = "PASS" if (is_cols_ok and is_null_ok and is_type_ok) else "FAIL"
    
    elapsed = time.time() - t0
    
    evidence_records.append({
        "dataset": ds,
        "est_nodes": int(est_nodes) if pd.notna(est_nodes) else -1,
        "nodes_raw": n_raw,
        "nodes_filtered": n_filtered,
        "nodes_removed": n_removed,
        "removal_pct": round(removal_pct, 2),
        "edges_count": len(edges_df),
        "pe_raw": round(pe_raw, 4) if pd.notna(pe_raw) else -1,
        "pe_filtered": round(pe_filtered, 4) if pd.notna(pe_filtered) else -1,
        "format_validation": val_status,
        "time_sec": round(elapsed, 2)
    })
    
    print(f"  [{ds}] raw: {n_raw:,} -> filtered: {n_filtered:,} (-{removal_pct:.1f}%) | PE: {pe_raw:.3f} -> {pe_filtered:.3f} | Edges: {len(edges_df):,} | Format: {val_status} ({elapsed:.2f}s)")

# 5. エビデンス CSV の保存
df_evidence = pd.DataFrame(evidence_records)
df_evidence.to_csv(evidence_csv_path, index=False, encoding="utf-8")
print(f"\n[+] エビデンス CSV を正常保存しました: {evidence_csv_path}")

print("=" * 80)
print("[*] 判定結果:")
print(f"  - 1. get_edgedetector_by_method('4dmahalanobis') 検証: PASS")
print(f"  - 2. 代表 5 データセット 総エッジ数: {df_evidence['edges_count'].sum():,} 本")
print(f"  - 3. 孤立ノード刈取によるP/E比圧縮: 平均 {df_evidence['pe_raw'].mean():.3f} -> {df_evidence['pe_filtered'].mean():.3f}")
all_pass = all(r["format_validation"] == "PASS" for r in evidence_records)
print(f"  - 4. Kaggle 提出フォーマット(全10列, NaNゼロ, int64): {'ALL PASS (100% 適合)' if all_pass else 'FAIL'}")
print(f"  => 総合: {'【 PASS: Step 1 DoD 全項目クリア 】' if all_pass else '【 FAIL 】'}")
print("=" * 80)
