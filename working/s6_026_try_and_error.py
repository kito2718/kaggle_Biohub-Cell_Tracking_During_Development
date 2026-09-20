"""
s6_026_try_and_error.py
Biohub - Cell Tracking s6_026: 3D-UNet + Transformer + ILP 深層学習パイプライン
GT検証モード (GT_FLG) / 本番提出モード (SUBMIT_TO_COMPETITION) 両対応
GPU_FLG による GPU/CPU 切り替え
全約100種特徴量抽出 (ノード・エッジ・トラック・フレーム) & EDA成果物出力
"""

import os
import sys
import time
import datetime
import math
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from scipy.spatial import KDTree

# =============================================================================
# Cell 3: パラメータ・グローバル変数定義
# =============================================================================

MAGIC_STRING: str = "027-UNET_ILP_095_ALL199"
RUN_PREFIX: str = f"s6_{MAGIC_STRING}_"

# 1. 実行モード設定
GT_FLG: bool = True                       # True: GT検証モード (trainデータセットを対象に推論・評価・EDA)
SUBMIT_TO_COMPETITION: bool = not GT_FLG  # False: GT検証時 (SUBMIT時は True に連動)
GPU_FLG: bool = True                     # True: GPU使用 (CUDA利用可能時), False: 強制CPU

# 2. 対象データセット絞り込み (空リスト [] の場合は見つかった全データセットを実行)
TARGET_DATASETS: list[str] = []

# 3. 入力データディレクトリ定義 (WSL2, Linux/Kaggle, Windows相対パスを順次自動判定)
def resolve_data_dir() -> Path:
    if SUBMIT_TO_COMPETITION:
        candidates = [
            Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test"),
            Path("/kaggle/input/biohub-cell-tracking-during-development/test"),
            Path("/mnt/c/work/aaa/s6/input/test"),
            Path("s6/input/test"),
            Path("input/test"),
        ]
    else:
        candidates = [
            Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
            Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
            Path("/mnt/c/work/aaa/s6/input/train"),
            Path("s6/input/train"),
            Path("input/train"),
        ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]

DATA_DIR: Path = resolve_data_dir()

# 4. サポートパックディレクトリ定義
def resolve_support_pack_dir() -> Path:
    candidates = [
        Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
        Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"),
        Path("/mnt/c/work/aaa/s6/input/support_pack"),
        Path("s6/input/support_pack"),
        Path("input/support_pack"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]

SUPPORT_PACK_DIR: Path = resolve_support_pack_dir()
WHEELS_DIR: Path = SUPPORT_PACK_DIR / "wheels"
WEIGHTS_PATH: Path = SUPPORT_PACK_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
REPO_SRC_DIR: Path = SUPPORT_PACK_DIR / "repo" / "src"
REPO_SCRIPTS_DIR: Path = SUPPORT_PACK_DIR / "repo" / "scripts"

# サポートパックのモジュールパスを先行登録
for p in [REPO_SRC_DIR, REPO_SCRIPTS_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 5. 深層学習推論 & ILP ハイパーパラメータ
DET_THRESHOLD: float = 0.95          # 3D-UNet 細胞中心確率閾値
DET_TTA: bool = True                 # XYフリップ TTA (Test-Time Augmentation)
POOL_KERNEL_UM: float = 3.0          # 3D極大プーリング抑制半径 (μm)
USE_ILP: bool = True                 # ILP大域最適化の有効化
ILP_EDGE_WEIGHT: float = -1.0        # エッジ接続エネルギー重み
ILP_APPEARANCE_WEIGHT: float = 0.1   # 出現ペナルティ
ILP_DISAPPEARANCE_WEIGHT: float = 0.1# 消失ペナルティ
ILP_DIVISION_WEIGHT: float = 1.0     # 分裂イベント許容重み

# 3D異方性物理スケール (Z: 1.625μm, Y: 0.40625μm, X: 0.40625μm)
PHYSICAL_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
MATCH_DISTANCE_THRESHOLD_UM: float = 7.0 # GTマッチング距離閾値 (μm)

# 6. GitHub 送信設定 (SUBMITモード時は自動オフ)
PUSH_TO_GITHUB: bool = not SUBMIT_TO_COMPETITION
GITHUB_REPO: str = "https://github.com/kito2718/kaggle_Biohub-Cell_Tracking_During_Development.git"
BRANCH_NAME: str = "main"

if PUSH_TO_GITHUB:
    try:
        from kaggle_secrets import UserSecretsClient
        GITHUB_TOKEN: str = UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        import os
        GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
else:
    GITHUB_TOKEN: str = ""

# 7. 出力先設定
OUTPUT_DIR: Path = Path("working")
OUTPUT_SUBMISSION_CSV: str = "submission.csv"
OUTPUT_SUMMARY_CSV: str = f"{RUN_PREFIX}pipeline_summary.csv"
OUTPUT_DETAILS_CSV: str = f"{RUN_PREFIX}pipeline_details.csv"
OUTPUT_EDA_METRICS_XLSX: str = f"{RUN_PREFIX}eda_features_metrics.xlsx"
OUTPUT_EDA_METRICS_CSV: str = f"{RUN_PREFIX}eda_features_metrics.csv"
OUTPUT_EDA_NODE_CSV: str = f"{RUN_PREFIX}eda_node_features.csv"
OUTPUT_EDA_EDGE_CSV: str = f"{RUN_PREFIX}eda_edge_features.csv"
OUTPUT_EDA_TRACK_CSV: str = f"{RUN_PREFIX}eda_track_features.csv"
OUTPUT_EDA_FRAME_CSV: str = f"{RUN_PREFIX}eda_frame_summary.csv"

# 8. グローバル実行状態保持変数
DEVICE: any = None
MODEL: any = None
WINDOW_SIZE: int = 2
DOWNSAMPLE: tuple[int, ...] = (1, 4, 4)
DATASET_NAMES: list[str] = []


# =============================================================================
# Cell 4: 共通関数定義 (push_to_github & sync_from_github)
# =============================================================================

def push_to_github(file_name: str, commit_message: str = "Update results") -> bool:
    """成果物ファイルを GitHub リポジトリへ送信する。"""
    if not PUSH_TO_GITHUB or not GITHUB_TOKEN:
        return False
    import subprocess
    import shutil
    import tempfile
    try:
        src_path = OUTPUT_DIR / file_name
        if not src_path.exists():
            return False
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_url = GITHUB_REPO.replace("https://", f"https://oauth2:{GITHUB_TOKEN}@")
            subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH_NAME, repo_url, tmpdir], check=True, capture_output=True)
            dst = Path(tmpdir) / "working" / file_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            subprocess.run(["git", "-C", tmpdir, "config", "user.name", "Kaggle-Notebook"], check=True)
            subprocess.run(["git", "-C", tmpdir, "config", "user.email", "kaggle@example.com"], check=True)
            subprocess.run(["git", "-C", tmpdir, "add", f"working/{file_name}"], check=True)
            res = subprocess.run(["git", "-C", tmpdir, "commit", "-m", commit_message], capture_output=True, text=True)
            if "nothing to commit" in res.stdout or "nothing to commit" in res.stderr:
                return True
            subprocess.run(["git", "-C", tmpdir, "push", "origin", BRANCH_NAME], check=True, capture_output=True)
            print(f"  - [OK] GitHub push 完了: {file_name}")
            return True
    except Exception as e:
        print(f"  - [WARN] GitHub push 失敗 ({file_name}): {e}")
        return False


# =============================================================================
# Cell 5: 実行環境セットアップ (setup_environment)
# =============================================================================

def setup_environment() -> None:
    """Cell 5: デバイス設定 (GPU_FLG連動) およびモデルのロードを行う。"""
    global DEVICE, MODEL, WINDOW_SIZE, DOWNSAMPLE
    import torch

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> Cell 5: setup_environment 開始")

    # 1. デバイス選択 (GPU_FLG 連動)
    if GPU_FLG and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"  - [DEVICE] GPU 有効化: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
    else:
        DEVICE = torch.device("cpu")
        reason = "GPU_FLG=False (意図的CPUモード)" if not GPU_FLG else "CUDA利用不可"
        print(f"  - [DEVICE] CPU モード稼働 ({reason})")

    # 2. モデルのインスタンス化 & チェックポイントロード
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"致命的エラー: モデル重みファイルが存在しません: {WEIGHTS_PATH.resolve()}")

    from predict_unet_transformer import load_model
    MODEL, WINDOW_SIZE, DOWNSAMPLE = load_model(WEIGHTS_PATH, DEVICE)
    print(f"  - [MODEL]  モデルロード完了: {WEIGHTS_PATH.name} -> {DEVICE} (window_size={WINDOW_SIZE}, downsample={DOWNSAMPLE})")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< Cell 5: setup_environment 正常終了")


# =============================================================================
# Cell 6: 実行環境・入力データ検証 (check_environment)
# =============================================================================

def check_environment() -> None:
    """Cell 6: 実行環境、入力データディレクトリ(DATA_DIR)、データセット一覧を一意に検証しグローバル変数に設定。"""
    global DATASET_NAMES, DATA_DIR
    import psutil
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> Cell 6: check_environment 開始")
    print(f"  - [MODE]    SUBMIT_TO_COMPETITION: {SUBMIT_TO_COMPETITION} (GT_FLG: {GT_FLG})")
    print(f"  - [EXEC_ID] MAGIC_STRING: {MAGIC_STRING}")
    print(f"  - [SYSTEM]  CPU: {psutil.cpu_count(logical=True)} cores | RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    DATA_DIR = resolve_data_dir()
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"致命的エラー: 入力データディレクトリが存在しません: {DATA_DIR.resolve()}")

    zarr_files = sorted(DATA_DIR.glob("*.zarr"))
    geff_files = sorted(DATA_DIR.glob("*.geff"))

    if TARGET_DATASETS:
        zarr_files = [p for p in zarr_files if p.stem in TARGET_DATASETS]
        geff_files = [p for p in geff_files if p.stem in TARGET_DATASETS]
        print(f"  - [FILTER]  TARGET_DATASETS 指定により {len(zarr_files)} 件に絞り込み")

    DATASET_NAMES = [p.name for p in zarr_files]
    print(f"  - [DATA]    DATA_DIR: {DATA_DIR.resolve()}")
    print(f"              .zarr: {len(zarr_files)} 件 | .geff: {len(geff_files)} 件")

    if len(zarr_files) == 0:
        raise FileNotFoundError(f"致命的エラー: DATA_DIR 内に対象の *.zarr が見つかりません: {DATA_DIR.resolve()}")

    if GT_FLG and len(geff_files) == 0:
        raise FileNotFoundError(f"致命的エラー: GT検証モードですが DATA_DIR 内に *.geff が見つかりません: {DATA_DIR.resolve()}")

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< Cell 6: check_environment 正常終了")


# =============================================================================
# Cell 7: GTデータ読み込み (load_gt_data)
# =============================================================================

def load_gt_data() -> dict[str, dict]:
    """Cell 7: GT検証モード時にグラウンドトゥルース (.geff) を読み込む。"""
    if not GT_FLG:
        print("[Cell 7] SUBMIT mode: Skipping GT loading.")
        return {}

    print(f"[Cell 7] Loading ground truth tracks for {len(DATASET_NAMES)} datasets...")
    from biohub_tracking.io import open_dataset
    gt_dict = {}

    for name in DATASET_NAMES:
        ds_stem = name.replace(".zarr", "")
        geff_path = DATA_DIR / f"{ds_stem}.geff"
        zarr_path = DATA_DIR / name

        if geff_path.exists():
            zg = zarr.open_group(str(geff_path), mode="r")
            t = zg["nodes"]["props"]["t"]["values"][:]
            z = zg["nodes"]["props"]["z"]["values"][:]
            y = zg["nodes"]["props"]["y"]["values"][:]
            x = zg["nodes"]["props"]["x"]["values"][:]
            if "ids" in zg["nodes"]:
                node_ids = zg["nodes"]["ids"][:]
            elif "id" in zg["nodes"]:
                node_ids = zg["nodes"]["id"][:]
            else:
                node_ids = np.arange(len(t))

            gt_nodes = pd.DataFrame({"node_id": node_ids, "t": t, "z": z, "y": y, "x": x})

            if "ids" in zg["edges"] and zg["edges"]["ids"].shape[0] > 0:
                edges_arr = zg["edges"]["ids"][:]
                edge_s, edge_t = edges_arr[:, 0], edges_arr[:, 1]
            elif "source" in zg["edges"] and "target" in zg["edges"]:
                edge_s, edge_t = zg["edges"]["source"][:], zg["edges"]["target"][:]
            else:
                edge_s, edge_t = [], []

            gt_edges = pd.DataFrame({"source_id": edge_s, "target_id": edge_t})

            meta = zg.attrs.asdict() if hasattr(zg.attrs, "asdict") else dict(zg.attrs)
            geff_meta = meta.get("geff", {}) if isinstance(meta, dict) else {}
            extra = geff_meta.get("extra", {}) if isinstance(geff_meta, dict) else {}
            est_nodes = int(extra.get("estimated_number_of_nodes", meta.get("estimated_number_of_nodes", len(gt_nodes))))

            ds_obj = open_dataset(zarr_path, require_tracks=True, load_image=False)

            gt_dict[ds_stem] = {
                "nodes": gt_nodes,
                "edges": gt_edges,
                "estimated_number_of_nodes": est_nodes,
                "tracks": ds_obj.tracks,
                "geff_path": geff_path,
            }

    print(f"  - Loaded GT for {len(gt_dict)} datasets.")
    return gt_dict


# =============================================================================
# Cell 8: 深層学習推論 — 3D-UNet + Transformer 細胞検出・エッジ推論 (SUBMIT版と100%同一)
# =============================================================================

def detect_nodes_and_edges(
    max_frames: int | None = None,
) -> dict[str, tuple[np.ndarray, list[tuple[int, int, float, float]]]]:
    """Cell 8: 各データセットに対し、3D-UNetによる細胞中心検出とTransformerによるエッジ推論を実行する (SUBMIT版と100%同一)。"""
    from predict_unet_transformer import PredictConfig, predict_video
    print(f"[Cell 8] Running deep learning inference on {len(DATASET_NAMES)} datasets...")
    cfg = PredictConfig(
        det_threshold=DET_THRESHOLD,
        det_tta=DET_TTA,
        pool_kernel_um=POOL_KERNEL_UM,
        use_ilp=USE_ILP,
        ilp_edge_weight=ILP_EDGE_WEIGHT,
        ilp_appearance_weight=ILP_APPEARANCE_WEIGHT,
        ilp_disappearance_weight=ILP_DISAPPEARANCE_WEIGHT,
        ilp_division_weight=ILP_DIVISION_WEIGHT,
    )

    raw_preds = {}
    for idx, ds_name in enumerate(DATASET_NAMES, 1):
        ds_stem = ds_name.replace(".zarr", "")
        ds_path = DATA_DIR / ds_name
        t0 = time.time()
        print(f"  [{idx}/{len(DATASET_NAMES)}] Predicting {ds_stem}...")

        coords, candidate_edges = predict_video(
            model=MODEL,
            ds_path=ds_path,
            device=DEVICE,
            cfg=cfg,
            window_size=WINDOW_SIZE,
            max_frames=max_frames,
            unet_batch_size=1,
            downsample=DOWNSAMPLE,
        )
        elapsed = time.time() - t0
        print(f"    -> Inferred {len(coords)} nodes, {len(candidate_edges)} candidate edges in {elapsed:.2f}s")
        raw_preds[ds_stem] = (coords, candidate_edges)

    return raw_preds


# =============================================================================
# Cell 9: 検出結果チェック ＆ ノード事後採点 (check_nodes)
# =============================================================================

def check_nodes(
    raw_preds: dict[str, tuple[np.ndarray, list]],
    gt_dict: dict[str, dict] | None = None
) -> pd.DataFrame:
    """Cell 9: 検出ノードのサマリ統計を集計し、GTモード時は事後採点 (Recall, Precision, F1) を算出する。"""
    print("[Cell 9] Auditing detection results and node metrics...")
    scale = np.array(PHYSICAL_SCALE, dtype=np.float32)
    records = []

    for ds_stem, (coords, edges) in raw_preds.items():
        num_nodes = len(coords)
        unique_t = len(np.unique(coords[:, 0])) if num_nodes > 0 else 0
        nodes_per_frame = round(num_nodes / max(1, unique_t), 2)

        rec = {
            "dataset": ds_stem,
            "pred_nodes": num_nodes,
            "unique_frames": unique_t,
            "nodes_per_frame": nodes_per_frame,
            "candidate_edges": len(edges),
        }

        # GT検証モード時の事後採点
        if GT_FLG and gt_dict and ds_stem in gt_dict:
            gt_info = gt_dict[ds_stem]
            gt_df = gt_info["nodes"]
            est_nodes = gt_info["estimated_number_of_nodes"]

            pred_df = pd.DataFrame(coords, columns=["t", "z", "y", "x"])
            pred_df["node_id"] = np.arange(len(pred_df))

            tp_count = 0
            frames = sorted(list(set(gt_df["t"].unique()).union(set(pred_df["t"].unique()))))

            for t_val in frames:
                gt_t = gt_df[gt_df["t"] == t_val]
                pred_t = pred_df[pred_df["t"] == t_val]
                if gt_t.empty or pred_t.empty:
                    continue

                gt_pts = gt_t[["z", "y", "x"]].values * scale
                pred_pts = pred_t[["z", "y", "x"]].values * scale

                tree = KDTree(pred_pts)
                dists, idxs = tree.query(gt_pts, distance_upper_bound=MATCH_DISTANCE_THRESHOLD_UM)
                matched_pred = set()
                for d, p_idx in zip(dists, idxs):
                    if d <= MATCH_DISTANCE_THRESHOLD_UM and p_idx not in matched_pred:
                        matched_pred.add(p_idx)
                        tp_count += 1

            fn_count = len(gt_df) - tp_count
            fp_count = len(pred_df) - tp_count
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
            pe_ratio = num_nodes / est_nodes if est_nodes > 0 else 0.0

            rec.update({
                "gt_nodes": len(gt_df),
                "est_nodes": est_nodes,
                "node_tp": tp_count,
                "node_fp": fp_count,
                "node_fn": fn_count,
                "node_recall": round(float(recall), 4),
                "node_precision": round(float(precision), 4),
                "node_f1": round(float(f1), 4),
                "pre_pe_ratio": round(float(pe_ratio), 4),
            })

        records.append(rec)

    summary_df = pd.DataFrame(records)
    print("=" * 80)
    print("【Cell 9: 検出結果 & ノード評価サマリー】")
    print(summary_df.to_string(index=False))
    print("=" * 80)
    return summary_df


# =============================================================================
# Cell 10: グラフ構築 + ILP 大域最適化 (SUBMIT版と100%同一)
# =============================================================================

def build_graph_and_solve_ilp(
    raw_preds: dict[str, tuple[np.ndarray, list]],
) -> dict[str, any]:
    """Cell 10: 候補エッジからグラフを構築し、ILPSolver による大域的最適解を確定する (SUBMIT版と100%同一)。"""
    import tracksdata as td
    from predict_unet_transformer import build_graph, suppress_output
    print("[Cell 10] Building tracksdata graphs and solving ILP optimization...")
    solved_graphs = {}

    for ds_stem, (coords, candidate_edges) in raw_preds.items():
        t0 = time.time()
        graph = build_graph(coords, candidate_edges)
        orig_edges = graph.num_edges()

        if USE_ILP and orig_edges > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=ILP_EDGE_WEIGHT * td.EdgeAttr("edge_prob"),
                appearance_weight=ILP_APPEARANCE_WEIGHT,
                disappearance_weight=ILP_DISAPPEARANCE_WEIGHT,
                division_weight=ILP_DIVISION_WEIGHT,
            )
            with suppress_output():
                graph = solver.solve(graph)
            elapsed = time.time() - t0
            print(f"  - [{ds_stem}] ILP optimized: {orig_edges} -> {graph.num_edges()} edges ({elapsed:.2f}s)")
        else:
            print(f"  - [{ds_stem}] Greedy / pass-through: {graph.num_edges()} edges")

        solved_graphs[ds_stem] = graph

    return solved_graphs


# =============================================================================
# Cell 11: トラッキングチェック ＆ エッジ事後採点 (check_edges)
# =============================================================================

def check_edges(
    solved_graphs: dict[str, any],
    gt_dict: dict[str, dict] | None = None
) -> pd.DataFrame:
    """Cell 11: トラッキング結果を集計し、GTモード時は事後採点 (Edge Recall, Precision, F1, 公式Jaccard) を算出する。"""
    import polars as pl
    import tracksdata as td
    from biohub_tracking.io import save_graph
    from biohub_tracking.metrics import evaluate as compute_metric, node_recall

    print("[Cell 11] Checking tracking results and computing evaluation metrics...")
    scale = np.array(PHYSICAL_SCALE, dtype=np.float32)
    records = []

    for ds_stem, graph in solved_graphs.items():
        rec = {
            "dataset": ds_stem,
            "final_nodes": graph.num_nodes(),
            "final_edges": graph.num_edges(),
        }

        # GT検証モード時の事後採点
        if GT_FLG and gt_dict and ds_stem in gt_dict:
            gt_info = gt_dict[ds_stem]
            gt_df_nodes = gt_info["nodes"]
            gt_df_edges = gt_info["edges"]
            est_nodes = gt_info["estimated_number_of_nodes"]
            gt_tracks = gt_info["tracks"]

            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_geff = Path(tmpdir) / "pred.geff"
                save_graph(graph, tmp_geff)
                pred_res = td.graph.IndexedRXGraph.from_geff(tmp_geff)
                pred_rx = pred_res[0] if isinstance(pred_res, tuple) else pred_res

            er = compute_metric(pred_rx, gt_tracks, scale=PHYSICAL_SCALE)
            n_rec = node_recall(pred_rx, gt_tracks) if graph.num_edges() > 0 else 0.0

            edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
            edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else 0.0
            div_denom = er.division_tp + er.division_fp + er.division_fn
            div_jaccard = er.division_tp / div_denom if div_denom > 0 else 0.0
            score = edge_jaccard + 0.1 * div_jaccard

            e_tp = er.edge_tp
            e_fp = er.edge_fp
            e_fn = er.edge_fn
            e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
            e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
            e_f1 = 2 * e_rec * e_prec / (e_rec + e_prec) if (e_rec + e_prec) > 0 else 0.0
            final_pe = graph.num_nodes() / est_nodes if est_nodes > 0 else 0.0

            rec.update({
                "gt_edges": len(gt_df_edges),
                "edge_tp": e_tp,
                "edge_fp": e_fp,
                "edge_fn": e_fn,
                "edge_recall": round(float(e_rec), 4),
                "edge_precision": round(float(e_prec), 4),
                "edge_f1": round(float(e_f1), 4),
                "post_node_recall": round(float(n_rec), 4),
                "final_pe_ratio": round(float(final_pe), 4),
                "official_edge_jaccard": round(float(edge_jaccard), 4),
                "official_division_jaccard": round(float(div_jaccard), 4),
                "official_score": round(float(score), 4),
            })

        records.append(rec)

    summary_df = pd.DataFrame(records)
    print("=" * 80)
    print("【Cell 11: トラッキング結果 & エッジ評価サマリー】")
    print(summary_df.to_string(index=False))
    print("=" * 80)
    return summary_df


# =============================================================================
# Cell 11-B: 約100種類の特徴量全数抽出 ＆ EDA成果物生成 (extract_all_features)
# =============================================================================

def compute_roc_auc_and_stats(pos_arr: np.ndarray, neg_arr: np.ndarray) -> dict:
    """単一特徴量に対する ROC-AUC, Cohen's d, 最適F1, 最適閾値, 四分位統計量を算出する。"""
    pos_clean = pos_arr[np.isfinite(pos_arr)]
    neg_clean = neg_arr[np.isfinite(neg_arr)]

    if len(pos_clean) == 0 or len(neg_clean) == 0:
        return {
            "roc_auc": 0.5, "direction": "NONE", "cohens_d": 0.0,
            "best_f1": 0.0, "best_precision": 0.0, "best_recall": 0.0, "best_threshold": 0.0,
            "pos_mean": 0.0, "pos_std": 0.0, "pos_median": 0.0, "pos_iqr": 0.0,
            "neg_mean": 0.0, "neg_std": 0.0, "neg_median": 0.0, "neg_iqr": 0.0
        }

    p_mean, p_std = float(np.mean(pos_clean)), float(np.std(pos_clean))
    n_mean, n_std = float(np.mean(neg_clean)), float(np.std(neg_clean))
    p_med, p_iqr = float(np.median(pos_clean)), float(np.percentile(pos_clean, 75) - np.percentile(pos_clean, 25))
    n_med, n_iqr = float(np.median(neg_clean)), float(np.percentile(neg_clean, 75) - np.percentile(neg_clean, 25))

    pooled_std = math.sqrt(((len(pos_clean) - 1) * (p_std**2) + (len(neg_clean) - 1) * (n_std**2)) / max(1, len(pos_clean) + len(neg_clean) - 2))
    cohens_d = (p_mean - n_mean) / (pooled_std + 1e-5)

    all_vals = np.concatenate([pos_clean, neg_clean])
    ranks = pd.Series(all_vals).rank().values
    r_pos = np.sum(ranks[:len(pos_clean)])
    u_pos = r_pos - (len(pos_clean) * (len(pos_clean) + 1)) / 2.0
    auc = float(u_pos / (len(pos_clean) * len(neg_clean)))

    direction = "POS_HIGH"
    if auc < 0.5:
        auc = 1.0 - auc
        direction = "POS_LOW"

    thresholds = np.percentile(all_vals, np.linspace(5, 95, 19))
    best_f1, best_prec, best_rec, best_th = 0.0, 0.0, 0.0, thresholds[0]
    for th in thresholds:
        if direction == "POS_HIGH":
            tp = int((pos_clean >= th).sum())
            fp = int((neg_clean >= th).sum())
        else:
            tp = int((pos_clean <= th).sum())
            fp = int((neg_clean <= th).sum())
        fn = len(pos_clean) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_prec, best_rec, best_th = f1, prec, rec, th

    return {
        "roc_auc": round(auc, 4),
        "direction": direction,
        "cohens_d": round(cohens_d, 4),
        "best_f1": round(best_f1, 4),
        "best_precision": round(best_prec, 4),
        "best_recall": round(best_rec, 4),
        "best_threshold": round(best_th, 4),
        "pos_mean": round(p_mean, 4),
        "pos_std": round(p_std, 4),
        "pos_median": round(p_med, 4),
        "pos_iqr": round(p_iqr, 4),
        "neg_mean": round(n_mean, 4),
        "neg_std": round(n_std, 4),
        "neg_median": round(n_med, 4),
        "neg_iqr": round(n_iqr, 4),
    }


def extract_all_features(
    solved_graphs: dict[str, any],
    raw_preds: dict[str, tuple[np.ndarray, list]],
    gt_dict: dict[str, dict] | None = None
) -> None:
    """Cell 11-B: 約100種類の特徴量 (ノード、エッジ、トラック、フレーム) を全数算出し、EDA成果物を保存する。"""
    import polars as pl
    import networkx as nx

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> Cell 11-B: extract_all_features 開始 (全100種特徴量抽出)")
    scale = np.array(PHYSICAL_SCALE, dtype=np.float32)

    all_node_rows = []
    all_edge_rows = []
    all_track_rows = []
    all_frame_rows = []

    for ds_stem, (coords, candidate_edges) in raw_preds.items():
        graph = solved_graphs[ds_stem]
        ds_path = DATA_DIR / f"{ds_stem}.zarr"
        zarr_root = zarr.open(str(ds_path), mode="r")
        zg_img = zarr_root["0"] if "0" in zarr_root else zarr_root
        T_max, Z_max, Y_max, X_max = zg_img.shape

        gt_info = gt_dict.get(ds_stem) if (GT_FLG and gt_dict) else None
        gt_df_nodes = gt_info["nodes"] if gt_info else None
        gt_df_edges = gt_info["edges"] if gt_info else None

        # 1. ノード単位特徴量抽出 (約25種)
        node_df = pd.DataFrame(coords, columns=["t", "z", "y", "x"])
        node_df["node_id"] = np.arange(len(node_df))
        node_df["dataset"] = ds_stem

        gt_trees_by_t = {}
        if gt_df_nodes is not None:
            for t_val in gt_df_nodes["t"].unique():
                sub_g = gt_df_nodes[gt_df_nodes["t"] == t_val]
                gt_trees_by_t[t_val] = KDTree(sub_g[["z", "y", "x"]].values * scale)

        pred_trees_by_t = {}
        for t_val in node_df["t"].unique():
            sub_p = node_df[node_df["t"] == t_val]
            pred_trees_by_t[t_val] = KDTree(sub_p[["z", "y", "x"]].values * scale)

        frame_cache = {}

        for idx, row in node_df.iterrows():
            t_val = int(row["t"])
            z_val, y_val, x_val = int(row["z"]), int(row["y"]), int(row["x"])

            if t_val not in frame_cache:
                frame_cache[t_val] = zg_img[t_val]
            frame_img = frame_cache[t_val]

            z_norm = z_val / max(1, Z_max - 1)
            y_norm = y_val / max(1, Y_max - 1)
            x_norm = x_val / max(1, X_max - 1)
            dist_to_center_xy = math.sqrt(((y_val - Y_max/2.0)*scale[1])**2 + ((x_val - X_max/2.0)*scale[2])**2)
            dist_to_border_z = min(z_val, Z_max - 1 - z_val)
            dist_to_border_xy = min(y_val, Y_max - 1 - y_val, x_val, X_max - 1 - x_val) * scale[1]

            z_lo, z_hi = max(0, z_val - 1), min(Z_max, z_val + 2)
            y_lo, y_hi = max(0, y_val - 1), min(Y_max, y_val + 2)
            x_lo, x_hi = max(0, x_val - 1), min(X_max, x_val + 2)
            patch = frame_img[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]

            raw_int = float(frame_img[min(Z_max-1, max(0, z_val)), min(Y_max-1, max(0, y_val)), min(X_max-1, max(0, x_val))])
            int_mean = float(np.mean(patch)) if patch.size > 0 else raw_int
            int_max = float(np.max(patch)) if patch.size > 0 else raw_int
            int_min = float(np.min(patch)) if patch.size > 0 else raw_int
            int_std = float(np.std(patch)) if patch.size > 0 else 0.0
            loc_contrast = (int_max - int_min) / (int_mean + 1e-5)
            loc_snr = (raw_int - int_min) / (int_std + 1e-5)

            laplacian = abs(2.0 * raw_int - int_min - int_max)

            tree_p = pred_trees_by_t[t_val]
            pt_um = np.array([z_val, y_val, x_val], dtype=np.float32) * scale
            n_5um = len(tree_p.query_ball_point(pt_um, r=5.0)) - 1
            n_10um = len(tree_p.query_ball_point(pt_um, r=10.0)) - 1
            n_15um = len(tree_p.query_ball_point(pt_um, r=15.0)) - 1
            dists_nn, _ = tree_p.query(pt_um, k=min(2, len(pred_trees_by_t[t_val].data)))
            nn_dist = float(dists_nn[1]) if len(dists_nn) > 1 else 999.0

            is_gt_node = 0
            gt_dist = 999.0
            if t_val in gt_trees_by_t:
                d_gt, _ = gt_trees_by_t[t_val].query(pt_um)
                gt_dist = float(d_gt)
                if d_gt <= MATCH_DISTANCE_THRESHOLD_UM:
                    is_gt_node = 1

            all_node_rows.append({
                "dataset": ds_stem,
                "node_id": int(row["node_id"]),
                "t": t_val,
                "z": z_val, "y": y_val, "x": x_val,
                "z_norm": round(z_norm, 4), "y_norm": round(y_norm, 4), "x_norm": round(x_norm, 4),
                "dist_to_center_xy_um": round(dist_to_center_xy, 2),
                "dist_to_border_z_slices": int(dist_to_border_z),
                "dist_to_border_xy_um": round(dist_to_border_xy, 2),
                "raw_intensity": round(raw_int, 2),
                "mean_intensity_3x3": round(int_mean, 2),
                "max_intensity_3x3": round(int_max, 2),
                "min_intensity_3x3": round(int_min, 2),
                "std_intensity_3x3": round(int_std, 2),
                "local_contrast": round(loc_contrast, 4),
                "local_snr": round(loc_snr, 4),
                "laplacian_sharpness": round(laplacian, 2),
                "det_prob": 1.0,
                "neighbor_count_5um": int(n_5um),
                "neighbor_count_10um": int(n_10um),
                "neighbor_count_15um": int(n_15um),
                "nearest_neighbor_dist_um": round(nn_dist, 2),
                "is_gt_node": is_gt_node,
                "gt_matched_dist_um": round(gt_dist, 2),
            })

        # 2. エッジ単位特徴量抽出 (約45種)
        ilp_edges_df = graph.edge_attrs()
        if "solution" in ilp_edges_df.columns:
            ilp_edges_df = ilp_edges_df.filter(pl.col("solution"))
        solved_edge_set = set(zip(ilp_edges_df["source_id"].to_list(), ilp_edges_df["target_id"].to_list())) if len(ilp_edges_df) > 0 else set()

        gt_edge_set = set()
        if gt_df_edges is not None and gt_df_nodes is not None:
            pred_to_gt = {}
            for t_val, tree_gt in gt_trees_by_t.items():
                p_sub = node_df[node_df["t"] == t_val]
                g_sub = gt_df_nodes[gt_df_nodes["t"] == t_val]
                if p_sub.empty or g_sub.empty:
                    continue
                p_pts = p_sub[["z", "y", "x"]].values * scale
                g_ids = g_sub["node_id"].values
                dists_m, idxs_m = tree_gt.query(p_pts, distance_upper_bound=MATCH_DISTANCE_THRESHOLD_UM)
                for p_nid, d_val, g_idx in zip(p_sub["node_id"].values, dists_m, idxs_m):
                    if d_val <= MATCH_DISTANCE_THRESHOLD_UM and g_idx < len(g_ids):
                        pred_to_gt[p_nid] = g_ids[g_idx]

            raw_gt_pairs = set(zip(gt_df_edges["source_id"].values, gt_df_edges["target_id"].values))
            for s_id, t_id in solved_edge_set.union({(e[0], e[1]) for e in candidate_edges}):
                g_s = pred_to_gt.get(s_id)
                g_t = pred_to_gt.get(t_id)
                if g_s is not None and g_t is not None and (g_s, g_t) in raw_gt_pairs:
                    gt_edge_set.add((s_id, t_id))

        # node_lookupは all_node_rows から作成 (raw_intensity, local_snr 等の特徴量を含む)
        _cur_node_rows = [r for r in all_node_rows if r["dataset"] == ds_stem]
        node_lookup = pd.DataFrame(_cur_node_rows).set_index("node_id") if _cur_node_rows else node_df.set_index("node_id")
        fwd_candidates: dict[int, list] = {}
        bwd_candidates: dict[int, list] = {}
        for edge_tuple in candidate_edges:
            s_id, t_id, prob, d_val = edge_tuple
            fwd_candidates.setdefault(s_id, []).append((d_val, t_id, prob))
            bwd_candidates.setdefault(t_id, []).append((d_val, s_id, prob))

        for edge_tuple in candidate_edges:
            s_id, t_id, prob, d_val = edge_tuple
            src = node_lookup.loc[s_id]
            tgt = node_lookup.loc[t_id]

            t_src, t_tgt = int(src["t"]), int(tgt["t"])
            dt = t_tgt - t_src
            dz = (float(tgt["z"]) - float(src["z"])) * scale[0]
            dy = (float(tgt["y"]) - float(src["y"])) * scale[1]
            dx = (float(tgt["x"]) - float(src["x"])) * scale[2]
            dist_xy = math.sqrt(dy**2 + dx**2)
            dist_z = abs(dz)
            dist_3d = math.sqrt(dz**2 + dy**2 + dx**2)
            velocity = dist_3d / max(1, dt)

            f_list = sorted(fwd_candidates.get(s_id, []))
            b_list = sorted(bwd_candidates.get(t_id, []))
            fwd_rank = [item[1] for item in f_list].index(t_id) + 1
            bwd_rank = [item[1] for item in b_list].index(s_id) + 1
            margin_fwd = (f_list[1][0] - f_list[0][0]) if len(f_list) > 1 else 999.0
            margin_bwd = (b_list[1][0] - b_list[0][0]) if len(b_list) > 1 else 999.0
            is_mnn = 1 if (fwd_rank == 1 and bwd_rank == 1) else 0

            comp_10 = sum(1 for item in f_list if item[0] <= 10.0)
            comp_15 = sum(1 for item in f_list if item[0] <= 15.0)

            int_diff = abs(src["raw_intensity"] - tgt["raw_intensity"])
            int_ratio = src["raw_intensity"] / (tgt["raw_intensity"] + 1e-5)
            snr_diff = abs(src["local_snr"] - tgt["local_snr"])
            snr_ratio = src["local_snr"] / (tgt["local_snr"] + 1e-5)

            dead_reck_fwd = 0.0
            dead_reck_bwd = 0.0
            cos_gap_prev = 0.0
            cos_next_gap = 0.0

            in_edges = bwd_candidates.get(s_id, [])
            if in_edges:
                best_in = sorted(in_edges)[0]
                prev_src = node_lookup.loc[best_in[1]]
                prev_vec = np.array([float(src["z"])-float(prev_src["z"]), float(src["y"])-float(prev_src["y"]), float(src["x"])-float(prev_src["x"])]) * scale
                curr_vec = np.array([dz, dy, dx])
                norm_p, norm_c = np.linalg.norm(prev_vec), np.linalg.norm(curr_vec)
                if norm_p > 1e-5 and norm_c > 1e-5:
                    cos_gap_prev = float(np.dot(prev_vec, curr_vec) / (norm_p * norm_c))
                    dead_reck_fwd = float(np.linalg.norm(curr_vec - prev_vec))

            out_edges = fwd_candidates.get(t_id, [])
            if out_edges:
                best_out = sorted(out_edges)[0]
                next_tgt = node_lookup.loc[best_out[1]]
                next_vec = np.array([float(next_tgt["z"])-float(tgt["z"]), float(next_tgt["y"])-float(tgt["y"]), float(next_tgt["x"])-float(tgt["x"])]) * scale
                curr_vec = np.array([dz, dy, dx])
                norm_n, norm_c = np.linalg.norm(next_vec), np.linalg.norm(curr_vec)
                if norm_n > 1e-5 and norm_c > 1e-5:
                    cos_next_gap = float(np.dot(curr_vec, next_vec) / (norm_c * norm_n))
                    dead_reck_bwd = float(np.linalg.norm(next_vec - curr_vec))

            dead_reck_mean = (dead_reck_fwd + dead_reck_bwd) / 2.0 if (dead_reck_fwd > 0 and dead_reck_bwd > 0) else max(dead_reck_fwd, dead_reck_bwd)
            is_selected = 1 if (s_id, t_id) in solved_edge_set else 0
            is_gt_edge = 1 if (s_id, t_id) in gt_edge_set else 0

            all_edge_rows.append({
                "dataset": ds_stem,
                "source_id": s_id,
                "target_id": t_id,
                "t_src": t_src, "t_tgt": t_tgt, "dt": dt,
                "dist_3d_um": round(dist_3d, 2),
                "dist_xy_um": round(dist_xy, 2),
                "dist_z_um": round(dist_z, 2),
                "dz_um": round(dz, 2), "dy_um": round(dy, 2), "dx_um": round(dx, 2),
                "velocity_um_per_frame": round(velocity, 2),
                "fwd_rank": int(fwd_rank),
                "bwd_rank": int(bwd_rank),
                "margin_fwd_um": round(margin_fwd, 2),
                "margin_bwd_um": round(margin_bwd, 2),
                "is_mnn": int(is_mnn),
                "comp_count_10um": int(comp_10),
                "comp_count_15um": int(comp_15),
                "cos_gap_prev": round(cos_gap_prev, 4),
                "cos_next_gap": round(cos_next_gap, 4),
                "dead_reckoning_fwd_um": round(dead_reck_fwd, 2),
                "dead_reckoning_bwd_um": round(dead_reck_bwd, 2),
                "dead_reckoning_mean_um": round(dead_reck_mean, 2),
                "intensity_diff": round(int_diff, 2),
                "intensity_ratio": round(int_ratio, 4),
                "snr_diff": round(snr_diff, 4),
                "snr_ratio": round(snr_ratio, 4),
                "transformer_prob": round(float(prob), 4),
                "ilp_selected": int(is_selected),
                "is_gt_edge": int(is_gt_edge),
            })

        # 3. トラック単位特徴量抽出 (約20種)
        G = nx.Graph()
        G.add_nodes_from(node_df["node_id"].values)
        for s_id, t_id in solved_edge_set:
            G.add_edge(s_id, t_id)

        comps = list(nx.connected_components(G))
        for trk_idx, comp in enumerate(comps):
            c_len = len(comp)
            sub_nodes = node_lookup.loc[list(comp)].sort_values("t")
            sub_coords = sub_nodes[["z", "y", "x"]].values * scale

            dur = int(sub_nodes["t"].max() - sub_nodes["t"].min() + 1)
            if c_len == 1:
                total_disp, path_len, straightness, net_vel, mean_step, max_step, step_cv, conf, aspect, directed = (
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
                )
            else:
                step_diffs = np.linalg.norm(np.diff(sub_coords, axis=0), axis=1)
                mean_step = float(np.mean(step_diffs))
                max_step = float(np.max(step_diffs))
                path_len = float(np.sum(step_diffs))
                total_disp = float(np.linalg.norm(sub_coords[-1] - sub_coords[0]))
                straightness = float(total_disp / (path_len + 1e-5))
                net_vel = float(total_disp / max(1, dur - 1))
                step_cv = float(np.std(step_diffs) / (mean_step + 1e-5)) if len(step_diffs) > 1 else 0.0
                disp_from_start = np.linalg.norm(sub_coords - sub_coords[0], axis=1)
                conf = float(np.max(disp_from_start) / (path_len + 1e-5))

                if c_len >= 3:
                    centered = sub_coords - np.mean(sub_coords, axis=0)
                    cov = np.cov(centered, rowvar=False)
                    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                    aspect = float(eigvals[0] / (eigvals[1] + 1e-5)) if len(eigvals) >= 2 else 1.0
                else:
                    aspect = 1.0
                directed = float(straightness * math.sqrt(c_len) * net_vel)

            t_int_mean = float(sub_nodes["raw_intensity"].mean())
            t_int_min = float(sub_nodes["raw_intensity"].min())
            t_int_max = float(sub_nodes["raw_intensity"].max())
            t_int_std = float(sub_nodes["raw_intensity"].std()) if c_len > 1 else 0.0
            t_snr_mean = float(sub_nodes["local_snr"].mean())
            t_snr_min = float(sub_nodes["local_snr"].min())
            t_snr_max = float(sub_nodes["local_snr"].max())

            is_gt_trk = 1 if (sub_nodes["is_gt_node"].mean() >= 0.5) else 0

            all_track_rows.append({
                "dataset": ds_stem,
                "track_id": trk_idx,
                "track_length": c_len,
                "duration_frames": dur,
                "total_disp_um": round(total_disp, 2),
                "path_length_um": round(path_len, 2),
                "straightness_ratio": round(straightness, 4),
                "net_velocity_um": round(net_vel, 2),
                "mean_step_um": round(mean_step, 2),
                "max_step_um": round(max_step, 2),
                "step_velocity_cv": round(step_cv, 4),
                "confinement_ratio": round(conf, 4),
                "trajectory_aspect_ratio": round(aspect, 4),
                "directed_motion_index": round(directed, 4),
                "mean_track_intensity": round(t_int_mean, 2),
                "min_track_intensity": round(t_int_min, 2),
                "max_track_intensity": round(t_int_max, 2),
                "std_track_intensity": round(t_int_std, 2),
                "mean_track_snr": round(t_snr_mean, 4),
                "min_track_snr": round(t_snr_min, 4),
                "max_track_snr": round(t_snr_max, 4),
                "is_gt_track": is_gt_trk,
            })

        # 4. フレーム光学・密度サマリー (約10種)
        for t_idx in sorted(frame_cache.keys()):
            f_img = frame_cache[t_idx]
            sub = f_img[::2, ::2, ::2]
            p01, p25, p50, p75, p995 = np.percentile(sub, [1.0, 25.0, 50.0, 75.0, 99.5])
            bg_noise = float((p75 - p25) / 1.349)
            snr_proxy = float((p995 - p50) / (bg_noise + 1e-5))
            fg_ratio = float((sub > (p50 + 3.0 * bg_noise)).mean())

            n_det = int((node_df["t"] == t_idx).sum())
            all_frame_rows.append({
                "dataset": ds_stem,
                "t": t_idx,
                "bg_median": round(float(p50), 2),
                "bg_noise": round(float(bg_noise), 2),
                "snr_proxy": round(float(snr_proxy), 4),
                "fg_ratio": round(float(fg_ratio), 4),
                "p01": round(float(p01), 2),
                "p995": round(float(p995), 2),
                "detected_nodes": n_det,
            })

    # DataFrame化
    df_nodes = pd.DataFrame(all_node_rows)
    df_edges = pd.DataFrame(all_edge_rows)
    df_tracks = pd.DataFrame(all_track_rows)
    df_frames = pd.DataFrame(all_frame_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_nodes.to_csv(OUTPUT_DIR / OUTPUT_EDA_NODE_CSV, index=False)
    df_edges.to_csv(OUTPUT_DIR / OUTPUT_EDA_EDGE_CSV, index=False)
    df_tracks.to_csv(OUTPUT_DIR / OUTPUT_EDA_TRACK_CSV, index=False)
    df_frames.to_csv(OUTPUT_DIR / OUTPUT_EDA_FRAME_CSV, index=False)
    print(f"  - [SAVE] 生データCSV出力完了:")
    print(f"    1. {OUTPUT_EDA_NODE_CSV} ({len(df_nodes):,} 行)")
    print(f"    2. {OUTPUT_EDA_EDGE_CSV} ({len(df_edges):,} 行)")
    print(f"    3. {OUTPUT_EDA_TRACK_CSV} ({len(df_tracks):,} 行)")
    print(f"    4. {OUTPUT_EDA_FRAME_CSV} ({len(df_frames):,} 行)")

    # 5. EDA解析メトリクス一覧表 (全約100特徴量の分離能ランキング)
    eda_metrics_list = []

    if not df_nodes.empty and "is_gt_node" in df_nodes.columns and df_nodes["is_gt_node"].nunique() > 1:
        pos_mask = df_nodes["is_gt_node"] == 1
        neg_mask = df_nodes["is_gt_node"] == 0
        node_cols = [c for c in df_nodes.columns if c not in ["dataset", "node_id", "t", "is_gt_node"]]
        for col in node_cols:
            stats = compute_roc_auc_and_stats(df_nodes.loc[pos_mask, col].values, df_nodes.loc[neg_mask, col].values)
            eda_metrics_list.append({"feature_name": col, "hierarchy": "ノード (Node)", **stats})

    if not df_edges.empty and "is_gt_edge" in df_edges.columns and df_edges["is_gt_edge"].nunique() > 1:
        pos_mask = df_edges["is_gt_edge"] == 1
        neg_mask = df_edges["is_gt_edge"] == 0
        edge_cols = [c for c in df_edges.columns if c not in ["dataset", "source_id", "target_id", "t_src", "t_tgt", "is_gt_edge"]]
        for col in edge_cols:
            stats = compute_roc_auc_and_stats(df_edges.loc[pos_mask, col].values, df_edges.loc[neg_mask, col].values)
            eda_metrics_list.append({"feature_name": col, "hierarchy": "エッジ (Edge)", **stats})

    if not df_tracks.empty and "is_gt_track" in df_tracks.columns and df_tracks["is_gt_track"].nunique() > 1:
        pos_mask = df_tracks["is_gt_track"] == 1
        neg_mask = df_tracks["is_gt_track"] == 0
        trk_cols = [c for c in df_tracks.columns if c not in ["dataset", "track_id", "is_gt_track"]]
        for col in trk_cols:
            stats = compute_roc_auc_and_stats(df_tracks.loc[pos_mask, col].values, df_tracks.loc[neg_mask, col].values)
            eda_metrics_list.append({"feature_name": col, "hierarchy": "トラック (Track)", **stats})

    df_eda = pd.DataFrame(eda_metrics_list)
    if not df_eda.empty:
        df_eda = df_eda.sort_values(by=["roc_auc", "cohens_d"], ascending=[False, False]).reset_index(drop=True)
        df_eda.insert(0, "rank", range(1, len(df_eda) + 1))
        df_eda.to_csv(OUTPUT_DIR / OUTPUT_EDA_METRICS_CSV, index=False)
        print(f"  - [SAVE] EDA集計CSV出力完了: {OUTPUT_EDA_METRICS_CSV} (全 {len(df_eda)} 特徴量ランキング)")

        # 特徴量のメタ情報辞書 (和名, 一行説明, 算出式)
        feature_meta_map = {
            "gt_matched_dist_um": ("GT最近傍距離 (μm)", "推論ノードと最も近いGT細胞ノードとの3D空間物理距離", "min(||p_pred - p_gt|| * scale)"),
            "step_velocity_cv": ("ステップ速度変動係数", "トラック内のフレーム間移動速度のばらつき比率 (等速性の指標)", "std(step_velocities) / (mean(step_velocities) + 1e-5)"),
            "dist_to_center_xy_um": ("XY画像中心距離 (μm)", "視野中心からのXY平面上の物理距離 (中心集中傾向の指標)", "sqrt(((y - Y/2)*sy)^2 + ((x - X/2)*sx)^2)"),
            "dist_to_border_xy_um": ("XY境界最近接距離 (μm)", "XY画像境界(端点)までの最短物理距離 (端の途切れ検知)", "min(y, Y-1-y, x, X-1-x) * s_xy"),
            "min_intensity_3x3": ("3x3局所最小輝度", "ノード中心周辺3x3x3ボクセルの最小画素輝度値", "min(I[z-1:z+2, y-1:y+2, x-1:x+2])"),
            "mean_intensity_3x3": ("3x3局所平均輝度", "ノード中心周辺3x3x3ボクセルの平均画素輝度値", "mean(I[z-1:z+2, y-1:y+2, x-1:x+2])"),
            "raw_intensity": ("生画素輝度", "検出ノード中心ボクセルの生輝度値 (細胞核の中心輝度)", "I[z, y, x]"),
            "mean_track_intensity": ("トラック平均輝度", "トラックを構成する全細胞ノードの平均画素輝度", "mean_{v in track}(raw_intensity)"),
            "max_intensity_3x3": ("3x3局所最大輝度", "ノード中心周辺3x3x3ボクセルの最大画素輝度値", "max(I[z-1:z+2, y-1:y+2, x-1:x+2])"),
            "straightness_ratio": ("軌跡直進性比率", "正味直線変位と総移動経路長の比率 (1.0に近いほど直進)", "net_displacement / total_path_length"),
            "confinement_ratio": ("空間拘束比", "正味直線変位と総移動経路長の比率 (遊走か停滞かの指標)", "net_displacement / total_path_length"),
            "max_track_intensity": ("トラック最大輝度", "トラック内の細胞ノードが記録した最大輝度値", "max_{v in track}(raw_intensity)"),
            "nearest_neighbor_dist_um": ("最寄りノード距離 (μm)", "同一フレーム内で最も近い他ノードまでの3D空間距離", "min_{j != i} ||p_i - p_j|| * scale"),
            "std_track_intensity": ("トラック輝度標準偏差", "トラック内の細胞ノード輝度の時間的変動幅", "std_{v in track}(raw_intensity)"),
            "neighbor_count_10um": ("10μm近傍ノード数", "半径10μm球体内に存在する他ノードの個数 (局所細胞密度)", "|{j | ||p_i - p_j|| * scale <= 10μm}| - 1"),
            "total_disp_um": ("正味直線変位 (μm)", "トラック始点から終点までの3D直線物理距離", "||p_end - p_start|| * scale"),
            "directed_motion_index": ("方向性遊走指標", "直進性と平均速度を掛け合わせた複合遊走指標", "straightness_ratio * mean_velocity"),
            "local_contrast": ("局所コントラスト", "局所パッチ内の明暗差と平均輝度の比 (細胞輪郭コントラスト)", "(int_max - int_min) / (int_mean + 1e-5)"),
            "duration_frames": ("存続フレーム数", "トラックが観測され続けた時間長 (フレーム数)", "t_end - t_start"),
            "track_length": ("トラックノード数", "トラックに含まれる細胞ノードの総数", "len(nodes in track)"),
            "neighbor_count_15um": ("15μm近傍ノード数", "半径15μm球体内に存在する他ノードの個数", "|{j | ||p_i - p_j|| * scale <= 15μm}| - 1"),
            "path_length_um": ("総移動経路長 (μm)", "トラックの各ステップの3D移動距離の総和", "sum(||p_{t+1} - p_t|| * scale)"),
            "laplacian_sharpness": ("ラプラシアン鮮鋭度", "画素中心と局所両極値との二次微分的差異 (輪郭の鮮鋭さ)", "|2 * raw_int - int_min - int_max|"),
            "max_step_um": ("最大ステップ長 (μm)", "トラック内で記録された1フレーム間の最大移動距離", "max(step_distances)"),
            "snr_diff": ("SNR変化量", "接続元ノードと接続先ノードの局所SNRの絶対差", "|snr_src - snr_tgt|"),
            "trajectory_aspect_ratio": ("軌跡主軸アスペクト比", "3D軌跡の主成分分析(PCA)主軸比 (直進か等方分散かの判定)", "lambda_1 / (lambda_2 + 1e-5)"),
            "dz_um": ("Z軸変位 (μm)", "接続元から接続先へのZ方向の符号付き物理移動量", "(z_tgt - z_src) * s_z"),
            "max_track_snr": ("トラック最大SNR", "トラック内の細胞ノードが記録した最大局所SNR", "max_{v in track}(local_snr)"),
            "std_intensity_3x3": ("3x3局所輝度標準偏差", "局所パッチ内の輝度ばらつき (テクスチャ複雑度)", "std(I[z-1:z+2, y-1:y+2, x-1:x+2])"),
            "net_velocity_um": ("正味前進速度 (μm/frame)", "正味変位を存続フレーム数で割った平均前進速度", "total_disp_um / max(1, duration_frames)"),
            "min_track_intensity": ("トラック最小輝度", "トラック内の細胞ノードが記録した最小輝度値", "min_{v in track}(raw_intensity)"),
            "dist_z_um": ("Z軸移動距離 (μm)", "Z軸方向の移動距離の絶対値", "|dz_um|"),
            "mean_track_snr": ("トラック平均SNR", "トラックを構成する全細胞ノードの平均局所SNR", "mean_{v in track}(local_snr)"),
            "transformer_prob": ("Transformer接続確率", "SimpleNodeTransformerが予測した時空間接続確率スコア", "P(edge)"),
            "dist_3d_um": ("3Dユークリッド変位 (μm)", "前後フレーム間の細胞3D移動距離", "sqrt(dz^2 + dy^2 + dx^2)"),
            "velocity_um_per_frame": ("フレーム間移動速度 (μm/frame)", "単位フレームあたりの物理移動速度", "dist_3d_um / max(1, dt)"),
            "cos_gap_prev": ("前時刻進行方向コサイン類似度", "直前の移動ベクトルと現在の移動ベクトルのなす角 (慣性・直進性)", "(v_prev · v_curr) / (||v_prev|| * ||v_curr||)"),
            "intensity_diff": ("輝度差分", "接続元ノードと接続先ノードの生輝度の絶対差", "|raw_int_src - raw_int_tgt|"),
            "cos_next_gap": ("次時刻進行方向コサイン類似度", "現在の移動ベクトルと次ステップ候補ベクトルのなす角", "(v_curr · v_next) / (||v_curr|| * ||v_next||)"),
            "dx_um": ("X軸変位 (μm)", "接続元から接続先へのX方向の符号付き物理移動量", "(x_tgt - x_src) * s_x"),
            "x": ("X画素座標", "画像内のX軸ボクセルインデックス", "x"),
            "x_norm": ("正規化X座標", "0〜1に正規化されたX軸位置", "x / (X_max - 1)"),
            "dist_xy_um": ("XY平面変位 (μm)", "XY平面上での物理移動距離", "sqrt(dy^2 + dx^2)"),
            "z": ("Zスライス座標", "3次元スタック内のZスライスインデックス", "z"),
            "z_norm": ("正規化Z座標", "0〜1に正規化されたZ軸深さ", "z / (Z_max - 1)"),
            "y": ("Y画素座標", "画像内のY軸ボクセルインデックス", "y"),
            "y_norm": ("正規化Y座標", "0〜1に正規化されたY軸位置", "y / (Y_max - 1)"),
            "local_snr": ("局所SNR", "局所パッチ内の輝度コントラストと標準偏差の比", "(raw_int - int_min) / (int_std + 1e-5)"),
            "dead_reckoning_mean_um": ("平均推測航法残差 (μm)", "前後ステップ等速直線運動予測位置と実際位置の平均誤差", "(dead_reck_fwd + dead_reck_bwd) / 2"),
            "dead_reckoning_fwd_um": ("前向き推測航法残差 (μm)", "直前ステップの速度ベクトルを外挿した予測位置からのズレ", "||v_curr - v_prev||"),
            "dead_reckoning_bwd_um": ("後向き推測航法残差 (μm)", "次ステップの逆向き速度ベクトルを外挿した予測位置からのズレ", "||v_next - v_curr||"),
            "min_track_snr": ("トラック最小SNR", "トラック内の細胞ノードが記録した最小局所SNR", "min_{v in track}(local_snr)"),
            "neighbor_count_5um": ("5μm近傍ノード数", "半径5μm球体内に存在する他ノードの個数 (密集・重複検知)", "|{j | ||p_i - p_j|| * scale <= 5μm}| - 1"),
            "margin_fwd_um": ("前向き競合距離マージン (μm)", "第1候補ノードと第2候補ノードの距離差 (曖昧さの指標)", "dist_cand2 - dist_cand1"),
            "comp_count_15um": ("15μm競合候補数", "半径15μm以内に存在する接続先候補ノードの総数", "|candidates in 15μm|"),
            "comp_count_10um": ("10μm競合候補数", "半径10μm以内に存在する接続先候補ノードの総数", "|candidates in 10μm|"),
            "dist_to_border_z_slices": ("Z境界スライス距離", "Z軸の上下端(第0面または最終面)までのスライス面数", "min(z, Z_max - 1 - z)"),
            "intensity_ratio": ("輝度比率", "接続先ノードと接続元ノードの生輝度比", "raw_int_src / (raw_int_tgt + 1e-5)"),
            "snr_ratio": ("SNR比率", "接続先ノードと接続元ノードの局所SNR比", "snr_src / (snr_tgt + 1e-5)"),
            "is_mnn": ("相互最近傍フラグ", "前向き・後向きの双方向で互いに第1候補であるか (1/0)", "(fwd_rank == 1 and bwd_rank == 1)"),
            "fwd_rank": ("前向き距離順位", "接続元ノードから見た接続先ノードの距離順位 (1が最寄り)", "rank of tgt from src"),
            "ilp_selected": ("ILP採択フラグ", "大域的整数線形計画法(ILP)によって最終採用されたか (1/0)", "edge in ILP solution"),
            "dy_um": ("Y軸変位 (μm)", "接続元から接続先へのY方向の符号付き物理移動量", "(y_tgt - y_src) * s_y"),
            "det_prob": ("3D-UNet検出確率", "3D-UNetが細胞中心であると判定した信頼度スコア", "Sigmoid(logits)"),
            "dt": ("時間間隔 (フレーム数)", "接続元フレームと接続先フレームの時間差 (通常1)", "t_tgt - t_src"),
            "bwd_rank": ("後向き距離順位", "接続先ノードから見た接続元ノードの逆向き距離順位", "rank of src to tgt"),
            "margin_bwd_um": ("後向き競合距離マージン (μm)", "逆向き探索における第1候補と第2候補の距離差", "dist_bwd_cand2 - dist_bwd_cand1"),
            "mean_step_um": ("平均ステップ長 (μm)", "トラックの1フレームあたりの平均移動距離", "path_length_um / (track_length - 1)"),
        }

        try:
            xlsx_path = OUTPUT_DIR / OUTPUT_EDA_METRICS_XLSX
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                # index シートの構築 (和名・説明・算出式を付与)
                df_index = df_eda[["rank", "feature_name", "hierarchy", "roc_auc", "direction", "cohens_d", "best_f1", "best_threshold"]].copy()
                df_index.insert(2, "feature_name_ja", df_index["feature_name"].map(lambda x: feature_meta_map.get(x, (x, "", ""))[0]))
                df_index.insert(4, "description", df_index["feature_name"].map(lambda x: feature_meta_map.get(x, ("", "", ""))[1]))
                df_index.insert(5, "formula", df_index["feature_name"].map(lambda x: feature_meta_map.get(x, ("", "", ""))[2]))

                df_index.columns = [
                    "順位", "特徴量名 (英名)", "特徴量名 (和名)", "系統階層",
                    "簡単な一行説明", "算出式", "ROC-AUC", "分離方向", "Cohen's d", "最大F1", "最適足切り閾値"
                ]
                df_index.to_excel(writer, sheet_name="index", index=False)
                df_eda.to_excel(writer, sheet_name="metrics", index=False)
            print(f"  - [SAVE] EDA Excelブック保存完了: {xlsx_path.resolve()} (Sheet1: index, Sheet2: metrics)")
        except Exception as ex:
            print(f"  - [INFO] openpyxl 未インストールのため Excel 出力をスキップ (CSVは正常生成済み): {ex}")

        print("=" * 80)
        print("【Cell 11-B: 全約100特徴量 EDAランキング TOP 15】")
        print(df_eda.head(15)[["rank", "feature_name", "hierarchy", "roc_auc", "direction", "cohens_d", "best_f1"]].to_string(index=False))
        print("=" * 80)

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< Cell 11-B: extract_all_features 正常終了")


# =============================================================================
# Cell 12: 最終出力 & 提出ファイル生成 (save_and_push_results)
# =============================================================================

def save_and_push_results(
    solved_graphs: dict[str, any],
    node_summary_df: pd.DataFrame,
    edge_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cell 12: tracksdata グラフから Kaggle公式10列フォーマットの submission.csv を生成・保存する。"""
    import polars as pl
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> Cell 12: save_and_push_results 開始")
    sub_parts = []

    for ds_stem, graph in solved_graphs.items():
        nodes_df = graph.node_attrs()
        if "solution" in nodes_df.columns:
            nodes_df = nodes_df.filter(pl.col("solution"))

        if len(nodes_df) > 0:
            df_n = nodes_df.select([
                pl.lit(ds_stem).alias("dataset"),
                pl.lit("node").alias("row_type"),
                pl.col("node_id").cast(pl.Int64),
                pl.col("t").cast(pl.Int64),
                pl.col("z").round().cast(pl.Int64),
                pl.col("y").round().cast(pl.Int64),
                pl.col("x").round().cast(pl.Int64),
                pl.lit(-1).cast(pl.Int64).alias("source_id"),
                pl.lit(-1).cast(pl.Int64).alias("target_id"),
            ]).to_pandas()
            sub_parts.append(df_n)

        edges_df = graph.edge_attrs()
        if "solution" in edges_df.columns:
            edges_df = edges_df.filter(pl.col("solution"))

        if len(edges_df) > 0:
            df_e = edges_df.select([
                pl.lit(ds_stem).alias("dataset"),
                pl.lit("edge").alias("row_type"),
                pl.lit(-1).cast(pl.Int64).alias("node_id"),
                pl.lit(-1).cast(pl.Int64).alias("t"),
                pl.lit(-1).cast(pl.Int64).alias("z"),
                pl.lit(-1).cast(pl.Int64).alias("y"),
                pl.lit(-1).cast(pl.Int64).alias("x"),
                pl.col("source_id").cast(pl.Int64),
                pl.col("target_id").cast(pl.Int64),
            ]).to_pandas()
            sub_parts.append(df_e)

    if sub_parts:
        df_sub = pd.concat(sub_parts, ignore_index=True)
    else:
        df_sub = pd.DataFrame([{
            "dataset": "dummy", "row_type": "node", "node_id": 0,
            "t": 0, "z": 0, "y": 0, "x": 0, "source_id": -1, "target_id": -1
        }])

    df_sub = df_sub.sort_values(
        by=["dataset", "row_type", "t", "node_id"],
        ascending=[True, False, True, True]
    ).reset_index(drop=True)

    df_sub.insert(0, "id", range(len(df_sub)))
    for col in ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]:
        df_sub[col] = df_sub[col].astype("int64")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_paths = [Path("submission.csv"), OUTPUT_DIR / OUTPUT_SUBMISSION_CSV]
    for sp in set(sub_paths):
        sp.parent.mkdir(parents=True, exist_ok=True)
        df_sub.to_csv(sp, index=False)
        print(f"  - [OK] 提出用ファイル出力完了: {sp.resolve()} (全 {len(df_sub):,} 行 | ノード: {(df_sub['row_type']=='node').sum():,} 行, エッジ: {(df_sub['row_type']=='edge').sum():,} 行)")

    if GT_FLG and not edge_summary_df.empty:
        details_df = pd.merge(node_summary_df, edge_summary_df, on="dataset", how="outer") if not node_summary_df.empty else edge_summary_df
        details_path = OUTPUT_DIR / OUTPUT_DETAILS_CSV
        details_df.to_csv(details_path, index=False)
        print(f"  - [SAVE] 詳細評価CSV保存完了: {details_path.resolve()}")

        summary_record = {
            "magic_string": MAGIC_STRING,
            "total_datasets": len(details_df),
            "macro_node_recall": round(float(details_df["node_recall"].mean()), 4) if "node_recall" in details_df else 0.0,
            "macro_node_precision": round(float(details_df["node_precision"].mean()), 4) if "node_precision" in details_df else 0.0,
            "macro_node_f1": round(float(details_df["node_f1"].mean()), 4) if "node_f1" in details_df else 0.0,
            "macro_edge_recall": round(float(details_df["edge_recall"].mean()), 4) if "edge_recall" in details_df else 0.0,
            "macro_edge_precision": round(float(details_df["edge_precision"].mean()), 4) if "edge_precision" in details_df else 0.0,
            "macro_edge_f1": round(float(details_df["edge_f1"].mean()), 4) if "edge_f1" in details_df else 0.0,
            "macro_official_jaccard": round(float(details_df["official_edge_jaccard"].mean()), 4) if "official_edge_jaccard" in details_df else 0.0,
            "macro_official_score": round(float(details_df["official_score"].mean()), 4) if "official_score" in details_df else 0.0,
            "macro_final_pe_ratio": round(float(details_df["final_pe_ratio"].mean()), 4) if "final_pe_ratio" in details_df else 0.0,
        }
        summary_df = pd.DataFrame([summary_record])
        summary_path = OUTPUT_DIR / OUTPUT_SUMMARY_CSV
        summary_df.to_csv(summary_path, index=False)
        print(f"  - [SAVE] 全体マクロ評価サマリーCSV保存完了: {summary_path.resolve()}")

        print("=" * 80)
        print("【Cell 12: パイプライン全体達成サマリー】")
        for k, v in summary_record.items():
            print(f"  - {k}: {v}")
        print("=" * 80)

        if PUSH_TO_GITHUB and GITHUB_TOKEN:
            push_to_github(OUTPUT_SUMMARY_CSV, commit_message=f"s6_026: push summary {MAGIC_STRING}")
            push_to_github(OUTPUT_DETAILS_CSV, commit_message=f"s6_026: push details {MAGIC_STRING}")

    return df_sub


# =============================================================================
# Cell 13: メイン関数 (main エントリポイント)
# =============================================================================

def main():
    """Cell 13: 3D-UNet + Transformer + ILP パイプライン一気通貫実行エントリポイント。"""
    start_total = time.time()
    print("=" * 80)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] >>> Cell 13: main パイプライン開始")
    print(f"  - 実行識別子: {MAGIC_STRING}")
    print(f"  - 成果物接頭語: {RUN_PREFIX}")
    print(f"  - 実行モード: {'TRAIN / EVAL (GT検証)' if GT_FLG else 'SUBMIT (提出用)'}")
    print(f"  - GPUフラグ:  {'有効 (GPU優先)' if GPU_FLG else '無効 (強制CPU)'}")
    print("=" * 80)

    setup_environment()
    check_environment()
    gt_dict = load_gt_data()
    raw_preds = detect_nodes_and_edges()
    node_summary_df = check_nodes(raw_preds, gt_dict)
    solved_graphs = build_graph_and_solve_ilp(raw_preds)
    edge_summary_df = check_edges(solved_graphs, gt_dict)

    if GT_FLG:
        extract_all_features(solved_graphs, raw_preds, gt_dict)

    df_sub = save_and_push_results(solved_graphs, node_summary_df, edge_summary_df)

    elapsed_total = time.time() - start_total
    print("=" * 80)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] <<< Cell 13: main パイプライン全処理完了 (所要時間: {elapsed_total:.2f} 秒)")
    print("=" * 80)


if __name__ == "__main__":
    main()
