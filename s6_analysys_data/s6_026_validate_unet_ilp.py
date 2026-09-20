"""s6_026_validate_unet_ilp.py

3D-UNet + Transformer + ILP 深層学習パイプラインのローカル検証スクリプト。
代表データセットに対して推論・グラフ構築・ILP大域最適化を実行し、
公式評価指標 (Node Recall, Edge Jaccard, Division Jaccard, Score) を算出する。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

# サポートパックの repo/src と repo/scripts を sys.path に追加
BASE_DIR = Path(r"c:\work\aaa\s6")
SUPPORT_PACK_DIR = BASE_DIR / "input" / "support_pack"
REPO_SRC = SUPPORT_PACK_DIR / "repo" / "src"
REPO_SCRIPTS = SUPPORT_PACK_DIR / "repo" / "scripts"

if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
if str(REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPTS))

import tracksdata as td
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate as compute_metric, node_recall, per_sample_metrics
from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video, suppress_output


def run_single_validation(
    dataset_name: str,
    data_dir: Path,
    weights_path: Path,
    cfg: PredictConfig,
    device: torch.device,
    max_frames: int | None = None,
) -> dict[str, float | str | int | bool]:
    """1つのデータセットに対して推論・最適化・GT評価を実行する。"""
    print(f"\n========================================================")
    print(f" Validating: {dataset_name} (max_frames={max_frames})")
    print(f" Config: det_thresh={cfg.det_threshold}, use_ilp={cfg.use_ilp}, device={device}")
    print(f"========================================================")

    start_time = time.time()
    ds_path = data_dir / dataset_name
    gt_geff_path = data_dir / f"{dataset_name}.geff"

    # 1. モデルロード
    print("[1/4] Loading model & weights...")
    model, window_size, downsample = load_model(weights_path, device)

    # 2. 推論 (UNet検出 + Transformerエッジ推論)
    print(f"[2/4] Running predict_video (window_size={window_size}, downsample={downsample})...")
    infer_start = time.time()
    coords, edges = predict_video(
        model=model,
        ds_path=ds_path,
        device=device,
        cfg=cfg,
        window_size=window_size,
        max_frames=max_frames,
        unet_batch_size=1,
        downsample=downsample,
    )
    infer_elapsed = time.time() - infer_start
    print(f"  -> Inferred {len(coords)} nodes, {len(edges)} candidate edges in {infer_elapsed:.2f}s")

    # 3. グラフ構築 + ILP 最適化
    print("[3/4] Building graph and solving...")
    graph = build_graph(coords, edges)
    ilp_start = time.time()
    if cfg.use_ilp and graph.num_edges() > 0:
        print("  -> Solving ILP (tracksdata.solvers.ILPSolver)...")
        solver = td.solvers.ILPSolver(
            edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=cfg.ilp_appearance_weight,
            disappearance_weight=cfg.ilp_disappearance_weight,
            division_weight=cfg.ilp_division_weight,
        )
        with suppress_output():
            graph = solver.solve(graph)
        ilp_elapsed = time.time() - ilp_start
        print(f"  -> ILP solved in {ilp_elapsed:.2f}s. Final edges: {graph.num_edges()}")
    else:
        print(f"  -> Greedy mode or no edges. Final edges: {graph.num_edges()}")

    # 4. GT評価
    metrics_result: dict[str, float | str | int | bool] = {
        "dataset": dataset_name,
        "max_frames": max_frames if max_frames is not None else -1,
        "det_threshold": cfg.det_threshold,
        "use_ilp": cfg.use_ilp,
        "pred_nodes": graph.num_nodes(),
        "pred_edges": graph.num_edges(),
        "infer_time_sec": round(infer_elapsed, 2),
        "total_time_sec": round(time.time() - start_time, 2),
    }

    if gt_geff_path.exists():
        print("[4/4] Evaluating against Ground Truth...")
        import tempfile
        import shutil
        from biohub_tracking.io import save_graph

        ds_gt = open_dataset(ds_path, require_tracks=True, load_image=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_geff = Path(tmpdir) / "pred.geff"
            save_graph(graph, tmp_geff)
            pred_result = td.graph.IndexedRXGraph.from_geff(tmp_geff)
            pred_rx = pred_result[0] if isinstance(pred_result, tuple) else pred_result

        gt_tracks = ds_gt.tracks

        # max_frames 指定時は GT 側も t < max_frames に制限して公正に評価
        if max_frames is not None:
            print(f"  -> Filtering GT tracks to t < {max_frames} using tracksdata filter...")
            sub_view = gt_tracks.filter(td.NodeAttr("t") < max_frames).subgraph()
            with tempfile.TemporaryDirectory() as tmpdir_gt:
                tmp_gt_geff = Path(tmpdir_gt) / "gt_sub.geff"
                save_graph(sub_view, tmp_gt_geff)
                gt_res = td.graph.IndexedRXGraph.from_geff(tmp_gt_geff)
                gt_tracks = gt_res[0] if isinstance(gt_res, tuple) else gt_res

        er = compute_metric(pred_rx, gt_tracks, scale=ds_gt.scale)
        rec = node_recall(pred_rx, gt_tracks) if graph.num_edges() > 0 and graph.num_nodes() > 0 else 0.0

        # スコア計算
        edge_denom = er.edge_tp + er.edge_fp + er.edge_fn
        edge_jaccard = er.edge_tp / edge_denom if edge_denom > 0 else 0.0
        div_denom = er.division_tp + er.division_fp + er.division_fn
        division_jaccard = er.division_tp / div_denom if div_denom > 0 else 0.0
        score = edge_jaccard + 0.1 * division_jaccard

        print(f"\n--- [Evaluation Results: {dataset_name}] ---")
        print(f"  Node Recall       : {rec:.4f}")
        print(f"  Edge Jaccard      : {edge_jaccard:.4f} (TP={er.edge_tp}, FP={er.edge_fp}, FN={er.edge_fn})")
        print(f"  Division Jaccard  : {division_jaccard:.4f} (TP={er.division_tp}, FP={er.division_fp}, FN={er.division_fn})")
        print(f"  Total Score       : {score:.4f}")
        print(f"--------------------------------------------")

        metrics_result.update({
            "node_recall": round(float(rec), 4),
            "edge_jaccard": round(float(edge_jaccard), 4),
            "edge_tp": er.edge_tp,
            "edge_fp": er.edge_fp,
            "edge_fn": er.edge_fn,
            "division_jaccard": round(float(division_jaccard), 4),
            "division_tp": er.division_tp,
            "division_fp": er.division_fp,
            "division_fn": er.division_fn,
            "score": round(float(score), 4),
        })
    else:
        print("[4/4] GT not found, skipping metric evaluation.")

    return metrics_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 3D-UNet + Transformer + ILP locally.")
    parser.add_argument("--dataset", type=str, default="44b6_0113de3b", help="Dataset name to test.")
    parser.add_argument("--max-frames", type=int, default=3, help="Max frames to test (for fast CPU check).")
    parser.add_argument("--det-threshold", type=float, default=0.95, help="Detection threshold.")
    parser.add_argument("--use-ilp", action="store_true", default=True, help="Use ILP solver.")
    parser.add_argument("--no-ilp", action="store_false", dest="use_ilp", help="Disable ILP solver (greedy).")
    args = parser.parse_args()

    data_dir = BASE_DIR / "input" / "train"
    weights_path = SUPPORT_PACK_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=-1.0,
        ilp_appearance_weight=0.1,
        ilp_disappearance_weight=0.1,
        ilp_division_weight=1.0,
    )

    result = run_single_validation(
        dataset_name=args.dataset,
        data_dir=data_dir,
        weights_path=weights_path,
        cfg=cfg,
        device=device,
        max_frames=args.max_frames,
    )

    # 結果をCSVに追記保存
    out_csv = BASE_DIR / "github" / "s6_analysys_data" / "s6_026_validation_results.csv"
    import pandas as pd
    df_new = pd.DataFrame([result])
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.to_csv(out_csv, index=False)
    else:
        df_new.to_csv(out_csv, index=False)
    print(f"\nSaved validation result to {out_csv}")


if __name__ == "__main__":
    main()
