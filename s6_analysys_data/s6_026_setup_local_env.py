"""s6_026_setup_local_env.py

ローカル環境の検証とセットアップ確認スクリプト。
Python バージョン、CUDA/GPU の利用可否、必須ライブラリのインストール状況を検査する。
"""

import sys
from pathlib import Path


def check_python_environment() -> None:
    """Pythonバージョンとプラットフォームを出力する。"""
    print(f"[Python] Version: {sys.version}")
    print(f"[Python] Executable: {sys.executable}")
    print(f"[Platform] OS: {sys.platform}")


def check_torch_and_gpu() -> bool:
    """PyTorchおよびCUDA/GPUの利用可否を検査する。"""
    try:
        import torch
        print(f"[Torch] Version: {torch.__version__}")
        has_cuda = torch.cuda.is_available()
        print(f"[CUDA] Available: {has_cuda}")
        if has_cuda:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"[CUDA] Device Count: {device_count}")
            print(f"[CUDA] Device Name: {device_name}")
        return has_cuda
    except ImportError as e:
        print(f"[Torch] Error: {e}")
        return False


def check_dependencies() -> dict[str, bool]:
    """必須・関連ライブラリのインポート可否を検査する。"""
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "polars",
        "zarr",
        "numcodecs",
        "torch",
        "tracksdata",
        "pyscipopt",
        "ilpy",
        "geff",
        "rustworkx",
        "skimage",
        "tqdm",
    ]
    status: dict[str, bool] = {}
    print("\n[Package Inspection]")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  - {pkg:<15}: OK (v{version})")
            status[pkg] = True
        except ImportError as e:
            print(f"  - {pkg:<15}: Missing ({e})")
            status[pkg] = False
    return status


def check_support_pack_paths() -> None:
    """サポートパックの配置状況を検査する。"""
    base_dir = Path(r"c:\work\aaa\s6")
    support_pack_dir = base_dir / "input" / "support_pack"
    weights_path = support_pack_dir / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    repo_src = support_pack_dir / "repo" / "src"

    print("\n[Support Pack Inspection]")
    print(f"  - Support Pack Dir: {support_pack_dir} (Exists: {support_pack_dir.exists()})")
    print(f"  - Weights File: {weights_path} (Exists: {weights_path.exists()})")
    print(f"  - Repo Src Dir: {repo_src} (Exists: {repo_src.exists()})")


def main() -> None:
    print("=== s6_026 Local Environment Inspection ===")
    check_python_environment()
    has_gpu = check_torch_and_gpu()
    dep_status = check_dependencies()
    check_support_pack_paths()
    print("============================================")


if __name__ == "__main__":
    main()
