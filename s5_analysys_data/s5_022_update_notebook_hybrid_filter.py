# -*- coding: utf-8 -*-
import json
import ast

nb_path = "s5/github/working/s5_001_try_and_error.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 3: MAGIC_STRING
cell3_src = "".join(nb["cells"][3]["source"])
cell3_src = cell3_src.replace('MAGIC_STRING          : str = "015DYNRADIUS"', 'MAGIC_STRING          : str = "021HYBRID_BGSUB"')
nb["cells"][3]["source"] = [cell3_src]

# 2. Update Cell 8: BlobDogNodeDetector optics & 2-stage hybrid filter
cell8_src = "".join(nb["cells"][8]["source"])

# Locate analyze_frame_optics and _get_frame_params
start_marker = "    @staticmethod\n    def analyze_frame_optics("
end_marker = "        return img_norm, min_sig, max_sig, th, ov\n"

idx_start = cell8_src.find(start_marker)
idx_end = cell8_src.find(end_marker, idx_start) + len(end_marker)

if idx_start == -1 or idx_end <= idx_start:
    raise ValueError(f"Could not locate markers in Cell 8 (start: {idx_start}, end: {idx_end})")

new_code_block = '''    @staticmethod
    def analyze_frame_optics(frame: np.ndarray) -> dict:
        """サブサンプリング配列による高速 (~4ms) 光学特徴量計測"""
        from scipy.ndimage import gaussian_filter
        sub = frame[::2, ::2, ::2]
        p01, p25, p50, p75, p995 = np.percentile(sub, [1.0, 25.0, 50.0, 75.0, 99.5])
        bg_noise = float((p75 - p25) / 1.349)
        snr_proxy = float((p995 - p50) / (bg_noise + 1e-5))
        dyn_range = float(p995 - p01)
        fg_ratio = float((sub > (p50 + 3.0 * bg_noise)).mean())
        vol_max = float(sub.max())
        max_to_med = float(vol_max / (p50 + 1e-5))
        bg_lowpass = gaussian_filter(sub, sigma=(1.5, 4.0, 4.0))
        bg_gradient_std = float(bg_lowpass.std() / (p50 + 1e-5))
        return {
            "p01": float(p01),
            "p995": float(p995),
            "bg_median": float(p50),
            "bg_noise": float(bg_noise),
            "snr_proxy": float(snr_proxy),
            "dyn_range": float(dyn_range),
            "fg_ratio": fg_ratio,
            "vol_max": vol_max,
            "max_to_med": max_to_med,
            "bg_gradient_std": bg_gradient_std
        }

    def _get_frame_params(self, frame: np.ndarray):
        """
        フレームごとの光学特性プレ解析に基づく2段階ハイブリッド適応フィルター:
        - 第1段階: snr_proxy <= 12.0 or bg_median >= 80.0 (救済対象スクリーニング)
          - False -> 方式0: 現行維持 (Global Norm, th=0.035) [高画質・クリーン群]
          - True  -> 第2段階: 局所ムラ・平坦判定
            - bg_gradient_std >= 1.0 and max_to_med >= 14.0
              -> 方式1: 標準背景差分 (sigma=(2, 8, 8), th=0.018) [重症ムラ・巨大蛍光塊]
            - else
              -> 方式2: マイルド背景差分 (sigma=(4, 16, 16), th=0.025) [広域背景・細胞核削れ防止]
        """
        from scipy.ndimage import gaussian_filter

        if self.dynamic_adaptive:
            optics = self.analyze_frame_optics(frame)
            snr_proxy = optics["snr_proxy"]
            bg_median = optics["bg_median"]
            bg_gradient_std = optics["bg_gradient_std"]
            max_to_med = optics["max_to_med"]

            # 第1段階: 適応型背景差分ルール (広域スクリーニング)
            is_triggered = (snr_proxy <= 12.0) or (bg_median >= 80.0)

            if not is_triggered:
                # 方式0: 現行維持 (Global Normalization)
                p_low = max(optics["p01"], optics["bg_median"] - 2.0 * optics["bg_noise"])
                p_high_pct = 99.8 if optics["fg_ratio"] < 0.05 else 99.3
                p_high = np.percentile(frame[::2, ::2, ::2], p_high_pct)
                th = float(np.clip(self.th_min + self.th_slope * (snr_proxy - 2.0), self.th_min, self.th_max))
                frame_target = frame
            else:
                # 第2段階: 照明ムラ限定フィルター (局所ムラ・平坦判定)
                if (bg_gradient_std >= 1.0) and (max_to_med >= 14.0):
                    # 方式1: 標準背景差分 (急峻なガウシアン引き算で重症ムラを完全除去)
                    bg = gaussian_filter(frame, sigma=(2.0, 8.0, 8.0))
                    vol_sub = np.maximum(0.0, frame.astype(np.float32, copy=False) - bg)
                    p_low = np.percentile(vol_sub, 1.0)
                    p_high = np.percentile(vol_sub, 99.5)
                    th = 0.018
                    frame_target = vol_sub
                else:
                    # 方式2: マイルド背景差分 (広域背景のみを緩やかに除去し細胞核中心の削れを防止)
                    bg = gaussian_filter(frame, sigma=(4.0, 16.0, 16.0))
                    vol_sub = np.maximum(0.0, frame.astype(np.float32, copy=False) - bg)
                    p_low = np.percentile(vol_sub, 1.0)
                    p_high = np.percentile(vol_sub, 99.5)
                    th = 0.025
                    frame_target = vol_sub

            min_sig = self.min_sigma
            max_sig = self.max_sigma
            ov = self.overlap_dense if optics["fg_ratio"] > 0.08 else self.overlap_sparse
        else:
            optics = {}
            p_low, p_high = np.percentile(frame, self.percentile_range)
            th = self.threshold
            min_sig = self.min_sigma
            max_sig = self.max_sigma
            ov = 0.50
            frame_target = frame

        if p_high > p_low:
            img_norm = np.clip((frame_target.astype(np.float32, copy=False) - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0).astype(np.float32)
        else:
            img_norm = np.zeros_like(frame, dtype=np.float32)

        return img_norm, min_sig, max_sig, th, ov
'''

cell8_src_new = cell8_src[:idx_start] + new_code_block + cell8_src[idx_end:]
nb["cells"][8]["source"] = [cell8_src_new]

# Verify AST syntax for all cells
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        c_src = "".join(cell.get("source", []))
        # filter IPython magic lines (% or !)
        clean_lines = []
        for line in c_src.split("\n"):
            if line.strip().startswith("%") or line.strip().startswith("!"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        ast.parse("\n".join(clean_lines))

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("SUCCESS: Notebook updated and all code cells compiled without error!")
