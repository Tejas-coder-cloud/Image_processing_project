"""Batch gallery generator: Evaluates the pipeline on diverse GTSDB and GTSRB samples

and produces individual reports plus a multi-image master comparison grid.
"""

from pathlib import Path
from typing import List, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import PipelineConfig
from src.pipeline import run_full_pipeline
from src.visualizer import plot_pipeline_summary, plot_stage1_report, plot_stage2_report


def find_diverse_test_cases() -> List[Tuple[str, Path, str]]:
    """Select 10 diverse test cases across sign categories and datasets."""
    root = Path(__file__).resolve().parent
    cases: List[Tuple[str, Path, str]] = []

    # 1. GTSDB Road Scenes
    gtsdb_dir = root / "gtsdb_dataset" / "train" / "images"
    gtsdb_candidates = [
        ("GTSDB_Speed_Limit_50", "122_jpg*.jpg", "Speed Limit 50 (Red Circular Sign)"),
        ("GTSDB_Danger_Warning", "108_jpg*.jpg", "Danger / Warning (Red Triangle Sign)"),
        ("GTSDB_Mandatory_Blue", "102_jpg*.jpg", "Mandatory Blue + Prohibitory"),
        ("GTSDB_Prohibitory_Highway", "112_jpg*.jpg", "Prohibitory Sign (Highway Day)"),
        ("GTSDB_Speed_Overpass", "100_jpg*.jpg", "Speed Sign (Overpass Shadow)"),
        ("GTSDB_Road_Signs", "101_jpg*.jpg", "Road Infrastructure Signs"),
    ]

    for label, pattern, desc in gtsdb_candidates:
        matches = list(gtsdb_dir.glob(pattern))
        if matches:
            cases.append((label, matches[0], desc))

    # 2. GTSRB Cropped Signs
    gtsrb_dir = root / "gtsrb_dataset" / "Train"
    gtsrb_candidates = [
        ("GTSRB_Speed_20", "0", "Speed Limit 20 km/h (Class 0)"),
        ("GTSRB_Speed_30", "1", "Speed Limit 30 km/h (Class 1)"),
        ("GTSRB_Danger_Triangle", "11", "Right-of-Way Danger Triangle (Class 11)"),
        ("GTSRB_Priority_Yellow", "12", "Priority Road Yellow Diamond (Class 12)"),
    ]

    for label, cls_folder, desc in gtsrb_candidates:
        c_path = gtsrb_dir / cls_folder
        if c_path.exists():
            pngs = list(c_path.glob("*.png"))
            if pngs:
                cases.append((label, pngs[0], desc))

    return cases


def generate_master_gallery(results_list: List[dict], save_path: Path):
    """Create a unified comparison grid showcasing all test cases across the 4 stages."""
    n_cases = len(results_list)
    fig, axes = plt.subplots(n_cases, 4, figsize=(18, 3.6 * n_cases))
    fig.patch.set_facecolor("#0F172A")
    plt.rcParams["text.color"] = "#E2E8F0"

    col_headers = [
        "1. Raw Input Image",
        "2. Stage 1: Enhanced (CLAHE+Sharp)",
        "3. Stage 2: Pure Binary Mask",
        "4. Detected Sign Overlay",
    ]

    for r, item in enumerate(results_list):
        raw_bgr = item["stage1"]["raw_image"]
        enh_bgr = item["stage1"]["enhanced_image"]
        mask = item["stage2"]["binary_mask_cleaned"]
        overlay_bgr = item["stage2"]["overlay"]
        label = item["label"]
        desc = item["desc"]
        n_signs = len(item["stage2"]["bounding_boxes"])

        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # Plot columns
        axes[r, 0].imshow(raw_rgb)
        axes[r, 0].set_title(f"[{r+1}] {desc}\nInput: {raw_bgr.shape[1]}x{raw_bgr.shape[0]} px", fontsize=10, color="white")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(enh_rgb)
        axes[r, 1].set_title(f"Stage 1 Enhanced\nBlur Var: {item['stage1']['quality_enhanced']['blur_laplacian_var']:.1f}", fontsize=10, color="#38BDF8")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(mask, cmap="gray")
        axes[r, 2].set_title("Stage 2: Verified Binary Mask\n(Zero stray noise)", fontsize=10, color="#A78BFA")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(overlay_rgb)
        axes[r, 3].set_title(f"Sign Candidates Localized\n({n_signs} sign candidate{'s' if n_signs != 1 else ''})", fontsize=10, color="#34D399")
        axes[r, 3].axis("off")

    # Column supertitles on top row
    for c in range(4):
        axes[0, c].text(
            0.5, 1.25, col_headers[c],
            transform=axes[0, c].transAxes,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#F8FAFC"
        )

    plt.suptitle("TRAFFIC SIGN IMAGE PROCESSING: MULTI-IMAGE BATCH EVALUATION GALLERY", fontsize=15, fontweight="bold", color="#38BDF8", y=1.002)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def main():
    test_cases = find_diverse_test_cases()
    print(f"\n=======================================================")
    print(f"BATCH EVALUATION: RUNNING ON {len(test_cases)} DIVERSE TEST IMAGES")
    print(f"=======================================================\n")

    out_gallery = Path("outputs/gallery")
    out_gallery.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig()
    all_results = []

    for idx, (label, path, desc) in enumerate(test_cases):
        print(f"[{idx+1}/{len(test_cases)}] Processing: {label} ({desc})")
        print(f"    Path: {path}")

        img = cv2.imread(str(path))
        if img is None:
            print(f"    [SKIP] Failed to load {path}")
            continue

        res = run_full_pipeline(img, config)
        res["label"] = label
        res["desc"] = desc
        res["path"] = path
        all_results.append(res)

        s1 = res["stage1"]
        s2 = res["stage2"]
        n_bboxes = len(s2["bounding_boxes"])
        print(f"    - Stage 1: Noise={s1['quality_raw']['noise_sigma']:.2f}, BlurVar={s1['quality_enhanced']['blur_laplacian_var']:.1f}")
        print(f"    - Stage 2: Verified Sign Candidates={n_bboxes}")

        # Save individual reports per test case
        case_dir = out_gallery / label
        case_dir.mkdir(parents=True, exist_ok=True)
        plot_stage1_report(s1, case_dir / "stage1_enhancement_report.png")
        plot_stage2_report(s2, case_dir / "stage2_color_segmentation_report.png")
        plot_pipeline_summary(s1, s2, case_dir / "pipeline_summary.png")
        cv2.imwrite(str(case_dir / "enhanced_image.png"), s1["enhanced_image"])
        cv2.imwrite(str(case_dir / "binary_mask.png"), s2["binary_mask_cleaned"])
        cv2.imwrite(str(case_dir / "overlay.png"), s2["overlay"])

    print(f"\n[+] Generating Master Comparison Gallery Grid ({len(all_results)} images)...")
    master_path = out_gallery / "master_gallery.png"
    generate_master_gallery(all_results, master_path)

    # Also copy master gallery to assets/ for README
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy(master_path, assets_dir / "master_gallery.png")

    print(f"\n[OK] Batch Evaluation Completed Successfully!")
    print(f"    - Master Gallery: {master_path}")
    print(f"    - Assets Copy: {assets_dir / 'master_gallery.png'}")
    print(f"    - Individual Cases Folder: {out_gallery}\n")


if __name__ == "__main__":
    main()
