"""Command-line runner for Stage 1 (Enhancement) and Stage 2 (Color Segmentation)."""

import argparse
from pathlib import Path
import sys
from typing import Optional
import cv2

from src.config import PipelineConfig, DatasetConfig
from src.pipeline import run_full_pipeline
from src.visualizer import plot_pipeline_summary, plot_stage1_report, plot_stage2_report


def find_default_sample(dataset_type: str = "gtsdb", config_paths: Optional[DatasetConfig] = None) -> Path:
    """Find a representative image from GTSDB or GTSRB datasets."""
    root = Path(__file__).resolve().parent
    paths = config_paths or DatasetConfig()

    if dataset_type.lower() == "gtsdb":
        search_dirs = [
            root / paths.gtsdb_dir,
            paths.gtsdb_dir,
            root / "gtsdb_dataset" / "train" / "images",
        ]
        for d in search_dirs:
            if d.exists():
                images = list(d.glob("*.ppm")) + list(d.glob("*.jpg")) + list(d.glob("*.png"))
                if images:
                    return images[0]

    elif dataset_type.lower() == "gtsrb":
        search_dirs = [
            root / paths.gtsrb_dir,
            paths.gtsrb_dir,
            root / "gtsrb_dataset" / "Train",
        ]
        for d in search_dirs:
            if d.exists():
                for class_id in ["14", "1", "0", "13"]:
                    c_dir = d / class_id
                    if c_dir.exists():
                        images = list(c_dir.glob("*.png")) + list(c_dir.glob("*.ppm"))
                        if images:
                            return images[0]

    sys.exit(f"Could not find sample images in {dataset_type} dataset folder.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 (Image Enhancement) and Stage 2 (Color Segmentation) Pipeline."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image. If omitted, picks a default sample from GTSDB/GTSRB.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gtsdb",
        choices=["gtsdb", "gtsrb"],
        help="Dataset to sample from if --image is not provided (default: gtsdb).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated visual reports and output masks.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.15,
        help="Gamma correction factor for Stage 1 (default: 1.15).",
    )
    parser.add_argument(
        "--clahe_clip",
        type=float,
        default=2.5,
        help="CLAHE clip limit in CIELAB space (default: 2.5).",
    )

    args = parser.parse_args()

    # Determine input image
    if args.image:
        image_path = Path(args.image)
    else:
        image_path = find_default_sample(args.dataset)

    if not image_path.exists():
        sys.exit(f"Error: Image not found at {image_path}")

    print(f"\n=======================================================")
    print(f"TRAFFIC SIGN IMAGE PROCESSING PIPELINE")
    print(f"=======================================================")
    print(f"Input image: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        sys.exit(f"Error: Unable to load image file {image_path}")

    h, w, c = image_bgr.shape
    print(f"Dimensions: {w} x {h} pixels ({c} channels)")

    # Configure pipeline
    config = PipelineConfig()
    config.stage1.gamma = args.gamma
    config.stage1.clahe_clip_limit = args.clahe_clip

    print("\n[+] Executing Stage 1: Image Enhancement...")
    results = run_full_pipeline(image_bgr, config)

    s1 = results["stage1"]
    q_raw = s1["quality_raw"]
    q_enh = s1["quality_enhanced"]

    print(f"    - Noise std (Immerkaer): {q_raw['noise_sigma']:.3f} -> {q_enh['noise_sigma']:.3f}")
    print(f"    - Blur variance (Laplacian): {q_raw['blur_laplacian_var']:.1f}")
    print(f"    - RMS Contrast: {q_raw['rms_contrast']:.1f} -> {q_enh['rms_contrast']:.1f}")

    print("\n[+] Executing Stage 2: Color Segmentation...")
    s2 = results["stage2"]
    bboxes = s2["bounding_boxes"]
    print(f"    - Candidate sign regions localized: {len(bboxes)}")
    for idx, (bx, by, bw, bh) in enumerate(bboxes[:5]):
        print(f"      Sign Candidate #{idx+1}: Bounding Box (x={bx}, y={by}, w={bw}, h={bh})")
    if len(bboxes) > 5:
        print(f"      ... and {len(bboxes) - 5} more.")

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save PNG files
    cv2.imwrite(str(out_dir / "enhanced_image.png"), s1["enhanced_image"])
    cv2.imwrite(str(out_dir / "binary_mask.png"), s2["binary_mask_cleaned"])
    cv2.imwrite(str(out_dir / "segmentation_overlay.png"), s2["overlay"])

    print("\n[+] Generating Visual Diagnostic Reports...")
    plot_stage1_report(s1, out_dir / "stage1_enhancement_report.png")
    plot_stage2_report(s2, out_dir / "stage2_color_segmentation_report.png")
    plot_pipeline_summary(s1, s2, out_dir / "pipeline_summary.png")

    print(f"\n[OK] Done! Reports and outputs successfully saved to:")
    print(f"    - {out_dir / 'stage1_enhancement_report.png'}")
    print(f"    - {out_dir / 'stage2_color_segmentation_report.png'}")
    print(f"    - {out_dir / 'pipeline_summary.png'}")
    print(f"    - {out_dir / 'enhanced_image.png'}")
    print(f"    - {out_dir / 'binary_mask.png'}")
    print(f"    - {out_dir / 'segmentation_overlay.png'}\n")


if __name__ == "__main__":
    main()
