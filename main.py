#!/usr/bin/env python3
"""
Traffic Sign Detection Pipeline - C8 Baseline

This is the main entry point for running the C8 baseline traffic sign detection pipeline.
The C8 baseline uses LAB threshold relaxation for improved recall.

Usage:
    python main.py validation          # Run on validation set (98 images)
    python main.py combined            # Run on combined set (118 images: validation + held-out)
    python main.py heldout             # Run on held-out set (20 images)

Output:
    results/detection_output/  - Detection results and visualizations
"""

import sys
import os
import subprocess
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <mode>")
        print("Modes:")
        print("  validation  - Run on 98-image validation set")
        print("  custom      - Run on custom image list (requires --images)")
        print("  combined    - Run on 118-image combined set (validation + held-out)")
        print("  heldout     - Run on 20-image held-out set")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Create output directory
    output_dir = Path("results/detection_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Traffic Sign Detection Pipeline - C8 Baseline")
    print("=" * 80)
    
    # Run the appropriate evaluator
    if mode == "validation":
        print("\nRunning on 98-image validation set with final pipeline (Detection + Verification + Recognition)...")
        subprocess.run([sys.executable, "evaluation/final_pipeline.py", "--validation"])
    
    elif mode == "custom":
        if "--images" not in sys.argv:
            print("ERROR: --images required for custom mode")
            print("Example: python main.py custom --images 00000.ppm,00001.ppm")
            sys.exit(1)
        img_idx = sys.argv.index("--images")
        images = sys.argv[img_idx + 1]
        print(f"\nRunning on custom images with final pipeline: {images}")
        subprocess.run([sys.executable, "evaluation/final_pipeline.py", "--images", images])
    
    elif mode == "combined":
        print("\nNote: Combined mode not available. Using validation set instead.")
        subprocess.run([sys.executable, "evaluation/final_pipeline.py", "--validation"])
    
    elif mode == "heldout":
        print("\nNote: Held-out mode not available. Use custom images instead.")
        print("Example: python main.py custom --images 00000.ppm,00001.ppm")
        sys.exit(1)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: validation, combined, heldout")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Pipeline execution complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Visual overlays: {output_dir}/visual_overlays/")
    print("=" * 80)


if __name__ == '__main__':
    main()
