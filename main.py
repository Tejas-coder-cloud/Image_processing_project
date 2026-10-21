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
    
    # Run the appropriate C8 evaluator
    if mode == "validation":
        print("\nRunning on 98-image validation set...")
        subprocess.run([sys.executable, "evaluation/run_c8_custom_image_list.py", "--validation"])
    
    elif mode == "combined":
        print("\nRunning on 118-image combined set (validation + held-out)...")
        subprocess.run([sys.executable, "evaluation/run_c8_custom_image_list.py", "--combined"])
    
    elif mode == "heldout":
        print("\nRunning on 20-image held-out set...")
        subprocess.run([sys.executable, "evaluation/run_c8_custom_image_list.py", "--heldout"])
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: validation, combined, heldout")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Pipeline execution complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
