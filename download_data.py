#!/usr/bin/env python3
"""
Download GTSDB Dataset

This script downloads the German Traffic Sign Detection Benchmark dataset.
Run this script after cloning the repository to set up the data.

The dataset will be downloaded to: data/TrainIJCNN2013/
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# Dataset URL
GTSDB_URL = "https://sid.erda.net/public/sid/D32E2FCB-7CC9-4903-9B32-2E46969A9DD8/TrainIJCNN2013.zip"

# Paths
DATA_DIR = Path("data")
DOWNLOAD_PATH = DATA_DIR / "TrainIJCNN2013.zip"
EXTRACT_PATH = DATA_DIR / "TrainIJCNN2013"


def download_dataset():
    """Download GTSDB dataset."""
    print("=" * 80)
    print("GTSDB Dataset Download")
    print("=" * 80)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading from: {GTSDB_URL}")
    print(f"Target: {DOWNLOAD_PATH}")
    
    try:
        urllib.request.urlretrieve(GTSDB_URL, DOWNLOAD_PATH)
        print(f"\nDownload complete: {DOWNLOAD_PATH}")
    except Exception as e:
        print(f"\nERROR: Failed to download dataset: {e}")
        print("Please download manually from:")
        print(f"  {GTSDB_URL}")
        return False
    
    # Extract
    print(f"\nExtracting to: {EXTRACT_PATH}")
    try:
        with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete")
    except Exception as e:
        print(f"ERROR: Failed to extract: {e}")
        return False
    
    # Cleanup
    print(f"\nCleaning up: {DOWNLOAD_PATH}")
    DOWNLOAD_PATH.unlink()
    
    print("\n" + "=" * 80)
    print("Dataset setup complete!")
    print("=" * 80)
    print(f"\nDataset location: {EXTRACT_PATH}")
    print(f"Images: {len(list(EXTRACT_PATH.glob('*.ppm')))}")
    print(f"Annotations: {EXTRACT_PATH / 'gt.txt'}")
    
    return True


if __name__ == '__main__':
    download_dataset()
