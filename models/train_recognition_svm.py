"""Train Multi-Class SVM for Sign Recognition

This script trains a multi-class SVM on GTSRB dataset for sign recognition.
The trained model will classify detected traffic signs into 43 sign classes.

Usage:
    python models/train_recognition_svm.py
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Configuration
RANDOM_SEED = 42
OUTPUT_DIR = Path('models/recognition_svm_v1')
GTSRB_DIR = Path('data/Train')
TEST_SIZE = 0.2
IMG_SIZE = (64, 64)  # Resize to 64x64 for HOG


def load_gtsrb_dataset(gtsrb_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Load GTSRB dataset from folder structure."""
    print("Loading GTSRB dataset...")
    
    images = []
    labels = []
    class_ids = []
    
    # Class folders in GTSRB (0-42, 43 classes total)
    class_folders = sorted([d for d in gtsrb_dir.iterdir() if d.is_dir()])
    
    for class_folder in class_folders:
        class_id = int(class_folder.name)
        class_ids.append(class_id)
        
        # Load all images in this class folder
        image_files = list(class_folder.glob('*.png'))
        print(f"  Class {class_id}: {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize to standard size
                img_resized = cv2.resize(img, IMG_SIZE)
                images.append(img_resized)
                labels.append(class_id)
    
    print(f"Total images loaded: {len(images)}")
    print(f"Total classes: {len(class_ids)}")
    
    return np.array(images), np.array(labels), class_ids


def extract_hog_features_batch(images: np.ndarray, hog_config: dict) -> np.ndarray:
    """Extract HOG features from a batch of images."""
    print("Extracting HOG features...")
    
    hog = cv2.HOGDescriptor(
        _winSize=tuple(hog_config['window_size']),
        _blockSize=(hog_config['pixels_per_cell'][0] * hog_config['cells_per_block'][0],
                   hog_config['pixels_per_cell'][1] * hog_config['cells_per_block'][1]),
        _blockStride=tuple(hog_config['pixels_per_cell']),
        _cellSize=tuple(hog_config['pixels_per_cell']),
        _nbins=hog_config['orientations'],
    )
    
    features_list = []
    for i, img in enumerate(images):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{len(images)}")
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG
        features = hog.compute(img_gray).flatten()
        features_list.append(features)
    
    return np.array(features_list)


def train_multi_class_svm(X_train, y_train):
    """Train multi-class SVM."""
    print("Training multi-class SVM...")
    
    # Use RBF kernel with balanced class weights
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        random_state=RANDOM_SEED,
        decision_function_shape='ovr'  # One-vs-rest for multi-class
    )
    
    svm.fit(X_train, y_train)
    print("SVM training complete")
    
    return svm


def main():
    print("=" * 80)
    print("Multi-Class SVM Training for Sign Recognition")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # HOG configuration
    hog_config = {
        "window_size": [64, 64],
        "orientations": 9,
        "pixels_per_cell": [8, 8],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys",
        "transform_sqrt": False,
    }
    
    # Load dataset
    images, labels, class_ids = load_gtsrb_dataset(GTSRB_DIR)
    
    # Extract HOG features
    features = extract_hog_features_batch(images, hog_config)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train SVM
    svm = train_multi_class_svm(X_train_scaled, y_train)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = svm.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))
    
    # Save confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'class_ids': class_ids
    }
    
    # Save model and metadata
    print(f"\nSaving model to: {OUTPUT_DIR}")
    joblib.dump(svm, OUTPUT_DIR / 'recognition_svm_model.joblib')
    joblib.dump(scaler, OUTPUT_DIR / 'recognition_scaler.joblib')
    
    with open(OUTPUT_DIR / 'recognition_hog_config.json', 'w') as f:
        json.dump(hog_config, f, indent=2)
    
    with open(OUTPUT_DIR / 'recognition_training_info.json', 'w') as f:
        json.dump({
            'num_classes': len(class_ids),
            'class_ids': class_ids,
            'total_samples': len(images),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'validation_accuracy': float(accuracy),
            'test_size': TEST_SIZE,
            'random_seed': RANDOM_SEED
        }, f, indent=2)
    
    with open(OUTPUT_DIR / 'recognition_confusion_matrix.json', 'w') as f:
        json.dump(cm_dict, f, indent=2)
    
    # Save training report
    report = classification_report(y_val, y_pred, zero_division=0, output_dict=True)
    with open(OUTPUT_DIR / 'recognition_training_report.md', 'w') as f:
        f.write("# Multi-Class SVM Recognition Training Report\n\n")
        f.write(f"## Training Summary\n\n")
        f.write(f"- Number of classes: {len(class_ids)}\n")
        f.write(f"- Total samples: {len(images)}\n")
        f.write(f"- Train samples: {len(X_train)}\n")
        f.write(f"- Validation samples: {len(X_val)}\n")
        f.write(f"- Validation accuracy: {accuracy:.4f}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(classification_report(y_val, y_pred, zero_division=0))
        f.write("\n```\n")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
