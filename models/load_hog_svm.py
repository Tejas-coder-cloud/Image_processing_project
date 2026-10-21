"""Helper script for loading and using the trained HOG+SVM verifier.

This script demonstrates how to:
1. Load the trained SVM model and scaler
2. Extract HOG features from a crop
3. Obtain decision scores and predictions

Usage:
    python load_hog_svm.py --crop_path path/to/crop.png
"""

import argparse
import json
import cv2
import numpy as np
import joblib
from pathlib import Path


def load_model(model_dir: Path):
    """Load the trained SVM model and scaler."""
    svm = joblib.load(model_dir / 'svm_model.joblib')
    scaler = joblib.load(model_dir / 'scaler.joblib')
    
    with open(model_dir / 'hog_config.json', 'r') as f:
        hog_config = json.load(f)
    
    return svm, scaler, hog_config


def extract_hog_features(image_path: Path, hog_config: dict) -> np.ndarray:
    """Extract HOG features from a single image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    hog = cv2.HOGDescriptor(
        _winSize=tuple(hog_config['window_size']),
        _blockSize=(hog_config['pixels_per_cell'][0] * hog_config['cells_per_block'][0],
                   hog_config['pixels_per_cell'][1] * hog_config['cells_per_block'][1]),
        _blockStride=(hog_config['pixels_per_cell'][0], hog_config['pixels_per_cell'][1]),
        _cellSize=tuple(hog_config['pixels_per_cell']),
        _nbins=hog_config['orientations'],
    )
    
    features = hog.compute(img).flatten()
    return features


def predict_crop(model_dir: Path, crop_path: Path):
    """Load model and predict on a single crop."""
    svm, scaler, hog_config = load_model(model_dir)
    
    # Extract features
    features = extract_hog_features(crop_path, hog_config)
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Get decision score
    decision_score = svm.decision_function(features_scaled)[0]
    
    # Get prediction
    prediction = svm.predict(features_scaled)[0]
    
    # Get probability
    probability = svm.predict_proba(features_scaled)[0]
    
    return {
        'decision_score': decision_score,
        'prediction': 'sign' if prediction == 1 else 'background',
        'probability_sign': probability[1] if prediction == 1 else probability[0],
        'probability_background': probability[0] if prediction == 1 else probability[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Load and use HOG+SVM verifier')
    parser.add_argument('--model_dir', type=Path, default=Path('experiments_phase_n1/hog_svm_v1'),
                        help='Directory containing trained model (default: experiments_phase_n1/hog_svm_v1)')
    parser.add_argument('--crop_path', type=Path, required=True,
                        help='Path to crop image to classify')
    
    args = parser.parse_args()
    
    result = predict_crop(args.model_dir, args.crop_path)
    
    print("Prediction Results:")
    print(f"Prediction: {result['prediction']}")
    print(f"Decision score: {result['decision_score']:.4f}")
    print(f"Probability (sign): {result['probability_sign']:.4f}")
    print(f"Probability (background): {result['probability_background']:.4f}")


if __name__ == '__main__':
    main()
