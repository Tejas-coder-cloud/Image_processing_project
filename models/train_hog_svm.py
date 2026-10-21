"""Phase N1.2: HOG Feature Extraction and Binary SVM Training.

This script:
1. Loads crops prepared in Phase N1.1
2. Extracts HOG features using the recorded configuration
3. Trains a classical SVM binary classifier (sign vs background)
4. Evaluates on validation set with metrics and confusion matrix
5. Saves the trained model and all metadata for later integration

Output directory: experiments_phase_n1/hog_svm_v1/
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
import sklearn

# Configuration
RANDOM_SEED = 42
OUTPUT_DIR = Path('models/hog_svm_v1')


def load_hog_config(config_path: Path) -> Dict[str, Any]:
    """Load HOG configuration from JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_crop_metadata(metadata_path: Path) -> List[Dict]:
    """Load crop metadata from CSV."""
    crops = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            crops.append(row)
    return crops


def extract_hog_features(
    image_path: Path,
    config: Dict[str, Any],
) -> np.ndarray:
    """Extract HOG features from a single image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # OpenCV HOG requires uint8 input
    # Note: transform_sqrt is not natively supported by OpenCV HOG
    # We skip it here since it would require custom implementation
    
    hog = cv2.HOGDescriptor(
        _winSize=tuple(config['window_size']),
        _blockSize=(config['pixels_per_cell'][0] * config['cells_per_block'][0],
                   config['pixels_per_cell'][1] * config['cells_per_block'][1]),
        _blockStride=(config['pixels_per_cell'][0], config['pixels_per_cell'][1]),
        _cellSize=tuple(config['pixels_per_cell']),
        _nbins=config['orientations'],
    )
    
    features = hog.compute(img).flatten()
    return features


def prepare_datasets(
    crops_dir: Path,
    metadata: List[Dict],
    hog_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training and validation datasets with HOG features.
    
    Returns (X_train, y_train, X_val, y_val)
    """
    print("Extracting HOG features...")
    
    # Verify feature dimension
    hog = cv2.HOGDescriptor(
        _winSize=tuple(hog_config['window_size']),
        _blockSize=(hog_config['pixels_per_cell'][0] * hog_config['cells_per_block'][0],
                   hog_config['pixels_per_cell'][1] * hog_config['cells_per_block'][1]),
        _blockStride=(hog_config['pixels_per_cell'][0], hog_config['pixels_per_cell'][1]),
        _cellSize=tuple(hog_config['pixels_per_cell']),
        _nbins=hog_config['orientations'],
    )
    feature_dim = hog.getDescriptorSize()
    print(f"Feature dimension: {feature_dim}")
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    
    for crop in metadata:
        split = crop['split']
        label = crop['label']
        crop_id = int(crop['crop_id'])
        
        # Determine crop path
        label_str = 'sign' if label == 'sign' else 'background'
        crop_path = crops_dir / split / label_str / f"crop_{crop_id:06d}.png"
        
        # Extract HOG features
        try:
            features = extract_hog_features(crop_path, hog_config)
        except Exception as e:
            print(f"Warning: Failed to extract features for {crop_path}: {e}")
            continue
        
        # Assign label
        y = 1 if label == 'sign' else -1
        
        if split == 'train':
            X_train.append(features)
            y_train.append(y)
        else:
            X_val.append(features)
            y_val.append(y)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature shape: {X_train.shape}")
    
    return X_train, y_train, X_val, y_val, feature_dim


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
) -> SVC:
    """Train an SVM classifier with RBF kernel."""
    np.random.seed(random_seed)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train SVM
    print("Training SVM...")
    svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=random_seed,
    )
    svm.fit(X_train_scaled, y_train)
    
    print("SVM training complete.")
    
    return svm, scaler


def evaluate_model(
    svm: SVC,
    scaler: StandardScaler,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate model on validation set."""
    X_val_scaled = scaler.transform(X_val)
    
    # Get decision scores
    decision_scores = svm.decision_function(X_val_scaled)
    
    # Apply threshold
    y_pred = (decision_scores >= threshold).astype(int)
    y_pred = np.where(y_pred == 1, 1, -1)  # Convert to -1/1 labels
    
    # Metrics
    cm = confusion_matrix(y_val, y_pred, labels=[-1, 1])
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=[-1, 1], average='binary', pos_label=1
    )
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics for each class
    precision_bg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_bg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_bg = 2 * precision_bg * recall_bg / (precision_bg + recall_bg) if (precision_bg + recall_bg) > 0 else 0
    
    precision_sign = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_sign = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_sign = 2 * precision_sign * recall_sign / (precision_sign + recall_sign) if (precision_sign + recall_sign) > 0 else 0
    
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': false_positive_rate,
        'decision_threshold': threshold,
        'precision_background': precision_bg,
        'recall_background': recall_bg,
        'f1_background': f1_bg,
        'precision_sign': precision_sign,
        'recall_sign': recall_sign,
        'f1_sign': f1_sign,
        'decision_scores': decision_scores,
        'y_true': y_val,
        'y_pred': y_pred,
    }


def main():
    # Set paths
    root = Path('.').resolve()
    n1_dir = root / 'experiments_phase_n1'
    data_prep_dir = n1_dir / 'data_preparation'
    crops_dir = data_prep_dir / 'crops'
    metadata_path = data_prep_dir / 'crop_metadata.csv'
    hog_config_path = data_prep_dir / 'hog_config.json'
    
    # Output directory
    out_dir = n1_dir / 'hog_svm_v1'
    if out_dir.exists():
        print(f"ERROR: Output directory already exists: {out_dir}")
        print("Please remove it or specify a different location.")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase N1.2: HOG Feature Extraction and SVM Training")
    print("=" * 80)
    
    # Load configuration and metadata
    print("\nStep 1: Loading configuration and metadata...")
    hog_config = load_hog_config(hog_config_path)
    print(f"HOG config: {hog_config}")
    
    metadata = load_crop_metadata(metadata_path)
    print(f"Total crops in metadata: {len(metadata)}")
    
    # Validate no evaluation images
    print("\nStep 2: Validating no evaluation images in metadata...")
    eval_images = {f'{i:05d}.ppm' for i in range(20)}
    eval_in_metadata = set()
    for crop in metadata:
        if crop['image_name'] in eval_images:
            eval_in_metadata.add(crop['image_name'])
    
    if eval_in_metadata:
        print(f"ERROR: Found {len(eval_in_metadata)} evaluation images in metadata: {eval_in_metadata}")
        return
    
    print("Validation passed: No evaluation images in crop metadata.")
    
    # Prepare datasets
    print("\nStep 3: Preparing datasets and extracting HOG features...")
    X_train, y_train, X_val, y_val, feature_dim = prepare_datasets(crops_dir, metadata, hog_config)
    
    # Train SVM
    print("\nStep 4: Training SVM classifier...")
    svm, scaler = train_svm(X_train, y_train, random_seed=RANDOM_SEED)
    
    # Evaluate on validation set
    print("\nStep 5: Evaluating on validation set...")
    results = evaluate_model(svm, scaler, X_val, y_val, threshold=0.0)  # Use 0.0 threshold for balanced SVM
    
    print("\n" + "=" * 80)
    print("Validation Results (Crop-Level Classification)")
    print("=" * 80)
    print(f"Decision threshold: {results['decision_threshold']}")
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"              -1 (BG)   +1 (Sign)")
    print(f"Actual -1 (BG)  {results['confusion_matrix'][0,0]:>6}     {results['confusion_matrix'][0,1]:>8}")
    print(f"Actual +1 (Sign) {results['confusion_matrix'][1,0]:>6}     {results['confusion_matrix'][1,1]:>8}")
    
    print(f"\nOverall Metrics (sign class as positive):")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"Background (negative class):")
    print(f"  Precision: {results['precision_background']:.4f}")
    print(f"  Recall: {results['recall_background']:.4f}")
    print(f"  F1: {results['f1_background']:.4f}")
    print(f"Sign (positive class):")
    print(f"  Precision: {results['precision_sign']:.4f}")
    print(f"  Recall: {results['recall_sign']:.4f}")
    print(f"  F1: {results['f1_sign']:.4f}")
    
    # Save model and artifacts
    print("\nStep 6: Saving model and artifacts...")
    
    # Save SVM model
    model_path = out_dir / 'svm_model.joblib'
    joblib.dump(svm, model_path)
    print(f"SVM model saved: {model_path}")
    
    # Save scaler
    scaler_path = out_dir / 'scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # Save HOG config (without transform_sqrt since OpenCV doesn't support it)
    hog_config_for_output = hog_config.copy()
    hog_config_for_output['transform_sqrt'] = False  # Override to match actual implementation
    hog_config_out = out_dir / 'hog_config.json'
    with open(hog_config_out, 'w') as f:
        json.dump(hog_config_for_output, f, indent=2)
    print(f"HOG config saved: {hog_config_out}")
    
    # Save training info
    training_info = {
        'random_seed': RANDOM_SEED,
        'feature_dimension': feature_dim,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_sign': int(np.sum(y_train == 1)),
        'train_background': int(np.sum(y_train == -1)),
        'val_sign': int(np.sum(y_val == 1)),
        'val_background': int(np.sum(y_val == -1)),
        'svm_params': {
            'kernel': svm.kernel,
            'C': svm.C,
            'gamma': svm.gamma,
            'class_weight': svm.class_weight,
            'probability': svm.probability,
        },
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__,
        'sklearn_version': __import__('sklearn').__version__,
    }
    
    training_info_path = out_dir / 'training_info.json'
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"Training info saved: {training_info_path}")
    
    # Save validation metrics
    metrics_path = out_dir / 'validation_metrics.json'
    metrics_to_save = {
        'decision_threshold': results['decision_threshold'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'false_positive_rate': float(results['false_positive_rate']),
        'precision_background': float(results['precision_background']),
        'recall_background': float(results['recall_background']),
        'f1_background': float(results['f1_background']),
        'precision_sign': float(results['precision_sign']),
        'recall_sign': float(results['recall_sign']),
        'f1_sign': float(results['f1_sign']),
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Validation metrics saved: {metrics_path}")
    
    # Save confusion matrix as CSV
    cm_csv = out_dir / 'confusion_matrix.csv'
    with open(cm_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['actual', 'predicted', 'count'])
        w.writerow(['background', 'background', int(results['confusion_matrix'][0, 0])])
        w.writerow(['background', 'sign', int(results['confusion_matrix'][0, 1])])
        w.writerow(['sign', 'background', int(results['confusion_matrix'][1, 0])])
        w.writerow(['sign', 'sign', int(results['confusion_matrix'][1, 1])])
    print(f"Confusion matrix CSV saved: {cm_csv}")
    
    # Generate report
    report_path = out_dir / 'training_report.md'
    with open(report_path, 'w') as f:
        f.write("# Phase N1.2: HOG + SVM Binary Verifier Training Report\n\n")
        f.write(f"**Date:** {Path.cwd()}\n")
        f.write(f"**Output Directory:** {out_dir}\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write(f"- Random seed: {RANDOM_SEED}\n")
        f.write(f"- Feature dimension: {feature_dim}\n")
        f.write(f"- Training samples: {len(X_train)}\n")
        f.write(f"- Validation samples: {len(X_val)}\n")
        f.write(f"- Train sign crops: {training_info['train_sign']}\n")
        f.write(f"- Train background crops: {training_info['train_background']}\n")
        f.write(f"- Val sign crops: {training_info['val_sign']}\n")
        f.write(f"- Val background crops: {training_info['val_background']}\n\n")
        
        f.write("## HOG Configuration\n\n")
        for key, value in hog_config.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        f.write("**Note:** OpenCV's HOG implementation does not support transform_sqrt natively. ")
        f.write("This parameter was recorded but not applied during feature extraction.\n\n")
        
        f.write("## SVM Configuration\n\n")
        for key, value in training_info['svm_params'].items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        f.write("## Validation Results (Crop-Level Classification)\n\n")
        f.write(f"**Decision threshold:** {results['decision_threshold']}\n\n")
        f.write("### Confusion Matrix\n\n")
        f.write("| Actual \\ Predicted | Background | Sign |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Background | {results['confusion_matrix'][0, 0]} | {results['confusion_matrix'][0, 1]} |\n")
        f.write(f"| Sign | {results['confusion_matrix'][1, 0]} | {results['confusion_matrix'][1, 1]} |\n\n")
        
        f.write("### Overall Metrics (Sign as Positive)\n\n")
        f.write(f"- Precision: {results['precision']:.4f}\n")
        f.write(f"- Recall: {results['recall']:.4f}\n")
        f.write(f"- F1 Score: {results['f1']:.4f}\n")
        f.write(f"- False Positive Rate: {results['false_positive_rate']:.4f}\n\n")
        
        f.write("### Per-Class Metrics\n\n")
        f.write("**Background (negative class):**\n")
        f.write(f"- Precision: {results['precision_background']:.4f}\n")
        f.write(f"- Recall: {results['recall_background']:.4f}\n")
        f.write(f"- F1: {results['f1_background']:.4f}\n\n")
        f.write("**Sign (positive class):**\n")
        f.write(f"- Precision: {results['precision_sign']:.4f}\n")
        f.write(f"- Recall: {results['recall_sign']:.4f}\n")
        f.write(f"- F1: {results['f1_sign']:.4f}\n\n")
        
        f.write("## Files Created\n\n")
        f.write(f"- SVM model: {model_path}\n")
        f.write(f"- Scaler: {scaler_path}\n")
        f.write(f"- HOG config: {hog_config_out}\n")
        f.write(f"- Training info: {training_info_path}\n")
        f.write(f"- Validation metrics: {metrics_path}\n")
        f.write(f"- Confusion matrix CSV: {cm_csv}\n")
        f.write(f"- This report: {report_path}\n\n")
        
        f.write("## Limitations\n\n")
        f.write("- **M3_D Hard Negatives:** The negative samples contain only randomly sampled background regions. ")
        f.write("No M3_D detector-generated hard negatives were included due to integration complexity. ")
        f.write("This may limit the classifier's ability to reject detector-specific false positives.\n")
        f.write("- **Crop-Level Metrics:** These are crop-level classification metrics, not detector-level metrics. ")
        f.write("The actual impact on M3_D detector performance will need to be evaluated in a subsequent phase.\n")
        f.write("- **No Threshold Tuning:** The decision threshold was set to 0.0 (SVM default for balanced classes). ")
        f.write("Threshold selection on validation data was not performed in this phase.\n")
    
    print(f"Report saved: {report_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
