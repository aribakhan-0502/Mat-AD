#!/usr/bin/env python3
"""Inference script for Mat-AD"""

import torch
from pathlib import Path
import sys
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from matad import Config, MatADInference, visualize_results

def main():
    # Configuration
    config = Config(
        dataset_root="/path/to/your/dataset",  # Update this
        hf_token=None  # Will be loaded from HF_TOKEN environment variable
    )
    
    # Paths
    weights_path = "./outputs/training/best_mrl_projector.pth"  # Update this
    category = "zipper"
    calibration_shots = 10
    
    # Initialize inference engine
    matad = MatADInference(config, weights_path)
    
    # Calibrate
    dataset_path = Path(config.dataset_root) / category
    golden_paths = sorted(list((dataset_path / "train" / "good").glob("*.png")))[:calibration_shots]
    matad.calibrate(golden_paths)
    
    # Test on a sample
    test_paths = sorted(list((dataset_path / "test").glob("*/*.png")))
    defect_paths = [p for p in test_paths if p.parent.name != 'good']
    
    if defect_paths:
        test_image_path = defect_paths[0]
    else:
        test_image_path = test_paths[0]
    
    # Run inference on all dimensions
    original_img = Image.open(test_image_path).convert('RGB').resize((224, 224))
    heatmaps, scores, latencies = [], [], []
    
    for dim_idx in range(len(config.mrl_dims)):
        img, heatmap, score, latency = matad.predict(test_image_path, dim_idx)
        heatmaps.append(heatmap)
        scores.append(score)
        latencies.append(latency)
    
    # Visualize
    visualize_results(original_img, heatmaps, scores, latencies)
    
    # Check anomaly status
    is_anomaly, final_score = matad.is_anomalous(test_image_path)
    print(f"Anomaly Score: {final_score:.4f}")
    print(f"Threshold: {matad.threshold:.4f}")
    print(f"Status: {'ANOMALY DETECTED' if is_anomaly else 'NORMAL'}")

if __name__ == "__main__":
    main()
