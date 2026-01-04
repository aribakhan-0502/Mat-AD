#!/usr/bin/env python3
"""Training script for Mat-AD"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from matad import Config, MatADTrainer

def main():
    # Configuration
    config = Config(
        dataset_root="/path/to/your/dataset",  # Update this
        output_dir="./outputs/training",      # Update this
        hf_token=None  # Will be loaded from HF_TOKEN environment variable
    )
    
    # All MVTec categories
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    # Initialize and train
    trainer = MatADTrainer(config)
    history = trainer.train(config.dataset_root, categories)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
