import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
import time

from .models import MRLProjector, MatADBackbone
from .config import Config
from .utils import create_heatmap

class MatADInference:
    """Inference engine for Mat-AD"""
    
    def __init__(self, config: Config, weights_path: str):
        self.config = config
        self.backbone = MatADBackbone(config)
        self.projector = MRLProjector(config).to(config.device).eval()
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=config.device)
        if 'model_state_dict' in checkpoint:
            self.projector.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.projector.load_state_dict(checkpoint)
        
        # Memory banks and threshold
        self.memory_banks: List[torch.Tensor] = []
        self.threshold: Optional[float] = None
    
    def extract_features(self, image: Image.Image) -> List[torch.Tensor]:
        """Extract MRL features from an image"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_t = transform(image).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            # Extract pyramid features
            pyramid_features = self.backbone.extract_pyramid_features(img_t)
            # Project to MRL dimensions
            mrl_features = self.projector(pyramid_features)
        
        return mrl_features
    
    def calibrate(self, golden_paths: List[Path]):
        """Calibrate with golden samples using leave-one-out strategy"""
        print(f"Calibrating with {len(golden_paths)} golden samples...")
        
        all_samples_features = []  # List of MRL feature lists per sample
        
        # Extract features for all golden samples with augmentations
        for path in golden_paths:
            img = Image.open(path).convert('RGB')
            # Augment with 4 rotations
            aug_features = []
            for angle in [0, 90, 180, 270]:
                rotated_img = img.rotate(angle)
                features = self.extract_features(rotated_img)
                aug_features.append(features)
            all_samples_features.append(aug_features)
        
        # Build memory banks for each dimension
        self.memory_banks = []
        for dim_idx in range(len(self.config.mrl_dims)):
            bank = torch.cat([
                sample[rot][dim_idx].squeeze(0)
                for sample in all_samples_features
                for rot in range(4)
            ], dim=0)
            self.memory_banks.append(bank)
        
        # n-1 LOO calibration using the largest dimension
        loo_distances = []
        for i in range(len(all_samples_features)):
            target = all_samples_features[i][0][0]  # 0th rotation, largest dim
            
            # Create bank excluding sample i
            other_banks = torch.cat([
                all_samples_features[j][a][0].squeeze(0)
                for j in range(len(all_samples_features))
                if j != i
                for a in range(4)
            ], dim=0)
            
            dist_map = self._compute_distance_map(target, other_banks)
            loo_distances.extend(dist_map.flatten())
        
        self.threshold = np.percentile(loo_distances, self.config.threshold_percentile)
        print(f"Calibration complete. Threshold: {self.threshold:.4f}")
    
    def _compute_distance_map(self, test_features: torch.Tensor, 
                             memory_bank: torch.Tensor) -> np.ndarray:
        """Compute distance map between test features and memory bank"""
        test_norm = F.normalize(test_features.squeeze(0), dim=-1)
        bank_norm = F.normalize(memory_bank, dim=-1)
        
        # Cosine similarity and convert to distance
        similarities = torch.matmul(test_norm, bank_norm.T)
        distances = 1 - similarities.max(dim=1)[0]
        
        # Reshape to 14x14 patch grid
        return distances.view(14, 14).cpu().numpy()
    
    def predict(self, image_path: Path, dim_idx: int = 0) -> Tuple:
        """Run inference on a single image"""
        start_time = time.time()
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        
        # Extract features
        features = self.extract_features(img)[dim_idx]
        
        # Compute distance map
        dist_map = self._compute_distance_map(features, self.memory_banks[dim_idx])
        
        # Calculate anomaly score (top-k average)
        flat_dists = dist_map.flatten()
        anomaly_score = np.mean(np.sort(flat_dists)[-self.config.top_k_for_score:])
        
        # Create heatmap
        heatmap = create_heatmap(dist_map, output_size=(224, 224))
        
        latency = (time.time() - start_time) * 1000
        
        return img.resize((224, 224)), heatmap, anomaly_score, latency
    
    def is_anomalous(self, image_path: Path, dim_idx: int = 0) -> Tuple[bool, float]:
        """Check if image is anomalous"""
        _, _, score, _ = self.predict(image_path, dim_idx)
        is_anomaly = score > self.threshold if self.threshold else False
        return is_anomaly, score
