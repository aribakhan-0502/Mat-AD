import torch
import torch.nn.functional as F
import numpy as np
import psutil
from typing import Tuple
import matplotlib.pyplot as plt

def get_resource_usage(device: str = "cuda") -> Tuple[float, float]:
    """Get RAM and VRAM usage"""
    ram_usage = psutil.virtual_memory().percent
    vram_usage = 0
    
    if device == "cuda" and torch.cuda.is_available():
        vram_usage = torch.cuda.memory_allocated(device) / 1024**3
    
    return ram_usage, vram_usage

def create_heatmap(distance_map: np.ndarray, output_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Create heatmap from distance map"""
    heatmap_tensor = torch.tensor(distance_map).unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=output_size,
        mode='bilinear',
        align_corners=False
    )
    return heatmap_resized.squeeze().numpy()

def visualize_results(original_img, heatmaps, scores, latencies, save_path=None):
    """Visualize inference results across MRL dimensions"""
    num_dims = len(heatmaps)
    fig, axes = plt.subplots(1, num_dims + 1, figsize=(4*(num_dims + 1), 4))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Each dimension
    for i in range(num_dims):
        axes[i+1].imshow(original_img)
        axes[i+1].imshow(heatmaps[i], cmap='jet', alpha=0.5)
        axes[i+1].set_title(f"Dim {i}\nScore: {scores[i]:.4f}\n{latencies[i]:.2f}ms")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
