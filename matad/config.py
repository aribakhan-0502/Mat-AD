import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    """Configuration for Mat-AD framework"""
    # Model
    dino_model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    mrl_dims: List[int] = (384, 192, 96)
    projector_input_dim: int = 768
    projector_hidden_dim: int = 512
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 5
    temperature: float = 0.07
    mrl_weights: List[float] = (1.0, 0.5, 0.2)
    
    # Inference
    calibration_shots: int = 10
    threshold_percentile: float = 99.9
    top_k_for_score: int = 5
    
    # Paths (to be configured by user)
    dataset_root: Optional[str] = None
    output_dir: Optional[str] = None
    weights_dir: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Security - NEVER store tokens in code
    hf_token: Optional[str] = None  # Load from environment variable
    
    def __post_init__(self):
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")
