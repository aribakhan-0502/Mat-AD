import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .config import Config

class MRLProjector(nn.Module):
    """Matryoshka Representation Learning Projector for DINOv2 features"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.projector_input_dim, config.projector_hidden_dim),
            nn.LayerNorm(config.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(config.projector_hidden_dim, config.mrl_dims[0])  # Output largest dimension
        )
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: [B, num_patches, input_dim]
        Returns:
            List of tensors for each MRL dimension
        """
        full_emb = self.net(x)
        return [full_emb[:, :, :d] for d in self.config.mrl_dims]

class MatADBackbone:
    """Wrapper for DINOv2 backbone with pyramid fusion"""
    def __init__(self, config: Config):
        self.config = config
        self.backbone = AutoModel.from_pretrained(
            config.dino_model_name,
            token=config.hf_token,
            trust_remote_code=True
        ).to(config.device).eval()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_pyramid_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and fuse layer 7 and 11 features"""
        with torch.no_grad():
            out = self.backbone(images, output_hidden_states=True)
            num_extra = 1 + getattr(self.backbone.config, 'num_register_tokens', 0)
            
            # Pyramid fusion: Layer 7 (mid) + Layer 11 (final)
            features = torch.cat([
                out.hidden_states[7][:, num_extra:, :],
                out.hidden_states[-1][:, num_extra:, :]
            ], dim=-1)
        
        return features
