import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List
import psutil

from .models import MRLProjector, MatADBackbone
from .data import MVTecDataset
from .config import Config
from .utils import get_resource_usage

class MatADTrainer:
    """Trainer for MRL Projector"""
    
    def __init__(self, config: Config):
        self.config = config
        self.backbone = MatADBackbone(config)
        self.model = MRLProjector(config).to(config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Ensure output directory exists
        if config.output_dir:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def compute_contrastive_loss(self, view1_features: List[torch.Tensor], 
                                view2_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute weighted contrastive loss across MRL dimensions"""
        loss = 0
        for i, (v1, v2) in enumerate(zip(view1_features, view2_features)):
            # Flatten for patch-level contrastive alignment
            v1_n = F.normalize(v1.reshape(-1, v1.shape[-1]), dim=1)
            v2_n = F.normalize(v2.reshape(-1, v2.shape[-1]), dim=1)
            
            # Maximize cosine similarity between augmented views
            sim = (v1_n * v2_n).sum(dim=1)
            loss += (1 - sim.mean()) * self.config.mrl_weights[i]
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(self.config.device), view2.to(self.config.device)
            
            # Extract pyramid features
            pyr1 = self.backbone.extract_pyramid_features(view1)
            pyr2 = self.backbone.extract_pyramid_features(view2)
            
            # Get MRL projections
            mrl_views1 = self.model(pyr1)
            mrl_views2 = self.model(pyr2)
            
            # Compute loss
            loss = self.compute_contrastive_loss(mrl_views1, mrl_views2)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return epoch_loss / len(dataloader)
    
    def train(self, dataset_root: str, categories: List[str]):
        """Full training pipeline"""
        print(f"Starting Mat-AD training on {len(categories)} categories")
        
        # Create dataset
        dataset = MVTecDataset(dataset_root, categories)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )
        
        history = []
        best_loss = float('inf')
        total_start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            avg_loss = self.train_epoch(dataloader, epoch)
            
            # Log metrics
            duration = time.time() - epoch_start_time
            ram, vram = get_resource_usage(self.config.device)
            
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'duration': duration,
                'ram_percent': ram,
                'vram_gb': vram
            })
            
            print(f"Epoch [{epoch+1}/{self.config.epochs}] "
                  f"Loss: {avg_loss:.6f} | Time: {duration:.2f}s")
            
            # Save best model
            if avg_loss < best_loss and self.config.output_dir:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'loss': avg_loss,
                    'epoch': epoch + 1
                }, Path(self.config.output_dir) / "best_mrl_projector.pth")
                print(f"  Saved best model (loss: {avg_loss:.6f})")
        
        # Save final model and logs
        if self.config.output_dir:
            output_dir = Path(self.config.output_dir)
            
            # Final model
            torch.save(self.model.state_dict(), output_dir / "final_mrl_projector.pth")
            
            # Training log
            pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
            
            # Training summary
            total_time = time.time() - total_start_time
            print(f"\nTraining complete in {total_time:.2f}s")
            print(f"Models saved to: {output_dir}")
        
        return history
