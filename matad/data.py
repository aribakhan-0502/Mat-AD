import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from typing import List, Optional
from .config import Config

class MVTecDataset(Dataset):
    """Unified MVTec AD dataset for training"""
    def __init__(self, root: str, categories: List[str], transform: Optional[T.Compose] = None):
        self.paths = []
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        for cat in categories:
            train_dir = Path(root) / cat / 'train' / 'good'
            if train_dir.exists():
                self.paths.extend(list(train_dir.glob("*.png")))
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        # Return two augmented views for contrastive learning
        return self.transform(img), self.transform(img)

class InferenceDataset(Dataset):
    """Dataset for inference with simple transform"""
    def __init__(self, paths: List[Path]):
        self.paths = paths
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), str(self.paths[idx])
