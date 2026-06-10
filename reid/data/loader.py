from typing import List, Tuple, Dict, Any, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from reid.data.transforms import get_transform

class CatDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[str], transform: Optional[Any] = None, label_to_idx: Optional[Dict[str, int]] = None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Label mapping (str -> int)
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            unique_labels = sorted(list(set(labels)))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.targets = [self.label_to_idx[l] for l in labels]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.targets[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label

class CatDataLoader:
    """
    Handles dataset loading, splitting, and DataLoader creation.
    """
    def __init__(self, dataset_path: str, imgsz: int = 384) -> None:
        self.dataset_path = dataset_path
        self.imgsz = imgsz
        self.transform = get_transform(imgsz)
        self.image_paths, self.labels = self._load_image_list()
        
        # Build self.label_to_idx on initialization
        unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    def _load_image_list(self) -> Tuple[List[str], List[str]]:
        min_data_size = 10
        image_paths = []
        labels = []
        
        # Structure: dataset_path/label/**/*.jpg
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset path {self.dataset_path} does not exist.")
            return [], []

        for label in sorted(os.listdir(self.dataset_path)):
            label_dir = os.path.join(self.dataset_path, label)
            if not os.path.isdir(label_dir):
                continue
                
            for root, _, files in os.walk(label_dir):
                if len(files) < min_data_size: continue
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, f))
                        labels.append(label)
        
        return image_paths, labels

    def get_loaders(self, batch_size: int = 16, test_size: float = 0.2) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if not self.image_paths:
            return None, None
            
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths, self.labels, test_size=test_size, stratify=self.labels, random_state=42
        )
        
        train_ds = CatDataset(train_paths, train_labels, transform=self.transform, label_to_idx=self.label_to_idx)
        val_ds = CatDataset(val_paths, val_labels, transform=self.transform, label_to_idx=self.label_to_idx)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    def get_reid_split(self, test_size: float = 0.5) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Splits data into gallery (for registration) and query (for validation testing).
        Ensures at least 1 image per identity is in the gallery.
        """
        gallery_data = []
        query_data = []
        
        unique_labels = sorted(list(set(self.labels)))
        for label in unique_labels:
            cat_paths = [p for p, l in zip(self.image_paths, self.labels) if l == label]
            if len(cat_paths) < 2:
                # If only 1 image, put it in gallery so we at least know the cat exists
                gallery_data.append((cat_paths[0], label))
                continue
                
            g, q = train_test_split(cat_paths, test_size=test_size, random_state=42)
            gallery_data.extend([(p, label) for p in g])
            query_data.extend([(p, label) for p in q])
            
        return gallery_data, query_data
