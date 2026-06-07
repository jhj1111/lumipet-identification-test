import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from reid.data.transforms import get_transform

class CatDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Label mapping (str -> int)
        unique_labels = sorted(list(set(labels)))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.targets = [self.label_to_idx[l] for l in labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
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
    def __init__(self, dataset_path, imgsz=384):
        self.dataset_path = dataset_path
        self.imgsz = imgsz
        self.transform = get_transform(imgsz)
        self.image_paths, self.labels = self._load_image_list()

    def _load_image_list(self):
        image_paths = []
        labels = []
        
        # Structure: dataset_path/label/*.jpg
        if not os.path.exists(self.dataset_path):
            print(f"Warning: Dataset path {self.dataset_path} does not exist.")
            return [], []

        for label in sorted(os.listdir(self.dataset_path)):
            label_dir = os.path.join(self.dataset_path, label)
            if not os.path.isdir(label_dir):
                continue
                
            for root, _, files in os.walk(label_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, f))
                        labels.append(label)
        
        return image_paths, labels

    def get_loaders(self, batch_size=16, test_size=0.2):
        if not self.image_paths:
            return None, None
            
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths, self.labels, test_size=test_size, stratify=self.labels, random_state=42
        )
        
        train_ds = CatDataset(train_paths, train_labels, transform=self.transform)
        val_ds = CatDataset(val_paths, val_labels, transform=self.transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
