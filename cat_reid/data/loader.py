import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from cat_reid.preprocessing.transform import get_transform

class CatDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: [(image_path, label_idx), ...]
        """
        self.data_list = data_list
        self.transform = transform if transform else get_transform()
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        img_path, label_idx = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label_idx

class CatDatasetLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cat_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label_to_idx = {folder: i for i, folder in enumerate(self.cat_folders)}
        self.idx_to_label = {i: folder for i, folder in enumerate(self.cat_folders)}
        
    def get_split_data(self, test_size=0.5, random_seed=42):
        gallery_data = [] 
        query_data = []
        
        for folder in self.cat_folders:
            cat_path = os.path.join(self.root_dir, folder)
            images = glob.glob(os.path.join(cat_path, "**", "*.*"), recursive=True)
            images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(images) < 2:
                if len(images) == 1:
                    gallery_data.append((images[0], folder))
                continue
                
            g_imgs, q_imgs = train_test_split(images, test_size=test_size, random_state=random_seed)
            
            for img in g_imgs:
                gallery_data.append((img, folder))
            for img in q_imgs:
                query_data.append((img, folder))
                
        return gallery_data, query_data

    def get_train_loaders(self, batch_size=32, test_size=0.2):
        """
        학습용 DataLoader를 반환합니다.
        """
        all_data = []
        for folder in self.cat_folders:
            cat_path = os.path.join(self.root_dir, folder)
            images = glob.glob(os.path.join(cat_path, "**", "*.*"), recursive=True)
            for img in images:
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_data.append((img, self.label_to_idx[folder]))
                    
        train_data, val_data = train_test_split(all_data, test_size=test_size, random_state=42)
        
        train_dataset = CatDataset(train_data)
        val_dataset = CatDataset(val_data)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
