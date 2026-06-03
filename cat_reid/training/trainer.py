import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from cat_reid.models.extractor import FeatureExtractor
from cat_reid.config import settings

class CatReIdModel(nn.Module):
    def __init__(self, base_model, embedding_dim=512, num_classes=69):
        super(CatReIdModel, self).__init__()
        self.backbone = base_model.model
        # Backbone 가중치 고정
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        
        embeddings = self.projection(features)
        # L2 정규화 (Re-ID용 특징 벡터)
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        logits = self.classifier(embeddings)
        return norm_embeddings, logits

def train(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    optimizer = optim.AdamW(model.projection.parameters(), lr=lr)
    
    model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, logits = model(imgs)
            # CrossEntropyLoss 사용
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/(pbar.n+1), 'acc': 100.*correct/total})
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, logits = model(imgs)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.projection.state_dict(), settings.LINEAR_MODEL_PATH)
            print(f"Saved best model to {settings.LINEAR_MODEL_PATH} with Acc: {val_acc:.2f}%")

    return model
