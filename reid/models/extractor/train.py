import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from reid.engine.trainer import BaseTrainer

class ExtractorTrainer(BaseTrainer):
    """
    Trainer for the Extractor (Projection layer fine-tuning).
    """
    def train(self, model_instance, train_loader, val_loader, epochs=5, lr=1e-3):
        device = model_instance.predictor.device if hasattr(model_instance, 'predictor') else 'cpu'
        model = model_instance.model
        
        # Classifier layer for training (not used during inference)
        num_classes = len(train_loader.dataset.label_to_idx)
        classifier = nn.Linear(512, num_classes).to(device)
        
        optimizer = optim.AdamW(list(model.projection.parameters()) + list(classifier.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.to(device)
        best_acc = 0.0

        for epoch in range(epochs):
            model.train()
            classifier.train()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                
                optimizer.zero_zero_grad() if hasattr(optimizer, 'zero_zero_grad') else optimizer.zero_grad()
                
                # Forward
                embeddings = model(imgs) # CombinedModel returns projected features
                logits = classifier(embeddings)
                
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'loss': loss.item()})

            # Simple Validation
            val_acc = self.validate(model, classifier, val_loader, device)
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.projection.state_dict(), model_instance.model_path)
                print(f"Saved best model weights to {model_instance.model_path}")

    @torch.no_grad()
    def validate(self, model, classifier, val_loader, device):
        model.eval()
        classifier.eval()
        correct = 0
        total = 0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            embeddings = model(imgs)
            logits = classifier(embeddings)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return 100. * correct / total
