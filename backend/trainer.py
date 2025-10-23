"""
Training Pipeline for EEG Transformer Model
Supports LOSO cross-validation, metrics tracking, callbacks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from typing import Dict, List, Tuple, Optional, Callable
import json
from pathlib import Path
import time


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG sequences"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, subjects: Optional[np.ndarray] = None):
        """
        X: (n_samples, seq_len, feature_dim)
        y: (n_samples,) class labels
        subjects: (n_samples,) subject IDs for LOSO
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.subjects = subjects
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Save best model callback"""
    
    def __init__(self, filepath: str, mode: str = 'min', save_best_only: bool = True):
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
    def __call__(self, model: nn.Module, score: float, epoch: int):
        if not self.save_best_only:
            torch.save(model.state_dict(), f"{self.filepath}_epoch{epoch}.pth")
            return
        
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.filepath)
            return
        
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score
        
        if improved:
            self.best_score = score
            torch.save(model.state_dict(), self.filepath)


class EEGTrainer:
    """Complete training pipeline"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, f1
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, f1
    
    def train(self, train_dataset: EEGDataset, val_dataset: Optional[EEGDataset],
             epochs: int = 50, batch_size: int = 16, lr: float = 1e-4,
             weight_decay: float = 1e-5, patience: int = 10,
             save_path: Optional[str] = None) -> Dict:
        """
        Full training loop with callbacks
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        early_stopping = EarlyStopping(patience=patience, mode='min')
        if save_path:
            checkpoint = ModelCheckpoint(save_path, mode='min')
        
        print(f"Training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, optimizer, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            
            # Validate
            if val_loader:
                val_loss, val_acc, val_f1 = self.validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f} - "
                      f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f}")
                
                # Callbacks
                if save_path:
                    checkpoint(self.model, val_loss, epoch)
                
                if early_stopping(val_loss):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")
        
        return {
            'history': self.history,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }
    
    def evaluate(self, test_dataset: EEGDataset, 
                class_names: List[str] = None) -> Dict:
        """
        Comprehensive evaluation on test set
        """
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            all_labels, all_preds, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC AUC (one-vs-rest)
        n_classes = len(np.unique(all_labels))
        if n_classes > 2:
            try:
                roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            except:
                roc_auc = None
        else:
            try:
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            except:
                roc_auc = None
        
        results = {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'roc_auc_macro': float(roc_auc) if roc_auc else None,
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist()
        }
        
        if class_names:
            results['class_names'] = class_names
        
        return results


def cross_validate_loso(model_class, X: np.ndarray, y: np.ndarray, 
                       subjects: np.ndarray, **train_kwargs) -> Dict:
    """
    Leave-One-Subject-Out cross-validation
    """
    logo = LeaveOneGroupOut()
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, subjects)):
        print(f"\n=== LOSO Fold {fold_idx + 1} ===")
        print(f"Test subject: {subjects[test_idx[0]]}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Split train into train/val
        n_val = int(len(X_train) * 0.1)
        val_idx = np.random.choice(len(X_train), n_val, replace=False)
        train_mask = np.ones(len(X_train), dtype=bool)
        train_mask[val_idx] = False
        
        X_tr, X_val = X_train[train_mask], X_train[val_idx]
        y_tr, y_val = y_train[train_mask], y_train[val_idx]
        
        # Create datasets
        train_ds = EEGDataset(X_tr, y_tr)
        val_ds = EEGDataset(X_val, y_val)
        test_ds = EEGDataset(X_test, y_test)
        
        # Initialize model
        model = model_class()
        trainer = EEGTrainer(model)
        
        # Train
        trainer.train(train_ds, val_ds, **train_kwargs)
        
        # Evaluate
        results = trainer.evaluate(test_ds)
        fold_results.append(results)
    
    # Aggregate results
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
    
    return {
        'fold_results': fold_results,
        'average_accuracy': float(avg_acc),
        'average_f1_macro': float(avg_f1),
        'n_folds': len(fold_results)
    }
