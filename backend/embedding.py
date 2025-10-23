"""
Transformer Input Embedding Module
Prepares feature sequences for Transformer input
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import json


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class EEGEmbedding:
    """Prepare EEG features for Transformer input"""
    
    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_dim = None
        
        # Projection layer (linear) to map features to d_model
        self.projection = None
        
    def fit(self, X: np.ndarray):
        """
        Fit scaler on training data
        X: shape (n_samples, n_features) or (n_samples, seq_len, n_features)
        """
        if len(X.shape) == 3:
            # Flatten to (n_samples * seq_len, n_features)
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
            n_features = X.shape[1]
        
        self.scaler.fit(X_flat)
        self.feature_dim = n_features
        self.fitted = True
        
        # Initialize projection layer
        self.projection = nn.Linear(n_features, self.d_model)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        if len(X.shape) == 3:
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)
    
    def build_sequence(self, features: np.ndarray, 
                       add_positional: bool = True) -> torch.Tensor:
        """
        Convert feature matrix to Transformer input
        features: (seq_len, feature_dim) or (batch, seq_len, feature_dim)
        Returns: Tensor shape (seq_len, batch, d_model) or (batch, seq_len, d_model)
        """
        if not self.fitted:
            raise ValueError("Must fit embedding first")
        
        # Scale
        features_scaled = self.transform(features)
        
        # Convert to tensor
        if len(features_scaled.shape) == 2:
            # Single sequence: (seq_len, feature_dim)
            x = torch.FloatTensor(features_scaled).unsqueeze(1)  # (seq_len, 1, feature_dim)
        else:
            # Batch: (batch, seq_len, feature_dim)
            x = torch.FloatTensor(features_scaled)
            x = x.transpose(0, 1)  # -> (seq_len, batch, feature_dim)
        
        # Project to d_model
        if self.projection is not None:
            with torch.no_grad():
                x = self.projection(x)  # (seq_len, batch, d_model)
        
        return x
    
    def save(self, path: str):
        """Save scaler and metadata"""
        metadata = {
            'd_model': self.d_model,
            'feature_dim': self.feature_dim,
            'fitted': self.fitted,
            'scaler_mean': self.scaler.mean_.tolist() if self.fitted else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.fitted else None
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f)
        
        # Save projection weights
        if self.projection is not None:
            torch.save(self.projection.state_dict(), path.replace('.json', '_proj.pth'))
    
    def load(self, path: str):
        """Load scaler and metadata"""
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        self.d_model = metadata['d_model']
        self.feature_dim = metadata['feature_dim']
        self.fitted = metadata['fitted']
        
        if self.fitted:
            self.scaler.mean_ = np.array(metadata['scaler_mean'])
            self.scaler.scale_ = np.array(metadata['scaler_scale'])
            
            # Load projection
            self.projection = nn.Linear(self.feature_dim, self.d_model)
            proj_path = path.replace('.json', '_proj.pth')
            self.projection.load_state_dict(torch.load(proj_path))


def prepare_transformer_input(features: np.ndarray, 
                              embedding: EEGEmbedding,
                              device: str = 'cpu') -> torch.Tensor:
    """
    Prepare complete Transformer input with positional encoding
    """
    x = embedding.build_sequence(features)
    x = x.to(device)
    return x
