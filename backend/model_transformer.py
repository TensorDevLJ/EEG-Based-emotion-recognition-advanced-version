"""
Transformer Model for EEG Depression Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
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
        return self.dropout(x)


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG-based depression classification
    """
    
    def __init__(self, 
                 feature_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_classes: int = 4,
                 max_seq_len: int = 1000):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            src: shape [seq_len, batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
            src_mask: optional attention mask
            return_attention: if True, return attention weights
        Returns:
            logits: shape [batch_size, num_classes]
        """
        # Handle batch_first input
        if len(src.shape) == 3 and src.shape[1] != src.shape[2]:
            # Assume [batch, seq, feat] -> convert to [seq, batch, feat]
            batch_first = True
            src = src.transpose(0, 1)
        else:
            batch_first = False
        
        # Project input
        src = self.input_projection(src)  # [seq_len, batch, d_model]
        src = src * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoding
        if return_attention:
            # Need to manually compute attention (simplified)
            encoded = self.transformer_encoder(src, src_mask)
            attention_weights = None  # Implement if needed
        else:
            encoded = self.transformer_encoder(src, src_mask)
            attention_weights = None
        
        # Global average pooling over sequence
        pooled = encoded.mean(dim=0)  # [batch, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def predict_proba(self, src: torch.Tensor) -> torch.Tensor:
        """Get class probabilities"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(src)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        probs = self.predict_proba(src)
        return torch.argmax(probs, dim=1)


def save_model(model: EEGTransformer, path: str, metadata: dict = None):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'feature_dim': model.feature_dim,
        'd_model': model.d_model,
        'num_classes': model.num_classes,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)


def load_model(path: str, device: str = 'cpu') -> EEGTransformer:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    model = EEGTransformer(
        feature_dim=checkpoint['feature_dim'],
        d_model=checkpoint['d_model'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('metadata', {})
