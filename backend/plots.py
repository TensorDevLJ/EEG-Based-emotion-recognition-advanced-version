"""
Visualization Module
Generate all plots for frontend display
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import io
import base64
from typing import Dict, List, Tuple, Optional
import pywt


class EEGPlotter:
    """Generate EEG analysis plots"""
    
    def __init__(self, fs: int = 256):
        self.fs = fs
        self.bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-45 Hz)': (30, 45)
        }
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def plot_time_series(self, data: np.ndarray, channel_idx: int = 0,
                        duration: float = 10.0, channel_name: str = None) -> str:
        """
        Plot time series of raw EEG signal
        data: (n_channels, n_samples) or (n_samples,)
        """
        if len(data.shape) == 2:
            signal_data = data[channel_idx]
        else:
            signal_data = data
        
        # Limit to duration seconds
        n_samples = int(duration * self.fs)
        signal_data = signal_data[:n_samples]
        time = np.arange(len(signal_data)) / self.fs
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, signal_data, linewidth=0.5, color='#2E86AB')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (μV)', fontsize=12)
        title = f'EEG Time Series - {channel_name}' if channel_name else f'EEG Time Series - Channel {channel_idx}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def plot_psd(self, data: np.ndarray, channel_idx: int = 0,
                channel_name: str = None) -> str:
        """
        Plot Power Spectral Density with band highlighting
        """
        if len(data.shape) == 2:
            signal_data = data[channel_idx]
        else:
            signal_data = data
        
        # Compute PSD
        freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=min(len(signal_data)//2, 512))
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot PSD
        ax.semilogy(freqs, psd, linewidth=2, color='#2E86AB')
        
        # Highlight bands
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
        for (band_name, (low, high)), color in zip(self.bands.items(), colors):
            ax.axvspan(low, high, alpha=0.2, color=color, label=band_name)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=12)
        title = f'Power Spectral Density - {channel_name}' if channel_name else f'PSD - Channel {channel_idx}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0.5, 45])
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def plot_bandpower_bar(self, band_powers: Dict[str, float]) -> str:
        """
        Plot band powers as bar chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bands = list(band_powers.keys())
        powers = list(band_powers.values())
        colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
        
        bars = ax.bar(bands, powers, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2e}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Power (μV²)', fontsize=12)
        ax.set_title('Band Power Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=15)
        
        return self._fig_to_base64(fig)
    
    def plot_scalogram(self, data: np.ndarray, channel_idx: int = 0,
                      channel_name: str = None) -> str:
        """
        Plot CWT scalogram (time-frequency representation)
        """
        if len(data.shape) == 2:
            signal_data = data[channel_idx]
        else:
            signal_data = data
        
        # Limit length for performance
        max_samples = int(10 * self.fs)
        signal_data = signal_data[:max_samples]
        
        # CWT
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(signal_data, scales, 'morl', 1/self.fs)
        
        # Time array
        time = np.arange(signal_data.shape[0]) / self.fs
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot scalogram
        im = ax.imshow(np.abs(coefficients), extent=[0, time[-1], frequencies[-1], frequencies[0]],
                      cmap='jet', aspect='auto', interpolation='bilinear')
        
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        title = f'Scalogram - {channel_name}' if channel_name else f'Scalogram - Channel {channel_idx}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0.5, 45])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude', fontsize=10)
        
        return self._fig_to_base64(fig)
    
    def plot_asymmetry(self, faa_value: float) -> str:
        """
        Plot frontal alpha asymmetry
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Bar plot
        labels = ['Left Hemisphere', 'Right Hemisphere']
        # Convert FAA to visual representation
        # FAA = log(R) - log(L), so R/L = exp(FAA)
        ratio = np.exp(faa_value)
        left_power = 1.0
        right_power = left_power * ratio
        
        values = [left_power, right_power]
        colors = ['#2E86AB', '#BC4B51']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add text
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0.5, max(values) * 1.1, f'FAA = {faa_value:.3f}',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Interpretation
        if faa_value > 0.3:
            interp = "Right-lateralized (Withdrawal/Negative affect)"
            color = '#BC4B51'
        elif faa_value < -0.3:
            interp = "Left-lateralized (Approach/Positive affect)"
            color = '#2E86AB'
        else:
            interp = "Balanced"
            color = '#6A994E'
        
        ax.text(0.5, max(values) * 0.05, interp,
               ha='center', fontsize=10, style='italic', color=color)
        
        ax.set_ylabel('Relative Alpha Power', fontsize=12)
        ax.set_title('Frontal Alpha Asymmetry', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(values) * 1.3])
        
        return self._fig_to_base64(fig)
    
    def plot_connectivity_matrix(self, matrix: np.ndarray, 
                                 labels: Optional[List[str]] = None) -> str:
        """
        Plot connectivity matrix heatmap
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is None:
            labels = [f'Ch{i}' for i in range(matrix.shape[0])]
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   xticklabels=labels, yticklabels=labels,
                   vmin=0, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Coherence'})
        
        ax.set_title('Inter-Channel Connectivity Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        return self._fig_to_base64(fig)
    
    def plot_training_metrics(self, history: Dict) -> str:
        """
        Plot training history (loss and accuracy)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#2E86AB')
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#BC4B51')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='#2E86AB')
        if 'val_acc' in history and history['val_acc']:
            ax2.plot(epochs, history['val_acc'], 's-', label='Val Acc', color='#BC4B51')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: List[str] = None) -> str:
        """
        Plot confusion matrix
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, ax=ax, cbar_kws={'label': 'Percentage'})
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        return self._fig_to_base64(fig)
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: List[str] = None) -> str:
        """
        Plot ROC curves (one-vs-rest for multi-class)
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        n_classes = y_proba.shape[1]
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Binarize labels
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        # Plot ROC for each class
        for i, color in zip(range(n_classes), colors):
            if np.sum(y_bin[:, i]) == 0:  # No samples of this class
                continue
                
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def plot_feature_importance(self, importance: np.ndarray,
                               feature_names: List[str],
                               top_k: int = 15) -> str:
        """
        Plot feature importance bar chart
        """
        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]
        top_names = [feature_names[i] if i < len(feature_names) else f'Feature {i}' 
                    for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_names))
        colors = plt.cm.viridis(top_importance / top_importance.max())
        
        ax.barh(y_pos, top_importance, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (SHAP value)', fontsize=12)
        ax.set_title('Top Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        return self._fig_to_base64(fig)
