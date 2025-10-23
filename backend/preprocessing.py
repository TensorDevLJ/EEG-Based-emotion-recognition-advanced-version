"""
EEG Signal Preprocessing Module
Handles loading, filtering, artifact removal, and epoching
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import FastICA
import mne
from typing import Tuple, Dict, Optional
import pandas as pd


class EEGPreprocessor:
    """Complete EEG preprocessing pipeline"""
    
    def __init__(self, sampling_rate: int = 256):
        self.fs = sampling_rate
        
    def load_eeg(self, file_path: str) -> Tuple[np.ndarray, Dict, Optional[np.ndarray]]:
        """
        Load pre-extracted EEG features from CSV
        Returns: (features, metadata, labels)
        """
        df = pd.read_csv(file_path)
        
        labels = None
        if 'Label' in df.columns:
            labels = df['Label'].values
            df = df.drop('Label', axis=1)
        
        features = df.values
        feature_names = df.columns.tolist()
        
        metadata = {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'feature_names': feature_names,
            'has_labels': labels is not None
        }
        
        return features, metadata, labels
    
    def reconstruct_signal_from_features(self, features: np.ndarray, feature_names: list, n_channels: int = 4) -> np.ndarray:
        """Reconstruct approximate signal from features for visualization"""
        n_samples = 2560
        signal = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            mean_key = f'mean_{ch}'
            std_key = f'std_{ch}'
            mean_val = features[0, feature_names.index(mean_key)] if mean_key in feature_names else 0
            std_val = features[0, feature_names.index(std_key)] if std_key in feature_names else 1
            signal[ch] = np.random.randn(n_samples) * std_val + mean_val
        
        return signal
    
    def load_eeg_from_bytes(self, file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Dict]:
        """Load EEG from uploaded bytes"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            data, metadata = self.load_eeg(tmp_path)
        finally:
            os.unlink(tmp_path)
            
        return data, metadata
    
    def bandpass_filter(self, data: np.ndarray, low: float = 0.5, 
                        high: float = 45.0, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter
        """
        nyq = 0.5 * self.fs
        low_norm = low / nyq
        high_norm = high / nyq
        b, a = butter(order, [low_norm, high_norm], btype='band')
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
            
        return filtered
    
    def notch_filter(self, data: np.ndarray, freq: float = 50.0, 
                     quality: float = 30.0) -> np.ndarray:
        """
        Apply notch filter for powerline noise (50/60 Hz)
        """
        b, a = iirnotch(freq, quality, self.fs)
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
            
        return filtered
    
    def run_ica(self, data: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Run ICA for artifact removal
        """
        if n_components is None:
            n_components = min(data.shape[0], 20)
            
        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        
        # ICA expects samples × features
        sources = ica.fit_transform(data.T)
        
        # Simple artifact removal: remove components with extreme kurtosis
        kurtosis = np.array([self._kurtosis(sources[:, i]) for i in range(sources.shape[1])])
        threshold = np.mean(np.abs(kurtosis)) + 2 * np.std(np.abs(kurtosis))
        
        # Zero out artifact components
        clean_sources = sources.copy()
        artifact_idx = np.abs(kurtosis) > threshold
        clean_sources[:, artifact_idx] = 0
        
        # Reconstruct
        cleaned = ica.inverse_transform(clean_sources).T
        
        return cleaned
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis"""
        from scipy.stats import kurtosis
        return kurtosis(x)
    
    def epoch_signal(self, data: np.ndarray, epoch_len: float = 2.0, 
                     overlap: float = 0.5) -> np.ndarray:
        """
        Split signal into epochs
        Returns: epochs array (n_epochs × channels × samples)
        """
        samples_per_epoch = int(epoch_len * self.fs)
        step = int(samples_per_epoch * (1 - overlap))
        
        n_channels, n_samples = data.shape
        epochs = []
        
        start = 0
        while start + samples_per_epoch <= n_samples:
            epoch = data[:, start:start + samples_per_epoch]
            epochs.append(epoch)
            start += step
            
        return np.array(epochs)
    
    def preprocess_pipeline(self, file_path: str, 
                           apply_ica: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete preprocessing pipeline
        Returns: (raw_data, epochs, metadata)
        """
        # Load
        data, metadata = self.load_eeg(file_path)
        
        # Filter
        filtered = self.bandpass_filter(data)
        filtered = self.notch_filter(filtered, freq=50.0)  # Europe
        filtered = self.notch_filter(filtered, freq=60.0)  # US
        
        # ICA artifact removal
        if apply_ica and data.shape[0] >= 4:  # Need multiple channels
            cleaned = self.run_ica(filtered)
        else:
            cleaned = filtered
            
        # Epoch
        epochs = self.epoch_signal(cleaned)
        
        return data, epochs, metadata
