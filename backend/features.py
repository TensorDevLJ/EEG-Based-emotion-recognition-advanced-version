"""
EEG Feature Extraction Module
Extracts spectral, temporal, and connectivity features
"""
from typing import Optional, Tuple, Dict, List
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.integrate import trapz
import pywt


class EEGFeatureExtractor:
    """Extract comprehensive EEG features"""
    
    # Frequency bands (Hz)
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    def __init__(self, fs: int = 256):
        self.fs = fs
        self.training_stats = None  # For normalization
        
    def compute_welch(self, epoch: np.ndarray, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method
        epoch: single channel data
        Returns: (freqs, psd)
        """
        if nperseg is None:
            nperseg = min(epoch.shape[0] // 2, 256)
            
        freqs, psd = signal.welch(epoch, fs=self.fs, nperseg=nperseg, 
                                  scaling='density')
        return freqs, psd
    
    def bandpower(self, freqs: np.ndarray, psd: np.ndarray, 
                  band: Tuple[float, float]) -> float:
        """
        Calculate absolute band power
        """
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return trapz(psd[idx], freqs[idx])
    
    def relative_power(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate relative band powers
        """
        total = sum(band_powers.values())
        return {band: power / total for band, power in band_powers.items()}
    
    def peak_alpha_frequency(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """
        Find individual alpha frequency (IAF)
        """
        alpha_band = self.BANDS['alpha']
        idx = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
        alpha_freqs = freqs[idx]
        alpha_psd = psd[idx]
        
        if len(alpha_psd) == 0:
            return (alpha_band[0] + alpha_band[1]) / 2
            
        peak_idx = np.argmax(alpha_psd)
        return alpha_freqs[peak_idx]
    
    def frontal_alpha_asymmetry(self, alpha_left: float, alpha_right: float) -> float:
        """
        Calculate frontal alpha asymmetry
        FAA = log(right) - log(left)
        Positive = relatively more right alpha = withdrawal/negative affect
        """
        eps = 1e-10
        return np.log(alpha_right + eps) - np.log(alpha_left + eps)
    
    def spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Calculate spectral entropy (normalized)
        """
        # Normalize to probability distribution
        psd_norm = psd / (np.sum(psd) + 1e-10)
        # Calculate entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        # Normalize by max possible entropy
        max_entropy = np.log2(len(psd))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def hjorth_parameters(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Calculate Hjorth parameters: activity, mobility, complexity
        """
        # First derivative
        dx = np.diff(epoch)
        # Second derivative
        ddx = np.diff(dx)
        
        # Activity (variance)
        activity = np.var(epoch)
        
        # Mobility
        mobility = np.sqrt(np.var(dx) / (activity + 1e-10))
        
        # Complexity
        complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10)) / (mobility + 1e-10)
        
        return {
            'activity': activity,
            'mobility': mobility,
            'complexity': complexity
        }
    
    def statistical_moments(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical moments
        """
        return {
            'mean': np.mean(epoch),
            'variance': np.var(epoch),
            'skewness': skew(epoch),
            'kurtosis': kurtosis(epoch)
        }
    
    def coherence_pair(self, epoch1: np.ndarray, epoch2: np.ndarray) -> Dict[str, float]:
        """
        Calculate coherence between two channels for each band
        """
        nperseg = min(len(epoch1) // 2, 256)
        f, Cxy = signal.coherence(epoch1, epoch2, fs=self.fs, nperseg=nperseg)
        
        coherences = {}
        for band_name, (low, high) in self.BANDS.items():
            idx = np.logical_and(f >= low, f <= high)
            coherences[f'coherence_{band_name}'] = np.mean(Cxy[idx])
            
        return coherences
    
    def plv_pair(self, epoch1: np.ndarray, epoch2: np.ndarray, 
                 band: Tuple[float, float]) -> float:
        """
        Calculate Phase Locking Value for a specific band
        """
        # Bandpass filter
        nyq = 0.5 * self.fs
        low, high = band[0] / nyq, band[1] / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        
        sig1_filt = signal.filtfilt(b, a, epoch1)
        sig2_filt = signal.filtfilt(b, a, epoch2)
        
        # Hilbert transform
        analytic1 = signal.hilbert(sig1_filt)
        analytic2 = signal.hilbert(sig2_filt)
        
        # Phase difference
        phase_diff = np.angle(analytic1) - np.angle(analytic2)
        
        # PLV
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def cwt_features(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Compute CWT scalogram features
        """
        # Use Morlet wavelet
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(epoch, scales, 'morl', 1/self.fs)
        
        # Energy in each band
        cwt_features = {}
        for band_name, (low, high) in self.BANDS.items():
            freq_idx = np.logical_and(frequencies >= low, frequencies <= high)
            if np.any(freq_idx):
                energy = np.mean(np.abs(coefficients[freq_idx, :]) ** 2)
                cwt_features[f'cwt_energy_{band_name}'] = energy
            else:
                cwt_features[f'cwt_energy_{band_name}'] = 0
                
        return cwt_features
    
    def extract_channel_features(self, epoch_channel: np.ndarray) -> Dict[str, float]:
        """
        Extract all features for a single channel
        """
        features = {}
        
        # PSD
        freqs, psd = self.compute_welch(epoch_channel)
        
        # Band powers
        band_powers = {}
        for band_name, band_range in self.BANDS.items():
            power = self.bandpower(freqs, psd, band_range)
            band_powers[band_name] = power
            features[f'power_{band_name}'] = power
        
        # Relative powers
        rel_powers = self.relative_power(band_powers)
        for band_name, rel_power in rel_powers.items():
            features[f'rel_power_{band_name}'] = rel_power
        
        # Band ratios
        features['theta_alpha_ratio'] = (band_powers['theta'] + 1e-10) / (band_powers['alpha'] + 1e-10)
        features['alpha_beta_ratio'] = (band_powers['alpha'] + 1e-10) / (band_powers['beta'] + 1e-10)
        features['theta_beta_ratio'] = (band_powers['theta'] + 1e-10) / (band_powers['beta'] + 1e-10)
        
        # Peak alpha frequency
        features['peak_alpha_freq'] = self.peak_alpha_frequency(freqs, psd)
        
        # Spectral entropy
        features['spectral_entropy'] = self.spectral_entropy(psd)
        
        # Hjorth parameters
        hjorth = self.hjorth_parameters(epoch_channel)
        features.update({f'hjorth_{k}': v for k, v in hjorth.items()})
        
        # Statistical moments
        moments = self.statistical_moments(epoch_channel)
        features.update({f'stat_{k}': v for k, v in moments.items()})
        
        # CWT features
        cwt_feats = self.cwt_features(epoch_channel)
        features.update(cwt_feats)
        
        return features
    
    def extract_epoch_features(self, epoch: np.ndarray, 
                               channel_names: Optional[list] = None) -> np.ndarray:
        """
        Extract features for entire epoch (all channels)
        epoch: shape (n_channels, n_samples)
        Returns: 1D feature vector
        """
        n_channels = epoch.shape[0]
        if channel_names is None:
            channel_names = [f'ch{i}' for i in range(n_channels)]
        
        all_features = {}
        
        # Per-channel features
        channel_features_list = []
        for i in range(n_channels):
            ch_features = self.extract_channel_features(epoch[i])
            channel_features_list.append(ch_features)
            # Add to dict with channel name
            for feat_name, value in ch_features.items():
                all_features[f'{channel_names[i]}_{feat_name}'] = value
        
        # Cross-channel features (connectivity)
        if n_channels >= 2:
            # Frontal alpha asymmetry (if we have F3/F4 or AF3/AF4)
            left_channels = [i for i, name in enumerate(channel_names) 
                           if any(x in name.upper() for x in ['F3', 'AF3', 'FP1'])]
            right_channels = [i for i, name in enumerate(channel_names) 
                            if any(x in name.upper() for x in ['F4', 'AF4', 'FP2'])]
            
            if left_channels and right_channels:
                # Use first found channels
                freqs_l, psd_l = self.compute_welch(epoch[left_channels[0]])
                freqs_r, psd_r = self.compute_welch(epoch[right_channels[0]])
                alpha_left = self.bandpower(freqs_l, psd_l, self.BANDS['alpha'])
                alpha_right = self.bandpower(freqs_r, psd_r, self.BANDS['alpha'])
                all_features['frontal_alpha_asymmetry'] = self.frontal_alpha_asymmetry(alpha_left, alpha_right)
            
            # Coherence and PLV for selected pairs (limit to avoid explosion)
            pairs = [(0, 1)]  # At minimum, first two channels
            if n_channels >= 4:
                pairs.extend([(0, 3), (1, 2)])  # Add a few more pairs
                
            for idx, (i, j) in enumerate(pairs):
                if i < n_channels and j < n_channels:
                    # Coherence
                    coh = self.coherence_pair(epoch[i], epoch[j])
                    for k, v in coh.items():
                        all_features[f'pair{idx}_{k}'] = v
                    
                    # PLV for alpha band
                    plv = self.plv_pair(epoch[i], epoch[j], self.BANDS['alpha'])
                    all_features[f'pair{idx}_plv_alpha'] = plv
        
        # Convert to vector (maintain consistent ordering)
        feature_vector = np.array([all_features[k] for k in sorted(all_features.keys())])
        
        return feature_vector
    
    def build_feature_matrix(self, epochs: np.ndarray, 
                            channel_names: Optional[list] = None) -> np.ndarray:
        """
        Build feature matrix for all epochs
        epochs: shape (n_epochs, n_channels, n_samples)
        Returns: (n_epochs, n_features)
        """
        feature_list = []
        for epoch in epochs:
            features = self.extract_epoch_features(epoch, channel_names)
            feature_list.append(features)
        
        return np.array(feature_list)


from typing import Optional
