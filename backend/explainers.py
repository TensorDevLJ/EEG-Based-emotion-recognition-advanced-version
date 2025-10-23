"""
Explainability Module
SHAP-based and rule-based explanations for model predictions
"""

import numpy as np
import torch
import shap
from typing import Dict, List, Tuple, Optional
import json


class EEGExplainer:
    """Generate explanations for EEG depression predictions"""
    
    # Population statistics (fitted from training data)
    def __init__(self):
        self.feature_stats = None  # Dict of {feature_name: {'mean': x, 'std': y}}
        self.feature_names = None
        self.class_names = ['Not Depressed', 'Mild', 'Moderate', 'Severe']
        
    def fit_population_stats(self, features: np.ndarray, feature_names: List[str]):
        """
        Compute population statistics for rule-based explanations
        features: (n_samples, n_features)
        """
        self.feature_names = feature_names
        self.feature_stats = {}
        
        for i, name in enumerate(feature_names):
            self.feature_stats[name] = {
                'mean': float(np.mean(features[:, i])),
                'std': float(np.std(features[:, i])),
                'median': float(np.median(features[:, i])),
                'q25': float(np.percentile(features[:, i], 25)),
                'q75': float(np.percentile(features[:, i], 75))
            }
    
    def compute_shap_values(self, model: torch.nn.Module, X_sample: np.ndarray,
                           X_background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute SHAP values for a sample
        X_sample: (1, seq_len, feature_dim) or (seq_len, feature_dim)
        X_background: background dataset for SHAP (subset of training data)
        Returns: SHAP values array
        """
        model.eval()
        
        # Use DeepExplainer or GradientExplainer
        if X_background is not None:
            background_tensor = torch.FloatTensor(X_background)
            explainer = shap.DeepExplainer(model, background_tensor)
        else:
            # Use a simpler approach - gradient-based
            def model_predict(x):
                with torch.no_grad():
                    return model(torch.FloatTensor(x)).numpy()
            
            # For now, return feature importance based on gradient
            # This is a simplified version
            return self._gradient_based_importance(model, X_sample)
        
        X_tensor = torch.FloatTensor(X_sample)
        shap_values = explainer.shap_values(X_tensor)
        
        return shap_values
    
    def _gradient_based_importance(self, model: torch.nn.Module, 
                                   X: np.ndarray) -> np.ndarray:
        """
        Simplified feature importance using gradients
        """
        model.eval()
        X_tensor = torch.FloatTensor(X).requires_grad_(True)
        
        output = model(X_tensor)
        predicted_class = torch.argmax(output, dim=1)
        
        # Backward pass
        model.zero_grad()
        output[0, predicted_class].backward()
        
        # Gradients as importance
        gradients = X_tensor.grad.abs().mean(dim=0).detach().numpy()  # Average over sequence
        
        return gradients
    
    def get_top_features(self, shap_values: np.ndarray, 
                        X_sample: np.ndarray, 
                        top_k: int = 5) -> List[Dict]:
        """
        Get top contributing features based on SHAP values
        """
        if len(shap_values.shape) > 1:
            # Average over sequence dimension
            importance = np.mean(np.abs(shap_values), axis=0)
        else:
            importance = np.abs(shap_values)
        
        # Get top indices
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        # Average X_sample over sequence if needed
        if len(X_sample.shape) > 1:
            X_mean = np.mean(X_sample, axis=0)
        else:
            X_mean = X_sample
        
        top_features = []
        for idx in top_indices:
            if self.feature_names and idx < len(self.feature_names):
                name = self.feature_names[idx]
            else:
                name = f"feature_{idx}"
            
            top_features.append({
                'feature': name,
                'value': float(X_mean[idx]),
                'shap_value': float(importance[idx]),
                'interpretation': self._interpret_feature_value(name, X_mean[idx])
            })
        
        return top_features
    
    def _interpret_feature_value(self, feature_name: str, value: float) -> str:
        """
        Interpret if feature value is high/low/normal
        """
        if not self.feature_stats or feature_name not in self.feature_stats:
            return "Unknown"
        
        stats = self.feature_stats[feature_name]
        z_score = (value - stats['mean']) / (stats['std'] + 1e-10)
        
        if z_score > 1.5:
            return f"High (z={z_score:.2f}, >95th percentile)"
        elif z_score > 0.5:
            return f"Moderately high (z={z_score:.2f})"
        elif z_score < -1.5:
            return f"Low (z={z_score:.2f}, <5th percentile)"
        elif z_score < -0.5:
            return f"Moderately low (z={z_score:.2f})"
        else:
            return f"Within normal range (z={z_score:.2f})"
    
    def rule_based_explanations(self, features_dict: Dict[str, float],
                                top_features: List[Dict]) -> str:
        """
        Generate natural language explanations based on rules
        """
        explanations = []
        
        # Check each top feature and provide context
        for feat_info in top_features:
            name = feat_info['feature']
            value = feat_info['value']
            interp = feat_info['interpretation']
            
            # Feature-specific rules
            if 'theta_alpha_ratio' in name:
                if 'High' in interp:
                    explanations.append(
                        f"**Theta/Alpha Ratio** is elevated ({value:.3f}). "
                        f"An elevated theta/alpha ratio (>1.2) is associated with increased depressive symptoms "
                        f"and cognitive slowing, commonly observed in depression."
                    )
            
            elif 'frontal_alpha_asymmetry' in name:
                if value > 0.3:
                    explanations.append(
                        f"**Frontal Alpha Asymmetry** shows right dominance ({value:.3f}). "
                        f"Right-lateralized frontal alpha activity is linked to withdrawal-related affect "
                        f"and negative emotional states, a biomarker for depression risk."
                    )
                elif value < -0.3:
                    explanations.append(
                        f"**Frontal Alpha Asymmetry** shows left dominance ({value:.3f}). "
                        f"Left-lateralized activity is associated with approach-related positive affect."
                    )
            
            elif 'spectral_entropy' in name:
                if 'Low' in interp:
                    explanations.append(
                        f"**Spectral Entropy** is reduced ({value:.3f}). "
                        f"Lower spectral entropy indicates reduced signal complexity, "
                        f"which correlates with depressive states and decreased neural variability."
                    )
            
            elif 'rel_power_alpha' in name:
                if 'Low' in interp:
                    explanations.append(
                        f"**Alpha Power** is below normal ({value:.3f}). "
                        f"Reduced alpha power can indicate decreased relaxation and increased cognitive load."
                    )
            
            elif 'beta' in name and 'power' in name:
                if 'High' in interp:
                    explanations.append(
                        f"**Beta Power** is elevated ({value:.3f}). "
                        f"Increased beta activity may reflect heightened anxiety, rumination, or stress."
                    )
            
            elif 'coherence' in name or 'plv' in name:
                if 'Low' in interp:
                    explanations.append(
                        f"**Neural Connectivity** ({name}) is reduced ({value:.3f}). "
                        f"Decreased connectivity between brain regions is often found in depression, "
                        f"reflecting impaired communication in emotional regulation networks."
                    )
            
            elif 'hjorth_complexity' in name:
                if 'Low' in interp:
                    explanations.append(
                        f"**Signal Complexity** (Hjorth) is low ({value:.3f}). "
                        f"Reduced complexity suggests less dynamic brain activity, characteristic of depressive states."
                    )
        
        # Combine explanations
        if not explanations:
            explanations.append(
                "The model detected patterns in your EEG signal that differ from healthy controls. "
                "Multiple subtle features contribute to this assessment."
            )
        
        full_text = "\n\n".join(explanations)
        
        return full_text
    
    def compute_depression_index(self, features_dict: Dict[str, float]) -> float:
        """
        Compute a Depression Index (DI) based on key features
        Returns: value between 0 and 1
        """
        # Weights based on literature (simplified)
        di = 0.0
        count = 0
        
        # Theta/Alpha ratio contribution
        if 'theta_alpha_ratio' in features_dict:
            ta_ratio = features_dict['theta_alpha_ratio']
            # Normalize: typical range 0.5-2.0, depressed >1.2
            ta_score = min((ta_ratio - 0.5) / 1.5, 1.0)
            di += max(0, ta_score) * 0.25
            count += 1
        
        # FAA contribution
        if 'frontal_alpha_asymmetry' in features_dict:
            faa = features_dict['frontal_alpha_asymmetry']
            # Positive FAA (right>left) -> depression
            faa_score = max(0, faa) / 1.0  # Normalize
            di += min(faa_score, 1.0) * 0.20
            count += 1
        
        # Spectral entropy (lower = more depressed)
        if 'spectral_entropy' in features_dict:
            se = features_dict['spectral_entropy']
            se_score = 1.0 - se  # Invert (lower entropy = higher score)
            di += se_score * 0.20
            count += 1
        
        # Beta power (higher = more anxious/depressed)
        beta_features = [k for k in features_dict.keys() if 'rel_power_beta' in k]
        if beta_features:
            beta_avg = np.mean([features_dict[k] for k in beta_features])
            # Typical rel_power ~0.2, depressed ~0.3+
            beta_score = min((beta_avg - 0.15) / 0.2, 1.0)
            di += max(0, beta_score) * 0.15
            count += 1
        
        # Connectivity (lower = more depressed)
        coherence_features = [k for k in features_dict.keys() if 'coherence' in k or 'plv' in k]
        if coherence_features:
            conn_avg = np.mean([features_dict[k] for k in coherence_features])
            conn_score = 1.0 - conn_avg  # Lower coherence = higher score
            di += conn_score * 0.20
            count += 1
        
        # Normalize
        if count > 0:
            di = di / (count / 5.0)  # Assuming 5 features ideally
        
        return min(max(di, 0.0), 1.0)
    
    def map_di_to_stage(self, di: float, probs: np.ndarray = None) -> str:
        """
        Map Depression Index to stage
        """
        if probs is not None:
            # Use model prediction
            predicted_idx = np.argmax(probs)
            return self.class_names[predicted_idx]
        
        # Use DI thresholds
        if di < 0.25:
            return "Not Depressed"
        elif di < 0.50:
            return "Mild"
        elif di < 0.75:
            return "Moderate"
        else:
            return "Severe"
    
    def save_stats(self, path: str):
        """Save population statistics"""
        data = {
            'feature_stats': self.feature_stats,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_stats(self, path: str):
        """Load population statistics"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.feature_stats = data['feature_stats']
        self.feature_names = data['feature_names']
        self.class_names = data.get('class_names', self.class_names)
