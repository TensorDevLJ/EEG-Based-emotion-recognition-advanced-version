# EEG Emotion Classification Backend

CNN+LSTM model for EEG emotion detection with 16 visualization graphs.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python api.py
```

API runs at `http://localhost:8000`

## Endpoints

- `POST /train` - Train model with CSV (requires Label column)
- `POST /predict` - Predict emotion from CSV features
- `GET /health` - Health check

## Features

Your CSV should contain pre-extracted features like:
- `mean_*`, `std_*`, `skew_*`, `kurt_*`
- `freq_*` (frequency domain)
- `covM_*` (covariance)
- `Label` (for training)

## 16 Graphs Generated

1. Raw Signal
2. Filtered Signal  
3. PSD
4. Band Powers
5. Correlation Heatmap
6. PCA Variance
7. Line Graph (mean activity)
8. Bar Chart (energy)
9. Radar Chart (skew/kurt)
10. Connectivity Heatmap
11. FFT Spectrum
12. Quadrant Scatter
13. Emotion Distribution
14. Training Metrics
15. Confusion Matrix
16. ROC Curves

## Model

CNN+LSTM hybrid with PCA preprocessing (50 components).
