"""
FastAPI Backend for EEG Emotion Classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import json
import os
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from model_cnn_lstm import CNN_LSTM_Model, save_model, load_model
from trainer_cnn_lstm import CNNLSTMTrainer, EEGDataset
from plots import EEGPlotter
from plot_extended import ExtendedEEGPlotter
from preprocessing import EEGPreprocessor

app = FastAPI(title="EEG Emotion Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "saved_models/model_latest.pth"
os.makedirs("saved_models", exist_ok=True)

preprocessor = EEGPreprocessor()
plotter = EEGPlotter()
ext_plotter = ExtendedEEGPlotter()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict emotion from CSV with pre-extracted features"""
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(400, "Model not trained. Use /train first.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, metadata = load_model(MODEL_PATH, device)
        
        if 'Label' in df.columns:
            df = df.drop('Label', axis=1)
        
        features = df.values
        feature_names = df.columns.tolist()
        
        # Apply stored scaler and PCA
        scaler = metadata.get('scaler')
        pca = metadata.get('pca')
        
        if scaler:
            features = scaler.transform(features)
        if pca:
            features = pca.transform(features)
        
        # Predict
        trainer = CNNLSTMTrainer(model, device)
        preds, probs = trainer.predict(features)
        
        # Generate all 16 graphs
        graphs = {}
        raw_signal = preprocessor.reconstruct_signal_from_features(df.values, feature_names)
        filtered = preprocessor.bandpass_filter(raw_signal)
        
        graphs["line_graph"] = ext_plotter.plot_line_graph(df.values, feature_names)
        graphs["bar_chart"] = ext_plotter.plot_bar_chart(df.values, feature_names)
        graphs["radar_chart"] = ext_plotter.plot_radar_chart(df.values, feature_names)
        graphs["connectivity_heatmap"] = ext_plotter.plot_connectivity_heatmap(df.values, feature_names)
        graphs["fft_spectrum"] = ext_plotter.plot_fft_spectrum(df.values, feature_names)
        
        probabilities = {f"Emotion_{i}": float(probs[0][i]) for i in range(probs.shape[1])}
        
        return {
            "predicted_class": int(preds[0]),
            "probabilities": probabilities,
            "graphs": graphs
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train")
async def train(file: UploadFile = File(...), config: str = Form('{}')):
    """Train model on uploaded CSV"""
    try:
        config_dict = json.loads(config)
        epochs = config_dict.get('epochs', 50)
        
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        labels = df['Label'].values
        features = df.drop('Label', axis=1).values
        
        # Preprocessing
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
        
        # Split and train
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        model = CNN_LSTM_Model(input_dim=features.shape[1], num_classes=len(np.unique(labels)))
        trainer = CNNLSTMTrainer(model)
        
        history = trainer.train(EEGDataset(X_train, y_train), EEGDataset(X_val, y_val), epochs=epochs)
        
        save_model(model, MODEL_PATH, {'scaler': scaler, 'pca': pca})
        
        return {"status": "success", "final_acc": history['val_acc'][-1]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
