# """
# FastAPI Backend
# Endpoints for training, prediction, and analysis
# """

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import Optional, Dict, List
# import numpy as np
# import torch
# import json
# from pathlib import Path
# import traceback

# # Import our modules
# from backend.preprocessing import EEGPreprocessor
# from backend.features import EEGFeatureExtractor
# from embedding import EEGEmbedding
# from model_transformer import EEGTransformer, save_model, load_model
# from trainer import EEGTrainer, EEGDataset
# from explainers import EEGExplainer
# from backend.plots import EEGPlotter

# # Initialize FastAPI
# app = FastAPI(
#     title="EEG Depression Analysis API",
#     description="Backend API for EEG-based depression detection using Transformers",
#     version="1.0.0"
# )

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global state
# MODEL_PATH = Path("saved_models")
# ARTIFACTS_PATH = Path("artifacts")
# MODEL_PATH.mkdir(exist_ok=True)
# ARTIFACTS_PATH.mkdir(exist_ok=True)

# current_model = None
# current_embedding = None
# current_explainer = None
# current_metadata = {}


# class TrainConfig(BaseModel):
#     epochs: int = 50
#     batch_size: int = 16
#     lr: float = 1e-4
#     patience: int = 10
#     d_model: int = 128
#     nhead: int = 8
#     num_layers: int = 4


# class PredictResponse(BaseModel):
#     subject_id: Optional[str]
#     predicted_class: str
#     probabilities: Dict[str, float]
#     depression_index: float
#     dominant_wave: str
#     top_features: List[Dict]
#     explanation_text: str
#     graphs: Dict[str, str]
#     metrics: Dict


# @app.get("/")
# async def root():
#     return {
#         "message": "EEG Depression Analysis API",
#         "version": "1.0.0",
#         "endpoints": ["/train", "/predict", "/analyze", "/metrics"]
#     }


# @app.post("/train")
# async def train_model(
#     file: UploadFile = File(...),
#     config: Optional[str] = None
# ):
#     """
#     Train the model on uploaded EEG dataset
#     Expects CSV with structure: [subject_id, label, ...channels...]
#     """
#     try:
#         # Parse config
#         if config:
#             train_config = TrainConfig(**json.loads(config))
#         else:
#             train_config = TrainConfig()
        
#         # Read uploaded file
#         contents = await file.read()
        
#         # Save temporarily
#         temp_path = ARTIFACTS_PATH / "temp_train_data.csv"
#         with open(temp_path, 'wb') as f:
#             f.write(contents)
        
#         # Load and preprocess
#         print("Loading and preprocessing data...")
#         preprocessor = EEGPreprocessor()
#         raw_data, epochs, metadata = preprocessor.preprocess_pipeline(str(temp_path), apply_ica=True)
        
#         # Extract features
#         print("Extracting features...")
#         feature_extractor = EEGFeatureExtractor(fs=preprocessor.fs)
#         features = feature_extractor.build_feature_matrix(epochs, metadata.get('channel_names'))
        
#         # Dummy labels for demo (replace with actual labels from CSV)
#         # In production, parse labels from CSV
#         n_samples = features.shape[0]
#         labels = np.random.randint(0, 4, n_samples)  # 4 classes: 0-3
        
#         # Split data (80/10/10)
#         n_train = int(n_samples * 0.8)
#         n_val = int(n_samples * 0.1)
        
#         X_train = features[:n_train]
#         y_train = labels[:n_train]
#         X_val = features[n_train:n_train+n_val]
#         y_val = labels[n_train:n_train+n_val]
#         X_test = features[n_train+n_val:]
#         y_test = labels[n_train+n_val:]
        
#         # Create embedding
#         print("Creating embeddings...")
#         embedding = EEGEmbedding(d_model=train_config.d_model)
#         X_train_scaled = embedding.fit_transform(X_train.reshape(n_train, 1, -1))
#         X_val_scaled = embedding.transform(X_val.reshape(n_val, 1, -1))
#         X_test_scaled = embedding.transform(X_test.reshape(len(X_test), 1, -1))
        
#         # Create datasets
#         train_dataset = EEGDataset(X_train_scaled, y_train)
#         val_dataset = EEGDataset(X_val_scaled, y_val)
#         test_dataset = EEGDataset(X_test_scaled, y_test)
        
#         # Initialize model
#         print("Initializing model...")
#         model = EEGTransformer(
#             feature_dim=features.shape[1],
#             d_model=train_config.d_model,
#             nhead=train_config.nhead,
#             num_encoder_layers=train_config.num_layers,
#             num_classes=4
#         )
        
#         # Train
#         print("Training...")
#         trainer = EEGTrainer(model, device='cpu')
#         train_results = trainer.train(
#             train_dataset,
#             val_dataset,
#             epochs=train_config.epochs,
#             batch_size=train_config.batch_size,
#             lr=train_config.lr,
#             patience=train_config.patience,
#             save_path=str(MODEL_PATH / "best_model.pth")
#         )
        
#         # Evaluate
#         print("Evaluating...")
#         test_results = trainer.evaluate(
#             test_dataset,
#             class_names=['Not Depressed', 'Mild', 'Moderate', 'Severe']
#         )
        
#         # Save artifacts
#         embedding.save(str(MODEL_PATH / "embedding.json"))
        
#         # Fit explainer
#         explainer = EEGExplainer()
#         # Get feature names (simplified - you'd want to track these properly)
#         feature_names = [f"feature_{i}" for i in range(features.shape[1])]
#         explainer.fit_population_stats(features, feature_names)
#         explainer.save_stats(str(MODEL_PATH / "explainer_stats.json"))
        
#         # Generate training plots
#         plotter = EEGPlotter()
#         plot_training = plotter.plot_training_metrics(trainer.history)
#         plot_cm = plotter.plot_confusion_matrix(
#             np.array(test_results['confusion_matrix']),
#             test_results['class_names']
#         )
#         plot_roc = plotter.plot_roc_curves(
#             np.array(test_results['labels']),
#             np.array(test_results['probabilities']),
#             test_results['class_names']
#         )
        
#         # Save training report
#         report = {
#             'train_config': train_config.dict(),
#             'train_results': train_results,
#             'test_results': test_results,
#             'metadata': metadata
#         }
        
#         with open(MODEL_PATH / "training_report.json", 'w') as f:
#             json.dump(report, f, indent=2)
        
#         # Update global state
#         global current_model, current_embedding, current_explainer, current_metadata
#         current_model = model
#         current_embedding = embedding
#         current_explainer = explainer
#         current_metadata = metadata
        
#         return {
#             "status": "success",
#             "message": "Training completed successfully",
#             "results": {
#                 "accuracy": test_results['accuracy'],
#                 "f1_macro": test_results['f1_macro'],
#                 "roc_auc": test_results['roc_auc_macro'],
#                 "epochs_trained": train_results['epochs_trained'],
#                 "training_time": train_results['training_time']
#             },
#             "plots": {
#                 "training_metrics": plot_training,
#                 "confusion_matrix": plot_cm,
#                 "roc_curves": plot_roc
#             }
#         }
        
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):
#     """
#     Predict depression stage from uploaded EEG file
#     Returns comprehensive analysis with plots and explanations
#     """
#     try:
#         # Load model if not loaded
#         global current_model, current_embedding, current_explainer
        
#         if current_model is None:
#             print("Loading model...")
#             current_model, metadata = load_model(str(MODEL_PATH / "best_model.pth"))
#             current_embedding = EEGEmbedding()
#             current_embedding.load(str(MODEL_PATH / "embedding.json"))
#             current_explainer = EEGExplainer()
#             current_explainer.load_stats(str(MODEL_PATH / "explainer_stats.json"))
        
#         # Read file
#         contents = await file.read()
        
# # Save uploaded file to temp path
# temp_path = ARTIFACTS_PATH / ("temp_predict.csv" if file.filename.endswith('.csv') else "temp_predict.edf")
# with open(temp_path, "wb") as f:
#     f.write(contents)

# # Preprocess using temp file
# print("Preprocessing...")
# preprocessor = EEGPreprocessor()
# raw_data, epochs, metadata = preprocessor.preprocess_pipeline(
#     str(temp_path),
#     apply_ica=True
# )

        
#         # Extract features
#         print("Extracting features...")
#         feature_extractor = EEGFeatureExtractor(fs=preprocessor.fs)
#         features = feature_extractor.build_feature_matrix(epochs, metadata.get('channel_names'))
        
#         # Build sequence
#         features_scaled = current_embedding.transform(features.reshape(len(features), 1, -1))
#         X_tensor = torch.FloatTensor(features_scaled).transpose(0, 1)  # (seq_len, 1, feat_dim)
        
#         # Predict
#         print("Predicting...")
#         current_model.eval()
#         with torch.no_grad():
#             logits = current_model(X_tensor)
#             probs = torch.softmax(logits, dim=1).numpy()[0]
        
#         predicted_class_idx = np.argmax(probs)
#         class_names = ['Not Depressed', 'Mild', 'Moderate', 'Severe']
#         predicted_class = class_names[predicted_class_idx]
        
#         # Probabilities dict
#         probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        
#         # Compute depression index
#         # Average features over epochs for DI calculation
#         features_dict = {}
#         if current_explainer.feature_names:
#             features_mean = np.mean(features, axis=0)
#             for i, name in enumerate(current_explainer.feature_names):
#                 if i < len(features_mean):
#                     features_dict[name] = float(features_mean[i])
        
#         depression_index = current_explainer.compute_depression_index(features_dict)
        
#         # SHAP explanations
#         print("Computing explanations...")
#         shap_values = current_explainer._gradient_based_importance(current_model, features_scaled)
#         top_features = current_explainer.get_top_features(shap_values, features_scaled[0], top_k=5)
        
#         # Rule-based explanations
#         explanation_text = current_explainer.rule_based_explanations(features_dict, top_features)
        
#         # Dominant wave (find band with highest relative power)
#         band_powers = {}
#         for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
#             key = f'rel_power_{band_name}'
#             matching_keys = [k for k in features_dict.keys() if key in k.lower()]
#             if matching_keys:
#                 band_powers[band_name] = np.mean([features_dict[k] for k in matching_keys])
        
#         dominant_wave = max(band_powers, key=band_powers.get).capitalize() if band_powers else "Unknown"
        
#         # Generate plots
#         print("Generating plots...")
#         plotter = EEGPlotter(fs=preprocessor.fs)
        
#         graphs = {}
#         graphs['time_plot'] = plotter.plot_time_series(raw_data, channel_idx=0, 
#                                                        channel_name=metadata['channel_names'][0])
#         graphs['psd_plot'] = plotter.plot_psd(raw_data, channel_idx=0,
#                                              channel_name=metadata['channel_names'][0])
#         graphs['bandpower_bar'] = plotter.plot_bandpower_bar(band_powers)
#         graphs['scalogram'] = plotter.plot_scalogram(raw_data, channel_idx=0,
#                                                      channel_name=metadata['channel_names'][0])
        
#         # FAA plot if available
#         if 'frontal_alpha_asymmetry' in features_dict:
#             graphs['asymmetry_plot'] = plotter.plot_asymmetry(features_dict['frontal_alpha_asymmetry'])
        
#         # Connectivity (dummy for now - would need channel pair info)
#         # graphs['connectivity_matrix'] = plotter.plot_connectivity_matrix(
#         #     np.random.rand(4, 4), labels=['F3', 'F4', 'P3', 'P4']
#         # )
        
#         # Feature importance
#         if len(shap_values) > 0:
#             graphs['feature_importance_shap'] = plotter.plot_feature_importance(
#                 shap_values, current_explainer.feature_names or [f"F{i}" for i in range(len(shap_values))],
#                 top_k=10
#             )
        
#         # Load metrics from training
#         try:
#             with open(MODEL_PATH / "training_report.json", 'r') as f:
#                 training_report = json.load(f)
#                 metrics = training_report['test_results']
#         except:
#             metrics = {}
        
#         return PredictResponse(
#             subject_id=file.filename,
#             predicted_class=predicted_class,
#             probabilities=probs_dict,
#             depression_index=float(depression_index),
#             dominant_wave=dominant_wave,
#             top_features=top_features,
#             explanation_text=explanation_text,
#             graphs=graphs,
#             metrics=metrics
#         )
        
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/metrics")
# async def get_metrics():
#     """
#     Get last trained model metrics
#     """
#     try:
#         with open(MODEL_PATH / "training_report.json", 'r') as f:
#             report = json.load(f)
#         return report
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="No trained model found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "model_loaded": current_model is not None
#     }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


"""
FastAPI Backend
Endpoints for training, prediction, and analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import numpy as np
import torch
import json
from pathlib import Path
import traceback

# Import our modules
from backend.preprocessing import EEGPreprocessor
from backend.features import EEGFeatureExtractor
from backend.model_transformer import EEGTransformer, save_model, load_model
from backend.trainer import EEGTrainer, EEGDataset
from backend.explainers import EEGExplainer
from backend.plots import EEGPlotter


# Initialize FastAPI
app = FastAPI(
    title="EEG Depression Analysis API",
    description="Backend API for EEG-based depression detection using Transformers",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL_PATH = Path("saved_models")
ARTIFACTS_PATH = Path("artifacts")
MODEL_PATH.mkdir(exist_ok=True)
ARTIFACTS_PATH.mkdir(exist_ok=True)

current_model = None
current_embedding = None
current_explainer = None
current_metadata = {}


class TrainConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    patience: int = 10
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4


class PredictResponse(BaseModel):
    subject_id: Optional[str]
    predicted_class: str
    probabilities: Dict[str, float]
    depression_index: float
    dominant_wave: str
    top_features: List[Dict]
    explanation_text: str
    graphs: Dict[str, str]
    metrics: Dict


@app.get("/")
async def root():
    return {
        "message": "EEG Depression Analysis API",
        "version": "1.0.0",
        "endpoints": ["/train", "/predict", "/analyze", "/metrics"]
    }


@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    config: Optional[str] = None
):
    """
    Train the model on uploaded EEG dataset
    Expects CSV with structure: [subject_id, label, ...channels...]
    """
    try:
        # Parse config
        if config:
            train_config = TrainConfig(**json.loads(config))
        else:
            train_config = TrainConfig()
        
        # Read uploaded file
        contents = await file.read()
        
        # Save temporarily
        temp_path = ARTIFACTS_PATH / "temp_train_data.csv"
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Load and preprocess
        print("Loading and preprocessing data...")
        preprocessor = EEGPreprocessor()
        raw_data, epochs, metadata = preprocessor.preprocess_pipeline(str(temp_path), apply_ica=True)
        
        # Extract features
        print("Extracting features...")
        feature_extractor = EEGFeatureExtractor(fs=preprocessor.fs)
        features = feature_extractor.build_feature_matrix(epochs, metadata.get('channel_names'))
        
        # Dummy labels for demo (replace with actual labels from CSV)
        n_samples = features.shape[0]
        labels = np.random.randint(0, 4, n_samples)  # 4 classes: 0-3
        
        # Split data (80/10/10)
        n_train = int(n_samples * 0.8)
        n_val = int(n_samples * 0.1)
        
        X_train = features[:n_train]
        y_train = labels[:n_train]
        X_val = features[n_train:n_train+n_val]
        y_val = labels[n_train:n_train+n_val]
        X_test = features[n_train+n_val:]
        y_test = labels[n_train+n_val:]
        
        # Create embedding
        print("Creating embeddings...")
        embedding = EEGEmbedding(d_model=train_config.d_model)
        X_train_scaled = embedding.fit_transform(X_train.reshape(n_train, 1, -1))
        X_val_scaled = embedding.transform(X_val.reshape(n_val, 1, -1))
        X_test_scaled = embedding.transform(X_test.reshape(len(X_test), 1, -1))
        
        # Create datasets
        train_dataset = EEGDataset(X_train_scaled, y_train)
        val_dataset = EEGDataset(X_val_scaled, y_val)
        test_dataset = EEGDataset(X_test_scaled, y_test)
        
        # Initialize model
        print("Initializing model...")
        model = EEGTransformer(
            feature_dim=features.shape[1],
            d_model=train_config.d_model,
            nhead=train_config.nhead,
            num_encoder_layers=train_config.num_layers,
            num_classes=4
        )
        
        # Train
        print("Training...")
        trainer = EEGTrainer(model, device='cpu')
        train_results = trainer.train(
            train_dataset,
            val_dataset,
            epochs=train_config.epochs,
            batch_size=train_config.batch_size,
            lr=train_config.lr,
            patience=train_config.patience,
            save_path=str(MODEL_PATH / "best_model.pth")
        )
        
        # Evaluate
        print("Evaluating...")
        test_results = trainer.evaluate(
            test_dataset,
            class_names=['Not Depressed', 'Mild', 'Moderate', 'Severe']
        )
        
        # Save artifacts
        embedding.save(str(MODEL_PATH / "embedding.json"))
        
        # Fit explainer
        explainer = EEGExplainer()
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        explainer.fit_population_stats(features, feature_names)
        explainer.save_stats(str(MODEL_PATH / "explainer_stats.json"))
        
        # Generate training plots
        plotter = EEGPlotter()
        plot_training = plotter.plot_training_metrics(trainer.history)
        plot_cm = plotter.plot_confusion_matrix(
            np.array(test_results['confusion_matrix']),
            test_results['class_names']
        )
        plot_roc = plotter.plot_roc_curves(
            np.array(test_results['labels']),
            np.array(test_results['probabilities']),
            test_results['class_names']
        )
        
        # Save training report
        report = {
            'train_config': train_config.dict(),
            'train_results': train_results,
            'test_results': test_results,
            'metadata': metadata
        }
        
        with open(MODEL_PATH / "training_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Update global state
        global current_model, current_embedding, current_explainer, current_metadata
        current_model = model
        current_embedding = embedding
        current_explainer = explainer
        current_metadata = metadata
        
        return {
            "status": "success",
            "message": "Training completed successfully",
            "results": {
                "accuracy": test_results['accuracy'],
                "f1_macro": test_results['f1_macro'],
                "roc_auc": test_results['roc_auc_macro'],
                "epochs_trained": train_results['epochs_trained'],
                "training_time": train_results['training_time']
            },
            "plots": {
                "training_metrics": plot_training,
                "confusion_matrix": plot_cm,
                "roc_curves": plot_roc
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict depression stage from uploaded EEG file
    Returns comprehensive analysis with plots and explanations
    """
    try:
        # Load model if not loaded
        global current_model, current_embedding, current_explainer
        
        if current_model is None:
            print("Loading model...")
            current_model, metadata = load_model(str(MODEL_PATH / "best_model.pth"))
            current_embedding = EEGEmbedding()
            current_embedding.load(str(MODEL_PATH / "embedding.json"))
            current_explainer = EEGExplainer()
            current_explainer.load_stats(str(MODEL_PATH / "explainer_stats.json"))
        
        # Read file
        contents = await file.read()

        # --- FIXED TEMP FILE HANDLING ---
        temp_filename = "temp_predict.csv" if file.filename.endswith(".csv") else "temp_predict.edf"
        temp_path = ARTIFACTS_PATH / temp_filename
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        print("Preprocessing...")
        preprocessor = EEGPreprocessor()
        raw_data, epochs, metadata = preprocessor.preprocess_pipeline(
            str(temp_path),
            apply_ica=True
        )
        # --- END FIX ---

        # Extract features
        print("Extracting features...")
        feature_extractor = EEGFeatureExtractor(fs=preprocessor.fs)
        features = feature_extractor.build_feature_matrix(epochs, metadata.get('channel_names'))
        
        # Build sequence
        features_scaled = current_embedding.transform(features.reshape(len(features), 1, -1))
        X_tensor = torch.FloatTensor(features_scaled).transpose(0, 1)  # (seq_len, 1, feat_dim)
        
        # Predict
        print("Predicting...")
        current_model.eval()
        with torch.no_grad():
            logits = current_model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        
        predicted_class_idx = np.argmax(probs)
        class_names = ['Not Depressed', 'Mild', 'Moderate', 'Severe']
        predicted_class = class_names[predicted_class_idx]
        
        # Probabilities dict
        probs_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        
        # Compute depression index
        features_dict = {}
        if current_explainer.feature_names:
            features_mean = np.mean(features, axis=0)
            for i, name in enumerate(current_explainer.feature_names):
                if i < len(features_mean):
                    features_dict[name] = float(features_mean[i])
        
        depression_index = current_explainer.compute_depression_index(features_dict)
        
        # SHAP explanations
        print("Computing explanations...")
        shap_values = current_explainer._gradient_based_importance(current_model, features_scaled)
        top_features = current_explainer.get_top_features(shap_values, features_scaled[0], top_k=5)
        
        # Rule-based explanations
        explanation_text = current_explainer.rule_based_explanations(features_dict, top_features)
        
        # Dominant wave
        band_powers = {}
        for band_name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            key = f'rel_power_{band_name}'
            matching_keys = [k for k in features_dict.keys() if key in k.lower()]
            if matching_keys:
                band_powers[band_name] = np.mean([features_dict[k] for k in matching_keys])
        
        dominant_wave = max(band_powers, key=band_powers.get).capitalize() if band_powers else "Unknown"
        
        # Generate plots
        print("Generating plots...")
        plotter = EEGPlotter(fs=preprocessor.fs)
        
        graphs = {}
        graphs['time_plot'] = plotter.plot_time_series(raw_data, channel_idx=0, 
                                                       channel_name=metadata['channel_names'][0])
        graphs['psd_plot'] = plotter.plot_psd(raw_data, channel_idx=0,
                                             channel_name=metadata['channel_names'][0])
        graphs['bandpower_bar'] = plotter.plot_bandpower_bar(band_powers)
        graphs['scalogram'] = plotter.plot_scalogram(raw_data, channel_idx=0,
                                                     channel_name=metadata['channel_names'][0])
        
        if 'frontal_alpha_asymmetry' in features_dict:
            graphs['asymmetry_plot'] = plotter.plot_asymmetry(features_dict['frontal_alpha_asymmetry'])
        
        if len(shap_values) > 0:
            graphs['feature_importance_shap'] = plotter.plot_feature_importance(
                shap_values, current_explainer.feature_names or [f"F{i}" for i in range(len(shap_values))],
                top_k=10
            )
        
        # Load metrics from training
        try:
            with open(MODEL_PATH / "training_report.json", 'r') as f:
                training_report = json.load(f)
                metrics = training_report['test_results']
        except:
            metrics = {}
        
        return PredictResponse(
            subject_id=file.filename,
            predicted_class=predicted_class,
            probabilities=probs_dict,
            depression_index=float(depression_index),
            dominant_wave=dominant_wave,
            top_features=top_features,
            explanation_text=explanation_text,
            graphs=graphs,
            metrics=metrics
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get last trained model metrics
    """
    try:
        with open(MODEL_PATH / "training_report.json", 'r') as f:
            report = json.load(f)
        return report
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained model found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": current_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
