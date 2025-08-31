from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Random Forest Bank APK Detector")

# Paths to save model
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
FEATURE_SELECTOR_PATH = os.path.join(MODEL_DIR, "feature_selector.pkl")

# Request schema
class Features(BaseModel):
    features: list  # list of feature lists

# Endpoint to save model
@app.post("/save_model")
def save_model():
    joblib.dump(rf_freq, MODEL_PATH)
    joblib.dump(freq_features, FEATURE_SELECTOR_PATH)
    return {"status": "success", "message": "Random Forest model saved!"}

# Endpoint to predict
@app.post("/predict")
def predict(data: Features):
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURE_SELECTOR_PATH)

    X = np.array(data.features)
    X_selected = X[:, selected_features]

    y_pred = model.predict(X_selected)
    y_prob = model.predict_proba(X_selected)[:, 1]

    pred_labels = ["Benign" if p==0 else "Malicious" for p in y_pred]
    results = [{"Prediction": label, "Probability": float(prob)} for label, prob in zip(pred_labels, y_prob)]
    return {"status": "success", "results": results}
