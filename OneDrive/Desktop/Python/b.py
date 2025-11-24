"""
Multi-Class Disease Prediction System
- Train with a CSV (or generate synthetic data if CSV not present)
- Save model to disk
- Provide a Flask API for predictions

Expected CSV format (example):
symptom_fever,symptom_cough,symptom_headache,symptom_fatigue,disease
1,0,1,0,Flu
0,1,0,0,Cold
...

If you don't have a CSV named 'data.csv' in the same folder, this script will generate synthetic data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from flask import Flask, request, jsonify

MODEL_PATH = "disease_model.joblib"
META_PATH = "meta.joblib"
CSV_PATH = "data.csv"
RANDOM_STATE = 42

def generate_synthetic_data(n_samples=1000, random_state=RANDOM_STATE):
    """
    Generates a simple synthetic dataset of binary symptoms and multiple diseases.
    Symptoms: fever, cough, headache, fatigue, nausea, shortness_of_breath, sore_throat
    Diseases: Flu, Common Cold, Migraine, Food Poisoning, Asthma
    """
    rng = np.random.RandomState(random_state)
    symptoms = [
        "fever", "cough", "headache", "fatigue",
        "nausea", "shortness_of_breath", "sore_throat"
    ]
    diseases = ["Flu", "Common Cold", "Migraine", "Food Poisoning", "Asthma"]
    rows = []
    for _ in range(n_samples):
        # sample disease first (so symptoms correlate with disease)
        disease = rng.choice(diseases, p=[0.25, 0.25, 0.2, 0.15, 0.15])
        s = {sym: 0 for sym in symptoms}
        # set symptom patterns per disease (simple deterministic-ish rules + noise)
        if disease == "Flu":
            s["fever"] = 1
            s["cough"] = rng.binomial(1, 0.7)
            s["fatigue"] = rng.binomial(1, 0.6)
        elif disease == "Common Cold":
            s["cough"] = 1
            s["sore_throat"] = rng.binomial(1, 0.6)
            s["fever"] = rng.binomial(1, 0.2)
        elif disease == "Migraine":
            s["headache"] = 1
            s["nausea"] = rng.binomial(1, 0.4)
            s["fatigue"] = rng.binomial(1, 0.3)
        elif disease == "Food Poisoning":
            s["nausea"] = 1
            s["fatigue"] = rng.binomial(1, 0.3)
            s["fever"] = rng.binomial(1, 0.2)
        elif disease == "Asthma":
            s["shortness_of_breath"] = 1
            s["cough"] = rng.binomial(1, 0.4)
        # add noise: flip some symptoms randomly
        for sym in symptoms:
            if rng.rand() < 0.05:
                s[sym] = 1 - s[sym]
        row = {f"symptom_{sym}": s[sym] for sym in symptoms}
        row["disease"] = disease
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def load_data(csv_path=CSV_PATH):
    if os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("data.csv not found — generating synthetic dataset for demo.")
        df = generate_synthetic_data()
        df.to_csv(csv_path, index=False)
        print(f"Synthetic data saved to {csv_path}")
    return df

def prepare_and_train(df, test_size=0.2, random_state=RANDOM_STATE):
    # Expect disease label column named 'disease', and symptom columns are the rest
    if "disease" not in df.columns:
        raise ValueError("Input dataframe must contain a 'disease' column as the target.")
    X = df.drop(columns=["disease"])
    y = df["disease"].astype(str)

    # For safety: ensure all features are numeric. If symptoms are strings, encode them.
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # Pipeline: scaler + classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=random_state))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model + meta (label encoder + feature names)
    joblib.dump(pipeline, MODEL_PATH)
    meta = {
        "label_encoder": le,
        "feature_names": list(X.columns)
    }
    joblib.dump(meta, META_PATH)
    print(f"Saved model -> {MODEL_PATH}, meta -> {META_PATH}")
    return pipeline, le, X.columns

def load_model(model_path=MODEL_PATH, meta_path=META_PATH):
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or meta file not found. Train first by running this script.")
    pipeline = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return pipeline, meta["label_encoder"], meta["feature_names"]

# Flask REST API
app = Flask(__name__)
# Load model on startup if present
MODEL_LOADED = False
MODEL = None
LABEL_ENCODER = None
FEATURE_NAMES = None

@app.route("/train", methods=["POST"])
def api_train():
    """
    Train endpoint: triggers training using CSV (or synthetic data).
    POST JSON body (optional):
    {
      "csv_path": "data.csv",
      "test_size": 0.2
    }
    """
    body = request.get_json(silent=True) or {}
    csv_path = body.get("csv_path", CSV_PATH)
    test_size = float(body.get("test_size", 0.2))
    df = load_data(csv_path)
    pipeline, le, feat_names = prepare_and_train(df, test_size=test_size)
    global MODEL_LOADED, MODEL, LABEL_ENCODER, FEATURE_NAMES
    MODEL = pipeline
    LABEL_ENCODER = le
    FEATURE_NAMES = list(feat_names)
    MODEL_LOADED = True
    return jsonify({
        "status": "trained",
        "model_path": MODEL_PATH,
        "meta_path": META_PATH,
        "features": FEATURE_NAMES,
        "classes": list(le.classes_)
    })

@app.route("/predict", methods=["POST"])
def api_predict():
    """
    Predict endpoint: Accepts JSON with features.
    Example JSON:
    {
      "symptom_fever": 1,
      "symptom_cough": 0,
      "symptom_headache": 1,
      ...
    }
    OR a list of records:
    {
      "records": [
         {"symptom_fever":1, "symptom_cough":0, ...},
         {"symptom_fever":0, "symptom_cough":1, ...}
      ]
    }
    """
    global MODEL_LOADED, MODEL, LABEL_ENCODER, FEATURE_NAMES
    if not MODEL_LOADED:
        try:
            MODEL, LABEL_ENCODER, FEATURE_NAMES = load_model()
            MODEL_LOADED = True
        except Exception as e:
            return jsonify({"error": "Model not loaded. Train first or call /train", "details": str(e)}), 400

    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Empty JSON body"}), 400

    records = None
    if "records" in payload and isinstance(payload["records"], list):
        records = payload["records"]
    else:
        # single record
        records = [payload]

    # Build DataFrame ensuring consistent feature order and missing features filled with 0
    df = pd.DataFrame(records)
    # If features in model not present, add them with zero
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    # Keep only feature columns and preserve order
    df = df[FEATURE_NAMES].apply(pd.to_numeric, errors='coerce').fillna(0)

    preds_enc = MODEL.predict(df)
    probs = MODEL.predict_proba(df) if hasattr(MODEL, "predict_proba") else None
    preds = LABEL_ENCODER.inverse_transform(preds_enc)

    response = []
    for i in range(len(preds)):
        item = {"prediction": preds[i]}
        if probs is not None:
            prob_dict = {cls: float(probs[i][idx]) for idx, cls in enumerate(LABEL_ENCODER.classes_)}
            item["probabilities"] = prob_dict
        response.append(item)

    # If single input, return single object
    if len(response) == 1:
        return jsonify(response[0])
    return jsonify(response)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})

if __name__ == "__main__":
    # When run directly, train (if model missing) and start flask
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        df = load_data(CSV_PATH)
        prepare_and_train(df)

    # Load model into memory
    MODEL, LABEL_ENCODER, FEATURE_NAMES = load_model()
    MODEL_LOADED = True
    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)
