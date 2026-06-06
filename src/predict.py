import os
import joblib
import numpy as np

from src.feature_extraction import extract_features

MODEL_PATH = "models/voice_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def predict(audio_file):

    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return

    if not os.path.exists(SCALER_PATH):
        print("Scaler not found!")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(audio_file)

    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    # Get probabilities
    probs = model.predict_proba(features)[0]

    print("\nPrediction Probabilities:")
    print("-" * 30)

    for label, prob in zip(model.classes_, probs):
        print(f"{label:10s}: {prob*100:.2f}%")

    result = model.classes_[np.argmax(probs)]
    confidence = np.max(probs)

    print("\nRecognized Command :", result)
    print(
        "Confidence         :",
        round(confidence * 100, 2),
        "%"
    )