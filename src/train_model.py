import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset"
CLASSES      = ["left", "no", "right", "start", "stop", "yes"]
N_MFCC       = 40
FEATURE_SIZE = N_MFCC * 6   # mean+std for mfcc, delta, delta2  →  240
SR           = 16000
MIN_SAMPLES  = 1000
TEST_SIZE    = 0.20
RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path: str) -> np.ndarray:
    """
    Returns a 1-D feature vector of length FEATURE_SIZE (240).
    On failure / too-short audio → returns zero vector of correct size.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR)

        # Normalize amplitude
        y = librosa.util.normalize(y)

        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Guard: audio too short to be useful
        if len(y) < MIN_SAMPLES:
            return np.zeros(FEATURE_SIZE)          # ✅ Fixed: was 120 (wrong size)

        mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta   = librosa.feature.delta(mfcc)
        delta2  = librosa.feature.delta(mfcc, order=2)

        features = np.concatenate([
            np.mean(mfcc.T,   axis=0), np.std(mfcc.T,   axis=0),
            np.mean(delta.T,  axis=0), np.std(delta.T,  axis=0),
            np.mean(delta2.T, axis=0), np.std(delta2.T, axis=0),
        ])

        return features

    except Exception as e:
        print(f"  [WARN] Feature extraction failed for {file_path}: {e}")
        return np.zeros(FEATURE_SIZE)


# -----------------------------
# LOAD DATASET
# -----------------------------
print("=" * 50)
print("  VOICE COMMAND CLASSIFIER — SVM TRAINER")
print("=" * 50)
print("\nLoading dataset...")

X, y_labels = [], []

for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"  [WARN] Folder not found: {folder}")
        continue

    wav_files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"  {label:>6}: {len(wav_files)} files found")

    for file in wav_files:
        features = extract_features(os.path.join(folder, file))
        X.append(features)
        y_labels.append(label)

X        = np.array(X)
y_labels = np.array(y_labels)

print(f"\nDataset loaded  →  {len(X)} samples | Feature size: {X.shape[1]}")

if len(X) == 0:
    raise RuntimeError("No data found. Check DATASET_PATH and folder names.")

# -----------------------------
# ENCODE LABELS
# -----------------------------
le = LabelEncoder()
y  = le.fit_transform(y_labels)
joblib.dump(le, "models/label_encoder.pkl")
print(f"Classes: {list(le.classes_)}")

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,       # 20% → more reliable evaluation
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\nSplit  →  Train: {len(X_train)} | Test: {len(X_test)}")

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved.")

# -----------------------------
# HYPERPARAMETER SEARCH
# -----------------------------
print("\nRunning GridSearchCV (this may take ~30 s)...")

param_grid = {
    "C":     [1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01],
}

grid_search = GridSearchCV(
    SVC(kernel="rbf", probability=True),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best params: C={best_params['C']}, gamma={best_params['gamma']}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_

# -----------------------------
# CROSS-VALIDATION
# -----------------------------
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------
# FINAL EVALUATION
# -----------------------------
y_pred        = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\n" + "=" * 50)
print("  EVALUATION ON TEST SET")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=CLASSES))
print("Confusion Matrix:")
print(confusion_matrix(y_test_labels, y_pred_labels, labels=CLASSES))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "models/voice_model.pkl")
print("\nModel saved  →  models/voice_model.pkl")
print("Scaler saved  →  models/scaler.pkl")
print("Encoder saved →  models/label_encoder.pkl")
print("\nDone!")