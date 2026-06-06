import numpy as np
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import os

# ==================================================
# CONFIG
# ==================================================
MODEL_PATH   = "models/voice_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

SAMPLE_RATE          = 16000
RECORD_DURATION      = 2       # seconds
CONFIDENCE_THRESHOLD = 0.45
N_MFCC               = 40
FEATURE_SIZE         = N_MFCC * 6   # 240
MIN_AUDIO_LEN        = 1000
VAD_RMS_THRESHOLD    = 0.008        # below this = silence / no speech

# ==================================================
# LOAD MODEL + SCALER + ENCODER
# ==================================================
print("Loading model...")

model   = joblib.load(MODEL_PATH)
scaler  = joblib.load(SCALER_PATH)
le      = joblib.load(ENCODER_PATH)          # ✅ load encoder, don't hardcode
CLASSES = list(le.classes_)                  # alphabetically sorted, guaranteed correct

print(f"Model loaded. Classes: {CLASSES}")

# ==================================================
# VOICE ACTIVITY DETECTION
# ==================================================
def has_speech(audio: np.ndarray, threshold: float = VAD_RMS_THRESHOLD) -> bool:
    """Returns True if audio contains actual speech (not silence/noise)."""
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"Audio RMS level: {rms:.4f}  (threshold: {threshold})")
    return rms > threshold

# ==================================================
# FEATURE EXTRACTION  (matches training exactly)
# ==================================================
def extract_features(file_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Normalize
        y = librosa.util.normalize(y)

        # Aggressive silence trim (top_db=25 removes more silence than default 60)
        y, _ = librosa.effects.trim(y, top_db=25)

        if len(y) < MIN_AUDIO_LEN:
            print("Warning: Audio too short after trimming.")
            return np.zeros(FEATURE_SIZE)

        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # Pad if too short for delta (fixes stop35.wav-style warnings)
        if mfcc.shape[1] < 9:
            mfcc = np.pad(mfcc, ((0, 0), (0, 9 - mfcc.shape[1])), mode='edge')

        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # ✅ Consistent with training: use .T then axis=0
        features = np.concatenate([
            np.mean(mfcc.T,   axis=0), np.std(mfcc.T,   axis=0),
            np.mean(delta.T,  axis=0), np.std(delta.T,  axis=0),
            np.mean(delta2.T, axis=0), np.std(delta2.T, axis=0),
        ])

        return features.astype(np.float32)

    except Exception as e:
        print("Feature extraction error:", e)
        return np.zeros(FEATURE_SIZE)

# ==================================================
# RECORD AUDIO
# ==================================================
def record_audio(
    filename: str = "recordings/test.wav",
    duration: int = RECORD_DURATION,
    fs: int = SAMPLE_RATE
) -> tuple[str, np.ndarray]:

    os.makedirs("recordings", exist_ok=True)

    print(f"\n{'='*40}")
    print("  Speak now (you have 2 seconds)...")
    print(f"{'='*40}")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    recording = np.squeeze(recording)

    # Save WAV
    if np.max(np.abs(recording)) > 0:
        recording_int16 = np.int16(recording / np.max(np.abs(recording)) * 32767)
    else:
        recording_int16 = np.int16(recording)

    write(filename, fs, recording_int16)
    print(f"Recording saved: {filename}")

    return filename, recording   # return raw float array for VAD check

# ==================================================
# PREDICT
# ==================================================
def predict(file_path: str, raw_audio: np.ndarray):

    # --- Voice Activity Detection ---
    if not has_speech(raw_audio):
        print("No speech detected. Please speak closer to the microphone.")
        return "unknown", 0.0, np.zeros(len(CLASSES))

    # --- Feature Extraction ---
    features = extract_features(file_path)
    print(f"Feature shape: {features.shape}")

    # --- Dimension Check ---
    expected = scaler.n_features_in_
    if len(features) != expected:
        raise ValueError(f"Feature mismatch: expected {expected}, got {len(features)}")

    features = features.reshape(1, -1)
    features = scaler.transform(features)

    # --- Predict ---
    probs    = model.predict_proba(features)[0]
    probs    = probs / probs.sum()               # re-normalize (safety)

    best_idx   = np.argmax(probs)
    best_label = CLASSES[best_idx]
    confidence = probs[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence * 100, probs

    return best_label, confidence * 100, probs

# ==================================================
# DISPLAY RESULTS
# ==================================================
def display_results(label: str, confidence: float, probs: np.ndarray):
    print("\nPrediction Probabilities")
    print("-" * 35)

    for cls, prob in zip(CLASSES, probs):
        bar    = "█" * int(prob * 30)
        marker = " ← predicted" if cls == label else ""
        print(f"  {cls:<6}: {prob*100:5.1f}%  {bar}{marker}")

    print("-" * 35)
    print(f"  Recognized : {label.upper()}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"  Threshold  : {CONFIDENCE_THRESHOLD*100:.0f}%")

# ==================================================
# MAIN LOOP
# ==================================================
if __name__ == "__main__":
    print("\nVoice Command Recognizer — Ready")
    print(f"Commands: {CLASSES}")
    print("Press Ctrl+C to quit.\n")

    while True:
        
        try:
            input("Press Enter to record...")

            file_path, raw_audio = record_audio()

            label, confidence, probs = predict(file_path, raw_audio)

            display_results(label, confidence, probs)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            