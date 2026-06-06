import librosa
import numpy as np

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=22050)

    audio, _ = librosa.effects.trim(audio)

    audio = librosa.util.normalize(audio)

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfccs)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(delta.T, axis=0)
    ])

    return features