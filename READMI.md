# Voice Recognition System
The Voice Recognition System records audio from the microphone, extracts audio features, and predicts predefined voice commands such as left, right, start, stop, yes, and no. The command with the highest confidence score is selected as the recognized command, while low-confidence predictions are classified as unknown.



# Voice Recognition System

Real-time voice command recognition using SVM + MFCC features.

## Commands
`left` | `no` | `right` | `start` | `stop` | `yes`

## Results
- Test Accuracy : 98.06%
- CV Accuracy   : 96.83% ± 1.46%
- Dataset       : 513 samples (6 classes)
- Model         : SVM (RBF kernel)
- Features      : MFCC + Delta + Delta2 (240 features)

## Run
```bash
python app.py
```
Opens at http://localhost:5000

works flow

1. Collect WAV files into dataset/start, dataset/stop, etc.
2. Install requirements.
3. Run training:
   python src/train_model.py
4. Run prediction:
   python main.py


