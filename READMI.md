# Voice Recognition System
The Voice Recognition System records audio from the microphone, extracts audio features, and predicts predefined voice commands such as left, right, start, stop, yes, and no. The command with the highest confidence score is selected as the recognized command, while low-confidence predictions are classified as unknown.

works flow

1. Collect WAV files into dataset/start, dataset/stop, etc.
2. Install requirements.
3. Run training:
   python src/train_model.py
4. Run prediction:
   python main.py


