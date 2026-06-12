# 🎙️ Voice Recognition System

A real-time voice command recognition system built using **Machine Learning (SVM)** and **MFCC audio features**, with a live web-based interface for instant predictions through the browser microphone.

---

## 📌 Project Overview

This project recognizes spoken voice commands in real time and classifies them into one of **11 predefined categories**. It is designed as an end-to-end machine learning pipeline — from voice data collection, feature extraction, model training, evaluation, and deployment through a Flask web application.

---

## 🗣️ Voice Commands (11 Classes)

```
left | right | yes | no
down | speak | rcb | class | mine
```

---

## 🧠 Model Performance

| Metric              | Value                         |
|---------------------|-------------------------------|
| Test Accuracy        | **99.40%**                     |
| Correct Predictions  | 199 / 201                      |
| Misclassified        | 2 / 201                        |
| Cross-Validation Acc | ~96–97% (±1.5%)               |
| Total Dataset Size   | ~1,100 voice samples /660 samples         |
| Model Used           | SVM (RBF Kernel)               |
| Feature Extraction   | MFCC + Delta + Delta² (240 features) |

---

## ⚙️ Tech Stack

- **Python**
- **Flask** – Web backend
- **Scikit-learn** – SVM model, GridSearchCV, evaluation
- **Librosa** – Audio processing & MFCC extraction
- **NumPy / Pandas** – Data handling
- **HTML / CSS / JavaScript** – Real-time web interface
- **ffmpeg** – Audio format conversion

---

## 🔬 How It Works

1. **Data Collection** (`collect_data.py`)
   Records voice samples for each command and saves them as `.wav` files.

2. **Feature Extraction**
   Extracts MFCC, Delta, and Delta² coefficients (mean & std) → 240-dimensional feature vector per sample.

3. **Model Training** (`train_voice_model.py`)
   - Splits data into train/test sets
   - Scales features using `StandardScaler`
   - Tunes hyperparameters using `GridSearchCV`
   - Trains an SVM (RBF kernel) classifier
   - Evaluates using accuracy, classification report, and confusion matrix

4. **Real-Time Inference** (`app.py`)
   - Flask backend receives audio from the browser microphone
   - Converts audio using ffmpeg
   - Extracts features and predicts the spoken command
   - Displays results live on the web UI with confidence scores

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Voice Data
```bash
python collect_data.py
```

### 3. Train the Model
```bash
python train_voice_model.py
```

### 4. Run the Web App
```bash
python app.py
```

The app will automatically open in your browser at:
```
http://localhost:5000
```

---

## 📁 Project Structure

```
Voice Recognition System/
│
├── app.py                  # Flask web application (real-time prediction)
├── train_voice_model.py    # Model training & evaluation script
├── collect_data.py         # Voice data collection script
│
├── models/
│   ├── voice_model.pkl     # Trained SVM model
│   ├── scaler.pkl          # Feature scaler
│   └── label_encoder.pkl   # Label encoder
│
├── dataset/                 # Recorded voice samples (per class folders)
│
└── README.md
```

---

## 📊 Sample Results

```
Accuracy : 0.9900
Correct  : 199 / 201
Errors   : 2 / 201

Classification Report:
              precision    recall  f1-score   support
        left       1.00      1.00      1.00        20
          no       1.00      1.00      1.00        20
       right       0.94      1.00      0.97        17
         yes       1.00      0.95      0.97        20
        down       1.00      1.00      1.00        20
       speak       1.00      1.00      1.00        19
         rcb       1.00      1.00      1.00        17
       class       1.00      0.94      0.97        16
        mine       1.00      1.00      1.00        16
```

---

## 🔮 Future Improvements

- Expand to support more voice commands
- Improve noise robustness with augmented training data
- Add support for continuous speech / multi-word commands
- Deploy on cloud for public access
- Mobile app integration

---

## 👤 Author

**Voice Recognition System** —  MCA Project  
Built with Python, Machine Learning, and real-time web technology.

---

## 📄 License

This project is open-source and available for educational use.
