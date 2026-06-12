"""
app.py — Voice Command Recognizer (single-file, no templates folder needed)
Run: python app.py
Opens at: http://localhost:5000
"""

import os
import subprocess
import tempfile
import numpy as np
import librosa
import joblib
from flask import Flask, render_template_string, request, jsonify

# ==================================================
# CONFIG
# ==================================================
MODEL_PATH   = "models/voice_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# ✅ Full ffmpeg path — no PATH setup needed
FFMPEG_PATH = r"C:\Users\user\Downloads\ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"

SAMPLE_RATE          = 16000
CONFIDENCE_THRESHOLD = 0.45
N_MFCC               = 40
FEATURE_SIZE         = N_MFCC * 6
MIN_AUDIO_LEN        = 1000
VAD_RMS_THRESHOLD    = 0.008

app = Flask(__name__)

# ==================================================
# LOAD MODEL
# ==================================================
print("Loading model...")
try:
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    le      = joblib.load(ENCODER_PATH)
    CLASSES = list(le.classes_)
    print(f"Model loaded. Classes: {CLASSES}")
except FileNotFoundError as e:
    print(f"ERROR: Model file not found — {e}")
    print("Make sure models/ folder has voice_model.pkl, scaler.pkl, label_encoder.pkl")
    exit(1)

# ==================================================
# HTML
# ==================================================
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Voice Command Recognizer</title>
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet"/>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#030a0f;--panel:#060f17;--border:#0a2a3a;
      --cyan:#00e5ff;--cyan-dim:#0097aa;--green:#00ff9d;
      --red:#ff3b5c;--yellow:#ffd600;--text:#c8e6f0;--text-dim:#4a7a8a;
      --glow:0 0 18px rgba(0,229,255,0.35);
      --font-mono:'Share Tech Mono',monospace;
      --font-disp:'Orbitron',sans-serif;
    }
    html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-mono);overflow-x:hidden}
    body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px);pointer-events:none;z-index:999}
    body::after{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,229,255,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
    .wrapper{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:32px 16px 48px}
    header{text-align:center;margin-bottom:36px}
    .logo-line{font-family:var(--font-disp);font-size:clamp(22px,4vw,36px);font-weight:900;letter-spacing:0.12em;color:var(--cyan);text-shadow:var(--glow);animation:flicker 6s infinite}
    .sub-line{font-size:11px;letter-spacing:0.3em;color:var(--text-dim);margin-top:6px;text-transform:uppercase}
    @keyframes flicker{0%,95%,100%{opacity:1}96%{opacity:0.6}97%{opacity:1}98%{opacity:0.4}99%{opacity:1}}
    .main-grid{display:grid;grid-template-columns:1fr 320px;gap:20px;width:100%;max-width:960px}
    @media(max-width:700px){.main-grid{grid-template-columns:1fr}}
    .panel{background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:24px;position:relative}
    .panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:0.5}
    .panel-title{font-family:var(--font-disp);font-size:10px;letter-spacing:0.25em;color:var(--cyan-dim);text-transform:uppercase;margin-bottom:20px}
    .record-section{display:flex;flex-direction:column;align-items:center;gap:28px}
    .record-btn{width:160px;height:160px;border-radius:50%;border:2px solid var(--cyan);background:transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;transition:all 0.2s;box-shadow:0 0 20px rgba(0,229,255,0.15),inset 0 0 20px rgba(0,229,255,0.05)}
    .record-btn:hover:not(:disabled){box-shadow:0 0 40px rgba(0,229,255,0.35),inset 0 0 30px rgba(0,229,255,0.1);transform:scale(1.03)}
    .record-btn.recording{border-color:var(--red);box-shadow:0 0 40px rgba(255,59,92,0.5),inset 0 0 30px rgba(255,59,92,0.1);animation:pulse-red 1s ease-in-out infinite}
    @keyframes pulse-red{0%,100%{box-shadow:0 0 30px rgba(255,59,92,0.4),inset 0 0 20px rgba(255,59,92,0.08)}50%{box-shadow:0 0 60px rgba(255,59,92,0.7),inset 0 0 40px rgba(255,59,92,0.15)}}
    .record-btn:disabled{opacity:0.5;cursor:not-allowed}
    .mic-icon{font-size:44px;line-height:1;filter:drop-shadow(0 0 8px var(--cyan))}
    .record-btn.recording .mic-icon{filter:drop-shadow(0 0 8px var(--red))}
    .btn-label{font-family:var(--font-disp);font-size:9px;letter-spacing:0.2em;color:var(--cyan)}
    .record-btn.recording .btn-label{color:var(--red)}
    .timer-wrap{width:100%}
    .timer-label{font-size:10px;color:var(--text-dim);letter-spacing:0.15em;margin-bottom:6px}
    .timer-bar-bg{width:100%;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
    .timer-bar{height:100%;width:0%;background:var(--cyan);border-radius:2px;transition:width 0.1s linear,background 0.3s}
    .timer-bar.recording{background:var(--red)}
    .status-line{font-size:11px;letter-spacing:0.15em;color:var(--text-dim);text-align:center;min-height:18px}
    .status-line.ok{color:var(--green)}.status-line.error{color:var(--red)}.status-line.working{color:var(--yellow)}
    .result-box{text-align:center;padding:20px 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border);margin:4px 0}
    .result-word{font-family:var(--font-disp);font-size:clamp(32px,7vw,52px);font-weight:900;letter-spacing:0.08em;color:var(--cyan);text-shadow:var(--glow);min-height:64px;display:flex;align-items:center;justify-content:center;transition:all 0.3s}
    .result-word.unknown{color:var(--text-dim);text-shadow:none}
    .result-word.success{animation:pop 0.3s ease}
    @keyframes pop{0%{transform:scale(0.85);opacity:0.5}60%{transform:scale(1.08)}100%{transform:scale(1);opacity:1}}
    .confidence-row{display:flex;align-items:center;justify-content:center;gap:10px;margin-top:10px}
    .conf-label{font-size:11px;color:var(--text-dim);letter-spacing:0.1em}
    .conf-value{font-family:var(--font-disp);font-size:20px;font-weight:700;color:var(--green)}
    .conf-value.low{color:var(--yellow)}.conf-value.unknown{color:var(--text-dim)}
    .prob-list{display:flex;flex-direction:column;gap:10px}
    .prob-row{display:flex;align-items:center;gap:10px}
    .prob-class{font-size:11px;letter-spacing:0.1em;color:var(--text-dim);width:44px;text-transform:uppercase}
    .prob-bar-bg{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
    .prob-bar{height:100%;border-radius:3px;background:var(--cyan-dim);transition:width 0.5s cubic-bezier(0.4,0,0.2,1),background 0.3s;width:0%}
    .prob-bar.top{background:var(--cyan);box-shadow:0 0 8px rgba(0,229,255,0.5)}
    .prob-pct{font-size:11px;color:var(--text-dim);width:38px;text-align:right}
    .prob-pct.top{color:var(--cyan)}
    .history-list{display:flex;flex-direction:column;gap:8px;max-height:340px;overflow-y:auto}
    .history-list::-webkit-scrollbar{width:4px}
    .history-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
    .history-item{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:rgba(0,229,255,0.03);border:1px solid var(--border);border-radius:3px;animation:slide-in 0.25s ease}
    @keyframes slide-in{from{opacity:0;transform:translateX(10px)}to{opacity:1;transform:translateX(0)}}
    .h-word{font-family:var(--font-disp);font-size:13px;font-weight:700;color:var(--cyan);letter-spacing:0.08em}
    .h-word.unknown{color:var(--text-dim)}
    .h-conf{font-size:10px;color:var(--text-dim)}
    .h-time{font-size:9px;color:var(--text-dim);opacity:0.6}
    .empty-history{color:var(--text-dim);font-size:11px;letter-spacing:0.1em;text-align:center;padding:20px 0}
    .rms-row{display:flex;align-items:center;gap:8px;margin-top:4px}
    .rms-label{font-size:10px;color:var(--text-dim);letter-spacing:0.1em;width:44px}
    .rms-bg{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
    .rms-fill{height:100%;border-radius:2px;background:var(--green);transition:width 0.3s;width:0%}
    .rms-val{font-size:10px;color:var(--text-dim);width:38px;text-align:right}
    /* error toast */
    .toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1a0a0f;border:1px solid var(--red);color:var(--red);font-size:11px;letter-spacing:0.1em;padding:10px 20px;border-radius:4px;display:none;z-index:1000}
    .toast.show{display:block;animation:fadein 0.3s ease}
    @keyframes fadein{from{opacity:0;transform:translateX(-50%) translateY(10px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}
    footer{margin-top:32px;font-size:10px;letter-spacing:0.2em;color:var(--text-dim);text-align:center;opacity:0.5}
  </style>
</head>
<body>
<div class="wrapper">
  <header>
    <div class="logo-line">⬡-VOICE COMMAND SYSYSTEM</div>
    <div class="sub-line">SVM · MFCC · Real-Time Recognition · v2.0</div>
  </header>
  <div class="main-grid">
    <div style="display:flex;flex-direction:column;gap:20px">
      <div class="panel">
        <div class="panel-title">// input module</div>
        <div class="record-section">
          <button class="record-btn" id="recordBtn" onclick="toggleRecord()">
            <span class="mic-icon">🎙</span>
            <span class="btn-label" id="btnLabel">RECORD</span>
          </button>
          <div class="timer-wrap">
            <div class="timer-label">RECORDING DURATION — 2.0s</div>
            <div class="timer-bar-bg"><div class="timer-bar" id="timerBar"></div></div>
          </div>
          <div class="rms-row">
            <div class="rms-label">RMS</div>
            <div class="rms-bg"><div class="rms-fill" id="rmsFill"></div></div>
            <div class="rms-val" id="rmsVal">—</div>
          </div>
          <div class="status-line" id="statusLine">READY — PRESS TO RECORD</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">// recognition output</div>
        <div class="result-box">
          <div class="result-word" id="resultWord">—</div>
          <div class="confidence-row">
            <span class="conf-label">CONFIDENCE</span>
            <span class="conf-value unknown" id="confValue">—</span>
          </div>
        </div>
        <div style="margin-top:20px">
          <div class="panel-title" style="margin-bottom:14px">// class probabilities</div>
          <div class="prob-list" id="probList">
            <div class="prob-row"><span class="prob-class">left</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-left"></div></div><span class="prob-pct" id="pct-left">0%</span></div>
            <div class="prob-row"><span class="prob-class">no</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-no"></div></div><span class="prob-pct" id="pct-no">0%</span></div>
            <div class="prob-row"><span class="prob-class">right</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-right"></div></div><span class="prob-pct" id="pct-right">0%</span></div>
            <div class="prob-row"><span class="prob-class">yes</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-yes"></div></div><span class="prob-pct" id="pct-yes">0%</span></div>
            <div class="prob-row"><span class="prob-class">down</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-down"></div></div><span class="prob-pct" id="pct-down">0%</span></div>
<div class="prob-row"><span class="prob-class">speak</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-speak"></div></div><span class="prob-pct" id="pct-speak">0%</span></div>
<div class="prob-row"><span class="prob-class">rcb</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-rcb"></div></div><span class="prob-pct" id="pct-rcb">0%</span></div>
<div class="prob-row"><span class="prob-class">class</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-class"></div></div><span class="prob-pct" id="pct-class">0%</span></div>
<div class="prob-row"><span class="prob-class">mine</span><div class="prob-bar-bg"><div class="prob-bar" id="bar-mine"></div></div><span class="prob-pct" id="pct-mine">0%</span></div>
          </div>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">// recognition log</div>
      <div class="history-list" id="historyList">
        <div class="empty-history" id="emptyMsg">NO ENTRIES YET</div>
      </div>
    </div>
  </div>
  <footer>MCA PROJECT · VOICE COMMAND RECOGNITION · SVM + MFCC PIPELINE</footer>
</div>
<div class="toast" id="toast"></div>
<script>
  const CLASSES = ["left","no","right","yes","down","speak","rcb","class","mine"];
  const RECORD_MS = 2000;

  let mediaRecorder=null,audioChunks=[],isRecording=false,timerInterval=null,stream=null;
  const recordBtn  = document.getElementById("recordBtn");
  const btnLabel   = document.getElementById("btnLabel");
  const statusLine = document.getElementById("statusLine");
  const timerBar   = document.getElementById("timerBar");
  const resultWord = document.getElementById("resultWord");
  const confValue  = document.getElementById("confValue");
  const rmsFill    = document.getElementById("rmsFill");
  const rmsVal     = document.getElementById("rmsVal");
  const histList   = document.getElementById("historyList");
  const toast      = document.getElementById("toast");

  // ── Toast error popup ──
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 4000);
  }

  function setStatus(msg, type="") {
    statusLine.textContent = msg;
    statusLine.className = "status-line " + type;
  }

  function resetBars() {
    CLASSES.forEach(c => {
      document.getElementById("bar-"+c).style.width = "0%";
      document.getElementById("bar-"+c).classList.remove("top");
      document.getElementById("pct-"+c).textContent = "0%";
      document.getElementById("pct-"+c).classList.remove("top");
    });
  }

  function updateBars(probs) {
    const maxCls = Object.entries(probs).reduce((a,b) => b[1]>a[1] ? b : a)[0];
    CLASSES.forEach(c => {
      const pct   = probs[c] || 0;
      const bar   = document.getElementById("bar-"+c);
      const pctEl = document.getElementById("pct-"+c);
      bar.style.width = pct + "%";
      pctEl.textContent = pct.toFixed(1) + "%";
      if (c === maxCls) { bar.classList.add("top"); pctEl.classList.add("top"); }
      else              { bar.classList.remove("top"); pctEl.classList.remove("top"); }
    });
  }

  function updateRMS(rms) {
    rmsFill.style.width = Math.min(rms * 500, 100) + "%";
    rmsVal.textContent  = rms.toFixed(4);
  }

  function addHistory(label, confidence) {
    const empty = document.getElementById("emptyMsg");
    if (empty) empty.remove();
    const now     = new Date();
    const timeStr = now.toLocaleTimeString("en-GB", {hour:"2-digit",minute:"2-digit",second:"2-digit"});
    const item    = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <span class="h-word ${label==='unknown'?'unknown':''}">${label.toUpperCase()}</span>
      <span class="h-conf">${confidence > 0 ? confidence.toFixed(1)+"%" : "—"}</span>
      <span class="h-time">${timeStr}</span>`;
    histList.insertBefore(item, histList.firstChild);
    while (histList.children.length > 20) histList.removeChild(histList.lastChild);
  }

  function startTimer() {
    let elapsed = 0;
    timerBar.classList.add("recording");
    timerInterval = setInterval(() => {
      elapsed += 50;
      timerBar.style.width = (elapsed / RECORD_MS * 100) + "%";
      if (elapsed >= RECORD_MS) clearInterval(timerInterval);
    }, 50);
  }

  function resetTimer() {
    clearInterval(timerInterval);
    timerBar.classList.remove("recording");
    timerBar.style.width = "0%";
  }

  function resetUI() {
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordBtn.disabled = false;
    btnLabel.textContent = "RECORD";
    resetTimer();
  }

  async function toggleRecord() {
    if (isRecording) return;

    // ── Request mic ──
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    } catch(e) {
      setStatus("MIC ACCESS DENIED", "error");
      showToast("Microphone access denied — allow mic in browser settings");
      return;
    }

    isRecording = true;
    audioChunks = [];
    recordBtn.classList.add("recording");
    recordBtn.disabled = true;
    btnLabel.textContent = "LISTENING";
    setStatus("RECORDING — SPEAK NOW", "working");
    resetBars();
    startTimer();

    // ── Check supported MIME ──
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : "";

    try {
      mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch(e) {
      setStatus("RECORDER ERROR", "error");
      showToast("MediaRecorder failed: " + e.message);
      resetUI();
      return;
    }

    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };

    mediaRecorder.onerror = e => {
      showToast("Recording error: " + e.error);
      resetUI();
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      resetUI();
      setStatus("PROCESSING...", "working");

      if (audioChunks.length === 0) {
        setStatus("NO AUDIO CAPTURED", "error");
        showToast("No audio captured — try again");
        return;
      }

      const blob     = new Blob(audioChunks, { type: mimeType || "audio/webm" });
      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      try {
        const resp = await fetch("/predict", { method: "POST", body: formData });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: "Server error " + resp.status }));
          setStatus("SERVER ERROR — " + (err.error || resp.status), "error");
          showToast(err.error || "Server returned " + resp.status);
          return;
        }

        const data = await resp.json();

        if (data.error) {
          setStatus("ERROR: " + data.error, "error");
          showToast(data.error);
          return;
        }

        updateRMS(data.rms);

        if (data.reason === "no_speech") {
          resultWord.textContent = "—";
          resultWord.className   = "result-word unknown";
          confValue.textContent  = "—";
          confValue.className    = "conf-value unknown";
          setStatus("NO SPEECH DETECTED — speak louder", "error");
          addHistory("unknown", 0);
          return;
        }

        updateBars(data.probs);

        const label = data.label;
        const conf  = data.confidence;

        resultWord.textContent = label.toUpperCase();
        resultWord.className   = "result-word " + (label === "unknown" ? "unknown" : "success");
        confValue.textContent  = label === "unknown" ? "—" : conf.toFixed(1) + "%";
        confValue.className    = "conf-value " + (label === "unknown" ? "unknown" : conf < 55 ? "low" : "");

        setStatus(
          label === "unknown" ? "LOW CONFIDENCE — below threshold (45%)" : "RECOGNIZED ✓",
          label === "unknown" ? "error" : "ok"
        );

        addHistory(label, conf);

      } catch(e) {
        setStatus("NETWORK ERROR", "error");
        showToast("Could not reach server — is Flask running?");
      }
    };

    mediaRecorder.start();
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
    }, RECORD_MS);
  }
</script>
</body>
</html>"""

# ==================================================
# AUDIO CONVERSION  ✅ Full path — no PATH env needed
# ==================================================
def convert_to_wav(input_path: str) -> str:
    if not os.path.exists(FFMPEG_PATH):
        raise RuntimeError(
            f"ffmpeg not found at:\n{FFMPEG_PATH}\n"
            "Update FFMPEG_PATH in app.py to your actual ffmpeg.exe location."
        )
    wav_path = input_path + ".wav"
    result = subprocess.run(
        [FFMPEG_PATH, "-y", "-i", input_path,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")
    return wav_path

# ==================================================
# FEATURE EXTRACTION
# ==================================================
def extract_features(audio_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=25)
        if len(y) < MIN_AUDIO_LEN:
            return np.zeros(FEATURE_SIZE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if mfcc.shape[1] < 9:
            mfcc = np.pad(mfcc, ((0,0),(0,9-mfcc.shape[1])), mode='edge')
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([
            np.mean(mfcc.T,axis=0), np.std(mfcc.T,axis=0),
            np.mean(delta.T,axis=0), np.std(delta.T,axis=0),
            np.mean(delta2.T,axis=0), np.std(delta2.T,axis=0),
        ])
        return features.astype(np.float32)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(FEATURE_SIZE)

# ==================================================
# ROUTES
# ==================================================
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]

    # Check file is not empty
    audio_file.seek(0, 2)
    size = audio_file.tell()
    audio_file.seek(0)
    if size == 0:
        return jsonify({"error": "Empty audio file received"}), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        raw_path = tmp.name

    wav_path = None
    try:
        # Convert webm → wav
        try:
            wav_path = convert_to_wav(raw_path)
        except RuntimeError as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500

        # VAD check
        y_raw, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(y_raw ** 2)))
        print(f"RMS: {rms:.4f}")

        if rms < VAD_RMS_THRESHOLD:
            return jsonify({
                "label": "unknown", "confidence": 0.0,
                "rms": round(rms,4), "reason": "no_speech",
                "probs": {cls: 0.0 for cls in CLASSES}
            })

        # Predict
        features = extract_features(wav_path).reshape(1, -1)
        features = scaler.transform(features)
        probs    = model.predict_proba(features)[0]
        probs    = probs / probs.sum()

        best_idx   = int(np.argmax(probs))
        best_label = CLASSES[best_idx]
        confidence = float(probs[best_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            best_label = "unknown"

        print(f"Predicted: {best_label} ({confidence*100:.1f}%)")

        return jsonify({
            "label":      best_label,
            "confidence": round(confidence * 100, 1),
            "rms":        round(rms, 4),
            "reason":     "ok",
            "probs":      {cls: round(float(p)*100,1) for cls,p in zip(CLASSES,probs)}
        })

    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        if raw_path and os.path.exists(raw_path):
            os.unlink(raw_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

# ==================================================
# MAIN  ✅ use_reloader=False — prevents PATH reset
# ==================================================
if __name__ == "__main__":
    import webbrowser, threading, time

    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "="*50)
    print("  Voice Command Recognizer — Web UI")
    print("  Open: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("="*50 + "\n")

    app.run(debug=True, port=5000, use_reloader=False)  # ✅ reloader disabled