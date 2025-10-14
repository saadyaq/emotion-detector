# 🎙️ Emotion Detector

Speech Emotion Recognition (SER) pipeline built on top of **RAVDESS** speech audio.  
The project now ships with a rich hand-crafted feature extractor, an automated
training CLI, model selection utilities, and a Streamlit demo app.

> **Latest offline evaluation (Random Forest)**  
> Accuracy **94.1 %**, Balanced Accuracy **93.8 %**, Weighted F1 **94.1 %**  
> See `models/training_metrics.json` for the full report.

---

## 📦 Repository Layout

```
emotion-detector/
├── app.py                 # Streamlit application
├── data/
│   └── metadata.csv       # Relative paths + emotion labels (RAVDESS format)
├── models/
│   ├── audio_pipeline.joblib  # Trained sklearn pipeline (scaler + classifier)
│   └── label_encoder.joblib   # LabelEncoder matching the pipeline
├── notebooks/             # Exploratory notebooks
├── requirements.txt       # Python dependencies
└── src/
    ├── audio_to_image.py  # Spectrogram helpers (for CNN experiments)
    ├── feature_extraction.py  # Rich feature extractor (MFCC, chroma, spectral…)
    ├── preprocess.py      # Utility functions (visualisation, scaling, PCA)
    ├── predict.py         # CLI predictions for batches of audio files
    └── train.py           # Training script with model selection + reporting
```

The RAVDESS `.wav` files themselves are **not** tracked.  
Populate the `data/` directory locally before training or inference.

---

## 🚀 Quickstart

### Prerequisites

**For WSL2 / Linux users:**
Install FFmpeg to enable audio format support (MP3, M4A, FLAC, etc.):

```bash
sudo apt update
sudo apt install -y ffmpeg
```

**Python environment setup:**

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Get the dataset

1. Download the RAVDESS speech-only archive (e.g. via [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) or [Zenodo](https://zenodo.org/record/1188976)).
2. Extract the folders (`Actor_01`, …, `Actor_24`) under `data/`.
3. Regenerate the metadata (optional if you keep the same layout):
   ```bash
   . .venv/bin/activate
   python - <<'PY'
   import csv
   from pathlib import Path

   EMOTION_MAP = {
       '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
       '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
   }

   root = Path("data")
   rows = []
   for wav in sorted(root.rglob("*.wav")):
       code = wav.stem.split("-")[2]
       emotion = EMOTION_MAP.get(code)
       if emotion:
           rows.append((wav.relative_to(root).as_posix(), emotion))

   with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as fp:
       writer = csv.writer(fp)
       writer.writerow(["file_path", "emotion"])
       writer.writerows(rows)
   PY
   ```

### 2. Train / retrain the model

```bash
. .venv/bin/activate
python -m src.train \
  --metadata data/metadata.csv \
  --audio-root data \
  --feature-jobs -1 \
  --cv-folds 5
```

Flags worth tweaking:

- `--feature-jobs`: `-1` uses all available CPU cores for feature extraction.
- `--skip-search`: default **off**; enable it to skip hyper-parameter search when iterating quickly.
- `--sample-limit`: limit number of audio samples (useful for smoke tests).

Artifacts written to `models/`:

| File | Description |
|------|-------------|
| `audio_pipeline.joblib` | Fitted `sklearn.pipeline.Pipeline` (scaler + classifier). |
| `label_encoder.joblib`  | `LabelEncoder` to map class IDs ↔ labels. |
| `training_metrics.json` | Metrics, confusion matrix, CV scores, feature order. |

### 3. Run the Streamlit demo

```bash
. .venv/bin/activate
streamlit run app.py
```

The app provides **two input methods**:

#### 📁 Upload Audio Files
- Upload pre-recorded audio files from your computer
- **Supported formats**: WAV, MP3, M4A, FLAC, OGG
- **Perfect for WSL2 users** where microphone access is limited
- Preview the audio before analysis

**How to record audio on Windows:**
1. Open **Voice Recorder** app (Windows + S, search "Voice Recorder")
2. Click the microphone button to start recording
3. Speak naturally (expressing different emotions)
4. Click stop when finished
5. Find your recording in `C:\Users\YourName\Documents\Sound recordings\`
6. Upload the file to the Streamlit app

#### 🎤 Direct Recording
- Record directly from your microphone (works best on native OS)
- Adjust recording duration (1-10 seconds)
- Select input device if multiple microphones are available
- Instant emotion prediction after recording

Both methods display the predicted emotion with top-3 probabilities.

### 4. CLI predictions

```bash
. .venv/bin/activate
python -m src.predict path/to/audio.wav --topk 5
python -m src.predict samples/*.wav --as-json > results.json
```

Use `--allow-mismatch` when running legacy models that expect fewer features.

---

## 🧠 Feature Engineering

`src/feature_extraction.py` extracts >200 descriptors per clip:

- MFCCs + deltas, chroma (STFT/CQT/CENS)
- Spectral stats (centroid, bandwidth, roll-off, contrast, flatness)
- Harmonic/percussive energy ratios & tonnetz
- RMS, zero-crossing, tempo, pitch statistics

All NaN/inf values are safely replaced with zero and the feature order is deterministic,
which ensures compatibility between training and inference.

---

## 🧪 Evaluation Snapshot

Trained on the full speech portion of RAVDESS (train/test split 80 / 20, stratified):

| Metric | Score |
|--------|-------|
| Accuracy | **94.10 %** |
| Balanced Accuracy | 93.81 % |
| Weighted F1 | 94.11 % |

Per-class precision/recall and confusion matrix are logged in `models/training_metrics.json`.

---

## 📝 Notes & Roadmap

### Known Issues & Solutions

**WSL2 Microphone Access:**
- Direct microphone recording doesn't work on WSL2 due to audio device limitations
- **Solution**: Use the file upload feature to analyze audio recorded on Windows
- Alternative: Configure PulseAudio bridge between Windows and WSL2 (advanced)

**Audio Format Support:**
- Requires FFmpeg for MP3, M4A, FLAC, OGG formats
- WAV files work without FFmpeg
- Install with: `sudo apt install ffmpeg` (Linux/WSL2)

### Future Improvements

- Data augmentation (noise, pitch/tempo shifts) for robustness
- Swap classifiers (XGBoost, LightGBM) or fine-tune end-to-end CNNs on mel-spectrograms
- Deploy Streamlit app to the cloud (Streamlit Community Cloud or Hugging Face Spaces)
- Consider mixed-language datasets or emotion intensity regression
- Add real-time audio streaming for live emotion detection

---

## 🙌 Credits

- **Saad Yaqine** — project lead & experimentation
- RAVDESS dataset: Livingstone, S. R., & Russo, F. A. (2018).  
- Librosa, Scikit-learn, Streamlit, and the Python audio community ❤️
