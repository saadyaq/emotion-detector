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

- Select your microphone input (works best on native OS, not WSL).
- Record a short snippet and the app will display the predicted emotion plus top‑3 probabilities.

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

- Data augmentation (noise, pitch/tempo shifts) for robustness.
- Swap classifiers (XGBoost, LightGBM) or fine-tune end-to-end CNNs on mel-spectrograms.
- Deploy Streamlit app to the cloud (Streamlit Community Cloud or Hugging Face Spaces).
- Consider mixed-language datasets or emotion intensity regression.

---

## 🙌 Credits

- **Saad Yaqine** — project lead & experimentation
- RAVDESS dataset: Livingstone, S. R., & Russo, F. A. (2018).  
- Librosa, Scikit-learn, Streamlit, and the Python audio community ❤️
