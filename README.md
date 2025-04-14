# 🎙️ Emotion Detector

This project aims to build a **Speech Emotion Recognition** (SER) system that can detect human emotions from vocal signals using both **classical ML** (Random Forest, MLP) and **deep learning** (CNN on spectrograms). We built an end-to-end pipeline, from raw audio to a real-time Streamlit application.

---

## 📁 Project Structure

```
emotion_detector/
├── data/                      # Raw and processed data
│   ├── raw/                  # RAVDESS dataset
│   ├── metadata.csv          # Paths + emotion labels
│   ├── X.npy / y.npy         # Features and targets (classical)
│   └── X_spectro.npy         # Image-based features
├── models/                   # Saved models (e.g. Random Forest)
│   └── rf.joblib
├── src/                      # Python modules
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── audio_to_image.py
│   ├── train.py
│   └── predict.py
├── notebooks/                # Jupyter notebooks
│   ├── exploration.ipynb
│   ├── build_dataset.ipynb
│   └── indexation.ipynb
├── app.py                    # Streamlit app for real-time prediction
├── requirements.txt          # Python dependencies
└── README.md                 # Project description
```

---

## 🚀 To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch the Streamlit app:

```bash
streamlit run app.py
```

---

## 📊 What We've Built

### 🧩 1. Dataset Preparation

- Used the **RAVDESS dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song).
- Created a structured `metadata.csv` file with paths and corresponding emotion labels.

### 🎵 2. Audio Feature Extraction

From `.wav` files, we extracted features using `librosa`:
- **MFCCs**
- **Chroma STFT**
- **Spectral Centroid**
- **Spectral Bandwidth**
- **Spectral Contrast**
- **Tempo**

This produced a numerical dataset saved as `X.npy` / `y.npy`.

### 🧠 3. Machine Learning Models

We trained various classifiers:
- **Logistic Regression**
- **MLP Classifier**
- **Random Forest (best result ≈ 44%)**

We standardized and optionally applied PCA to reduce dimensions.

### 🖼️ 4. CNN with Spectrograms

We created mel-spectrogram images for each `.wav` using `librosa`:
- Normalized and padded to fixed size (128x143x1)
- Saved as numpy array `X_spectro.npy`

Then trained a CNN with TensorFlow/Keras:
- 3 convolutional layers
- BatchNorm, Dropout
- Accuracy ~ 28%

### 🌐 5. Streamlit App

- Record from microphone
- Extract features
- Load trained model
- Predict & display emotion

> Note: works only on native OS (not WSL) for microphone.

---

## 📌 Improvements Planned

- Improve CNN architecture (attention, deeper layers, transfer learning)
- Use data augmentation
- Try other audio features (e.g., pitch, energy)
- Deploy the app online (e.g., Streamlit Cloud or Hugging Face Spaces)

---

## 📚 References

- RAVDESS dataset: https://zenodo.org/record/1188976
- Librosa documentation: https://librosa.org
- TensorFlow, Scikit-learn, Streamlit

---

## ✨ Author

**Saad Yaqine**
