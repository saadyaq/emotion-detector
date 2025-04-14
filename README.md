# 🎤 Emotion Detector (Multilingual)

This project detects emotions in multilingual speech using audio signal processing, `librosa`, and machine learning.

## 🔍 Features
- Raw `.wav` audio exploration
- Waveform and Mel spectrogram visualization
- MFCC and feature extraction with `librosa`
- Ready for model training and Streamlit app

## 📁 Structure
emotion_detector/ │ ├── data/ # Raw and processed data │ ├── raw/ # RAVDESS dataset │ ├── metadata.csv # Paths + emotion labels │ ├── X.npy / y.npy # Features and targets (classical) │ └── X_spectro.npy # Image-based features │ ├── models/ # Saved models (e.g. Random Forest) │ └── rf.joblib │ ├── src/ # Python modules │ ├── feature_extraction.py │ ├── preprocess.py │ ├── audio_to_image.py │ └── train.py │ ├── notebooks/ # Jupyter notebooks │ ├── exploration.ipynb │ ├── build_dataset.ipynb │ └── indexation.ipynb │ ├── app.py # Streamlit app for real-time prediction ├── requirements.txt # Python dependencies └── README.md # Project description
## 🚀 To Run
```bash
pip install -r requirements.txt## 🔧 What We've Built

### 🗂️ 1. Dataset Preparation
- Used the **RAVDESS dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song).
- Created a structured `metadata.csv` with file paths and corresponding emotion labels.

### 🎧 2. Audio Feature Extraction
From each `.wav` file, we extracted features using `librosa`:
- **MFCCs**
- **Chroma STFT**
- **Spectral Centroid**
- **Spectral Bandwidth**
- **Spectral Contrast**
- **Tempo**

These were used as inputs to classical machine learning models.

### 🧠 3. Model Training
We trained and evaluated multiple models:
- ✅ **Random Forest** (best performance: ~44% accuracy)
- MLP (Multi-layer Perceptron)
- Logistic Regression
- CNN with spectrograms (TensorFlow)
- PyTorch (experimental)

The **Random Forest** achieved the best balance of simplicity and performance.

### 📷 4. Spectrogram Approach (CNN)
- Converted audio signals to **Mel spectrograms**.
- Normalized and padded them to fixed shapes.
- Built a CNN in TensorFlow/Keras.
- While results were below the RF model, this approach has strong potential.

### 🌐 5. Real-Time Web App (Streamlit)
Built a live Streamlit interface:
- 🎙️ Records audio using the microphone.
- 🎛️ Extracts features in real-time.
- 🤖 Uses the pre-trained Random Forest to predict emotion.
- 📈 Displays prediction on the interface.

---

## 🧪 How to Run

```bash
# Create & activate a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

---
## 📈 Known Challenges

- Dataset is small (~1400 files)
- Emotion categories are unbalanced (e.g., fewer neutral samples)
- Emotion in speech is subtle and subjective
- CNNs require larger data or transfer learning

---

## 🚀 Future Work

- Improve CNN with deeper architecture & data augmentation
- Use **pre-trained models** (YAMNet, wav2vec, etc.)
- Try **RNNs / LSTMs** for temporal modeling
- Multi-modal fusion (e.g., facial + vocal emotion)
- Deploy Streamlit on the cloud (e.g., Streamlit Cloud / Heroku / EC2)
- Add **language detection** and **multi-language emotion support**

---

## 🤝 Credits

This project was developed by **Saad Yaqine** as part of a personal initiative to combine signal processing, machine learning, and real-time applications.

📬 saadyaqine91@gmail.com

---

## 🔗 References

- [RAVDESS dataset](https://zenodo.org/record/1188976)
- [Librosa documentation](https://librosa.org/doc/latest/index.html)
- [Streamlit](https://streamlit.io)
- Scikit-learn, TensorFlow, PyTorch, and more!
---


