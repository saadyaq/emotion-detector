import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from joblib import load
from src.feature_extraction import extract_features  # ta fonction MFCC/chroma etc.

st.title("🎙️ Détecteur d'Émotions Vocales")

duration = st.slider("Durée d'enregistrement (secondes)", 1, 10, 3)
fs = 22050  # fréquence d'échantillonnage standard

if st.button("🎤 Enregistrer"):
    st.info("Enregistrement en cours...")
    devices = sd.query_devices()
    st.write(devices)
    sd.default.device = 1  # ou l'index du périphérique avec un micro actif

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    filepath = "audio_input.wav"
    write(filepath, fs, recording)
    st.success("Enregistrement terminé !")

    # Extraire les features
    features = extract_features(filepath)
    if features:
        X_input = np.array(features.values()).reshape(1, -1)

        # Charger le modèle
        model = load("models/rf.joblib")
        prediction = model.predict(X_input)[0]

        st.subheader("🧠 Émotion détectée :")
        st.success(prediction)
    else:
        st.error("Impossible d'extraire les features.")
