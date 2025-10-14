import numpy as np
import sounddevice as sd
import streamlit as st
from joblib import load
from pathlib import Path
from scipy.io.wavfile import write

from src.feature_extraction import FeatureExtractor


MODEL_CANDIDATES = [
    Path("models/audio_pipeline.joblib"),
    Path("models/rf.joblib"),  # legacy fallback
]
LABEL_ENCODER_PATH = Path("models/label_encoder.joblib")
EXTRACTOR = FeatureExtractor()


def load_model_artifacts():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model = load(candidate)
            label_encoder = load(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH.exists() else None
            return model, label_encoder, candidate
    return None, None, None


def choose_input_device() -> int:
    devices = sd.query_devices()
    input_devices = [(idx, dev["name"]) for idx, dev in enumerate(devices) if dev["max_input_channels"] > 0]
    if not input_devices:
        st.warning("Aucun microphone détecté. Veuillez en connecter un puis relancer l'application.")
        return -1

    labels = [f"{idx} • {name}" for idx, name in input_devices]
    default_index = 0
    selection = st.selectbox("Microphone d'entrée", labels, index=default_index)
    return input_devices[labels.index(selection)][0]


def validate_dimensions(model, feature_vector: np.ndarray) -> bool:
    n_features = feature_vector.shape[-1]
    expected = None

    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if clf is not None and hasattr(clf, "n_features_in_"):
            expected = clf.n_features_in_

    if expected is not None and expected != n_features:
        st.error(
            f"Le modèle chargé attend {expected} features mais l'extracteur en produit {n_features}. "
            "Relancez l'entraînement (`python -m src.train`) pour générer un nouveau modèle compatible."
        )
        return False
    return True


def predict_emotion(model, label_encoder, feature_vector: np.ndarray):
    prediction = model.predict(feature_vector)[0]
    if label_encoder is not None and isinstance(prediction, (np.integer, np.int32, np.int64, int)):
        prediction_label = label_encoder.inverse_transform([prediction])[0]
    else:
        prediction_label = prediction

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)[0]

    return prediction_label, probabilities


def render_probabilities(probabilities: np.ndarray, label_encoder) -> None:
    if probabilities is None or label_encoder is None:
        return
    labels = label_encoder.classes_
    top_indices = np.argsort(probabilities)[::-1][:3]
    st.markdown("**Top 3 probabilités :**")
    for idx in top_indices:
        st.write(f"- {labels[idx]} : {probabilities[idx]:.2%}")


def main() -> None:
    st.set_page_config(page_title="Détecteur d'Émotions Vocales", page_icon="🎙️")
    st.title("🎙️ Détecteur d'Émotions Vocales")

    model, label_encoder, model_path = load_model_artifacts()
    if model is None:
        st.error("Aucun modèle trouvé dans le dossier `models/`. Lancez `python -m src.train` avant d'utiliser l'application.")
        return

    st.caption(f"Modèle chargé : `{model_path}`")

    duration = st.slider("Durée d'enregistrement (secondes)", 1, 10, 3)
    fs = 22050

    device_index = choose_input_device()
    if device_index == -1:
        return

    if st.button("🎤 Enregistrer"):
        st.info("Enregistrement en cours...")
        sd.default.device = (device_index, None)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        filepath = Path("audio_input.wav")
        write(filepath.as_posix(), fs, recording.squeeze())
        st.success("Enregistrement terminé !")

        feature_vector = EXTRACTOR.feature_vector(filepath)
        if feature_vector is None:
            st.error("Impossible d'extraire les features à partir de l'audio.")
            return

        feature_vector = feature_vector.reshape(1, -1)
        if not validate_dimensions(model, feature_vector):
            return

        prediction_label, probabilities = predict_emotion(model, label_encoder, feature_vector)
        st.subheader("🧠 Émotion détectée")
        st.success(prediction_label)
        render_probabilities(probabilities, label_encoder)


if __name__ == "__main__":
    main()
