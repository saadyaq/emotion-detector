from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from joblib import load

from src.feature_extraction import FeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict emotions from audio files using a trained model.")
    parser.add_argument("audio", nargs="+", help="Paths to audio files to classify.")
    parser.add_argument("--model", type=Path, default=Path("models/audio_pipeline.joblib"), help="Path to the trained pipeline.")
    parser.add_argument("--label-encoder", type=Path, default=Path("models/label_encoder.joblib"), help="Path to the label encoder joblib (optional).")
    parser.add_argument("--topk", type=int, default=3, help="Number of top probabilities to display.")
    parser.add_argument("--as-json", action="store_true", help="Output predictions in JSON format.")
    parser.add_argument("--allow-mismatch", action="store_true", help="Skip feature dimension check (useful for legacy models).")
    return parser.parse_args()


def load_label_encoder(path: Path):
    if path.exists():
        return load(path)
    return None


def validate_dimensions(model, feature_vector: np.ndarray) -> Optional[int]:
    expected = None
    if hasattr(model, "n_features_in_"):
        expected = model.n_features_in_
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf")
        if clf is not None and hasattr(clf, "n_features_in_"):
            expected = clf.n_features_in_
    return expected


def format_topk(probabilities: np.ndarray, label_encoder, topk: int) -> List[dict]:
    if probabilities is None or label_encoder is None:
        return []
    indices = np.argsort(probabilities)[::-1][:topk]
    return [
        {"label": label_encoder.classes_[idx], "probability": float(probabilities[idx])}
        for idx in indices
    ]


def main() -> None:
    args = parse_args()
    model_path: Path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load(model_path)
    label_encoder = load_label_encoder(args.label_encoder)
    extractor = FeatureExtractor()

    results = []
    for audio_path in args.audio:
        path = Path(audio_path)
        if not path.exists():
            print(f"[WARN] Audio file not found: {path}")
            continue

        feature_vector = extractor.feature_vector(path)
        if feature_vector is None:
            print(f"[WARN] Failed to extract features for {path}")
            continue

        feature_vector = feature_vector.reshape(1, -1)
        expected = validate_dimensions(model, feature_vector)
        if expected is not None and expected != feature_vector.shape[1] and not args.allow_mismatch:
            raise ValueError(
                f"Model expects {expected} features but extractor produced {feature_vector.shape[1]}. "
                "Retrain the model or pass --allow-mismatch to skip this check."
            )

        prediction = model.predict(feature_vector)[0]
        if label_encoder is not None and isinstance(prediction, (np.integer, np.int32, np.int64, int)):
            label = label_encoder.inverse_transform([prediction])[0]
        else:
            label = prediction

        probabilities = model.predict_proba(feature_vector)[0] if hasattr(model, "predict_proba") else None
        topk = format_topk(probabilities, label_encoder, args.topk)

        result = {
            "file": path.as_posix(),
            "label": label,
            "probability": float(probabilities[np.argmax(probabilities)]) if probabilities is not None else None,
            "topk": topk,
        }
        results.append(result)

    if args.as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for entry in results:
            print(f"{entry['file']} → {entry['label']}")
            if entry["topk"]:
                for item in entry["topk"]:
                    print(f"  - {item['label']}: {item['probability']:.2%}")


if __name__ == "__main__":
    main()
