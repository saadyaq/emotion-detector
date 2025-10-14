from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import librosa
import numpy as np


def _safe_flatten(values: np.ndarray) -> np.ndarray:
    """Flatten and replace NaN/inf values with zeros."""
    flat = np.asarray(values).reshape(-1)
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class FeatureExtractor:
    """
    Rich audio feature extractor geared towards speech emotion recognition.

    Parameters
    ----------
    sr:
        Target sampling rate. `None` keeps the native sampling rate.
    n_mfcc:
        Number of MFCC coefficients to compute.
    hop_length:
        Number of samples between successive frames.
    n_mels:
        Number of Mel bands to generate for mel-based features.
    """

    sr: Optional[int] = 16000
    n_mfcc: int = 20
    hop_length: int = 512
    n_mels: int = 128
    _feature_names: List[str] = field(default_factory=list, init=False)

    def extract(self, file_path: Sequence[str] | str | Path) -> Optional[OrderedDict[str, float]]:
        """
        Extract a comprehensive feature dictionary from an audio file.

        Returns
        -------
        OrderedDict[str, float] or None
            Feature vector keyed by feature names. Returns None if extraction fails.
        """
        path = Path(file_path)
        try:
            y, sr = librosa.load(path.as_posix(), sr=self.sr)
        except librosa.util.exceptions.ParameterError as err:
            print(f"[WARN] Librosa parameter error for {path}: {err}")
            return None
        except FileNotFoundError:
            print(f"[WARN] Audio file not found: {path}")
            return None
        except Exception as err:
            print(f"[WARN] Error loading {path}: {err}")
            return None

        if y.size == 0:
            print(f"[WARN] Empty audio signal for {path}")
            return None

        features: "OrderedDict[str, float]" = OrderedDict()

        # Core time-domain descriptors
        duration = librosa.get_duration(y=y, sr=sr)
        features["duration"] = float(duration)

        stft = librosa.stft(y=y, hop_length=self.hop_length)
        magnitude = np.abs(stft)

        self._add_stats(features, "rms", librosa.feature.rms(S=magnitude), include_min_max=True)
        self._add_stats(
            features,
            "zero_crossing_rate",
            librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length),
            include_min_max=True,
        )

        # Spectral descriptors
        self._add_stats(
            features,
            "spectral_centroid",
            librosa.feature.spectral_centroid(S=magnitude, sr=sr),
            include_min_max=True,
        )
        self._add_stats(
            features,
            "spectral_bandwidth",
            librosa.feature.spectral_bandwidth(S=magnitude, sr=sr),
            include_min_max=True,
        )
        self._add_stats(
            features,
            "spectral_rolloff",
            librosa.feature.spectral_rolloff(S=magnitude, sr=sr),
            include_min_max=True,
        )
        self._add_stats(
            features,
            "spectral_flatness",
            librosa.feature.spectral_flatness(S=magnitude),
            include_min_max=True,
        )
        self._add_stats_matrix(features, "spectral_contrast", librosa.feature.spectral_contrast(S=magnitude, sr=sr))

        # Harmonic / percussive separation for tonnetz & energy ratios
        y_harm, y_perc = librosa.effects.hpss(y)
        self._add_stats_matrix(features, "tonnetz", librosa.feature.tonnetz(y=y_harm, sr=sr))

        harmonic_energy = np.mean(np.abs(y_harm)) if y_harm.size else 0.0
        percussive_energy = np.mean(np.abs(y_perc)) if y_perc.size else 0.0
        features["harmonic_energy_mean"] = float(harmonic_energy)
        features["percussive_energy_mean"] = float(percussive_energy)
        features["harmonic_to_percussive_ratio"] = float((np.sum(np.abs(y_harm)) + 1e-6) / (np.sum(np.abs(y_perc)) + 1e-6))

        # Chroma-based descriptors
        self._add_stats_matrix(features, "chroma_stft", librosa.feature.chroma_stft(S=magnitude, sr=sr))
        self._add_stats_matrix(features, "chroma_cqt", librosa.feature.chroma_cqt(y=y, sr=sr))
        self._add_stats_matrix(features, "chroma_cens", librosa.feature.chroma_cens(y=y, sr=sr))

        # MFCCs and their deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        self._add_stats_matrix(features, "mfcc", mfcc)

        if mfcc.shape[0] > 0:
            self._add_stats_matrix(features, "mfcc_delta", librosa.feature.delta(mfcc))
            self._add_stats_matrix(features, "mfcc_delta2", librosa.feature.delta(mfcc, order=2))

        # Pitch-related features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > magnitudes.max() * 0.1] if magnitudes.size else np.array([])
        pitch_values = pitch_values[pitch_values > 0]
        pitch_values = _safe_flatten(pitch_values) if pitch_values.size else np.array([0.0])
        self._add_stats(features, "pitch", pitch_values, include_min_max=True)

        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        features["tempo"] = float(tempo)

        if not self._feature_names:
            self._feature_names = list(features.keys())

        return features

    def feature_vector(self, file_path: Sequence[str] | str | Path) -> Optional[np.ndarray]:
        """
        Return the feature vector as a numpy array in the deterministic order.
        """
        features = self.extract(file_path)
        if features is None:
            return None
        return np.array([features[name] for name in self.feature_names], dtype=np.float32)

    @property
    def feature_names(self) -> Sequence[str]:
        if not self._feature_names:
            raise RuntimeError(
                "No feature names registered yet. "
                "Call 'extract' at least once before requesting names."
            )
        return self._feature_names

    def _add_stats(
        self,
        features: "OrderedDict[str, float]",
        prefix: str,
        values: np.ndarray,
        include_min_max: bool = False,
    ) -> None:
        arr = _safe_flatten(values)
        features[f"{prefix}_mean"] = float(np.mean(arr))
        features[f"{prefix}_std"] = float(np.std(arr))
        if include_min_max:
            features[f"{prefix}_min"] = float(np.min(arr))
            features[f"{prefix}_max"] = float(np.max(arr))

    def _add_stats_matrix(
        self,
        features: "OrderedDict[str, float]",
        prefix: str,
        matrix: np.ndarray,
    ) -> None:
        mat = np.asarray(matrix)
        if mat.ndim == 1:
            self._add_stats(features, prefix, mat)
            return

        for idx, row in enumerate(mat):
            self._add_stats(features, f"{prefix}_{idx + 1}", row)


_DEFAULT_EXTRACTOR = FeatureExtractor()


def extract_features(file_path: Sequence[str] | str | Path) -> Optional[OrderedDict[str, float]]:
    """
    Convenience wrapper to use a module-level feature extractor.
    """
    return _DEFAULT_EXTRACTOR.extract(file_path)


def extract_feature_vector(file_path: Sequence[str] | str | Path) -> Optional[np.ndarray]:
    """
    Convenience helper returning the feature vector as a numpy array.
    """
    return _DEFAULT_EXTRACTOR.feature_vector(file_path)


__all__ = ["FeatureExtractor", "extract_features", "extract_feature_vector"]
