from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.feature_extraction import FeatureExtractor


PATH_CANDIDATES = ("file_path", "filepath", "path", "audio_path", "relative_path")
LABEL_CANDIDATES = ("label", "emotion", "target", "class")


@dataclass
class TrainingArtifacts:
    pipeline_path: Path
    label_encoder_path: Path
    metrics_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"), help="CSV with file paths and labels.")
    parser.add_argument("--audio-root", type=Path, default=None, help="Base directory to resolve audio paths.")
    parser.add_argument("--path-column", type=str, default=None, help="Column holding audio relative paths.")
    parser.add_argument("--label-column", type=str, default=None, help="Column holding emotion labels.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory where model artifacts are saved.")
    parser.add_argument("--pipeline-name", type=str, default="audio_pipeline.joblib", help="Filename for the trained pipeline.")
    parser.add_argument("--label-encoder-name", type=str, default="label_encoder.joblib", help="Filename for the label encoder.")
    parser.add_argument("--metrics-name", type=str, default="training_metrics.json", help="Filename for evaluation metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out fraction for evaluation.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds during model selection.")
    parser.add_argument("--search-iters", type=int, default=20, help="Randomized search iterations per candidate model.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for model selection.")
    parser.add_argument("--skip-search", action="store_true", help="Skip hyperparameter search and use default settings.")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity during training.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional cap on the number of samples to load.")
    return parser.parse_args()


def _infer_column(df: pd.DataFrame, candidates: Sequence[str], override: Optional[str], role: str) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"{role} column '{override}' not found in CSV.")
        return override

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(f"Could not infer a {role} column. Available columns: {', '.join(df.columns)}")


def _count_param_space(param_grid: Dict[str, Sequence]) -> int:
    total = 1
    for values in param_grid.values():
        total *= max(1, len(values))
    return max(1, total)


def _resolve_audio_path(base: Path, relative_path: str) -> Path:
    rel = Path(str(relative_path))
    if rel.is_absolute():
        return rel
    return base / rel


def _collect_features(
    metadata: pd.DataFrame,
    path_col: str,
    label_col: str,
    extractor: FeatureExtractor,
    audio_root: Path,
    sample_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    used_paths: List[Path] = []

    iterator = metadata[[path_col, label_col]].itertuples(index=False, name=None)
    if sample_limit is not None:
        iterator = list(iterator)[:sample_limit]

    total = sample_limit or len(metadata)

    for path_value, label_value in tqdm(iterator, desc="Extracting features", total=total):
        file_path = _resolve_audio_path(audio_root, path_value)
        feature_vector = extractor.feature_vector(file_path)
        if feature_vector is None:
            continue
        features.append(feature_vector)
        labels.append(label_value)
        used_paths.append(file_path)

    if not features:
        raise RuntimeError("No features were extracted. Check your paths and audio files.")

    X = np.vstack(features).astype(np.float32)
    y = np.array(labels)
    return X, y, used_paths


def _build_candidate_models(random_state: int) -> Dict[str, Dict]:
    common_scaler = ("scaler", StandardScaler())

    hist_boost = Pipeline(
        [
            common_scaler,
            (
                "clf",
                HistGradientBoostingClassifier(
                    random_state=random_state,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            common_scaler,
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    svc = Pipeline(
        [
            common_scaler,
            (
                "clf",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {
        "hist_gradient_boosting": {
            "pipeline": hist_boost,
            "param_distributions": {
                "clf__learning_rate": np.linspace(0.03, 0.15, 5),
                "clf__max_depth": [None, 3, 5, 7],
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [10, 20, 40],
                "clf__l2_regularization": np.linspace(0.0, 0.2, 5),
            },
        },
        "random_forest": {
            "pipeline": rf,
            "param_distributions": {
                "clf__n_estimators": [400, 600, 800],
                "clf__max_depth": [None, 25, 40, 60],
                "clf__min_samples_split": [2, 4, 6],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.3, 0.5, 0.7],
            },
        },
        "svc_rbf": {
            "pipeline": svc,
            "param_distributions": {
                "clf__C": np.logspace(-2, 2, 9),
                "clf__gamma": np.logspace(-4, -1, 8),
            },
        },
    }


def _fit_and_select_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidates: Dict[str, Dict],
    cv_folds: int,
    search_iters: int,
    n_jobs: int,
    random_state: int,
    skip_search: bool,
    verbose: bool,
) -> Tuple[str, Pipeline, Dict[str, Dict]]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_model_name: Optional[str] = None
    best_model: Optional[Pipeline] = None
    best_score = -np.inf
    evaluations: Dict[str, Dict] = {}

    for name, config in candidates.items():
        pipeline = config["pipeline"]
        param_distributions = config.get("param_distributions", {})

        if skip_search or not param_distributions:
            estimator = clone(pipeline)
            estimator.fit(X_train, y_train)
            cross_val_f1: List[float] = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                est_fold = clone(pipeline).fit(X_tr, y_tr)
                preds = est_fold.predict(X_val)
                cross_val_f1.append(f1_score(y_val, preds, average="weighted"))

            mean_score = float(np.mean(cross_val_f1))
            evaluations[name] = {
                "mode": "default",
                "cv_f1_weighted_mean": mean_score,
                "cv_f1_weighted_std": float(np.std(cross_val_f1)),
            }
        else:
            n_iter = min(search_iters, _count_param_space(param_distributions))
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="f1_weighted",
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=1 if verbose else 0,
            )
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
            mean_score = float(search.best_score_)
            evaluations[name] = {
                "mode": "random_search",
                "best_params": search.best_params_,
                "cv_f1_weighted_mean": mean_score,
            }

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = estimator

    if best_model is None or best_model_name is None:
        raise RuntimeError("Model selection failed to produce a fitted pipeline.")

    return best_model_name, best_model, evaluations


def _evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, object]:
    predictions = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    if probs is not None:
        metrics["mean_max_probability"] = float(np.mean(np.max(probs, axis=1)))

    return metrics


def _prepare_output_paths(args: argparse.Namespace) -> TrainingArtifacts:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return TrainingArtifacts(
        pipeline_path=output_dir / args.pipeline_name,
        label_encoder_path=output_dir / args.label_encoder_name,
        metrics_path=output_dir / args.metrics_name,
    )


def main() -> None:
    args = parse_args()

    metadata_path: Path = args.metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    path_col = _infer_column(metadata, PATH_CANDIDATES, args.path_column, "path")
    label_col = _infer_column(metadata, LABEL_CANDIDATES, args.label_column, "label")

    audio_root = args.audio_root if args.audio_root else metadata_path.parent
    extractor = FeatureExtractor()

    X, y_labels, used_paths = _collect_features(
        metadata=metadata,
        path_col=path_col,
        label_col=label_col,
        extractor=extractor,
        audio_root=audio_root,
        sample_limit=args.sample_limit,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    candidates = _build_candidate_models(random_state=args.random_state)
    best_name, best_model, evaluations = _fit_and_select_model(
        X_train=X_train,
        y_train=y_train,
        candidates=candidates,
        cv_folds=args.cv_folds,
        search_iters=args.search_iters,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        skip_search=args.skip_search,
        verbose=args.verbose,
    )

    test_metrics = _evaluate_model(best_model, X_test=X_test, y_test=y_test, label_encoder=label_encoder)

    final_pipeline = clone(best_model)
    final_pipeline.fit(X, y)

    artifacts = _prepare_output_paths(args)
    dump(final_pipeline, artifacts.pipeline_path)
    dump(label_encoder, artifacts.label_encoder_path)

    metrics_payload = {
        "best_model_name": best_name,
        "model_selection": evaluations,
        "test_metrics": test_metrics,
        "label_mapping": {int(label_encoder.transform([cls])[0]): cls for cls in label_encoder.classes_},
        "feature_names": list(extractor.feature_names),
        "n_samples": int(X.shape[0]),
        "used_audio_paths": [path.as_posix() for path in used_paths],
        "settings": {
            "metadata": metadata_path.as_posix(),
            "audio_root": audio_root.as_posix(),
            "path_column": path_col,
            "label_column": label_col,
            "test_size": args.test_size,
            "cv_folds": args.cv_folds,
            "search_iters": args.search_iters,
            "random_state": args.random_state,
            "skip_search": args.skip_search,
        },
    }

    with artifacts.metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    if args.verbose:
        print(f"Best model: {best_name}")
        print(json.dumps(test_metrics, indent=2))
        print(f"Artifacts saved to: {artifacts.pipeline_path}")


if __name__ == "__main__":
    main()
