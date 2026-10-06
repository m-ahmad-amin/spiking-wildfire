import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler


def fit_logistic(features: np.ndarray, labels: np.ndarray, seed: int):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(scaled, labels)
    return model, scaler


def predict_proba(model, scaler, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.array([], dtype=float)
    return model.predict_proba(scaler.transform(features))[:, 1]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def f1_at_half(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(f1_score(labels, scores >= 0.5))
