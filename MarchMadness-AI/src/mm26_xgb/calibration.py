from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(slots=True)
class ProbabilityCalibrator:
    method: str
    model: Any

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        probs = np.asarray(raw_prob, dtype=float)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            out = np.asarray(self.model.predict(probs), dtype=float)
        elif self.method == "sigmoid":
            out = np.asarray(self.model.predict_proba(probs.reshape(-1, 1))[:, 1], dtype=float)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method!r}")
        return np.clip(out, 1e-6, 1 - 1e-6)



def fit_probability_calibrator(
    raw_prob: np.ndarray,
    target: np.ndarray,
    method: str = "isotonic",
) -> ProbabilityCalibrator | None:
    probs = np.asarray(raw_prob, dtype=float)
    y = np.asarray(target, dtype=int)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    if method in {"", "none", None}:  # type: ignore[arg-type]
        return None

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probs, y)
        return ProbabilityCalibrator(method="isotonic", model=model)

    if method == "sigmoid":
        model = LogisticRegression(max_iter=500)
        model.fit(probs.reshape(-1, 1), y)
        return ProbabilityCalibrator(method="sigmoid", model=model)

    raise ValueError(f"Unsupported calibration method: {method!r}")
