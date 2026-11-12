from __future__ import annotations

from math import erf, sqrt

import numpy as np


QUANTILE_Z = {
    0.10: -1.2815515655446004,
    0.25: -0.6744897501960817,
    0.50: 0.0,
    0.75: 0.6744897501960817,
    0.90: 1.2815515655446004,
}



def _label(alpha: float) -> str:
    return f"Q{int(round(alpha * 100)):02d}"



def enforce_non_crossing_quantiles(pred_matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred_matrix, dtype=float)
    return np.sort(arr, axis=1)



def robust_sigma_from_quantiles(q_map: dict[float, np.ndarray]) -> np.ndarray:
    candidates: list[np.ndarray] = []
    if 0.25 in q_map and 0.75 in q_map:
        candidates.append((np.asarray(q_map[0.75]) - np.asarray(q_map[0.25])) / 1.3489795003921634)
    if 0.10 in q_map and 0.90 in q_map:
        candidates.append((np.asarray(q_map[0.90]) - np.asarray(q_map[0.10])) / 2.5631031310892007)
    if 0.10 in q_map and 0.50 in q_map:
        candidates.append((np.asarray(q_map[0.50]) - np.asarray(q_map[0.10])) / 1.2815515655446004)
    if 0.50 in q_map and 0.90 in q_map:
        candidates.append((np.asarray(q_map[0.90]) - np.asarray(q_map[0.50])) / 1.2815515655446004)

    if not candidates:
        n = len(next(iter(q_map.values())))
        return np.full(n, 10.0, dtype=float)

    sigma = np.nanmedian(np.vstack(candidates), axis=0)
    sigma = np.where(np.isfinite(sigma), sigma, 10.0)
    return np.clip(sigma, 1.0, None)



def normal_cdf(x: np.ndarray | float, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    std_arr = np.asarray(std, dtype=float)
    std_arr = np.where(std_arr <= 0, 1.0, std_arr)
    z = (x_arr - mean_arr) / (std_arr * sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(erf)(z))



def cover_prob_team_a(mean_margin: np.ndarray, sigma_margin: np.ndarray, team_a_spread: np.ndarray) -> np.ndarray:
    threshold = -np.asarray(team_a_spread, dtype=float)
    return 1.0 - normal_cdf(threshold, mean_margin, sigma_margin)



def over_prob(mean_total: np.ndarray, sigma_total: np.ndarray, total_line: np.ndarray) -> np.ndarray:
    threshold = np.asarray(total_line, dtype=float)
    return 1.0 - normal_cdf(threshold, mean_total, sigma_total)



def reconcile_scores(
    win_prob: np.ndarray,
    margin_pred: np.ndarray,
    total_pred: np.ndarray,
    clip_min: int,
    clip_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prob = np.asarray(win_prob, dtype=float)
    margin = np.asarray(margin_pred, dtype=float)
    total = np.asarray(total_pred, dtype=float)

    sign = np.where(prob >= 0.5, 1.0, -1.0)
    margin = np.where(np.sign(margin) == 0, sign, margin)
    disagree = np.sign(margin) != sign
    margin = np.where(disagree, np.maximum(np.abs(margin), 1.0) * sign, margin)

    team_a = np.rint((total + margin) / 2.0)
    team_b = np.rint((total - margin) / 2.0)
    team_a = np.clip(team_a, clip_min, clip_max)
    team_b = np.clip(team_b, clip_min, clip_max)

    ties = team_a == team_b
    team_a = np.where(ties & (prob >= 0.5), team_a + 1, team_a)
    team_b = np.where(ties & (prob < 0.5), team_b + 1, team_b)

    pred_margin = team_a - team_b
    pred_total = team_a + team_b
    return team_a.astype(int), team_b.astype(int), pred_margin.astype(float), pred_total.astype(float)



def quantile_columns(alphas: list[float], prefix: str) -> list[str]:
    return [f"{prefix}{_label(alpha)}" for alpha in alphas]



def quantile_label(alpha: float) -> str:
    return _label(alpha)
