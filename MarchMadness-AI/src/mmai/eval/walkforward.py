"""Walk-forward backtest (plan W5) and gate G1.

The ratings predictions are as-of by construction. The only things fit across seasons, always
on seasons strictly before the test season, are scalar mappings: the margin sigma, the
overtime adjustment, and the point scales of the Elo / Massey / seed baselines.

Orientation: team A is the home team when there is one, else the lower TeamID (see data/games.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from mmai.eval import metrics as M
from mmai.model.baselines import elo, massey_consensus, raw_net_rating, seed_numbers
from mmai.model.ratings import RatingsConfig, asof_season, run_ratings

FLOOR_CONFIG = RatingsConfig(target="points", carryover=0.0, prior_games=0.0, ridge=2.0)
BASE_COLUMNS = ["game_id", "season", "daynum", "date", "phase", "neutral", "team_a", "team_b", "venue_a",
                "score_a", "score_b", "margin_a", "total", "numot"]
CONTEXT_COLUMNS = ["conf_game", "rest_a", "rest_b"]


def context_features(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Design matrices for the context adjustment (plan W3 "context GLM"): totals and margins.
    Everything here is known before tip-off from the schedule alone."""
    ct = d["phase"].eq("conf_tourney").to_numpy(float)
    nc = d["phase"].eq("ncaa").to_numpy(float)
    regular = d["phase"].eq("regular").to_numpy(float)
    neutral = d["neutral"].to_numpy(float)
    conf = d["conf_game"].fillna(False).to_numpy(float) if "conf_game" in d else np.zeros(len(d))
    v = d["venue_a"].to_numpy(float)
    ra = d["rest_a"].fillna(7).to_numpy(float) if "rest_a" in d else np.full(len(d), 7.0)
    rb = d["rest_b"].fillna(7).to_numpy(float) if "rest_b" in d else np.full(len(d), 7.0)
    b2b_a, b2b_b = (ra <= 1).astype(float), (rb <= 1).astype(float)
    x_total = np.column_stack([ct, nc, neutral * regular, conf, b2b_a + b2b_b])
    x_margin = np.column_stack([v * conf, v * (1 - conf) * regular, v * (1 - regular),
                                np.clip(ra, 0, 4) - np.clip(rb, 0, 4), b2b_a - b2b_b])
    return x_total, x_margin


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def after_jan15(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["date"]) >= pd.to_datetime(df["season"].astype(str) + "-01-15")


EVAL_SETS = {
    "after_jan15": after_jan15,
    "neutral": lambda d: d["neutral"].astype(bool),
    "conf_tourney": lambda d: d["phase"].eq("conf_tourney"),
    "ncaa": lambda d: d["phase"].eq("ncaa"),
}

# model -> (margin column, total column, win-probability column, sigma column)
MODELS = {
    "engine": ("eng_margin", "eng_total", "eng_p", "eng_sigma"),
    "engine_plain": ("eng_margin_plain", "eng_total_plain", "eng_p_plain", "eng_sigma_plain"),
    "ridge_floor": ("floor_margin", "floor_total", "floor_p", None),
    "raw_net": ("raw_margin", "raw_total", None, None),
    "elo": ("elo_margin", None, "elo_p", None),
    "massey": ("massey_margin", None, "massey_p", None),
    "seed": ("seed_margin", None, "seed_p", None),
    "closing_line": ("line_margin", "line_total", None, None),
}


def assemble(games: pd.DataFrame, cfg: RatingsConfig, *, massey: pd.DataFrame | None = None,
             seeds: pd.DataFrame | None = None, lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per game with every model's raw (pre-mapping) prediction."""
    df = games[BASE_COLUMNS + [c for c in CONTEXT_COLUMNS if c in games]].copy()
    eng = run_ratings(games, cfg)[["game_id", "pred_margin_a", "pred_total_reg"]]
    df = df.merge(eng.rename(columns={"pred_margin_a": "eng_margin", "pred_total_reg": "eng_total_reg"}),
                  on="game_id", how="left")
    floor = run_ratings(games, FLOOR_CONFIG)[["game_id", "pred_margin_a", "pred_total_reg"]]
    df = df.merge(floor.rename(columns={"pred_margin_a": "floor_margin", "pred_total_reg": "floor_total"}),
                  on="game_id", how="left")
    raw = raw_net_rating(games)
    raw["raw_margin"] = raw["raw_pts_a"] - raw["raw_pts_b"]
    raw["raw_total_reg"] = raw["raw_pts_a"] + raw["raw_pts_b"]
    df = df.merge(raw[["game_id", "raw_margin", "raw_total_reg"]], on="game_id", how="left")
    df = df.merge(elo(games), on="game_id", how="left")
    if massey is not None:
        df = df.merge(massey_consensus(massey, games), on="game_id", how="left")
    if seeds is not None:
        s = seed_numbers(seeds).set_index(["season", "team"])["seed"]
        sa = pd.Series(list(zip(df["season"], df["team_a"]))).map(s)
        sb = pd.Series(list(zip(df["season"], df["team_b"]))).map(s)
        df["seed_diff_a"] = np.where(df["phase"].eq("ncaa"), (sb - sa).to_numpy(), np.nan)
    if lines is not None and len(lines):
        df = df.merge(lines[["game_id", "line_margin_a", "line_total", "line_kind"]], on="game_id", how="left")
    return df


def _ot_mult(margin, total_reg, sigma, c_ot, m_ot):
    p_ot = np.clip(c_ot * norm.pdf(np.asarray(margin) / sigma) / sigma, 0.0, 0.5)
    return np.asarray(total_reg) * (1.0 + p_ot * m_ot / 8.0)


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def fit_mappings(train: pd.DataFrame) -> dict:
    t = train[after_jan15(train) & train["eng_margin"].notna()]
    mp: dict = {}
    sigma = float((t["margin_a"] - t["eng_margin"]).std())
    mp["sigma_plain"] = sigma
    dens = norm.pdf(t["eng_margin"] / sigma) / sigma
    ot = t["numot"] > 0
    mp["c_ot"] = float(ot.mean() / dens.mean())
    mp["m_ot"] = float(t.loc[ot, "numot"].mean()) if ot.any() else 1.0
    # context adjustment: ridge on the as-of residuals of all earlier-season games
    tr = train[train["eng_margin"].notna()]
    x_total, x_margin = context_features(tr)
    total_plain = _ot_mult(tr["eng_margin"], tr["eng_total_reg"], sigma, mp["c_ot"], mp["m_ot"])
    mp["ctx_total"] = _ridge(x_total, (tr["total"] - total_plain).to_numpy(float))
    mp["ctx_margin"] = _ridge(x_margin, (tr["margin_a"] - tr["eng_margin"]).to_numpy(float))
    adj = tr["eng_margin"].to_numpy(float) + x_margin @ mp["ctx_margin"]
    late = after_jan15(tr).to_numpy()
    mp["sigma"] = float(np.std(tr["margin_a"].to_numpy(float)[late] - adj[late], ddof=1))
    mp["floor_sigma"] = float((t["margin_a"] - t["floor_margin"]).std())
    e = t.dropna(subset=["elo_diff_a"])
    mp["elo_scale"] = float(_ols(e[["elo_diff_a"]].to_numpy(), e["margin_a"].to_numpy())[0])
    mp["elo_sigma"] = float((e["margin_a"] - mp["elo_scale"] * e["elo_diff_a"]).std())
    if "massey_diff_a" in t:
        mm = t.dropna(subset=["massey_diff_a"])
        if len(mm) > 100:
            b = _ols(mm[["massey_diff_a", "venue_a"]].to_numpy(float), mm["margin_a"].to_numpy(float))
            mp["massey_b"], mp["massey_h"] = float(b[0]), float(b[1])
            mp["massey_sigma"] = float((mm["margin_a"] - mm[["massey_diff_a", "venue_a"]].to_numpy(float) @ b).std())
    if "seed_diff_a" in train:
        sd = train.dropna(subset=["seed_diff_a"])
        if len(sd) > 50:
            x, y = sd["seed_diff_a"].to_numpy(float), sd["margin_a"].to_numpy(float)
            mp["seed_a"] = float(_ols(x[:, None], y)[0])
            won = (y > 0).astype(float)

            def nll(c):
                return M.log_loss(1 / (1 + np.exp(-c * x)), won)

            mp["seed_c"] = float(minimize_scalar(nll, bounds=(0.0, 2.0), method="bounded").x)
    return mp


def apply_mappings(test: pd.DataFrame, mp: dict) -> pd.DataFrame:
    d = test.copy()
    s0, s = mp["sigma_plain"], mp["sigma"]
    # plain ratings engine (+ overtime expectation)
    d["eng_margin_plain"] = d["eng_margin"]
    d["eng_sigma_plain"] = s0
    d["eng_total_plain"] = _ot_mult(d["eng_margin_plain"], d["eng_total_reg"], s0, mp["c_ot"], mp["m_ot"])
    d["eng_p_plain"] = norm.cdf(d["eng_margin_plain"] / s0)
    # shipped engine = ratings + context adjustment (fit on earlier seasons only)
    x_total, x_margin = context_features(d)
    d["eng_margin"] = d["eng_margin_plain"] + x_margin @ mp["ctx_margin"]
    d["eng_total"] = d["eng_total_plain"] + x_total @ mp["ctx_total"]
    d["eng_sigma"] = s
    d["eng_p"] = norm.cdf(d["eng_margin"] / s)
    d["floor_p"] = norm.cdf(d["floor_margin"] / mp["floor_sigma"])
    d["raw_total"] = _ot_mult(d["raw_margin"], d["raw_total_reg"], s0, mp["c_ot"], mp["m_ot"])
    d["elo_margin"] = mp["elo_scale"] * d["elo_diff_a"]
    d["elo_p"] = norm.cdf(d["elo_margin"] / mp["elo_sigma"])
    if "massey_b" in mp:
        d["massey_margin"] = mp["massey_b"] * d["massey_diff_a"] + mp["massey_h"] * d["venue_a"]
        d["massey_p"] = norm.cdf(d["massey_margin"] / mp["massey_sigma"])
    if "seed_a" in mp:
        d["seed_margin"] = mp["seed_a"] * d["seed_diff_a"]
        d["seed_p"] = 1 / (1 + np.exp(-mp["seed_c"] * d["seed_diff_a"]))
    if "line_margin_a" in d:
        d["line_margin"] = d["line_margin_a"]
    return d


def walk_forward(df: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Test rows for ``seasons``, each mapped with parameters fit on strictly earlier seasons."""
    parts, maps = [], {}
    for s in seasons:
        train, test = df[df["season"] < s], df[df["season"] == s]
        if test.empty or train.empty:
            continue
        maps[s] = fit_mappings(train)
        parts.append(apply_mappings(test, maps[s]))
    return pd.concat(parts, ignore_index=True), maps


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per (eval set, model), each model on the rows where it has a prediction."""
    rows = []
    y_win = (df["margin_a"] > 0).astype(float)
    for set_name, sel in EVAL_SETS.items():
        mask = sel(df)
        for model, (mcol, tcol, pcol, scol) in MODELS.items():
            if mcol not in df:
                continue
            m = mask & df[mcol].notna()
            if m.sum() < 20:
                continue
            e = df.loc[m, "margin_a"] - df.loc[m, mcol]
            row = {"set": set_name, "model": model, "n": int(m.sum()), "margin_rmse": M.rmse(e),
                   "margin_mae": M.mae(e), "margin_bias": M.bias(e)}
            if tcol and tcol in df:
                mt = m & df[tcol].notna()
                et = df.loc[mt, "total"] - df.loc[mt, tcol]
                row.update(total_rmse=M.rmse(et), total_mae=M.mae(et), total_bias=M.bias(et))
            if pcol and pcol in df:
                row["brier"] = M.brier(df.loc[m, pcol], y_win[m])
                row["log_loss"] = M.log_loss(df.loc[m, pcol], y_win[m])
            if scol and scol in df:
                yy, mu, sg = df.loc[m, "margin_a"], df.loc[m, mcol], df.loc[m, scol]
                row.update(cov50=M.coverage(yy, mu, sg, 0.5), cov80=M.coverage(yy, mu, sg, 0.8),
                           crps=M.crps_normal(yy, mu, sg))
            rows.append(row)
    return pd.DataFrame(rows)


def per_season(df: pd.DataFrame, set_name: str = "after_jan15") -> pd.DataFrame:
    d = df[EVAL_SETS[set_name](df) & df["eng_margin"].notna()]
    em, et = d["margin_a"] - d["eng_margin"], d["total"] - d["eng_total"]
    parts = pd.DataFrame({"season": d["season"], "em": em, "em2": em**2, "et": et, "et2": et**2,
                          "br": (d["eng_p"] - (d["margin_a"] > 0).astype(float)) ** 2})
    agg = parts.groupby("season").agg(n=("em", "size"), margin_rmse=("em2", "mean"), margin_bias=("em", "mean"),
                                      total_rmse=("et2", "mean"), total_bias=("et", "mean"), brier=("br", "mean"))
    agg["margin_rmse"] = np.sqrt(agg["margin_rmse"])
    agg["total_rmse"] = np.sqrt(agg["total_rmse"])
    return agg


def time_travel_check(games: pd.DataFrame, cfg: RatingsConfig, season: int, cut_day: int) -> float:
    """Max |difference| in predictions for games on days <= cut_day between the full season and
    the season truncated after cut_day. Zero means no future game leaks into any prediction."""
    prev = games[games["season"] == season - 1]
    prior = asof_season(prev, cfg)[1].as_prior(cfg.carryover) if len(prev) else None
    gs = games[games["season"] == season]
    full, _ = asof_season(gs, cfg, prior)
    trunc, _ = asof_season(gs[gs["daynum"] <= cut_day], cfg, prior)
    merged = trunc.merge(full, on="game_id", suffixes=("_t", "_f"))
    if len(merged) != len(trunc):
        return float("inf")
    return float(max((merged["pred_pts_a_t"] - merged["pred_pts_a_f"]).abs().max(),
                     (merged["pred_pts_b_t"] - merged["pred_pts_b_f"]).abs().max()))


def gate_g1(test: pd.DataFrame, prereg: dict, time_travel_max_diff: float) -> list[dict]:
    g = prereg["gates"]["G1_ratings"]
    res = [{"check": "time-travel test: identical predictions", "value": time_travel_max_diff,
            "threshold": "== 0 (< 1e-9)", "passed": bool(time_travel_max_diff < 1e-9)}]
    sel = after_jan15(test) & test["eng_margin"].notna()

    def rmse_gap(col, need):  # plain ratings vs the baseline: context must not flatter the ratings gate
        m = sel & test[col].notna()
        d, lo, hi = M.paired_bootstrap(M.rmse, (test.loc[m, "margin_a"] - test.loc[m, col]).to_numpy(),
                                       (test.loc[m, "margin_a"] - test.loc[m, "eng_margin_plain"]).to_numpy(), n=500)
        return d, lo, hi, int(m.sum())

    for col, key, label in (("raw_margin", "margin_rmse_beats_raw_net_rating_by", "raw net rating"),
                            ("elo_margin", "margin_rmse_beats_elo_by", "Elo")):
        d, lo, hi, n = rmse_gap(col, g[key])
        res.append({"check": f"margin RMSE improvement over {label} (after Jan 15, n={n})",
                    "value": f"{d:.3f} [{lo:.3f}, {hi:.3f}]", "threshold": f">= {g[key]}", "passed": bool(d >= g[key])})
    if "line_margin" in test and (sel & test["line_margin"].notna()).sum() >= 200:
        m = sel & test["line_margin"].notna()
        gap = (M.rmse(test.loc[m, "margin_a"] - test.loc[m, "eng_margin_plain"])
               - M.rmse(test.loc[m, "margin_a"] - test.loc[m, "line_margin"]))
        res.append({"check": f"margin RMSE gap to closing line (n={int(m.sum())})", "value": round(gap, 3),
                    "threshold": f"<= {g['margin_rmse_within_closing_line']}",
                    "passed": bool(gap <= g["margin_rmse_within_closing_line"])})
    else:
        res.append({"check": "margin RMSE gap to closing line", "value": "n/a", "threshold": "needs lines",
                    "passed": None})
    ps = per_season(test)
    res.append({"check": "max per-season |margin bias| (after Jan 15)", "value": round(ps["margin_bias"].abs().max(), 3),
                "threshold": f"< {g['per_season_abs_margin_bias_max']}",
                "passed": bool(ps["margin_bias"].abs().max() < g["per_season_abs_margin_bias_max"])})
    res.append({"check": "max per-season |total bias| (after Jan 15)", "value": round(ps["total_bias"].abs().max(), 3),
                "threshold": f"< {g['per_season_abs_total_bias_max']}",
                "passed": bool(ps["total_bias"].abs().max() < g["per_season_abs_total_bias_max"])})
    nc = test["phase"].eq("ncaa") & test["eng_total"].notna()
    if nc.any():
        tb = M.bias(test.loc[nc, "total"] - test.loc[nc, "eng_total"])
        res.append({"check": f"pooled NCAA tournament |total bias| (n={int(nc.sum())})", "value": round(abs(tb), 3),
                    "threshold": f"< {g['pooled_tourney_abs_total_bias_max']}",
                    "passed": bool(abs(tb) < g["pooled_tourney_abs_total_bias_max"])})
    else:
        res.append({"check": "pooled NCAA tournament |total bias|", "value": "n/a", "threshold": "needs NCAA games",
                    "passed": None})
    tol = g["interval_coverage_tolerance_pp"] / 100.0
    for level in (0.5, 0.8):
        c = M.coverage(test.loc[sel, "margin_a"], test.loc[sel, "eng_margin"], test.loc[sel, "eng_sigma"], level)
        res.append({"check": f"{int(level * 100)}% margin interval coverage", "value": round(c, 3),
                    "threshold": f"{level:.2f} ± {tol:.2f}", "passed": bool(abs(c - level) <= tol)})
    return res


def gate_g2(test: pd.DataFrame, prereg: dict, *, new: str = "engine", old: str = "engine_plain",
            component: str = "context adjustment") -> list[dict]:
    """G2 for an added component: CI-significant improvement in margin or total RMSE (or Brier),
    without hurting pooled NCAA-tournament metrics by more than 1 SE."""
    g = prereg["gates"]["G2_added_component"]
    (nm, nt, _, _), (om, ot_, _, _) = MODELS[new], MODELS[old]
    sel = after_jan15(test) & test[nm].notna() & test[om].notna()
    nc = test["phase"].eq("ncaa") & test[nm].notna() & test[om].notna()
    res = []
    improved = False
    for label, y, a_col, b_col, need in (("margin RMSE", "margin_a", om, nm, g["min_improvement"]["rmse"]),
                                         ("total RMSE", "total", ot_, nt, g["min_improvement"]["rmse"])):
        e_old = (test.loc[sel, y] - test.loc[sel, a_col]).to_numpy()
        e_new = (test.loc[sel, y] - test.loc[sel, b_col]).to_numpy()
        d, lo, hi = M.paired_bootstrap(M.rmse, e_old, e_new, n=500)
        ok = d >= need and lo > 0
        improved |= ok
        res.append({"check": f"{component}: {label} improvement (after Jan 15, n={int(sel.sum())})",
                    "value": f"{d:.3f} [{lo:.3f}, {hi:.3f}]", "threshold": f">= {need} with CI > 0", "passed": bool(ok)})
        if nc.sum() >= 50:
            e_old_t = (test.loc[nc, y] - test.loc[nc, a_col]).to_numpy()
            e_new_t = (test.loc[nc, y] - test.loc[nc, b_col]).to_numpy()
            d_t, lo_t, hi_t = M.paired_bootstrap(M.rmse, e_old_t, e_new_t, n=500)
            se = (hi_t - lo_t) / (2 * 1.96)
            ok_t = d_t >= -g["max_pooled_tourney_degradation_se"] * se
            res.append({"check": f"{component}: NCAA tournament {label} change (n={int(nc.sum())})",
                        "value": f"{d_t:.3f} [{lo_t:.3f}, {hi_t:.3f}]", "threshold": f">= -1 SE ({-se:.3f})",
                        "passed": bool(ok_t)})
    res.append({"check": f"{component}: ships", "value": "", "threshold": "an improvement + no tourney harm",
                "passed": bool(improved and all(r["passed"] for r in res if "NCAA" in r["check"]))})
    return res
