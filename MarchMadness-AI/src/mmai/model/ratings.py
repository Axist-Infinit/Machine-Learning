"""As-of team ratings: the core of the score model (plan W3).

For each gender and season, a weighted ridge regression on points per 100 possessions

    ppp(offense o vs defense d) = mu + O[o] + D[d] + h * venue        (two rows per game)

and a multiplicative tempo model

    log(possessions per 40 min) = tau + T[a] + T[b]                    (one row per game)

re-solved for every day that has games, using only games from *earlier* days. Team effects
are shrunk toward a prior (last season's rating x ``carryover``) with weight ``ridge`` plus a
``prior_games`` term that fades out by ``prior_fade_day``. Weights are "game equivalents":
each game row counts ~1, so ``ridge=2`` acts like two extra games at the prior.

Expected points for team a vs b (venue v from a's side, regulation only):

    poss = exp(tau + T[a] + T[b]);   pts_a = poss * (mu + O[a] + D[b] + h*v) / 100

``target="points"`` fits raw points (overtime included) and skips tempo: that is the
2026-11-11 "ridge floor" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

DEFAULTS = {"ppp": {"mu": 100.0, "hca": 3.0}, "points": {"mu": 70.0, "hca": 2.0}}
DEFAULT_LOG_TEMPO = float(np.log(68.0))


@dataclass(frozen=True)
class RatingsConfig:
    target: str = "ppp"            # "ppp" (+ tempo) or "points"
    ridge: float = 2.0             # shrinkage toward the prior, in game equivalents
    prior_games: float = 6.0       # extra prior weight at DayNum 0 ...
    prior_fade_day: int = 90       # ... fading linearly to 0 by this DayNum (~Feb 1)
    carryover: float = 0.6         # share of last season's (centered) rating kept in the prior
    recency: str = "none"          # none | exp | trank
    half_life: float = 60.0        # days, for recency="exp"
    trank_full: float = 40.0       # recency="trank": full weight up to this age (days) ...
    trank_floor: float = 0.6       # ... falling linearly to this weight ...
    trank_floor_at: float = 80.0   # ... at this age, flat afterwards
    global_prior_weight: float = 25.0  # pseudo-rows anchoring mu, h and tau to last season


def recency_weights(age_days, cfg: RatingsConfig) -> np.ndarray:
    age = np.asarray(age_days, dtype=float)
    if cfg.recency == "none":
        return np.ones_like(age)
    if cfg.recency == "exp":
        return 0.5 ** (age / cfg.half_life)
    if cfg.recency == "trank":
        frac = np.clip((age - cfg.trank_full) / (cfg.trank_floor_at - cfg.trank_full), 0.0, 1.0)
        return 1.0 - frac * (1.0 - cfg.trank_floor)
    raise ValueError(f"unknown recency {cfg.recency!r}")


@dataclass
class TeamRatings:
    teams: np.ndarray
    off: np.ndarray
    deff: np.ndarray
    tempo: np.ndarray
    mu: float
    hca: float
    tau: float
    target: str = "ppp"

    def lookup(self) -> dict[int, int]:
        return {int(t): i for i, t in enumerate(self.teams)}

    def as_prior(self, carryover: float) -> TeamRatings:
        """Next season's prior: centered effects shrunk by ``carryover``; league constants kept."""
        def shrink(x):
            return carryover * (x - x.mean()) if len(x) else x
        return TeamRatings(self.teams.copy(), shrink(self.off), shrink(self.deff), shrink(self.tempo),
                           self.mu, self.hca, self.tau, self.target)

    def expected_points(self, a_idx: np.ndarray, b_idx: np.ndarray, venue_a: np.ndarray):
        """Regulation expected points (pts_a, pts_b) and possessions per 40 (NaN for points target)."""
        ra_off, rb_off = self.off[a_idx], self.off[b_idx]
        ra_def, rb_def = self.deff[a_idx], self.deff[b_idx]
        v = np.asarray(venue_a, dtype=float)
        rate_a = self.mu + ra_off + rb_def + self.hca * v
        rate_b = self.mu + rb_off + ra_def - self.hca * v
        if self.target == "points":
            return rate_a, rate_b, np.full(len(rate_a), np.nan)
        poss = np.exp(self.tau + self.tempo[a_idx] + self.tempo[b_idx])
        return poss * rate_a / 100.0, poss * rate_b / 100.0, poss


class SeasonRater:
    """One season's games as regression rows; ``solve(day)`` returns ratings as of ``day``
    (built from games with ``daynum < day`` only)."""

    def __init__(self, games: pd.DataFrame, cfg: RatingsConfig, prior: TeamRatings | None = None):
        self.cfg = cfg
        g = games.sort_values("daynum", kind="stable")
        self.teams = np.unique(np.concatenate([g["w_team"].to_numpy(), g["l_team"].to_numpy()]))
        idx = {int(t): i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.n = n
        wi = g["w_team"].map(idx).to_numpy()
        li = g["l_team"].map(idx).to_numpy()
        v_w = np.where(g["w_loc"].eq("H"), 1.0, np.where(g["w_loc"].eq("A"), -1.0, 0.0))
        day = g["daynum"].to_numpy()
        m = len(g)

        # rating rows: winner on offense, then loser on offense (row order follows day order)
        order = np.argsort(np.concatenate([day, day]), kind="stable")
        off_idx = np.concatenate([wi, li])[order]
        def_idx = np.concatenate([li, wi])[order]
        venue = np.concatenate([v_w, -v_w])[order]
        pts = np.concatenate([g["w_score"].to_numpy(), g["l_score"].to_numpy()]).astype(float)[order]
        poss = np.concatenate([g["poss"].to_numpy(), g["poss"].to_numpy()]).astype(float)[order]
        self.row_day = np.concatenate([day, day])[order]
        self.y = 100.0 * pts / poss if cfg.target == "ppp" else pts
        rows = np.repeat(np.arange(2 * m), 4)
        cols = np.column_stack([np.zeros(2 * m, int), np.ones(2 * m, int), 2 + off_idx, 2 + n + def_idx]).ravel()
        vals = np.column_stack([np.ones(2 * m), venue, np.ones(2 * m), np.ones(2 * m)]).ravel()
        self.X = sparse.csr_matrix((vals, (rows, cols)), shape=(2 * m, 2 + 2 * n))

        # tempo rows: one per game
        self.t_day = day
        if cfg.target == "ppp":
            rows_t = np.repeat(np.arange(m), 3)
            cols_t = np.column_stack([np.zeros(m, int), 1 + wi, 1 + li]).ravel()
            self.Xt = sparse.csr_matrix((np.ones(3 * m), (rows_t, cols_t)), shape=(m, 1 + n))
            self.yt = np.log(g["poss40"].to_numpy().astype(float))

        defaults = DEFAULTS[cfg.target]
        self.beta0 = np.zeros(2 + 2 * n)
        self.tbeta0 = np.zeros(1 + n)
        if prior is not None:
            pidx = prior.lookup()
            hit = np.array([pidx.get(int(t), -1) for t in self.teams])
            known = hit >= 0
            self.beta0[:2] = [prior.mu, prior.hca]
            self.beta0[2:2 + n][known] = prior.off[hit[known]]
            self.beta0[2 + n:][known] = prior.deff[hit[known]]
            self.tbeta0[0] = prior.tau
            self.tbeta0[1:][known] = prior.tempo[hit[known]]
        else:
            self.beta0[:2] = [defaults["mu"], defaults["hca"]]
            self.tbeta0[0] = DEFAULT_LOG_TEMPO

    def _ridge_solve(self, X, y, w, beta0, team_lambda: float, n_global: int) -> np.ndarray:
        p = X.shape[1]
        lam = np.full(p, team_lambda)
        lam[:n_global] = self.cfg.global_prior_weight
        if X.shape[0] == 0:  # no earlier games yet: the prior is the answer
            return beta0.copy()
        A = (X.T @ sparse.diags(w) @ X).toarray() + np.diag(lam)
        b = X.T @ (w * y) + lam * beta0
        return np.linalg.solve(A, b)

    def solve(self, day: int) -> TeamRatings:
        cfg = self.cfg
        team_lambda = cfg.ridge + cfg.prior_games * max(0.0, 1.0 - day / cfg.prior_fade_day)
        k = int(np.searchsorted(self.row_day, day, side="left"))  # rows strictly before `day`
        w = recency_weights(day - self.row_day[:k], cfg)
        beta = self._ridge_solve(self.X[:k], self.y[:k], w, self.beta0, team_lambda, 2)
        n = self.n
        if cfg.target == "ppp":
            kt = int(np.searchsorted(self.t_day, day, side="left"))
            wt = recency_weights(day - self.t_day[:kt], cfg)
            tb = self._ridge_solve(self.Xt[:kt], self.yt[:kt], wt, self.tbeta0, team_lambda, 1)
            tau, tempo = tb[0], tb[1:]
        else:
            tau, tempo = np.nan, np.zeros(n)
        return TeamRatings(self.teams, beta[2:2 + n], beta[2 + n:], tempo, float(beta[0]), float(beta[1]),
                           float(tau), cfg.target)


def asof_season(games: pd.DataFrame, cfg: RatingsConfig, prior: TeamRatings | None = None,
                predict: pd.DataFrame | None = None) -> tuple[pd.DataFrame, TeamRatings]:
    """Predict every game in ``predict`` (default: ``games``) with ratings as of its own day,
    fit on ``games`` from earlier days. Returns (predictions, end-of-season ratings)."""
    rater = SeasonRater(games, cfg, prior)
    target = games if predict is None else predict
    idx = {int(t): i for i, t in enumerate(rater.teams)}
    out = []
    for day, dg in target.groupby("daynum", sort=True):
        r = rater.solve(int(day))
        a = dg["team_a"].map(idx)
        b = dg["team_b"].map(idx)
        ok = a.notna() & b.notna()
        if not ok.any():
            continue
        dg = dg[ok]
        pa, pb, poss = r.expected_points(a[ok].astype(int).to_numpy(), b[ok].astype(int).to_numpy(),
                                         dg["venue_a"].to_numpy())
        out.append(pd.DataFrame({"game_id": dg["game_id"].to_numpy(), "pred_pts_a": pa, "pred_pts_b": pb,
                                 "pred_poss40": poss}))
    final = rater.solve(int(games["daynum"].max()) + 1)
    preds = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["game_id", "pred_pts_a", "pred_pts_b", "pred_poss40"])
    return preds, final


def run_ratings(games: pd.DataFrame, cfg: RatingsConfig) -> pd.DataFrame:
    """As-of predictions for every game of every season, carrying priors season to season."""
    out, prior, prev_season = [], None, None
    for season, gs in games.groupby("season", sort=True):
        use_prior = prior.as_prior(cfg.carryover) if prior is not None and prev_season == season - 1 else None
        preds, prior = asof_season(gs, cfg, use_prior)
        prev_season = season
        out.append(preds)
    preds = pd.concat(out, ignore_index=True)
    preds["pred_margin_a"] = preds["pred_pts_a"] - preds["pred_pts_b"]
    preds["pred_total_reg"] = preds["pred_pts_a"] + preds["pred_pts_b"]
    return preds
