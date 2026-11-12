"""A synthetic league with known team strengths, in the unified game-table schema."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

MU, HCA, LOG_TEMPO = 102.0, 3.5, float(np.log(68.0))


def make_league(n_teams=30, seasons=(2020,), n_days=110, games_per_day=10, seed=0, home_share=0.8,
                carry=0.7, noise_ppp=9.0):
    rng = np.random.default_rng(seed)
    teams = np.arange(1101, 1101 + n_teams)
    off, deff, tempo = rng.normal(0, 6, n_teams), rng.normal(0, 6, n_teams), rng.normal(0, 0.06, n_teams)
    rows, truth = [], {}
    for season in seasons:
        if season != seasons[0]:  # strengths persist partly from season to season
            off = carry * off + rng.normal(0, 6 * np.sqrt(1 - carry**2), n_teams)
            deff = carry * deff + rng.normal(0, 6 * np.sqrt(1 - carry**2), n_teams)
        truth[season] = {"off": off.copy(), "deff": deff.copy(), "tempo": tempo.copy()}
        zero = date(season - 1, 11, 2)
        for day in range(n_days):
            perm = rng.permutation(n_teams)[: 2 * games_per_day].reshape(-1, 2)
            for i, j in perm:
                v = 1 if rng.random() < home_share else 0  # i is home when v == 1
                poss40 = float(np.exp(LOG_TEMPO + tempo[i] + tempo[j] + rng.normal(0, 0.03)))
                ppp_i = MU + off[i] + deff[j] + HCA * v + rng.normal(0, noise_ppp)
                ppp_j = MU + off[j] + deff[i] - HCA * v + rng.normal(0, noise_ppp)
                si, sj = int(np.rint(poss40 * ppp_i / 100)), int(np.rint(poss40 * ppp_j / 100))
                numot = 0
                if si == sj:
                    si, numot = si + 1, 1
                    poss = poss40 * 45 / 40
                else:
                    poss = poss40
                w, lo = (i, j) if si > sj else (j, i)
                w_score, l_score = max(si, sj), min(si, sj)
                w_loc = "N" if v == 0 else ("H" if w == i else "A")
                a, b = (i, j) if v == 1 else ((i, j) if teams[i] < teams[j] else (j, i))
                rows.append({
                    "season": season, "daynum": day, "date": zero + timedelta(days=day),
                    "phase": "regular", "w_team": int(teams[w]), "l_team": int(teams[lo]),
                    "w_score": w_score, "l_score": l_score, "w_loc": w_loc, "numot": numot,
                    "poss": poss, "poss40": poss40, "neutral": v == 0,
                    "team_a": int(teams[a]), "team_b": int(teams[b]), "venue_a": v,
                    "score_a": si if a == i else sj, "score_b": sj if a == i else si,
                })
    g = pd.DataFrame(rows)
    g["margin_a"] = g["score_a"] - g["score_b"]
    g["total"] = g["score_a"] + g["score_b"]
    g.insert(0, "game_id", [f"{s}_{d}_{a}_{b}" for s, d, a, b in zip(g["season"], g["daynum"], g["team_a"], g["team_b"])])
    return g, teams, truth
