"""Play-efficiency experiment: opponent-adjusted offense/defense PPA per play (predicted points added, garbage
time excluded), each season seeded by last season's ratings (kept at KEEP) nudged by recruiting talent, turned
into a margin and blended with our score model. Every prediction uses only games before that week.
Writes <work>/ppa_preds.json. Usage: python research/ppa_model.py research <work dir> <repo root>"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); import tune as T
from multiprocessing import Pool
W, ROOT = sys.argv[2], sys.argv[3]
FCS = T.FCS
KEEP, RIDGE = 0.6, 4.0

def ppa_games(y, fbs):
    rows = json.load(open(f"{ROOT}/data/history/{y}_ppa.json"))["games"]
    out = {}
    for wk, team, opp, o, op, orr, d, dp, dr in rows:
        out[(wk, team if team in fbs else FCS, opp if opp in fbs else FCS)] = (o, d)
    return out

def fit(gs, prior):
    """gs: list of (off_team, def_team, ppa per play). y = mu + off + def, ridge toward prior."""
    teams = sorted({t for g in gs for t in g[:2]}); ix = {t: i for i, t in enumerate(teams)}; n = len(teams)
    A = np.zeros((2 * n + 1, 2 * n + 1)); b = np.zeros(2 * n + 1)
    for o, d, y in gs:
        cols = [ix[o], n + ix[d], 2 * n]
        for i in cols:
            b[i] += y
            for j in cols: A[i, j] += 1
    for t, i in ix.items():
        A[i, i] += RIDGE; b[i] += RIDGE * prior[0].get(t, 0); A[n + i, n + i] += RIDGE; b[n + i] += RIDGE * prior[1].get(t, 0)
    A[2 * n, 2 * n] += 1e-6
    x = np.linalg.solve(A, b)
    return {t: x[ix[t]] for t in teams}, {t: x[n + ix[t]] for t in teams}, x[2 * n]

def season(y):
    G = T.load(f"{ROOT}/data/history/{y}.json") if y < 2026 else T.load(f"{ROOT}/data/games.json")
    fbs = {g["home"] for g in G if g["home"] != FCS} | {g["away"] for g in G if g["away"] != FCS}
    P, Pprev = ppa_games(y, fbs), ppa_games(y - 1, fbs)
    o0, d0, _ = fit([(t, op, v[0]) for (w, t, op), v in Pprev.items()] + [(op, t, v[1]) for (w, t, op), v in Pprev.items()], ({}, {}))
    tal = dict(json.load(open(f"{ROOT}/data/history/{y}_talent.json"))["talent"])
    tv = np.array([tal[t] for t in fbs if t in tal]); tz = {t: (tal[t] - tv.mean()) / tv.std() for t in fbs if t in tal}
    prior = ({t: KEEP * o0.get(t, 0) + 0.02 * tz.get(t, 0) for t in fbs | {FCS}},
             {t: KEEP * d0.get(t, 0) - 0.02 * tz.get(t, 0) for t in fbs | {FCS}})
    out = []
    for wk in sorted({g["week"] for g in G}):
        rows = [(t, op, v[0]) for (w, t, op), v in P.items() if w < wk]
        off, de, mu = fit(rows, prior) if rows else (prior[0], prior[1], 0.0)
        for g in G:
            if g["week"] != wk or not g["fbs"]: continue
            h, a = g["home"], g["away"]
            net = lambda t, u: (off.get(t, prior[0].get(t, 0)) + de.get(u, prior[1].get(u, 0))) - (off.get(u, prior[0].get(u, 0)) + de.get(t, prior[1].get(t, 0)))
            out.append(dict(season=y, week=wk, home=h, away=a, neutral=g["neutral"], ppa_net=float(net(h, a)), act=g["hp"] - g["ap"]))
    return out

if __name__ == "__main__":
    with Pool(4) as pool: res = [r for s in pool.map(season, [2022, 2023, 2024, 2025, 2026]) for r in s]
    json.dump(res, open(f"{W}/ppa_preds.json", "w")); print(len(res), "games")
