"""Positional matchups experiment: for every FBS game of 2022-26, predict from earlier games (plus last season)
the score margin (m1), the passing/rushing efficiency margin (eff) and a fuller matchup margin (full) that adds
sack, tackle-for-loss, pressure, turnover and third-down rates. Writes <work>/matchups.json for blend tuning.
Usage: python research/matchups.py research <work dir>"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); import tune as T
from multiprocessing import Pool
P = dict(cap=45, lam=2, keep=0.9, keep_eff=0.6, eff_w=0, eff_lam=50)
RIDGE = 50

def units(g, s):
    """Offense s vs. defense d=1-s rates for one game, each (value, weight) or None."""
    if not g["box"] or not g["box2"]: return {}
    b, c, d = g["box"], g["box2"], 1 - s
    pa, ra, py, ry = b[4*s+1], b[4*s+3], b[4*s], b[4*s+2]
    sacks, tfl, hur = c[6*d], c[6*d+1], c[6*d+2]
    to, conv, tried = c[6*s+3], c[6*s+4], c[6*s+5]
    drop, plays = pa + sacks, pa + ra
    out = {}
    if pa: out["ypa"] = (py / pa, pa)
    if ra: out["ypc"] = (ry / ra, ra)
    if drop: out["sack"] = (sacks / drop, drop); out["hurry"] = (hur / drop, drop)
    if plays: out["tfl"] = (tfl / plays, plays); out["to"] = (to / plays, plays)
    if tried: out["third"] = (conv / tried, tried)
    return out
FEATS = ["ypa", "ypc", "sack", "tfl", "hurry", "to", "third"]

def season(y):
    G = T.load(f"data/history/{y}.json") if y < 2026 else T.load("data/games.json")
    prev = T.load(f"data/history/{y-1}.json")
    pri = T.season_priors(prev, P); out = []
    for wk in sorted({g["week"] for g in G}):
        tr = [g for g in G if g["week"] < wk]
        M = T.build(tr, pri, P)
        bx = [g for g in tr if g["box"] and g["box2"]]
        src = [g for g in prev + tr if g["box"] and g["box2"]]
        # points per unit rate, from last season + this season so far
        X = []; yv = []
        for g in src:
            for s in (0, 1):
                u = units(g, s)
                if all(f in u for f in FEATS): X.append([1] + [u[f][0] for f in FEATS]); yv.append(g["ap"] if s else g["hp"])
        X = np.array(X); yv = np.array(yv, float)
        cols_base = [0, 1, 2]; cols_full = list(range(len(FEATS) + 1))
        cb = np.linalg.lstsq(X[:, cols_base], yv, rcond=None)[0]; cf = np.linalg.lstsq(X[:, cols_full], yv, rcond=None)[0]
        fits = {}
        if len(bx) >= 50:
            for f in FEATS:
                fits[f] = T.fit_od(bx, lambda g, s, f=f: units(g, s).get(f), RIDGE, {}, {})
        for g in G:
            if g["week"] != wk or not g["fbs"]: continue
            h, a, neu = g["home"], g["away"], g["neutral"]
            m1 = float(T.margin(M, h, a, neu, P, None))
            row = dict(season=y, week=wk, home=h, away=a, act=g["hp"] - g["ap"], m1=m1)
            if fits:
                ex = {f: (T.exp_od(fits[f], h, a, not neu), T.exp_od(fits[f], a, h, False)) for f in FEATS}
                pts = lambda coef, cols, side: coef[0] + sum(coef[i] * ex[FEATS[c - 1]][side] for i, c in enumerate(cols) if c > 0)
                row["eff"] = float(pts(cb, cols_base, 0) - pts(cb, cols_base, 1))
                row["full"] = float(pts(cf, cols_full, 0) - pts(cf, cols_full, 1))
                row["coef_full"] = [float(x) for x in cf]
            out.append(row)
    return out

if __name__ == "__main__":
    with Pool(4) as pool: res = pool.map(season, [2022, 2023, 2024, 2025, 2026])
    json.dump([r for s in res for r in s], open(sys.argv[2] + "/matchups.json", "w"))
    print({y: len(s) for y, s in zip([2022, 2023, 2024, 2025, 2026], res)})
