"""Backtest harness: predict each week of each season from earlier games + last season's ratings."""
import sys, json, math, itertools, numpy as np
from pathlib import Path
FCS = "FCS (pooled)"
def load(path):
    G = []
    for r in json.load(open(path))["games"]:
        if not (r[6] or r[7]) or r[3] == r[4] and False: continue
        G.append(dict(week=r[0], home=r[1] if r[6] else FCS, away=r[2] if r[7] else FCS, hp=r[3], ap=r[4],
                      neutral=bool(r[5]), fbs=bool(r[6] and r[7]), box=r[8] if len(r) > 8 else None))
    return G
def teams_of(games): ts = sorted({g["home"] for g in games} | {g["away"] for g in games}); return ts, {t: i for i, t in enumerate(ts)}

def fit_margin(games, cap, lam, prior, default):
    """ridge toward prior: min sum (y - r_h + r_a - hfa)^2 + lam * sum (r - p)^2"""
    ts, ix = teams_of(games); n = len(ts)
    A = np.zeros((n + 1, n + 1)); b = np.zeros(n + 1)
    for g in games:
        h, a = ix[g["home"]], ix[g["away"]]; x = 0 if g["neutral"] else 1; y = max(-cap, min(cap, g["hp"] - g["ap"]))
        for i, ci in ((h, 1), (a, -1), (n, x)):
            b[i] += ci * y
            for j, cj in ((h, 1), (a, -1), (n, x)): A[i, j] += ci * cj
    p = np.array([prior.get(t, default) for t in ts])
    A[np.arange(n), np.arange(n)] += lam; b[:n] += lam * p; A[n, n] += 1e-6
    x = np.linalg.solve(A, b)
    return {t: x[i] for i, t in enumerate(ts)}, x[n]

def fit_od(games, val, lam, prior_o, prior_d, default=0.0):
    ts, ix = teams_of(games); n = len(ts); N = 2 * n + 2
    A = np.zeros((N, N)); b = np.zeros(N)
    for g in games:
        h, a = ix[g["home"]], ix[g["away"]]
        for side in (0, 1):
            v = val(g, side)
            if v is None: continue
            y, w = v; o, d = (h, a) if side == 0 else (a, h)
            cols = [o, n + d, 2 * n] + ([2 * n + 1] if side == 0 and not g["neutral"] else [])
            for i in cols:
                b[i] += w * y
                for j in cols: A[i, j] += w
    po = np.array([prior_o.get(t, default) for t in ts]); pd = np.array([prior_d.get(t, -default) for t in ts])
    A[np.arange(2 * n), np.arange(2 * n)] += lam; b[:n] += lam * po; b[n:2 * n] += lam * pd
    A[2 * n, 2 * n] += 1e-6; A[2 * n + 1, 2 * n + 1] += 1e-6
    x = np.linalg.solve(A, b)
    return dict(off={t: x[i] for i, t in enumerate(ts)}, defn={t: x[n + i] for i, t in enumerate(ts)}, mu=x[2 * n], hp=x[2 * n + 1])
exp_od = lambda M, o, d, home: M["mu"] + M["off"].get(o, 0) + M["defn"].get(d, 0) + (M["hp"] if home else 0)
rate = lambda y, att: (y / att, att) if att else None

def season_priors(prev, P):
    """End-of-season ratings from last season, regressed toward the mean by P['keep']."""
    R, _ = fit_margin(prev, P["cap"], 1.0, {}, 0.0)
    pr = {t: P["keep"] * r for t, r in R.items()}
    out = dict(R=pr, Rdef=P["keep"] * R.get(FCS, -15.0))
    bx = [g for g in prev if g["box"]]
    if bx:
        ypa = fit_od(bx, lambda g, s: rate(g["box"][4 * s], g["box"][4 * s + 1]), 50, {}, {})
        ypc = fit_od(bx, lambda g, s: rate(g["box"][4 * s + 2], g["box"][4 * s + 3]), 50, {}, {})
        k = P["keep_eff"]
        out["ypa"] = ({t: k * v for t, v in ypa["off"].items()}, {t: k * v for t, v in ypa["defn"].items()})
        out["ypc"] = ({t: k * v for t, v in ypc["off"].items()}, {t: k * v for t, v in ypc["defn"].items()})
    return out

def build(train, pri, P):
    R, hfa = fit_margin(train, P["cap"], P["lam"], pri["R"] if pri else {}, pri["Rdef"] if pri else 0.0) if train else ({}, 2.5)
    M = dict(R=R, hfa=hfa, pri=pri)
    bx = [g for g in train if g["box"]]
    if P["eff_w"] and (len(bx) >= 50 or (pri and "ypa" in pri)):
        po = pri["ypa"] if pri and "ypa" in pri else ({}, {}); pc = pri["ypc"] if pri and "ypc" in pri else ({}, {})
        if bx:
            ypa = fit_od(bx, lambda g, s: rate(g["box"][4 * s], g["box"][4 * s + 1]), P["eff_lam"], *po)
            ypc = fit_od(bx, lambda g, s: rate(g["box"][4 * s + 2], g["box"][4 * s + 3]), P["eff_lam"], *pc)
            M["ypa"], M["ypc"] = ypa, ypc
    return M

def margin(M, h, a, neutral, P, coef):
    def r(t):
        if t in M["R"]: return M["R"][t]
        pri = M["pri"]; return pri["R"].get(t, pri["Rdef"]) if pri else 0.0
    m = r(h) - r(a) + (0 if neutral else M["hfa"])
    if "ypa" in M and P["eff_w"]:
        pts = lambda o, d, home: coef[0] + coef[1] * exp_od(M["ypa"], o, d, home) + coef[2] * exp_od(M["ypc"], o, d, home)
        m = (1 - P["eff_w"]) * m + P["eff_w"] * (pts(h, a, not neutral) - pts(a, h, False))
    return m

def run(seasons, P, first_week=1):
    """seasons: list of (games, prev_games or None). Returns list of (pred_margin, actual, week)."""
    out = []
    for G, prev in seasons:
        pri = season_priors(prev, P) if prev and P["keep"] > 0 else None
        # points per YPA / YPC: stable relation, take it from last season if present else this season so far
        last = max(g["week"] for g in G)
        for wk in sorted({g["week"] for g in G}):
            if wk < first_week: continue
            tr = [g for g in G if g["week"] < wk]
            if not tr and not pri: continue
            M = build(tr, pri, P)
            src = [g for g in (prev or []) + tr if g["box"]]
            coef = None
            if src:
                X = []; y = []
                for g in src:
                    for s in (0, 1):
                        pa, ra = g["box"][4 * s + 1], g["box"][4 * s + 3]
                        if pa and ra: X.append([1, g["box"][4 * s] / pa, g["box"][4 * s + 2] / ra]); y.append(g["ap"] if s else g["hp"])
                X = np.array(X); coef = np.linalg.solve(X.T @ X, X.T @ np.array(y, float))
            for g in G:
                if g["week"] == wk and g["fbs"] and g["hp"] != g["ap"]:
                    out.append((margin(M, g["home"], g["away"], g["neutral"], P, coef) if coef is not None or not P["eff_w"] else margin(M, g["home"], g["away"], g["neutral"], dict(P, eff_w=0), None), g["hp"] - g["ap"], wk))
    return out

def score(res, sd=None):
    p = np.array([r[0] for r in res]); a = np.array([r[1] for r in res])
    sd = sd or float(np.std(a - p))
    pr = 0.5 * (1 + np.vectorize(math.erf)(p / (sd * math.sqrt(2)))); pr = np.clip(pr, 1e-4, 1 - 1e-4)
    win = (a > 0).astype(float)
    return dict(n=len(res), hit=float(np.mean((p > 0) == (a > 0))), mae=float(np.mean(np.abs(p - a))),
                ll=float(-np.mean(win * np.log(pr) + (1 - win) * np.log(1 - pr))), sd=sd)
