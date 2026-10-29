"""Pre-game features for spread and over/under picks, 2022-26: for each FBS game, from earlier games plus last
season only: our predicted total (points model, with and without last season as a prior), each team's pace
(plays per game) and recent form. Joins our margin (preds.json) and the betting lines. Writes <work>/picks.json.
Usage: python research/picks_features.py research <work dir> <repo root>"""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); import tune as T
from multiprocessing import Pool
W, ROOT = sys.argv[2], sys.argv[3]
KEEP_PTS = 0.6

def pts_model(games, prior=None):
    o = prior[0] if prior else {}; d = prior[1] if prior else {}
    return T.fit_od(games, lambda g, s: ((g["ap"] if s else g["hp"]), 1), 2, o, d)

def plays(g, s): return (g["box"][4*s+1] + g["box"][4*s+3]) if g["box"] else None

def season(y):
    G = T.load(f"{ROOT}/data/history/{y}.json") if y < 2026 else T.load(f"{ROOT}/data/games.json")
    prev = T.load(f"{ROOT}/data/history/{y-1}.json")
    P0 = pts_model(prev)
    prior = ({t: KEEP_PTS * v for t, v in P0["off"].items()}, {t: KEEP_PTS * v for t, v in P0["defn"].items()})
    out = []
    for wk in sorted({g["week"] for g in G}):
        tr = [g for g in G if g["week"] < wk]
        Mp = pts_model(tr, prior) if tr else None
        Mn = pts_model(tr) if len(tr) > 100 else None
        pace = {}
        for g in tr:
            for s, t in ((0, g["home"]), (1, g["away"])):
                if plays(g, s): pace.setdefault(t, []).append(plays(g, s))
        for g in G:
            if g["week"] != wk or not g["fbs"]: continue
            h, a, neu = g["home"], g["away"], g["neutral"]
            def tot(M):
                if not M: return None
                return T.exp_od(M, h, a, not neu) + T.exp_od(M, a, h, False)
            pre = lambda t: T.exp_od(dict(off=prior[0], defn=prior[1], mu=P0["mu"], hp=P0["hp"]), t, a if t == h else h, False)
            out.append(dict(season=y, week=wk, home=h, away=a, neutral=neu, act_total=g["hp"] + g["ap"],
                            tot_prior=tot(Mp) if Mp else (pre(h) + pre(a)), tot_now=tot(Mn),
                            pace_h=float(np.mean(pace[h])) if h in pace else None, pace_a=float(np.mean(pace[a])) if a in pace else None))
    return out

if __name__ == "__main__":
    with Pool(4) as pool: res = [r for s in pool.map(season, [2022, 2023, 2024, 2025, 2026]) for r in s]
    J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(f"{W}/joined.json"))}
    keep = []
    for r in res:
        j = J.get((r["season"], r["week"], r["home"], r["away"]))
        if j: keep.append(dict(r, **{k: j[k] for k in ("ours", "mkt", "total", "open", "act")}))
    json.dump(keep, open(f"{W}/picks.json", "w"))
    print(len(keep), "games with lines and features")
