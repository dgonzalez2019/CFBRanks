import sys, json; sys.path.insert(0, sys.argv[1]); import tune as T
import numpy as np
from multiprocessing import Pool
P = dict(cap=45, lam=2, keep=0.9, keep_eff=0.6, eff_w=0.25, eff_lam=50)
H = {y: T.load(f"data/history/{y}.json") for y in range(2021, 2026)}
H[2026] = T.load("data/games.json")
def season(y):
    G, prev = H[y], H[y - 1]
    pri = T.season_priors(prev, P); out = []
    for wk in sorted({g["week"] for g in G}):
        tr = [g for g in G if g["week"] < wk]
        M = T.build(tr, pri, P)
        src = [g for g in prev + tr if g["box"]]
        X = []; yv = []
        for g in src:
            for s in (0, 1):
                pa, ra = g["box"][4 * s + 1], g["box"][4 * s + 3]
                if pa and ra: X.append([1, g["box"][4 * s] / pa, g["box"][4 * s + 2] / ra]); yv.append(g["ap"] if s else g["hp"])
        X = np.array(X); coef = np.linalg.solve(X.T @ X, X.T @ np.array(yv, float))
        for g in G:
            if g["week"] == wk and g["fbs"]:
                out.append(dict(season=y, week=wk, home=g["home"], away=g["away"], neutral=g["neutral"],
                                ours=float(T.margin(M, g["home"], g["away"], g["neutral"], P, coef)), act=g["hp"] - g["ap"]))
    return out
with Pool(4) as pool: res = pool.map(season, [2022, 2023, 2024, 2025, 2026])
json.dump([r for s in res for r in s], open(sys.argv[2], "w"))
print({y: len(s) for y, s in zip([2022, 2023, 2024, 2025, 2026], res)})
