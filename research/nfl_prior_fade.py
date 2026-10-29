"""NFL: should last season's prior fade as this season's games pile up? Same week-by-week setup as nfl_tune.py
(each week predicted from earlier games plus last season's prior), 2016-2025. The prior's ridge pull is scaled by
max(PRIOR_FLOOR, 1 - games per team / PRIOR_FADE); PRIOR_FADE 0 = no fading (the old setting).
Also reports how much weight the prior still has at week 9 and how well each week's ratings rank teams by their
results over the rest of that season (Spearman vs. rest-of-season point margin per game).
Usage: python research/nfl_prior_fade.py [workers]"""
import sys, json, itertools, pathlib, numpy as np
from multiprocessing import Pool
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cfb_sor as m
m.use_league("nfl")
H = ROOT / "data" / "nfl" / "history"
SEASONS = range(2016, 2026)
GAMES = {y: m.load_json(H / f"{y}.json") for y in range(2015, 2026)}

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])

def run(cfg):
    keep, lam, fade, floor = cfg
    m.PRIOR_KEEP, m.PRIOR_LAM, m.PRIOR_FADE, m.PRIOR_FLOOR = keep, lam, fade, floor
    out, rho = [], []
    for y in SEASONS:
        prior = m.season_prior(GAMES[y - 1]); G = GAMES[y]
        for wk in sorted({g["week"] for g in G}):
            tr = [g for g in G if g["week"] < wk]
            if not tr: continue
            M = m.fit_predictor(tr, prior)
            for g in G:
                if g["week"] != wk or g["home"] not in M["T"] or g["away"] not in M["T"]: continue
                p = m.predict(M, g["home"], g["away"], "N" if g["neutral"] else "A")
                out.append((wk, p["margin"], g["hp"] - g["ap"]))
            if wk in (5, 9, 13):  # ranking quality: ratings now vs. margin per game over the rest of the regular season
                rest = [g for g in G if wk <= g["week"] <= 18]; mg = {}
                for g in rest:
                    for t, d in ((g["home"], g["hp"] - g["ap"]), (g["away"], g["ap"] - g["hp"])): mg.setdefault(t, []).append(d)
                ts = [t for t in mg if t in M["T"]]
                rho.append((wk, spearman([M["T"][t]["r"] for t in ts], [np.mean(mg[t]) for t in ts])))
    return cfg, out, rho

def score(o, lo, hi):
    o = [x for x in o if lo <= x[0] <= hi]; a = np.array([x[2] for x in o]); p = np.array([x[1] for x in o]); nz = a != 0
    return np.mean(np.sign(p[nz]) == np.sign(a[nz])), np.mean(np.abs(p - a))

if __name__ == "__main__":
    grid = [(0.5, 8.0, 0, 0.0)] + list(itertools.product((0.5, 0.7), (8.0, 12.0), (8, 12, 17), (0.0, 0.2)))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool: res = pool.map(run, grid)
    print(f"{'keep':>5}{'lam':>5}{'fade':>5}{'floor':>6} | all win/MAE | wk1-4 | wk5-9 | wk10+ | rank rho wk5/9/13")
    rows = []
    for cfg, o, rho in res:
        r = [score(o, *b) for b in ((1, 99), (1, 4), (5, 9), (10, 99))]
        rr = [np.mean([v for w, v in rho if w == k]) for k in (5, 9, 13)]
        rows.append((r[0][1], cfg, r, rr))
    for mae, cfg, r, rr in sorted(rows, key=lambda x: x[0]):
        print(f"{cfg[0]:>5}{cfg[1]:>5}{cfg[2]:>5}{cfg[3]:>6} | " + " | ".join(f"{w:.1%}/{e:.2f}" for w, e in r) + " | " + "/".join(f"{v:.3f}" for v in rr))
    json.dump([dict(cfg=c, all=r[0], wk1_4=r[1], wk5_9=r[2], wk10=r[3], rho=rr) for _, c, r, rr in rows], open(ROOT / "research" / "nfl_prior_fade.json", "w"), indent=1)
