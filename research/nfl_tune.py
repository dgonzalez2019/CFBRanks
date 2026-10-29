"""NFL: tune the Matchup model (the power rating) on 2016-2025. Each week's games are predicted from earlier games
that season plus last season's ratings (season_prior), exactly as the site does it; the closing line is the
benchmark. Grid over PRIOR_KEEP, PRIOR_LAM, the margin cap and EFF_WEIGHT.
Usage: python research/nfl_tune.py [workers]"""
import sys, json, itertools, pathlib, numpy as np
from multiprocessing import Pool
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cfb_sor as m
H = ROOT / "data" / "nfl" / "history"
SEASONS = range(2016, 2026)
GAMES = {y: m.load_json(H / f"{y}.json") for y in range(2015, 2026)}
LINES = {y: {(r[0], r[1], r[2]): -r[4] for r in json.load(open(H / f"{y}_lines.json"))["games"] if r[4] is not None} for y in SEASONS}

def run(cfg):
    keep, lam, cap, eff = cfg
    m.PRIOR_KEEP, m.PRIOR_LAM, m.PRIOR_CAP, m.EFF_WEIGHT = keep, lam, cap, eff
    out = []
    for y in SEASONS:
        prior = m.season_prior(GAMES[y - 1]); G = GAMES[y]
        for wk in sorted({g["week"] for g in G}):
            tr = [g for g in G if g["week"] < wk]
            if not tr: continue
            M = m.fit_predictor(tr, prior)
            for g in G:
                if g["week"] != wk or g["home"] not in M["T"] or g["away"] not in M["T"]: continue
                p = m.predict(M, g["home"], g["away"], "N" if g["neutral"] else "A")
                out.append((y, wk, p["margin"], g["hp"] - g["ap"], LINES[y].get((wk, g["home"], g["away"])), p["pa"]))
    return cfg, out

def score(o, lo=1, hi=99):
    o = [x for x in o if lo <= x[1] <= hi]
    act = np.array([x[3] for x in o]); ours = np.array([x[2] for x in o]); nz = act != 0
    win = np.mean(np.sign(ours[nz]) == np.sign(act[nz])); mae = np.mean(np.abs(ours - act))
    pa = np.clip([x[5] for x in o], 1e-6, 1 - 1e-6); ll = -np.mean(np.where(act > 0, np.log(pa), np.log(1 - pa))[nz])
    return len(o), win, mae, ll

if __name__ == "__main__":
    grid = list(itertools.product((0.3, 0.5, 0.7, 0.9), (2, 4, 8, 16), (21, 28, 35), (0.0, 0.25, 0.5)))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool: res = pool.map(run, grid)
    _, base = res[0]
    L = [x for x in base if x[4] is not None]
    act = np.array([x[3] for x in L]); line = np.array([x[4] for x in L]); nz = act != 0
    print(f"closing line, same games: winners {np.mean(np.sign(line[nz]) == np.sign(act[nz])):.1%}, avg miss {np.mean(np.abs(line - act)):.2f} (n={len(L)})")
    rows = sorted(((score(o), cfg, o) for cfg, o in res), key=lambda r: r[0][2])
    print(f"{'keep':>5}{'lam':>5}{'cap':>5}{'eff':>6} | {'all: n':>7} {'win':>6} {'MAE':>6} {'LL':>6} | wk1-4 win/MAE | wk5+ win/MAE")
    for (n, w, mae, ll), cfg, o in rows[:12] + [r for r in rows if r[1] == (0.9, 2, 35, 0.25)]:
        a = score(o, 1, 4); b = score(o, 5)
        print(f"{cfg[0]:>5}{cfg[1]:>5}{cfg[2]:>5}{cfg[3]:>6} | {n:>7} {w:>6.1%} {mae:>6.2f} {ll:>6.4f} | {a[1]:.1%} / {a[2]:.2f} | {b[1]:.1%} / {b[2]:.2f}")
    best = rows[0][1]
    print("best by MAE:", best, "| CFB settings (0.9, 2, 35, 0.25) shown last for comparison")
    json.dump(dict(best=best), open(ROOT / "research" / "nfl_tune_best.json", "w"))
