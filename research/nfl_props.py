"""NFL player props: fit and test the projection model in scripts/props_model.py on nflverse weekly player stats.

Each player-game (regular season + playoffs) is projected from earlier games only (see props_model.py). Fit on
2016-2022, tested on 2023-2025. There are no free historical prop lines, so this can't test whether picks beat the
sportsbooks: it tests projection accuracy (vs. the plain season average) and whether the over probabilities are
calibrated on held-out games at lines around the projection. Writes research/nfl_props.json (coefficients and
quantiles, fit on 2016-2025 for live use; the printed test numbers come from the 2016-2022 fit).
Usage: python research/nfl_props.py <dir with stats_player_week_YYYY.csv files>"""
import sys, json, pathlib, numpy as np
from collections import defaultdict
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import props_model as pm
from fetch_nfl import TEAMS

def season_rows(d, y):
    cur = pm.load_rows(open(pathlib.Path(d) / f"stats_player_week_{y}.csv").read(), TEAMS)
    prev = pm.load_rows(open(pathlib.Path(d) / f"stats_player_week_{y-1}.csv").read(), TEAMS)
    imp = pm.implied_points(json.load(open(ROOT / "data" / "nfl" / "history" / f"{y}_lines.json"))["games"])
    S = pm.Season(cur, prev); out = []
    for g in cur:
        it = imp.get((g["week"], g["team"]))
        if it is None or not g["opp"]: continue
        for mk, (col, *_) in pm.MARKETS.items():
            f = S.features(g["id"], g["pos"], g["opp"], g["week"], mk, it, g["team"])
            if f: out.append(dict(f, mk=mk, y=g[col]))
    return out

def fit(R, k=None, count=False):
    X = np.array([r["x"][:k] for r in R]); y = np.array([r["y"] for r in R])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    M = dict(coef=[float(c) for c in coef])
    if count:  # small counts (TDs, interceptions): Poisson around the projection, scale chosen by likelihood
        M["floor"] = 0.02; p = np.array([pm.project(M, r["x"][:k]) for r in R])
        ll = lambda s_: np.sum(y * np.log(s_ * p) - s_ * p)
        M["poisson"] = float(max(np.arange(0.6, 1.41, 0.02), key=ll))
    M["edges"], M["q"] = pm.fit_quantiles([pm.project(M, r["x"][:k]) for r in R], y)
    return M

if __name__ == "__main__":
    d = sys.argv[1]
    rows = {y: season_rows(d, y) for y in range(2016, 2026)}
    TRAIN, TEST = range(2016, 2023), range(2023, 2026)
    res = {}
    for mk in pm.MARKETS:
        tr = [r for y in TRAIN for r in rows[y] if r["mk"] == mk]; te = [r for y in TEST for r in rows[y] if r["mk"] == mk]
        cnt = mk in pm.COUNT_MK
        M = fit(tr, count=cnt)
        # teammate absences: the same model without the two absence terms, on the same held-out games
        M4 = fit(tr, 4, cnt); p4 = np.array([max(M4.get("floor", .5), float(np.dot(M4["coef"], r["x"][:4]))) for r in te])
        pa = np.array([pm.project(M, r["x"]) for r in te]); ya = np.array([r["y"] for r in te])
        sub = lambda m: f"{np.mean(np.abs(pa[m] - ya[m])):.2f} vs {np.mean(np.abs(p4[m] - ya[m])):.2f} (n={int(m.sum())})"
        vac = np.array([r["vac"] for r in te]); qo = np.array([r["qb_out"] for r in te])
        print(f"  teammates out: all {sub(np.ones(len(te), bool))} | 15%+ of volume out {sub(vac >= .15)} | main QB out {sub(qo == 1)}")
        pt = np.array([pm.project(M, r["x"]) for r in te]); yt = np.array([r["y"] for r in te]); av = np.array([r["avg"] for r in te])
        cal = defaultdict(lambda: [0.0, 0, 0])
        for p_, y_ in zip(pt, yt):
            for L in ((0.5, 1.5, 2.5) if cnt else [np.floor(p_ * k) + 0.5 for k in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)]):
                po = pm.p_over(M, p_, L); b = min(9, int(po * 10))
                cal[b][0] += po; cal[b][1] += y_ > L; cal[b][2] += 1
        # picks at lines near the projection: our side's record by our probability
        side = defaultdict(lambda: [0, 0])
        for p_, y_ in zip(pt, yt):
            for L in ((0.5, 1.5) if cnt else [np.floor(p_ * k) + 0.5 for k in (0.85, 0.95, 1.05, 1.15)]):
                po = pm.p_over(M, p_, L); c = max(po, 1 - po)
                won = (y_ > L) == (po > 0.5); b = "55-60%" if c < .6 else "60-65%" if c < .65 else "65%+" if c >= .65 else ""
                if c >= .55: side[b][0] += won; side[b][1] += 1
        print(f"\n== {mk}: train n={len(tr)}, test n={len(te)} | coef {np.round(M['coef'], 3)}")
        print(f"  held-out avg miss: model {np.mean(np.abs(pt - yt)):.2f} vs season average {np.mean(np.abs(av - yt)):.2f}")
        print("  calibration (predicted over % -> actual):", ", ".join(f"{100*c[0]/c[2]:.0f}->{100*c[1]/c[2]:.0f} (n={c[2]})" for _, c in sorted(cal.items()) if c[2] >= 50))
        print("  our side at lines 0.85-1.15x the projection:", ", ".join(f"{k}: {v[0]/v[1]:.1%} (n={v[1]})" for k, v in sorted(side.items())))
        full = fit([r for y in range(2016, 2026) for r in rows[y] if r["mk"] == mk], count=cnt)
        if cnt: print(f"  Poisson scale {M['poisson']:.2f}")
        cal_out = {f"{10*b}-{10*b+10}": [round(c[0] / c[2], 3), round(c[1] / c[2], 3), c[2]] for b, c in sorted(cal.items()) if c[2] >= 50}
        res[mk] = dict(full, test=dict(mae=float(np.mean(np.abs(pt - yt))), mae_avg=float(np.mean(np.abs(av - yt))), n=len(te), calibration=cal_out))
    res["settings"] = dict(DECAY=pm.DECAY, PRIOR_GAMES=pm.PRIOR_GAMES, OPP_SHRINK=pm.OPP_SHRINK, BINS=pm.BINS)
    json.dump(res, open(ROOT / "research" / "nfl_props.json", "w"), indent=1)
