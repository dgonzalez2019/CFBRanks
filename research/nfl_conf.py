"""NFL confidence scores (spreads and totals), built like research/conf_score.py and conf_total.py.

For every 2016-2026 game with a line (week 2 on), the pre-game inputs are rebuilt from earlier games only: our margin
and total (Matchup model with the NFL settings and last season's prior), each team's last result against the spread
and the total, their season over rate, pace (plays per game from box scores) and the EPA efficiency model's margin
(ppa_model on nflverse EPA per play, last season as its prior; its points scale is fit on the training seasons).
Logistic models on our side of the line / total; each season is scored by a model fit on the other 2016-2025
seasons, 2026 by the 2022-2025... no: by the full 2016-2025 fit. Prints top 10% / 5% records and writes
research/nfl_conf.json with the final coefficients.
Usage: python research/nfl_conf.py"""
import sys, json, pathlib, math, numpy as np
from collections import defaultdict
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import cfb_sor as m
m.use_league("nfl")
D = ROOT / "data" / "nfl"; H = D / "history"
TOTAL_MID = 45.0  # NFL totals sit lower than CFB's; the "high" feature is centered here

def season_data(y):
    if y < 2026:
        G = m.load_json(H / f"{y}.json"); L = json.load(open(H / f"{y}_lines.json"))["games"]; P = json.load(open(H / f"{y}_ppa.json"))["games"]
    else:
        G = m.load_json(D / "games.json"); L = json.load(open(D / "lines.json"))["games"]; P = json.load(open(D / "ppa.json"))["games"]
    return G, {(r[0], r[1], r[2]): (-r[4], r[5]) for r in L if r[4] is not None}, P

def rows_for(y):
    G, LN, PP = season_data(y)
    prev_G = m.load_json(H / f"{y-1}.json"); prev_P = json.load(open(H / f"{y-1}_ppa.json"))["games"]
    prior = m.season_prior(prev_G); teams = {g["home"] for g in G} | {g["away"] for g in G}
    out = []
    for wk in sorted({g["week"] for g in G}):
        tr = [g for g in G if g["week"] < wk]
        if not tr: continue
        M = m.fit_predictor(tr, prior)
        P = m.ppa_model([r for r in PP if r[0] < wk], prev_P, {}, teams)
        off, de, pri = P; o = lambda t: off.get(t, pri[0].get(t, 0)); d = lambda t: de.get(t, pri[1].get(t, 0))
        hist = defaultdict(list)  # team -> [(week, ats cover, over margin, plays)]
        for g in sorted(tr, key=lambda g: g["week"]):
            ln = LN.get((g["week"], g["home"], g["away"]))
            act = g["hp"] - g["ap"]
            for t, sign, s in ((g["home"], 1, 0), (g["away"], -1, 1)):
                cover = None if not ln or ln[0] is None else sign * (act - ln[0])
                over = None if not ln or ln[1] is None else (g["hp"] + g["ap"]) - ln[1]
                plays = (g["box"][4 * s + 1] + g["box"][4 * s + 3]) if g.get("box") else None
                hist[t].append((g["week"], cover, over, plays))
        for g in G:
            if g["week"] != wk: continue
            h, a = g["home"], g["away"]; ln = LN.get((wk, h, a))
            if not ln or h not in M["T"] or a not in M["T"]: continue
            p = m.predict(M, h, a, "N" if g["neutral"] else "A")
            def form(t):
                e = hist[t]; ou = [x[2] for x in e if x[2] is not None]; pl = [x[3] for x in e if x[3]]
                return (e[-1][1] if e and e[-1][1] is not None else 0.0, ou[-1] if ou else 0.0,
                        (sum(v > 0 for v in ou) + 1) / (len(ou) + 2), (sum(pl) / len(pl)) if pl else None)
            fh, fa = form(h), form(a)
            out.append(dict(season=y, week=wk, home=h, away=a, act=g["hp"] - g["ap"], act_total=g["hp"] + g["ap"],
                            line=ln[0], total=ln[1], ours=p["margin"], ours_total=p["sa"] + p["sb"],
                            epa=(o(h) + d(a)) - (o(a) + d(h)), ats_h=fh[0], ats_a=fa[0], ou_h=fh[1], ou_a=fa[1],
                            fr_h=fh[2], fr_a=fa[2], pace=[x for x in (fh[3], fa[3]) if x]))
    return out

def fit(X, y, l2=10.0):
    X = np.array(X); y = np.array(y, float); b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b)); Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b -= np.linalg.solve(Hm, X.T @ (p - y) + l2 * np.r_[0, b[1:]])
    return b

def spread_rows(R, A, B):
    out = []
    for r in R:
        if r["act"] == r["line"] or r["ours"] == r["line"]: continue
        s = 1 if r["ours"] > r["line"] else -1
        la, lo = (r["ats_h"], r["ats_a"]) if s > 0 else (r["ats_a"], r["ats_h"])
        pm = A * r["epa"] + B
        bad = -(la - lo) / 14; dis = -s * (pm - r["line"]) / 7; gap = abs(r["ours"] - r["line"]) / 7
        out.append(dict(r, x=[1, bad, dis, gap, float(bad > 0 and dis > 0)], won=(r["act"] > r["line"]) == (s > 0)))
    return out

def total_rows(R, pace_avg):
    out = []
    for r in R:
        if r["total"] is None or r["act_total"] == r["total"] or r["ours_total"] == r["total"]: continue
        s = 1 if r["ours_total"] > r["total"] else -1
        pace = sum(r["pace"]) / len(r["pace"]) if r["pace"] else pace_avg
        x = [1, abs(r["ours_total"] - r["total"]) / 7, -s * (r["ou_h"] + r["ou_a"]) / 14, -s * (r["fr_h"] + r["fr_a"] - 1),
             s * (pace - pace_avg) / 10, -s * (r["total"] - TOTAL_MID) / 10]
        out.append(dict(r, x=x, won=(r["act_total"] > r["total"]) == (s > 0)))
    return out

def epa_scale(R):
    X = np.array([[r["epa"], 1] for r in R]); y = np.array([r["act"] for r in R], float)
    return np.linalg.lstsq(X, y, rcond=None)[0]

def topk(p, won, frac):
    o = np.argsort(-p)[:max(1, int(round(len(p) * frac)))]; w = int(won[o].sum()); return w, len(o) - w

if __name__ == "__main__":
    ALL = {y: rows_for(y) for y in range(2016, 2027)}
    print("games:", {y: len(v) for y, v in ALL.items()})
    TRAIN = range(2016, 2026)
    pace_avg = float(np.mean([v for y in TRAIN for r in ALL[y] for v in r["pace"]]))
    res = {}
    for kind in ("spread", "total"):
        print(f"\n== {kind} ==  (held out: each season scored by a model fit on the other 2016-2025 seasons; 2026 by all of 2016-2025)")
        tot = {"top10": [0, 0], "top5": [0, 0]}; by = {}
        for y in list(TRAIN) + [2026]:
            trn = [r for t in TRAIN if t != y for r in ALL[t]]
            A, B = epa_scale(trn)
            mk = (lambda R: spread_rows(R, A, B)) if kind == "spread" else (lambda R: total_rows(R, pace_avg))
            tr, te = mk(trn), mk(ALL[y])
            if not te: continue
            b = fit([r["x"] for r in tr], [r["won"] for r in tr])
            p = 1 / (1 + np.exp(-np.array([r["x"] for r in te]) @ b)); won = np.array([r["won"] for r in te])
            t10, t5 = topk(p, won, .10), topk(p, won, .05); by[y] = t10
            if y != 2026:
                for k, v in (("top10", t10), ("top5", t5)): tot[k][0] += v[0]; tot[k][1] += v[1]
            print(f"  {y}: our side {won.mean():.1%} (n={len(te)}) | top 10% {t10[0]}-{t10[1]} ({t10[0]/max(1,sum(t10)):.0%}) | top 5% {t5[0]}-{t5[1]}")
        print(f"  2016-2025 held out: top 10% {tot['top10'][0]}-{tot['top10'][1]} ({tot['top10'][0]/sum(tot['top10']):.1%}), "
              f"top 5% {tot['top5'][0]}-{tot['top5'][1]} ({tot['top5'][0]/sum(tot['top5']):.1%})")
        trn = [r for t in TRAIN for r in ALL[t]]; A, B = epa_scale(trn)
        rows = spread_rows(trn, A, B) if kind == "spread" else total_rows(trn, pace_avg)
        bfin = fit([r["x"] for r in rows], [r["won"] for r in rows])
        print("  final coefficients (2016-2025):", np.round(bfin, 4), "| EPA scale A, B:", round(float(A), 3), round(float(B), 3))
        res[kind] = dict(coef=[float(v) for v in bfin], held=tot, by_season={str(k): v for k, v in by.items()}, epa_a=float(A), epa_b=float(B))
    res["pace_avg"] = pace_avg; res["total_mid"] = TOTAL_MID
    json.dump(res, open(ROOT / "research" / "nfl_conf.json", "w"), indent=1)
