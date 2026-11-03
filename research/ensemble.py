"""Ensemble confidence: many simple pick models, weighted by how well each has done on every earlier game.

Each "expert" is a one-signal model of whether the home team covers the spread (or the game goes over the total):
p = logistic(b * x), with x the signal from the home team's (or the over's) side, standardized, and b refit every
week on all earlier games (ridge). One expert is the site's current confidence score (the multi-signal logistic in
track_picks.py), also refit every week on earlier games only. The ensemble's chance is the weighted average of the
experts' chances; weights follow each expert's record on every earlier game ("multiplicative weights"):
    w_i  proportional to  exp(-eta * (cumulative loss of expert i - best expert's))
with loss = log loss of its chance (eta = 0: plain average). Our pick is the ensemble's side, its confidence the
ensemble's chance for that side; each week's top 10% / 5% by confidence are graded like the site's flags.

Seasons: the first season only trains the experts; eta is chosen on the next (tuning) season(s); every later season is
the test, compared with the current confidence score refit the same way (expanding window) on the same games.
Usage: python research/ensemble.py cfb <work> <repo>    (work = the research work dir: joined.json, ppa_preds.json, picks.json)
       python research/ensemble.py nfl <work> <repo>    (work holds nflgames/games.csv from nflverse/nfldata)
Writes research/ensemble_<league>.json."""
import sys, json, runpy, io, contextlib, csv, pathlib, numpy as np
from collections import defaultdict
LEAGUE, W, ROOT = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "research"))
import os
DROP = set(filter(None, os.environ.get("ENSEMBLE_DROP", "").split(",")))   # experts to leave out (e.g. wind,cold)
ETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

# ---------------------------------------------------------------- data: one row per game and market
def team_history(rows):
    """Per season and team: [(week, cover margin, game margin)] from earlier rows (home perspective flipped for away)."""
    h = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["season"], r["week"])):
        h[(r["season"], r["home"])].append((r["week"], r["act"] - r["mkt"], r["act"]))
        h[(r["season"], r["away"])].append((r["week"], -(r["act"] - r["mkt"]), -r["act"]))
    return h

def form(h, season, team, week):
    e = [x for x in h[(season, team)] if x[0] < week]
    cover = (sum(x[1] > 0 for x in e) + 1) / (len(e) + 2)
    return cover, (e[-1][2] if e else 0.0), (e[-1][1] if e else 0.0)

def cfb_rows():
    sys.argv = ["situational", str(ROOT)]
    with contextlib.redirect_stdout(io.StringIO()):
        S = runpy.run_path(str(ROOT / "research" / "situational.py"))
    J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(W / "joined.json"))}
    PPA = {(p["season"], p["week"], p["home"], p["away"]): p["ppa_net"] for p in json.load(open(W / "ppa_preds.json"))}
    spreads = []
    for r in S["R"]:
        k = (r["season"], r["week"], r["home"], r["away"]); j = J.get(k)
        if r["week"] < 2 or not j or k not in PPA or r["act"] == r["mkt"]: continue
        spreads.append(dict(season=r["season"], week=r["week"], home=r["home"], away=r["away"], act=r["act"], mkt=r["mkt"],
                            ours=j["ours"], effraw=PPA[k], y=float(r["act"] > r["mkt"]),
                            x=dict(model=j["ours"] - r["mkt"], last_ats=r["last_ats_h"] - r["last_ats_a"],
                                   move=(r["mkt"] - r["open"]) if r.get("open") is not None else 0.0,
                                   rest=r["rest_h"] - r["rest_a"], travel=r["travel_a"] - r["travel_h"],
                                   form=r["form_h"] - r["form_a"], fav=r["mkt"], home=0.0 if r["neutral"] else 1.0,
                                   confdog=-np.sign(r["mkt"]) if r["conf"] else 0.0,
                                   last_margin=r["last_margin_h"] - r["last_margin_a"])))
    h = team_history(spreads)
    for r in spreads:
        ch, *_ = form(h, r["season"], r["home"], r["week"]); ca, *_ = form(h, r["season"], r["away"], r["week"])
        r["x"]["cover_form"] = ch - ca
    # totals (picks.json + over/under history as in conf_total.py)
    sys.argv = ["conf_total", str(W), str(ROOT)]
    with contextlib.redirect_stdout(io.StringIO()):
        T = runpy.run_path(str(ROOT / "research" / "conf_total.py"))
    HIST, pace_avg = T["HIST"], T["pace_avg"]
    global PACE_AVG; PACE_AVG = pace_avg
    totals = []
    for r in json.load(open(W / "picks.json")):
        if r["tot_now"] is None or r["total"] is None or r["act_total"] == r["total"] or r["week"] < 2: continue
        def prev(t):
            e = [v for wk, v in HIST[r["season"]][t] if wk < r["week"]]
            return (e[-1] if e else 0.0), (sum(v > 0 for v in e) + 1) / (len(e) + 2)
        (lh, fh), (la, fa) = prev(r["home"]), prev(r["away"])
        pace = ((r["pace_h"] or pace_avg) + (r["pace_a"] or pace_avg)) / 2
        totals.append(dict(season=r["season"], week=r["week"], home=r["home"], away=r["away"], y=float(r["act_total"] > r["total"]),
                           gapraw=r["tot_now"] - r["total"], mid=55.0,
                           x=dict(gap=r["tot_now"] - r["total"], last=lh + la, form=fh + fa - 1, pace=pace - pace_avg,
                                  high=r["total"] - 55, prior=r["tot_prior"] - r["total"])))
    return spreads, totals

def nfl_rows():
    cache = W / "nfl_conf_rows.json"
    if cache.exists(): R = json.load(open(cache))
    else:
        sys.argv = ["nfl_conf"]
        with contextlib.redirect_stdout(io.StringIO()):
            NC = runpy.run_path(str(ROOT / "research" / "nfl_conf.py"))
        R = [r for y in range(2016, 2027) for r in NC["rows_for"](y)]
        json.dump(R, open(cache, "w"))
    from fetch_nfl import TEAMS
    global PACE_AVG; PACE_AVG = 60.8
    meta, qb = {}, {}
    for g in sorted(csv.DictReader(open(W / "nflgames" / "games.csv")), key=lambda g: (g["season"], g["gameday"])):
        if not g["week"]: continue
        y, wk, h, a = int(g["season"]), int(g["week"]), TEAMS.get(g["home_team"]), TEAMS.get(g["away_team"])
        chg = {}
        for t, q in ((h, g["home_qb_name"]), (a, g["away_qb_name"])):
            chg[t] = float(bool((y, t) in qb and q and qb[(y, t)] != q)); qb[(y, t)] = q or qb.get((y, t))
        num = lambda v: float(v) if v not in ("", "NA", None) else None
        meta[(y, wk, h, a)] = dict(rest=(num(g["home_rest"]) or 7) - (num(g["away_rest"]) or 7), div=g["div_game"] == "1",
                                   neutral=g["location"] == "Neutral", qbchg=chg[a] - chg[h],
                                   dome=g["roof"] in ("dome", "closed"), wind=num(g["wind"]), temp=num(g["temp"]))
    spreads, totals = [], []
    for r in R:
        m = meta.get((r["season"], r["week"], r["home"], r["away"]))
        if not m: continue
        if r["act"] != r["line"]:
            spreads.append(dict(season=r["season"], week=r["week"], home=r["home"], away=r["away"], act=r["act"], mkt=r["line"],
                                ours=r["ours"], effraw=r["epa"], y=float(r["act"] > r["line"]), ats_h=r["ats_h"], ats_a=r["ats_a"],
                                x=dict(model=r["ours"] - r["line"], last_ats=r["ats_h"] - r["ats_a"], rest=m["rest"],
                                       fav=r["line"], home=0.0 if m["neutral"] else 1.0,
                                       divdog=-np.sign(r["line"]) if m["div"] else 0.0, qbchg=m["qbchg"])))
        if r["total"] is not None and r["act_total"] != r["total"] and r["ours_total"] != r["total"]:
            pace = sum(r["pace"]) / len(r["pace"]) if r["pace"] else 60.8
            outdoor = not m["dome"]
            totals.append(dict(season=r["season"], week=r["week"], home=r["home"], away=r["away"], y=float(r["act_total"] > r["total"]),
                               gapraw=r["ours_total"] - r["total"], mid=45.0,
                               x=dict(gap=r["ours_total"] - r["total"], last=r["ou_h"] + r["ou_a"], form=r["fr_h"] + r["fr_a"] - 1,
                                      pace=pace - 60.8, high=r["total"] - 45, dome=float(m["dome"]),
                                      wind=(m["wind"] or 0.0) if outdoor else 0.0,
                                      cold=max(0.0, 50 - m["temp"]) if outdoor and m["temp"] is not None else 0.0)))
    h = team_history(spreads)
    for r in spreads:
        ch, mh, _ = form(h, r["season"], r["home"], r["week"]); ca, ma, _ = form(h, r["season"], r["away"], r["week"])
        r["x"]["cover_form"] = ch - ca; r["x"]["last_margin"] = mh - ma
    return spreads, totals

# ---------------------------------------------------------------- models
def logit_fit(X, y, l2):
    b = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ b)); H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (p - y) + l2 * b); b -= step
        if np.abs(step).max() < 1e-7: break
    return b

def composite(market, past, rows):
    """The site's current confidence score, refit on past rows: P(home covers / over) for each row."""
    def feats(r, eff_ab=None):
        if market == "spread":
            s = 1 if r["ours"] > r["mkt"] else -1
            A, B = eff_ab; em = A * r["effraw"] + B
            bad = -s * r["x"]["last_ats"] / 14; dis = -s * (em - r["mkt"]) / 7; gap = abs(r["ours"] - r["mkt"]) / 7
            return s, [1, bad, dis, gap, float(bad > 0 and dis > 0)]
        s = 1 if r["gapraw"] > 0 else -1
        return s, [1, abs(r["gapraw"]) / 7, -s * r["x"]["last"] / 14, -s * r["x"]["form"], s * r["x"]["pace"] / 10, -s * (r["x"]["high"]) / 10]
    ab = None
    if market == "spread":
        Xe = np.array([[r["effraw"], 1] for r in past]); ab = np.linalg.lstsq(Xe, np.array([r["act"] for r in past]), rcond=None)[0]
    F = [feats(r, ab) for r in past]; X = np.array([f[1] for f in F])
    won = np.array([float((r["y"] > .5) == (f[0] > 0)) for r, f in zip(past, F)])
    # same ridge as the site's fit (no penalty on the constant)
    b = np.zeros(X.shape[1]); P = 10.0 * np.diag(np.r_[0, np.ones(X.shape[1] - 1)])
    for _ in range(40):
        p = 1 / (1 + np.exp(-X @ b)); H = (X * (p * (1 - p))[:, None]).T @ X + P
        b -= np.linalg.solve(H, X.T @ (p - won) + P @ b)
    out = []
    for r in rows:
        s, f = feats(r, ab); pc = 1 / (1 + np.exp(-np.dot(b, f)))
        out.append(pc if s > 0 else 1 - pc)
    return np.array(out), ab

def experts_predict(market, past, rows, names):
    """Each one-signal expert's P(home covers / over) for rows, fit on past rows only. Returns {name: array}."""
    P = {}
    y = np.array([r["y"] for r in past])
    for n in names:
        xp = np.array([r["x"][n] for r in past], float); sd = xp.std() or 1.0
        b = logit_fit((xp / sd)[:, None], y, 5.0)[0]
        P[n] = 1 / (1 + np.exp(-b * np.array([r["x"][n] for r in rows], float) / sd))
    if market == "spread":   # efficiency model vs the line, points scale refit on past games
        pc, (A, B) = composite(market, past, rows); P["composite"] = pc
        xe = lambda rr: np.array([(A * r["effraw"] + B - r["mkt"]) for r in rr])
        xp = xe(past); sd = xp.std() or 1.0; b = logit_fit((xp / sd)[:, None], y, 5.0)[0]
        P["eff"] = 1 / (1 + np.exp(-b * xe(rows) / sd))
    else:
        P["composite"], _ = composite(market, past, rows)
    return P

# ---------------------------------------------------------------- replay
def replay(rows, market, first_scored):
    """Week by week: every expert's chances for that week's games from earlier games only. Returns list of dicts."""
    names = sorted(n for n in rows[0]["x"] if n not in DROP)
    weeks = sorted({(r["season"], r["week"]) for r in rows})
    out = []
    for sw in weeks:
        if sw[0] < first_scored: continue
        past = [r for r in rows if (r["season"], r["week"]) < sw]; cur = [r for r in rows if (r["season"], r["week"]) == sw]
        if len(past) < 300 or not cur: continue
        P = experts_predict(market, past, cur, names)
        for i, r in enumerate(cur):
            out.append(dict(season=r["season"], week=r["week"], y=r["y"], p={n: float(v[i]) for n, v in P.items()}))
    return out

def weights_run(scored, eta, final_loss=None):
    """Ensemble chance per game: weights from each expert's cumulative log loss on all earlier scored games.
    final_loss (a dict) receives every expert's cumulative log loss after the last week."""
    names = sorted(scored[0]["p"]); L = defaultdict(float); res = []
    for sw in sorted({(s["season"], s["week"]) for s in scored}):
        cur = [s for s in scored if (s["season"], s["week"]) == sw]
        m = min(L[n] for n in names) if L else 0.0
        w = np.array([np.exp(-eta * (L[n] - m)) for n in names]); w /= w.sum()
        for s in cur:
            p = float(np.dot(w, [s["p"][n] for n in names])); res.append(dict(s, ens=p, w=dict(zip(names, w.round(4).tolist()))))
        for s in cur:
            for n in names:
                q = min(max(s["p"][n], 1e-6), 1 - 1e-6); L[n] += -(s["y"] * np.log(q) + (1 - s["y"]) * np.log(1 - q))
    if final_loss is not None: final_loss.update(L)
    return res

def grade(res, key, seasons):
    """Pick = side of `key` chance; weekly top 10% / 5% by confidence; overall; log loss."""
    rs = [r for r in res if r["season"] in seasons]
    if not rs: return None
    tot = {"all": [0, 0], "top10": [0, 0], "top5": [0, 0]}; ll = 0.0
    for sw in sorted({(r["season"], r["week"]) for r in rs}):
        cur = [r for r in rs if (r["season"], r["week"]) == sw]
        conf = [max(c, 1 - c) for c in (r[key] for r in cur)]
        won = [(r[key] > .5) == (r["y"] > .5) for r in cur]
        order = np.argsort(conf)[::-1]
        for name, frac in (("all", 1.0), ("top10", .10), ("top5", .05)):
            k = len(cur) if frac == 1.0 else max(1, round(len(cur) * frac))
            for i in order[:k]: tot[name][0] += won[i]; tot[name][1] += 1
    for r in rs:
        q = min(max(r[key], 1e-6), 1 - 1e-6); ll += -(r["y"] * np.log(q) + (1 - r["y"]) * np.log(1 - q))
    return dict({k: v + [round(v[0] / v[1], 4) if v[1] else None] for k, v in tot.items()}, logloss=round(ll / len(rs), 5), n=len(rs))

if __name__ == "__main__":
    spreads, totals = cfb_rows() if LEAGUE == "cfb" else nfl_rows()
    seasons = sorted({r["season"] for r in spreads})
    if LEAGUE == "cfb": first, tune, test = 2023, (2023,), tuple(range(2024, 2027))
    else: first, tune, test = 2017, (2017, 2018), tuple(range(2019, 2027))
    fmt = lambda g: (f"all {g['all'][0]}-{g['all'][1]-g['all'][0]} ({g['all'][2]:.1%}) | top 10% {g['top10'][0]}-{g['top10'][1]-g['top10'][0]} ({g['top10'][2]:.1%}) "
                     f"| top 5% {g['top5'][0]}-{g['top5'][1]-g['top5'][0]} ({g['top5'][2]:.1%}) | log loss {g['logloss']:.4f}")
    out = {}
    for market, rows in (("spread", spreads), ("total", totals)):
        print(f"\n===== {LEAGUE.upper()} {market}s: {len(rows)} games, seasons {seasons[0]}-{seasons[-1]}; tune {tune}, test {test[0]}-{test[-1]}")
        scored = replay(rows, market, first)
        names = sorted(scored[0]["p"])
        print("experts:", ", ".join(names))
        runs = {eta: weights_run(scored, eta) for eta in ETAS}
        print("tuning season(s):")
        for eta in ETAS: print(f"  eta {eta:<5} {fmt(grade(runs[eta], 'ens', tune))}")
        best = min(ETAS, key=lambda e: grade(runs[e], "ens", tune)["logloss"])
        print(f"chosen eta (lowest tuning log loss): {best}")
        print("test seasons (pooled):")
        base = runs[best]
        for n in names: base = [dict(r, **{"x_" + n: r["p"][n]}) for r in base]
        rows_t = {"ensemble (chosen eta)": "ens", **{f"expert: {n}": "x_" + n for n in names}}
        res_t = {}
        for label, key in rows_t.items():
            g = grade(base, key, test); res_t[label] = g; print(f"  {label:<26} {fmt(g)}")
        for eta in ETAS:
            g = grade(runs[eta], "ens", test); res_t[f"ensemble eta {eta}"] = g
        print("  ensemble at every eta (test):", " | ".join(f"{e}: top10 {res_t[f'ensemble eta {e}']['top10'][2]:.1%}" for e in ETAS))
        print("by test season, top 10% (ensemble vs current score):")
        by = {}
        for y in test:
            ge, gc = grade(base, "ens", (y,)), grade(base, "x_composite", (y,))
            if ge: by[y] = dict(ensemble=ge["top10"], composite=gc["top10"]); print(f"  {y}: ensemble {ge['top10'][0]}-{ge['top10'][1]-ge['top10'][0]} ({ge['top10'][2]:.0%}) | current {gc['top10'][0]}-{gc['top10'][1]-gc['top10'][0]} ({gc['top10'][2]:.0%})")
        last = base[-1]["w"]
        print("final weights:", ", ".join(f"{n} {w:.2f}" for n, w in sorted(last.items(), key=lambda kv: -kv[1])))
        out[market] = dict(eta=best, test=res_t, by_season=by, final_weights=last, experts=names)
        if market == "total":   # live version for scripts/track_picks.py: final fits on every game, running losses
            L = {}; weights_run(scored, best, L); y = np.array([r["y"] for r in rows]); live = {}
            for n in names:
                if n == "composite": continue
                xp = np.array([r["x"][n] for r in rows], float); sd = float(xp.std() or 1.0)
                live[n] = dict(b=float(logit_fit((xp / sd)[:, None], y, 5.0)[0]), sd=sd)
            out[market]["live"] = dict(eta=best, experts=live, loss={n: round(v, 4) for n, v in L.items()},
                                       mid=rows[0]["mid"], pace_avg=float(PACE_AVG), through=max((r["season"], r["week"]) for r in rows))
    json.dump(out, open(ROOT / "research" / f"ensemble_{LEAGUE}.json", "w"), indent=1)
