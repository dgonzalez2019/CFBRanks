"""Walk-forward comparison of every confidence approach we have, on every season (Nov 2026 redo after a poor live month).

Each week of each season, every approach is refit on earlier games only and ranks that week's games; we grade each
week's top 10% / top 5% and every game, then pool by season. Approaches (spreads and totals):
  current    the site's multi-signal confidence score (composite expert in research/ensemble.py)
  ensemble   plain average of the one-signal experts (research/ensemble.py)
  stack      one L2 logistic on every expert signal at once (penalty chosen on the tuning season(s))
  leader     each week, the single expert with the lowest log loss on all earlier games
and simple market rules with no ranking (favorite at 3/7, conference-game underdog, big underdog 14+, line movement).
Seasons: CFB 2022 trains only, 2023 tunes, 2024-2026 test; NFL 2016 trains, 2017-18 tune, 2019-2026 test.
Usage: python research/redo.py cfb|nfl <work> <repo>   (work = research work dir; nfl also needs <work>/nflgames/games.csv)
Writes research/redo_<league>.json."""
import sys, json, math, pathlib, numpy as np
from collections import defaultdict
LEAGUE, W, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, str(pathlib.Path(ROOT) / "research"))
sys.argv = ["ensemble", LEAGUE, W, ROOT]
import ensemble as E

def stack_probs(past, rows, names, lam):
    X = lambda rr: np.array([[r["x"][n] for n in names] for r in rr], float)
    Xp = X(past); mu = Xp.mean(0); sd = Xp.std(0); sd[sd == 0] = 1
    Z = (Xp - mu) / sd; y = np.array([r["y"] for r in past])
    b = E.logit_fit(np.c_[np.ones(len(Z)), Z], y, lam)
    Zr = (X(rows) - mu) / sd
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(Zr)), Zr] @ b)))

def walk(rows, market, first, lams):
    """Per game: chances from every approach. Expert fits from E.experts_predict; stack at several penalties."""
    names = sorted(n for n in rows[0]["x"] if n not in E.DROP)
    out = []; L = defaultdict(float)
    for sw in sorted({(r["season"], r["week"]) for r in rows}):
        if sw[0] < first: continue
        past = [r for r in rows if (r["season"], r["week"]) < sw]; cur = [r for r in rows if (r["season"], r["week"]) == sw]
        if len(past) < 300 or not cur: continue
        P = E.experts_predict(market, past, cur, names)
        ex = [n for n in P]
        leader = min(ex, key=lambda n: L[n]) if L else "composite"
        S = {lam: stack_probs(past, cur, names + (["eff"] if market == "spread" and "eff" not in names else []), lam) if False else stack_probs(past, cur, names, lam) for lam in lams}
        for i, r in enumerate(cur):
            d = dict(season=r["season"], week=r["week"], y=r["y"], row=r,
                     current=float(P["composite"][i]), ensemble=float(np.mean([P[n][i] for n in ex])), leader=float(P[leader][i]))
            for lam in lams: d[f"stack{lam}"] = float(S[lam][i])
            out.append(d)
        for i, r in enumerate(cur):
            for n in ex:
                q = min(max(float(P[n][i]), 1e-6), 1 - 1e-6); L[n] += -(r["y"] * math.log(q) + (1 - r["y"]) * math.log(1 - q))
    return out

def grade(res, key, seasons):
    rs = [r for r in res if r["season"] in seasons]
    t = {"all": [0, 0], "top10": [0, 0], "top5": [0, 0]}
    for sw in sorted({(r["season"], r["week"]) for r in rs}):
        cur = [r for r in rs if (r["season"], r["week"]) == sw]
        conf = np.array([max(r[key], 1 - r[key]) for r in cur]); won = [(r[key] > .5) == (r["y"] > .5) for r in cur]
        order = np.argsort(-conf)
        for k, frac in (("all", 1), ("top10", .1), ("top5", .05)):
            n = len(cur) if frac == 1 else max(1, round(len(cur) * frac))
            for i in order[:n]: t[k][0] += won[i]; t[k][1] += 1
    return t

def z(w, n, p0=0.524):
    return (w / n - p0) / math.sqrt(p0 * (1 - p0) / n) if n else 0.0

def rule_rows(rows):
    """Market rules on spreads (no ranking): returns {rule: [(season, won)]}."""
    out = defaultdict(list)
    for r in rows:
        m = r["mkt"]; y = r["y"] > .5   # home covered
        def add(name, side_home):
            if side_home is None: return
            out[name].append((r["season"], y == side_home))
        add("favorite at 3 or 7", (m > 0) if any(abs(abs(m) - k) <= .5 for k in (3, 7)) else None)
        add("underdog at 3 or 7", (m < 0) if any(abs(abs(m) - k) <= .5 for k in (3, 7)) else None)
        add("underdog of 14+", (m < 0) if abs(m) >= 14 else None)
        add("home underdog", True if m < 0 and r["x"].get("home", 1) == 1 else None)
        mv = r["x"].get("move", 0.0)
        add("follow line move (1+ pt)", (mv > 0) if abs(mv) >= 1 else None)
        if "confdog" in r["x"] and r["x"]["confdog"] != 0: add("conference-game underdog", r["x"]["confdog"] > 0)
        if "divdog" in r["x"] and r["x"]["divdog"] != 0: add("division-game underdog", r["x"]["divdog"] > 0)
    return out

if __name__ == "__main__":
    spreads, totals = E.cfb_rows() if LEAGUE == "cfb" else E.nfl_rows()
    if LEAGUE == "cfb": first, tune, test = 2023, (2023,), (2024, 2025, 2026)
    else: first, tune, test = 2017, (2017, 2018), tuple(range(2019, 2027))
    LAMS = (10.0, 100.0, 1000.0)
    report = {}
    for market, rows in (("spread", spreads), ("total", totals)):
        res = walk(rows, market, first, LAMS)
        lam = max(LAMS, key=lambda l: (lambda t: t["top10"][0] / max(1, t["top10"][1]))(grade(res, f"stack{l}", tune)))
        keys = ["current", "ensemble", "leader", f"stack{lam}"]
        print(f"\n===== {LEAGUE.upper()} {market}s ({len(rows)} games; stack penalty {lam} chosen on {tune}) =====")
        print(f"{'approach':<14}" + "".join(f"{y:>12}" for y in test) + f"{'test pooled':>22}{'z vs 52.4%':>11}")
        rep = {}
        for k in keys:
            line = f"{k:<14}"; per = {}
            for y in test:
                t = grade(res, k, (y,))["top10"]; per[y] = t; line += f"{t[0]:>5}-{t[1]-t[0]:<3}{(t[0]/t[1] if t[1] else 0):>4.0%}"
            T = grade(res, k, test)
            line += f"   top10 {T['top10'][0]}-{T['top10'][1]-T['top10'][0]} {T['top10'][0]/T['top10'][1]:.1%}{z(*T['top10']):>8.2f}"
            print(line)
            print(f"{'':<14}top5 {T['top5'][0]}-{T['top5'][1]-T['top5'][0]} ({T['top5'][0]/T['top5'][1]:.1%}) | all games {T['all'][0]}-{T['all'][1]-T['all'][0]} ({T['all'][0]/T['all'][1]:.1%})")
            rep[k] = dict(by_season=per, pooled=T, z_top10=z(*T["top10"]))
        # final stack fit on every game, for live use (scripts/track_picks.py uses the NFL totals one)
        names = sorted(n for n in rows[0]["x"] if n not in E.DROP)
        X = np.array([[r["x"][n] for n in names] for r in rows], float); mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
        b = E.logit_fit(np.c_[np.ones(len(X)), (X - mu) / sd], np.array([r["y"] for r in rows]), lam)
        live = dict(names=names, mu=mu.round(5).tolist(), sd=sd.round(5).tolist(), b=b.round(6).tolist(), lam=lam,
                    pace_avg=float(getattr(E, "PACE_AVG", 0.0)), mid=rows[0].get("mid"))
        print("final stack (all games):", ", ".join(f"{n} {c:+.3f}" for n, c in zip(["const"] + names, b)))
        report[market] = dict(stack_lambda=lam, approaches=rep, live_stack=live)
    rr = rule_rows(spreads); seasons = sorted({r["season"] for r in spreads})
    print(f"\n===== {LEAGUE.upper()} spread rules, every game (no ranking) =====")
    rules = {}
    for name, v in sorted(rr.items()):
        per = {y: (sum(w for s, w in v if s == y), sum(1 for s, w in v if s == y)) for y in seasons}
        W_, N = sum(w for _, w in v), len(v)
        print(f"  {name:<28}" + " ".join(f"{y}:{a}-{n-a}" for y, (a, n) in per.items()) + f" | all {W_}-{N-W_} ({W_/N:.1%}) z {z(W_, N):.2f}")
        rules[name] = dict(by_season={y: list(per[y]) for y in seasons}, all=[W_, N], z=z(W_, N))
    report["rules"] = rules
    json.dump(report, open(pathlib.Path(ROOT) / "research" / f"redo_{LEAGUE}.json", "w"), indent=1, default=str)
