"""Over/under confidence score: probability our side of the closing total wins (over when our predicted total is above
the line, under when below), from a logistic model on pick-side features. Trained on 2022-24, scored once on 2025 and
2026; also leave-one-season-out over 2022-25. Reports single features on the training seasons, top 5% / 10% hit rates
and calibration. Writes <work>/conf_total_coef.json.
Usage: python research/conf_total.py <work> <repo>   (needs <work>/picks.json from picks_features.py)

Features (s = +1 when we take the over, -1 the under):
  gap   = |our total - line| / 7
  last  = -s * (home's last-game total minus its closing total + away's) / 14   (>0: we fade both teams' last game)
  form  = -s * (home's season over rate + away's - 1), each shrunk by 2 games toward 50%
  pace  = s * (average plays per game of the two teams - league average) / 10
  prior = s * (total from last season's points model - line) / 7   (>0: the slower prior model agrees with our side)
  high  = -s * (line - 55) / 10   (>0: under on a high total or over on a low one)"""
import sys, json, numpy as np
from collections import defaultdict
W, ROOT = sys.argv[1], sys.argv[2]
P = json.load(open(f"{W}/picks.json"))

def season_results(y):
    """Each game's actual total minus its closing total, keyed by (week, home, away)."""
    if y < 2026:
        L = {(r[0], r[1], r[2]): r[4] for r in json.load(open(f"{ROOT}/data/history/{y}_lines.json"))["games"]}
        G = json.load(open(f"{ROOT}/data/history/{y}.json"))["games"]
    else:
        L = {(r[0], r[1], r[2]): r[5] for r in json.load(open(f"{ROOT}/data/lines.json"))["games"]}
        G = json.load(open(f"{ROOT}/data/games.json"))["games"]
    out = {}
    for g in G:
        k = (g[0], g[1], g[2]); t = L.get(k)
        out[k] = None if t is None else (g[3] + g[4]) - t
    return out

OU = {y: season_results(y) for y in (2022, 2023, 2024, 2025, 2026)}
def history(y):
    """Per team, (week, over margin) for each game with a total, in week order."""
    h = defaultdict(list)
    for (wk, home, away), v in sorted(OU[y].items()):
        if v is None: continue
        h[home].append((wk, v)); h[away].append((wk, v))
    return h
HIST = {y: history(y) for y in OU}
pace_avg = np.mean([x for r in P for x in (r["pace_h"], r["pace_a"]) if x])

R = []
for r in P:
    if r["tot_now"] is None or r["total"] is None or r["act_total"] == r["total"] or r["tot_now"] == r["total"]: continue
    s = 1 if r["tot_now"] > r["total"] else -1
    def prev(t):
        e = [v for wk, v in HIST[r["season"]][t] if wk < r["week"]]
        return (e[-1] if e else 0.0), (sum(v > 0 for v in e) + 1) / (len(e) + 2)
    (lh, fh), (la, fa) = prev(r["home"]), prev(r["away"])
    pace = ((r["pace_h"] or pace_avg) + (r["pace_a"] or pace_avg)) / 2
    R.append(dict(season=r["season"], week=r["week"], won=(r["act_total"] > r["total"]) == (s > 0),
                  gap=abs(r["tot_now"] - r["total"]) / 7, last=-s * (lh + la) / 14, form=-s * (fh + fa - 1),
                  pace=s * (pace - pace_avg) / 10, prior=s * (r["tot_prior"] - r["total"]) / 7, high=-s * (r["total"] - 55) / 10))
NAMES = ["gap", "last", "form", "pace", "high"]  # "prior" (last-season points model) added nothing held out; dropped
print("games:", {y: sum(r["season"] == y for r in R) for y in (2022, 2023, 2024, 2025, 2026)})
print("our side of the total, all games:", {y: f"{np.mean([r['won'] for r in R if r['season'] == y]):.1%}" for y in (2022, 2023, 2024, 2025, 2026)})

print("\nsingle features, training seasons 2022-24 (games where the feature is in its top third vs bottom third):")
tr = [r for r in R if r["season"] <= 2024]
for f in NAMES:
    v = np.array([r[f] for r in tr]); w = np.array([r["won"] for r in tr]); lo, hi = np.quantile(v, [1/3, 2/3])
    print(f"  {f:<6} top third {w[v >= hi].mean():.1%} | bottom third {w[v <= lo].mean():.1%}")

FEATS = lambda r: [1] + [r[f] for f in NAMES]
def fit(rows, l2=10.0):
    X = np.array([FEATS(r) for r in rows]); y = np.array([r["won"] for r in rows], float); b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b)); H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b -= np.linalg.solve(H, X.T @ (p - y) + l2 * np.r_[0, b[1:]])
    return b
prob = lambda b, rows: 1 / (1 + np.exp(-np.array([FEATS(r) for r in rows]) @ b))
def report(label, b, test):
    p = prob(b, test); won = np.array([r["won"] for r in test]); order = np.argsort(-p); out = []
    for frac in (0.05, 0.10):
        top = order[:max(1, int(len(test) * frac))]; out.append(f"top {int(frac*100)}%: {won[top].sum()}-{(~won[top]).sum()} ({won[top].mean():.0%})")
    print(f"  {label}: " + " | ".join(out) + f" | other 90%: {won[order[int(len(test) * .10):]].mean():.1%}")
    return p, won, order

b = fit(tr)
print("\ntrained on 2022-24, coefficients [const, " + ", ".join(NAMES) + "]:", np.round(b, 3))
pall, wall, held = [], [], defaultdict(lambda: [0, 0])
for y in (2025, 2026):
    test = [r for r in R if r["season"] == y]; p, w, o = report(f"test {y}", b, test); pall += list(p); wall += list(w)
print("leave one season out (train on the other 2022-25 seasons; 2026 scored by the 2022-25 fit):")
for y in (2022, 2023, 2024, 2025, 2026):
    test = [r for r in R if r["season"] == y]
    p, w, o = report(f"test {y}", fit([r for r in R if r["season"] <= 2025 and r["season"] != y]), test)
    for frac, k in ((0.10, "top10"), (0.05, "top5")):
        top = o[:max(1, int(len(test) * frac))]; held[k][0] += int(w[top].sum()); held[k][1] += int((~w[top]).sum())
    held[y] = [int(w[o[:max(1, int(len(test) * .10))]].sum()), int((~w[o[:max(1, int(len(test) * .10))]]).sum())]
print("held out, all seasons: top 10%", held["top10"], "top 5%", held["top5"])
pall, wall = np.array(pall), np.array(wall)
print("calibration on 2025-26 (trained 2022-24):")
for lo, hi in ((0, .48), (.48, .50), (.50, .52), (.52, .54), (.54, 1)):
    s = (pall >= lo) & (pall < hi)
    if s.sum(): print(f"  score {lo:.0%}-{hi:.0%}: {s.sum()} games, won {wall[s].mean():.1%}")
bfin = fit([r for r in R if r["season"] <= 2025])
json.dump(dict(coef=list(map(float, bfin)), names=NAMES, pace_avg=float(pace_avg),
               held={str(k): v for k, v in held.items()}), open(f"{W}/conf_total_coef.json", "w"))
print("final model (2022-25) coefficients:", np.round(bfin, 4), "| pace avg", round(float(pace_avg), 2))
