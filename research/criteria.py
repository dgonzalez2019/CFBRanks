"""Which pre-game criteria separated our correct picks from wrong ones in the NIL era? Every criterion is a simple
yes/no condition on the game; its hit rate (our side of the closing line) is found on one season and checked on the
other, both ways round, and also vs the opening line. A depth-2 decision tree is fit the same way.
Usage: python research/criteria.py <work dir> <repo>"""
import sys, json, runpy, io, contextlib, numpy as np
W, ROOT = sys.argv[1], sys.argv[2]
sys.argv = ["situational", ROOT]
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(f"{ROOT}/research/situational.py")
J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(f"{W}/joined.json"))}
PPA = {(p["season"], p["week"], p["home"], p["away"]): p["ppa_net"] for p in json.load(open(f"{W}/ppa_preds.json"))}
L26 = {(r[0], r[1], r[2]): (-r[10] if len(r) > 10 and r[10] is not None else None) for r in json.load(open(f"{ROOT}/data/lines.json"))["games"]}
R = []
for r in S["R"]:
    if r["season"] not in (2025, 2026) or r["week"] < 2: continue
    k = (r["season"], r["week"], r["home"], r["away"]); j = J.get(k)
    if not j or k not in PPA or r["act"] == r["mkt"]: continue
    opn = r["open"] if r["season"] == 2025 else L26.get((r["week"], r["home"], r["away"]))
    R.append(dict(r, ours=j["ours"], ppa=PPA[k], open=opn))
for y in (2025, 2026):
    tr = [r for r in R if r["season"] == y]; a, b = np.polyfit([r["ppa"] for r in tr], [r["act"] for r in tr], 1)
    for r in tr: r["ppa_m"] = a * r["ppa"] + b  # season-specific scale; only its side vs the line is used
for r in R:
    r["pick_home"] = r["ours"] > r["mkt"]; r["gap"] = abs(r["ours"] - r["mkt"])
    r["fav_home"] = r["mkt"] > 0
    r["correct"] = (r["act"] > r["mkt"]) == r["pick_home"]
side = lambda r: 1 if r["pick_home"] else -1
C = {
    "gap 3+": lambda r: r["gap"] >= 3, "gap 5+": lambda r: r["gap"] >= 5, "gap 7+": lambda r: r["gap"] >= 7, "gap under 3": lambda r: r["gap"] < 3,
    "pick the favorite": lambda r: r["pick_home"] == r["fav_home"], "pick the underdog": lambda r: r["pick_home"] != r["fav_home"],
    "pick the home team": lambda r: r["pick_home"] and not r["neutral"], "pick the road team": lambda r: not r["pick_home"] and not r["neutral"],
    "spread under 7": lambda r: abs(r["mkt"]) < 7, "spread 7-14": lambda r: 7 <= abs(r["mkt"]) < 14, "spread 14+": lambda r: abs(r["mkt"]) >= 14,
    "weeks 2-5": lambda r: r["week"] <= 5, "weeks 6-9": lambda r: 6 <= r["week"] <= 9, "weeks 10+": lambda r: r["week"] >= 10,
    "conference game": lambda r: r["conf"], "non-conference": lambda r: not r["conf"],
    "picked team has more rest": lambda r: side(r) * (r["rest_h"] - r["rest_a"]) > 0, "picked team has less rest": lambda r: side(r) * (r["rest_h"] - r["rest_a"]) < 0,
    "picked team travels less": lambda r: side(r) * (r["travel_a"] - r["travel_h"]) > 0.3,
    "total under 48": lambda r: r["total"] is not None and r["total"] < 48, "total 58+": lambda r: r["total"] is not None and r["total"] >= 58,
    "efficiency model agrees": lambda r: (r["ppa_m"] > r["mkt"]) == r["pick_home"], "efficiency model disagrees": lambda r: (r["ppa_m"] > r["mkt"]) != r["pick_home"],
    "picked team failed to cover last week": lambda r: side(r) * (r["last_ats_h"] - r["last_ats_a"]) < 0,
    "picked team covered last week": lambda r: side(r) * (r["last_ats_h"] - r["last_ats_a"]) > 0,
    "picked team won its last game": lambda r: (r["last_margin_h"] if r["pick_home"] else r["last_margin_a"]) > 0,
    "picked team lost its last game": lambda r: (r["last_margin_h"] if r["pick_home"] else r["last_margin_a"]) < 0,
    "line moved toward our pick": lambda r: r["open"] is not None and r["mkt"] != r["open"] and ((r["mkt"] > r["open"]) == r["pick_home"]),
    "line moved against our pick": lambda r: r["open"] is not None and r["mkt"] != r["open"] and ((r["mkt"] > r["open"]) != r["pick_home"]),
}
def rate(rs): n = len(rs); w = sum(r["correct"] for r in rs); return w, n - w
pct = lambda wl: f"{wl[0]}-{wl[1]} {wl[0]/max(1,sum(wl)):.0%}"
y25, y26 = [r for r in R if r["season"] == 2025], [r for r in R if r["season"] == 2026]
print(f"games: 2025 {len(y25)} (our side {pct(rate(y25))}), 2026 {len(y26)} (our side {pct(rate(y26))})\n")
rows = []
for name, f in C.items():
    a, b = rate([r for r in y25 if f(r)]), rate([r for r in y26 if f(r)])
    rows.append((name, a, b))
print(f"{'criterion (pick = our side of closing line)':<42} {'2025':>12} {'2026':>12}  both")
for name, a, b in sorted(rows, key=lambda x: -(x[1][0] + x[2][0]) / max(1, sum(x[1]) + sum(x[2]))):
    both = (a[0] + b[0], a[1] + b[1])
    print(f"{name:<42} {pct(a):>12} {pct(b):>12}  {pct(both)}")
# pairs: best combination found on one season, checked on the other
names = list(C)
def best_pairs(train, test, minn=40):
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            f = lambda r, i=i, j=j: C[names[i]](r) and C[names[j]](r)
            t = rate([r for r in train if f(r)])
            if sum(t) >= minn: out.append((t[0] / sum(t), names[i], names[j], t, rate([r for r in test if f(r)])))
    return sorted(out, reverse=True)[:5]
for a_, b_, lab in ((y25, y26, "found on 2025 -> checked on 2026"), (y26, y25, "found on 2026 -> checked on 2025")):
    print(f"\nbest two-criterion combinations, {lab} (min 40 games):")
    for p, n1, n2, t, c in best_pairs(a_, b_):
        print(f"  {n1} + {n2}: {pct(t)} -> {pct(c)}")

# The two combinations that carried from 2026 to 2025, checked on 2022-24, which neither was found on.
PRE = []
for r in S["R"]:
    if r["season"] not in (2022, 2023, 2024) or r["week"] < 2: continue
    k = (r["season"], r["week"], r["home"], r["away"]); j = J.get(k)
    if not j or k not in PPA or r["act"] == r["mkt"]: continue
    PRE.append(dict(r, ours=j["ours"], ppa=PPA[k]))
for y in (2022, 2023, 2024):
    tr = [r for r in PRE if r["season"] == y]; a, b = np.polyfit([r["ppa"] for r in tr], [r["act"] for r in tr], 1)
    for r in tr: r["ppa_m"] = a * r["ppa"] + b
for r in PRE:
    r["pick_home"] = r["ours"] > r["mkt"]; r["gap"] = abs(r["ours"] - r["mkt"]); r["fav_home"] = r["mkt"] > 0
    r["correct"] = (r["act"] > r["mkt"]) == r["pick_home"]
print("\nchecked on 2022-24 (neither combination was found on these seasons):")
for n1, n2 in (("picked team has less rest", "picked team failed to cover last week"),
               ("efficiency model disagrees", "picked team failed to cover last week")):
    f = lambda r: C[n1](r) and C[n2](r)
    cells = [pct(rate([r for r in PRE if r["season"] == y and f(r)])) for y in (2022, 2023, 2024)]
    allp = rate([r for r in PRE if f(r)])
    print(f"  {n1} + {n2}: " + " | ".join(f"{y}: {c}" for y, c in zip((2022, 2023, 2024), cells)) + f" | 2022-24: {pct(allp)}")
