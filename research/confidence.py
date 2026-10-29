"""What makes a high-confidence pick? Candidate confidence measures, each chosen on 2022-24 and scored once on 2025
and 2026: the top 10% of games by each measure, our side vs the opening line (and vs the closing line for the
market-confirmation measure, which needs the later line). Usage: python research/confidence.py <work dir> <repo>"""
import sys, json, numpy as np
W, ROOT = sys.argv[1], sys.argv[2]
G = json.load(open(f"{W}/joined.json"))
L26 = {(r[0], r[1], r[2]): (-r[10] if len(r) > 10 and r[10] is not None else None) for r in json.load(open(f"{ROOT}/data/lines.json"))["games"]}
for g in G:
    if g["season"] == 2026: g["open"] = L26.get((g["week"], g["home"], g["away"]))
PPA = {(p["season"], p["week"], p["home"], p["away"]): p["ppa_net"] for p in json.load(open(f"{W}/ppa_preds.json"))}
G = [dict(g, ppa=PPA.get((g["season"], g["week"], g["home"], g["away"]))) for g in G]
G = [g for g in G if g["open"] is not None and g["ppa"] is not None and g["week"] >= 2]
TRAIN = (2022, 2023, 2024)
tr = [g for g in G if g["season"] in TRAIN]
# PPA net per play -> points, fit on training seasons
a, b = np.polyfit([g["ppa"] for g in tr], [g["act"] for g in tr], 1)
for g in G: g["ppa_m"] = a * g["ppa"] + b
# our typical miss by week bucket and spread size, from training seasons (for the uncertainty-scaled measure)
wb = lambda w: 0 if w <= 4 else 1 if w <= 9 else 2
sb = lambda s: 0 if abs(s) < 7 else 1 if abs(s) < 14 else 2 if abs(s) < 21 else 3
sd = {}
for i in range(3):
    for j in range(4):
        r = [g["act"] - g["ours"] for g in tr if wb(g["week"]) == i and sb(g["open"]) == j]
        sd[(i, j)] = float(np.std(r)) if len(r) > 30 else 16.0
side = lambda x: 1 if x > 0 else -1
M = {
    "1. gap from opening line (current)": lambda g: abs(g["ours"] - g["open"]),
    "2. gap / typical miss for that game": lambda g: abs(g["ours"] - g["open"]) / sd[(wb(g["week"]), sb(g["open"]))],
    "3. gap, efficiency model agrees": lambda g: abs(g["ours"] - g["open"]) if side(g["ours"] - g["open"]) == side(g["ppa_m"] - g["open"]) else 0,
    "4. like 3, skip weeks <=2 and spreads 21+": lambda g: (abs(g["ours"] - g["open"]) if side(g["ours"] - g["open"]) == side(g["ppa_m"] - g["open"]) else 0) if g["week"] > 2 and abs(g["open"]) < 21 else 0,
    "5. gap, line moved our way (graded vs close)": lambda g: abs(g["ours"] - g["open"]) if g["mkt"] != g["open"] and side(g["mkt"] - g["open"]) == side(g["ours"] - g["open"]) else 0,
}
def top(gs, f, line):
    s = sorted([g for g in gs if f(g) > 0 and g["act"] != g[line]], key=lambda g: -f(g))
    n = max(1, len([g for g in gs if g["act"] != g[line]]) // 10); t = s[:n]
    w = sum((g["act"] > g[line]) == (g["ours"] > g["open"]) for g in t)
    return w, len(t) - w
pct = lambda wl: f"{wl[0]}-{wl[1]} {wl[0]/max(1,sum(wl)):.0%}"
print(f"{'top 10% by':<46} {'2022':>10} {'2023':>10} {'2024':>10} {'train':>11} | {'2025':>10} {'2026':>10}")
for name, f in M.items():
    line = "mkt" if name.startswith("5") else "open"
    cells = [top([g for g in G if g["season"] == y], f, line) for y in (2022, 2023, 2024, 2025, 2026)]
    trn = (sum(c[0] for c in cells[:3]), sum(c[1] for c in cells[:3]))
    print(f"{name:<46} " + " ".join(f"{pct(c):>10}" for c in cells[:3]) + f" {pct(trn):>11} | " + " ".join(f"{pct(c):>10}" for c in cells[3:]))
print("\ntypical miss (SD) by week bucket x spread size, from 2022-24:", {k: round(v, 1) for k, v in sd.items()})
