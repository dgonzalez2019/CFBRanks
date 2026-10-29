import sys, json, numpy as np
from collections import defaultdict
S, ROOT = sys.argv[1], sys.argv[2]
preds = json.load(open(f"{S}/preds.json"))
lines = {}
for y in range(2022, 2026):
    for r in json.load(open(f"{ROOT}/data/history/{y}_lines.json"))["games"]:
        lines[(y, r[0], r[1], r[2])] = dict(mkt=-r[3], total=r[4], open=-r[5] if r[5] is not None else None)
for r in json.load(open(f"{ROOT}/data/lines.json"))["games"]:
    lines[(2026, r[0], r[1], r[2])] = dict(mkt=-r[4], total=r[5], open=None)
G = []
for p in preds:
    l = lines.get((p["season"], p["week"], p["home"], p["away"]))
    if l: G.append(dict(p, **l))
print("games with a line:", {y: sum(g["season"] == y for g in G) for y in range(2022, 2027)})
TRAIN = [2022, 2023, 2024, 2025]

def su(gs, k):  # straight-up winners right
    gs = [g for g in gs if g["act"] != 0 and g[k] != 0]; return np.mean([(g[k] > 0) == (g["act"] > 0) for g in gs])
def mae(gs, k): return np.mean([abs(g[k] - g["act"]) for g in gs])
def ats(gs, side):  # side(g) -> +1 home side vs line, -1 away side, 0 no pick. Returns (n, hit rate)
    picks = [(side(g), g) for g in gs]; picks = [(s, g) for s, g in picks if s and g["act"] != g["mkt"]]
    return len(picks), (np.mean([(g["act"] > g["mkt"]) == (s > 0) for s, g in picks]) if picks else float("nan"))

print("\n== 1. Straight-up and margin accuracy, by season ==")
for y in TRAIN + [2026]:
    gs = [g for g in G if g["season"] == y]
    print(f"{y}: n={len(gs)}  winners ours {su(gs,'ours'):.1%} market {su(gs,'mkt'):.1%} | avg miss ours {mae(gs,'ours'):.2f} market {mae(gs,'mkt'):.2f}")

print("\n== 2. Blend: margin = w*market + (1-w)*ours, w chosen on the other seasons ==")
W = np.linspace(0, 1, 21)
for g in G: g["blend_by_w"] = {round(w, 2): w * g["mkt"] + (1 - w) * g["ours"] for w in W}
tot = defaultdict(list)
for test in TRAIN + [2026]:
    train = [g for g in G if g["season"] in TRAIN and g["season"] != test]
    w = min(W, key=lambda w: mae([dict(g, b=g["blend_by_w"][round(w, 2)]) for g in train], "b"))
    gs = [dict(g, b=g["blend_by_w"][round(w, 2)]) for g in G if g["season"] == test]
    print(f"test {test}: w={w:.2f}  winners blend {su(gs,'b'):.1%} market {su(gs,'mkt'):.1%} ours {su(gs,'ours'):.1%} | miss blend {mae(gs,'b'):.2f} market {mae(gs,'mkt'):.2f}")
    if test in TRAIN: tot["b"] += gs

print("\n== 3. Against the spread: the side our model favors vs. the line (break-even 52.4%) ==")
lean = lambda t: (lambda g: (1 if g["ours"] - g["mkt"] > 0 else -1) if abs(g["ours"] - g["mkt"]) >= t else 0)
for t in (0, 3, 5, 7, 10):
    row = [ats([g for g in G if g["season"] == y], lean(t)) for y in TRAIN + [2026]]
    allp = ats([g for g in G if g["season"] in TRAIN], lean(t))
    print(f"edge >= {t:>2}: 2022-25 {allp[0]:4} games {allp[1]:.1%} | " + " ".join(f"{y}:{n}/{h:.0%}" for y, (n, h) in zip(TRAIN + [2026], row)))

print("\n== 4. Situations (2022-25 pooled, then each season) for edge >= 3 ==")
def sit(name, cond):
    f = lambda g: lean(3)(g) if cond(g) else 0
    allp = ats([g for g in G if g["season"] in TRAIN], f)
    row = [ats([g for g in G if g["season"] == y], f) for y in TRAIN + [2026]]
    print(f"{name:<38} {allp[0]:4} games {allp[1]:.1%} | " + " ".join(f"{y}:{n}/{h:.0%}" for y, (n, h) in zip(TRAIN + [2026], row)))
fav_side = lambda g: 1 if g["mkt"] > 0 else -1
sit("we like the favorite more", lambda g: np.sign(g["ours"] - g["mkt"]) == fav_side(g))
sit("we like the underdog more", lambda g: np.sign(g["ours"] - g["mkt"]) != fav_side(g))
sit("we like the home team more", lambda g: g["ours"] > g["mkt"] and not g["neutral"])
sit("we like the road team more", lambda g: g["ours"] < g["mkt"] and not g["neutral"])
sit("spread under 7", lambda g: abs(g["mkt"]) < 7)
sit("spread 7-14", lambda g: 7 <= abs(g["mkt"]) < 14)
sit("spread 14-21", lambda g: 14 <= abs(g["mkt"]) < 21)
sit("spread 21+", lambda g: abs(g["mkt"]) >= 21)
sit("weeks 1-4", lambda g: g["week"] <= 4)
sit("weeks 5-9", lambda g: 5 <= g["week"] <= 9)
sit("weeks 10+", lambda g: g["week"] >= 10)
sit("line moved toward our side (open->close)", lambda g: g["open"] is not None and np.sign(g["mkt"] - g["open"]) == np.sign(g["ours"] - g["mkt"]))
sit("line moved away from our side", lambda g: g["open"] is not None and np.sign(g["mkt"] - g["open"]) == -np.sign(g["ours"] - g["mkt"]) and g["mkt"] != g["open"])
json.dump([{k: v for k, v in g.items() if k != "blend_by_w"} for g in G], open(f"{S}/joined.json", "w"))
