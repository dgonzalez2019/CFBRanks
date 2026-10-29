"""Scores the positional-matchups experiment: margin = (1-w)*score margin + w*matchup margin, with w picked on the
other seasons (leave one season out), for the passing/rushing-only margin ("eff") and the full one ("full").
Usage: python research/matchups_eval.py <work dir>"""
import sys, json, numpy as np
R = [r for r in json.load(open(sys.argv[1] + "/matchups.json")) if "eff" in r and r["act"] != 0]
TRAIN = [2022, 2023, 2024, 2025]; W = np.round(np.linspace(0, 0.8, 17), 2)
def score(rs, k, w):
    p = np.array([(1 - w) * r["m1"] + w * r[k] for r in rs]); a = np.array([r["act"] for r in rs])
    return float(np.mean(np.abs(p - a))), float(np.mean((p > 0) == (a > 0)))
for k in ("eff", "full"):
    print(f"\n== {k} ==")
    held = []
    for test in TRAIN + [2026]:
        tr = [r for r in R if r["season"] in TRAIN and r["season"] != test]
        w = min(W, key=lambda w: score(tr, k, w)[0])
        te = [r for r in R if r["season"] == test]
        m0, h0 = score(te, k, 0); m, h = score(te, k, w)
        print(f"test {test}: w={w:.2f}  miss {m:.2f} (score only {m0:.2f})  winners {h:.1%} (score only {h0:.1%})  n={len(te)}")
        if test in TRAIN: held += [(m0, m, h0, h, len(te))]
    n = sum(x[4] for x in held)
    print("held-out 2022-25 avg: miss %.3f -> %.3f, winners %.2f%% -> %.2f%%" % tuple(sum(x[i] * x[4] for x in held) / n * (100 if i in (2, 3) else 1) for i in range(4)))
# feature coefficients (points per unit) across seasons, late-season fits
F = ["ypa", "ypc", "sack", "tfl", "hurry", "to", "third"]
for y in TRAIN + [2026]:
    c = [r["coef_full"] for r in R if r["season"] == y]
    if c: print(y, "points per unit (last fit):", dict(zip(["const"] + F, np.round(c[-1], 2))))
