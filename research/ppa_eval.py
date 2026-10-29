"""Scores the play-efficiency experiment against opening and closing lines. Margin = a*ours + b*ppa_net*k + c*home,
fit on training seasons only (leave one season out over 2022-25, and 2022-24 -> 2025-26).
Usage: python research/ppa_eval.py <work dir>"""
import sys, json, numpy as np
W = sys.argv[1]
J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(f"{W}/joined.json"))}
R = []
for p in json.load(open(f"{W}/ppa_preds.json")):
    j = J.get((p["season"], p["week"], p["home"], p["away"]))
    if j: R.append(dict(p, ours=j["ours"], mkt=j["mkt"], open=j.get("open")))
X = lambda rs, cols: np.array([[1.0 if c == "c" else (0.0 if r["neutral"] else 1.0) if c == "home" else r[c] for c in cols] for r in rs])
def fit(rs, cols):
    A = X(rs, cols); y = np.array([r["act"] for r in rs], float); return np.linalg.lstsq(A, y, rcond=None)[0]
def grade(rs, pred, line):
    p = [(m, r) for m, r in zip(pred, rs) if r.get(line) is not None and r["act"] != r[line]]
    out = {}
    for t in (0, 3, 5):
        s = [(m, r) for m, r in p if abs(m - r[line]) >= t]; w = sum((r["act"] > r[line]) == (m > r[line]) for m, r in s)
        out[t] = (w, len(s) - w)
    mae = np.mean([abs(m - r["act"]) for m, r in zip(pred, rs)]); hit = np.mean([(m > 0) == (r["act"] > 0) for m, r in zip(pred, rs) if r["act"]])
    return out, mae, hit
fmt = lambda o: " ".join(f"{t}+:{w}-{l} {w/max(1,w+l):.1%}" for t, (w, l) in o.items())
for name, cols in [("score model only", ["c", "ours"]), ("PPA only", ["c", "ppa_net", "home"]), ("score + PPA", ["c", "ours", "ppa_net", "home"])]:
    print(f"\n== {name}")
    for test in [[2022], [2023], [2024], [2025], [2025, 2026]]:
        train = [r for r in R if r["season"] in (2022, 2023, 2024, 2025) and r["season"] not in test] if test != [2025, 2026] else [r for r in R if r["season"] in (2022, 2023, 2024)]
        b = fit(train, cols); te = [r for r in R if r["season"] in test]; pred = X(te, cols) @ b
        oc, mae, hit = grade(te, pred, "mkt"); oo, _, _ = grade(te, pred, "open")
        print(f"  test {'+'.join(map(str,test))}: miss {mae:.2f} winners {hit:.1%} | vs CLOSING {fmt(oc)} | vs OPENING {fmt(oo)}")
