"""Writes data/research.json for the Lines tab from the outputs of vs_market.py and qb_injuries.py.
Usage: python research/summary.py <work dir> <repo root>"""
import sys, json, numpy as np
S, ROOT = sys.argv[1], sys.argv[2]
G = [g for g in json.load(open(f"{S}/joined.json")) if g["season"] < 2026]
F = json.load(open(f"{S}/qb_first_missed.json"))
seasons = sorted({g["season"] for g in G})
def su(gs, k): gs = [g for g in gs if g["act"] and g[k]]; return round(float(np.mean([(g[k] > 0) == (g["act"] > 0) for g in gs])), 4)
def mae(gs, k): return round(float(np.mean([abs(g[k] - g["act"]) for g in gs])), 2)
def ats(gs, t):
    p = [g for g in gs if abs(g["ours"] - g["mkt"]) >= t and g["act"] != g["mkt"]]
    w = sum((g["act"] > g["mkt"]) == (g["ours"] > g["mkt"]) for g in p); return [int(w), len(p) - int(w)]
out = dict(
    seasons=seasons,
    accuracy=[dict(season=y, n=len(gs), line_winners=su(gs, "mkt"), ours_winners=su(gs, "ours"), line_miss=mae(gs, "mkt"), ours_miss=mae(gs, "ours"))
              for y in seasons for gs in [[g for g in G if g["season"] == y]]],
    ats=[dict(edge=t, total=ats(G, t), by_season={str(y): ats([g for g in G if g["season"] == y], t) for y in seasons}) for t in (0, 3, 5, 7, 10)],
    qb_out=dict(by_season={str(y): [sum(r["mresid"] < 0 for r in F if r["season"] == y and r["mresid"]), sum(r["mresid"] > 0 for r in F if r["season"] == y and r["mresid"])] for y in seasons},
                by_value={str(lo): [sum(r["mresid"] < 0 for r in F if r["val"] >= lo and r["mresid"]), sum(r["mresid"] > 0 for r in F if r["val"] >= lo and r["mresid"])] for lo in (0, 4, 8)},
                drop_vs_model=round(float(-np.mean([r["resid"] for r in F])), 2), n=len(F)))
json.dump(out, open(f"{ROOT}/data/research.json", "w"), separators=(",", ":"))
print(json.dumps(out)[:400])
