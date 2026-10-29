"""Leak-free check of the one criterion that survived research/criteria.py: our pick failed to cover its last game AND
the play-efficiency model is on the other side of the line from us. The efficiency model's points-per-PPA scale is fit
on OTHER seasons only (criteria.py fit it on the same season, using later games). Usage: python research/criteria_check.py <work> <repo>"""
import sys, json, runpy, io, contextlib, numpy as np
W, ROOT = sys.argv[1], sys.argv[2]
sys.argv = ["situational", ROOT]
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(f"{ROOT}/research/situational.py")
J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(f"{W}/joined.json"))}
PPA = {(p["season"], p["week"], p["home"], p["away"]): p["ppa_net"] for p in json.load(open(f"{W}/ppa_preds.json"))}
R = []
for r in S["R"]:
    k = (r["season"], r["week"], r["home"], r["away"]); j = J.get(k)
    if r["week"] < 2 or not j or k not in PPA or r["act"] == r["mkt"]: continue
    R.append(dict(r, ours=j["ours"], ppa=PPA[k]))
for y in range(2022, 2027):
    other = [r for r in R if r["season"] != y and r["season"] != 2026]  # never fit on 2026 or the season scored
    a, b = np.polyfit([r["ppa"] for r in other], [r["act"] for r in other], 1)
    for r in R:
        if r["season"] == y: r["ppa_m"] = a * r["ppa"] + b
rec = {}
for r in R:
    pick_home = r["ours"] > r["mkt"]; side = 1 if pick_home else -1
    flag = (r["ppa_m"] > r["mkt"]) != pick_home and side * (r["last_ats_h"] - r["last_ats_a"]) < 0
    if flag:
        w = (r["act"] > r["mkt"]) == pick_home; s = rec.setdefault(r["season"], [0, 0]); s[0 if w else 1] += 1
tot = [sum(v[0] for y, v in rec.items() if y < 2026), sum(v[1] for y, v in rec.items() if y < 2026)]
n_all = {y: sum(1 for r in R if r["season"] == y) for y in rec}
for y in sorted(rec): print(f"{y}: {rec[y][0]}-{rec[y][1]} ({rec[y][0]/sum(rec[y]):.1%})  flagged {sum(rec[y])} of {n_all[y]} games ({sum(rec[y])/n_all[y]:.0%})")
print(f"2022-25 (seasons it was not found on): {tot[0]}-{tot[1]} ({tot[0]/sum(tot):.1%})")
