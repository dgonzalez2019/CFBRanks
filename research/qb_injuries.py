import sys, json, numpy as np; sys.path.insert(0, sys.argv[2] + "/scripts")
import cfb_sor as m
from collections import defaultdict
S, ROOT = sys.argv[1], sys.argv[2]
preds = {(p["season"], p["week"], p["home"], p["away"]): p for p in json.load(open(f"{S}/preds.json"))}
lines = {(g["season"], g["week"], g["home"], g["away"]): g["mkt"] for g in json.load(open(f"{S}/joined.json"))}
rows = []
for y in (2022, 2023, 2024, 2025):
    games = m.load_json(f"{ROOT}/data/history/{y}.json")
    Q = json.load(open(f"{ROOT}/data/history/{y}_qb.json"))["games"]
    att = defaultdict(lambda: defaultdict(int)); team_att = defaultdict(int); played = defaultdict(dict)
    for wk, h, a, hp, ap in Q:
        for team, ps in ((h, hp), (a, ap)):
            for name, n in ps: att[team][name] += n; team_att[team] += n
            played[team][wk] = {name: n for name, n in ps}
    m.INJ_SHARE = m.INJ_SHARE_QB = 1.0  # raw values; the fitted slope is the share
    V = m.player_values(json.load(open(f"{ROOT}/data/history/{y}_players.json"))["players"], [g for g in games if g["home_fbs"] or g["away_fbs"]])
    qb1 = {}
    for team, d in att.items():
        name, n = max(d.items(), key=lambda kv: kv[1])
        if n >= 0.5 * team_att[team]:  # a clear starter
            val = next((p["value"] for p in V.get(team, []) if p["name"] == name), None)
            if val is not None: qb1[team] = (name, val)
    for (s, wk, h, a), p in preds.items():
        if s != y: continue
        for team, sign in ((h, 1), (a, -1)):
            if team not in qb1 or wk not in played[team]: continue
            name, val = qb1[team]; n_now = played[team][wk].get(name, 0)
            prev_wks = [w for w in played[team] if w < wk]
            first_miss = n_now < 5 and prev_wks and played[team][max(prev_wks)].get(name, 0) >= 5
            rows.append(dict(season=y, team=team, qb=name, val=val, out=n_now == 0, first=bool(first_miss) and n_now == 0,
                             resid=sign * (p["act"] - p["ours"]), mresid=(sign * (p["act"] - lines[(s, wk, h, a)])) if (s, wk, h, a) in lines else None))
R = [r for r in rows]
inn = [r["resid"] for r in R if not r["out"]]
print(f"team-games: {len(R)}; starter played {len(inn)} (avg vs model {np.mean(inn):+.2f}); starter out {sum(r['out'] for r in R)}; first game out {sum(r['first'] for r in R)}")
F = [r for r in R if r["first"]]
v = np.array([r["val"] for r in F]); res = np.array([r["resid"] for r in F]) - np.mean(inn)
print(f"first games missed: n={len(F)}, team did {np.mean(res):+.2f} pts vs model (SE {np.std(res)/np.sqrt(len(F)):.2f}); raw value of those QBs avg {v.mean():.2f}")
k = -np.sum(v * res) / np.sum(v * v); se = np.sqrt(np.sum((res + k * v) ** 2) / (len(F) - 1) / np.sum(v * v))
print(f"fitted share of raw value lost: {k:.2f} (SE {se:.2f})  [current setting 0.5]")
for lo, hi in ((0, 4), (4, 8), (8, 99)):
    sel = [r for r in F if lo <= r["val"] < hi]
    if sel: print(f"  raw value {lo}-{hi}: n={len(sel):3} lost {-np.mean([r['resid'] for r in sel]) + np.mean(inn):.2f} pts vs raw value avg {np.mean([r['val'] for r in sel]):.2f}")
M = [r for r in F if r["mresid"] is not None]
print(f"vs betting line in those games: team did {np.mean([r['mresid'] for r in M]):+.2f} pts vs line (n={len(M)}) -> the market already prices most of it")
all_out = [r for r in R if r["out"]]
print(f"all games out (incl. later in a stretch): n={len(all_out)}, vs model {np.mean([r['resid'] for r in all_out]) - np.mean(inn):+.2f}")
print("\nATS: bet AGAINST the team in the first game its starting QB misses")
for y in (2022, 2023, 2024, 2025, None):
    sel = [r for r in F if r["mresid"] is not None and r["mresid"] != 0 and (y is None or r["season"] == y)]
    w = sum(r["mresid"] < 0 for r in sel)
    print(f"  {y or '2022-25'}: {w}-{len(sel)-w} ({w/len(sel):.1%})")
for lo in (0, 4, 8):
    sel = [r for r in F if r["mresid"] is not None and r["mresid"] != 0 and r["val"] >= lo]
    w = sum(r["mresid"] < 0 for r in sel); print(f"  raw value >= {lo}: {w}-{len(sel)-w} ({w/len(sel):.1%})")
json.dump([dict(season=r["season"], val=r["val"], resid=float(r["resid"] - np.mean(inn)), mresid=r["mresid"]) for r in F], open(f"{S}/qb_first_missed.json", "w"))
