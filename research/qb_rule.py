"""Out-of-sample test of the QB rule: bet against a team in the first game its main QB (most pass attempts in
the season's earlier games) doesn't throw a pass. Uses only pre-game information to name the starter.
Usage: python research/qb_rule.py <work dir> <repo root>"""
import sys, json
from collections import defaultdict
W, ROOT = sys.argv[1], sys.argv[2]
J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(f"{W}/joined.json"))}
OUT = {}
for y in (2022, 2023, 2024, 2025, 2026):
    try: Q = json.load(open(f"{ROOT}/data/history/{y}_qb.json"))["games"]
    except OSError: print(y, "no passer data"); continue
    Q.sort(key=lambda r: r[0])
    att = defaultdict(lambda: defaultdict(int)); last = {}; w = l = 0; picks = []
    for wk, h, a, hp, ap in Q:
        for team, ps, sign in ((h, hp, 1), (a, ap, -1)):
            now = {n: k for n, k in ps}
            if att[team]:
                qb1, n1 = max(att[team].items(), key=lambda kv: kv[1])
                started_last = last.get(team, {}).get(qb1, 0) >= 5
                g = J.get((y, wk, h, a))
                if n1 >= 40 and started_last and now.get(qb1, 0) == 0 and g and g["act"] != g["mkt"]:
                    covered_against = (sign * (g["act"] - g["mkt"])) < 0
                    w += covered_against; l += not covered_against; picks.append((wk, team, qb1))
            for n, k in ps: att[team][n] += k
            last[team] = now
    print(f"{y}: bet against team whose starter sat: {w}-{l} ({w/max(1,w+l):.1%})"); OUT[str(y)] = [int(w), int(l)]
json.dump(OUT, open(f"{W}/qb_rule.json", "w"))
