"""Pre-registered check: does the score model's edge against OPENING lines (2025: 55.8% at 3+ points, 57.8% at 5+,
model trained on 2022-24) hold in 2026, which no setting was tuned on? Uses the week-by-week 2026 predictions in
<work>/preds.json and the opening spreads in data/lines.json. Usage: python research/opening_2026.py <work> <repo>"""
import sys, json
W, ROOT = sys.argv[1], sys.argv[2]
P = {(p["week"], p["home"], p["away"]): p for p in json.load(open(f"{W}/preds.json")) if p["season"] == 2026}
L = json.load(open(f"{ROOT}/data/lines.json"))["games"]
rows = []
for r in L:
    if r[6] is None or len(r) < 11 or r[10] is None: continue
    p = P.get((r[0], r[1], r[2]))
    if p and p["week"] >= 2: rows.append(dict(week=r[0], ours=p["ours"], open=-r[10], close=-r[4], act=r[6] - r[7]))
print(f"2026 games with an opening line and a pre-game prediction: {len(rows)} (weeks {min(r['week'] for r in rows)}-{max(r['week'] for r in rows)})")
for line in ("open", "close"):
    for t in (0, 3, 5, 7):
        s = [r for r in rows if abs(r["ours"] - r[line]) >= t and r["act"] != r[line]]
        w = sum((r["act"] > r[line]) == (r["ours"] > r[line]) for r in s)
        print(f"  vs {'OPENING' if line == 'open' else 'closing'} line, edge {t}+: {w}-{len(s)-w} ({w/max(1,len(s)):.1%})")
