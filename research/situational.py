"""Situational model, no team ratings: rest, travel, last game vs the spread, season form vs the spread, and the
line itself. Angles and model are learned on 2022-24 and scored once on 2025 and 2026, against closing and
opening lines. Usage: python research/situational.py <repo root>"""
import sys, json, math, numpy as np
from collections import defaultdict
from datetime import datetime
ROOT = sys.argv[1]
V = {r[0]: r for r in json.load(open(f"{ROOT}/data/history/venues.json"))["venues"]}

def km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p = math.pi / 180; h = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))

def season_rows(y):
    meta = json.load(open(f"{ROOT}/data/history/{y}_meta.json"))["games"]
    if y < 2026:
        lines = {(r[0], r[1], r[2]): (-r[3], r[4], -r[5] if r[5] is not None else None) for r in json.load(open(f"{ROOT}/data/history/{y}_lines.json"))["games"]}
        res = {(r[0], r[1], r[2]): r[3] - r[4] for r in json.load(open(f"{ROOT}/data/history/{y}.json"))["games"]}
    else:
        L = json.load(open(f"{ROOT}/data/lines.json"))["games"]
        lines = {(r[0], r[1], r[2]): (-r[4], r[5], None) for r in L}
        res = {(r[0], r[1], r[2]): r[3] - r[4] for r in json.load(open(f"{ROOT}/data/games.json"))["games"]}
    # each team's home stadium = its most used non-neutral home venue
    homeven = defaultdict(lambda: defaultdict(int))
    for wk, h, a, st, vid, neu, conf, hc, ac in meta:
        if not neu and vid in V: homeven[h][vid] += 1
    home_loc = {t: V[max(d, key=d.get)][2:4] for t, d in homeven.items()}
    meta.sort(key=lambda r: r[3] or "")
    last = {}; ats_hist = defaultdict(list); rows = []
    for wk, h, a, st, vid, neu, conf, hc, ac in meta:
        if not st: continue
        day = datetime.fromisoformat(st.replace("Z", "+00:00")); key = (wk, h, a)
        ln = lines.get(key); act = res.get(key)
        f = {}
        for side, t in (("h", h), ("a", a)):
            prev = last.get(t)
            f[f"rest_{side}"] = min((day - prev["day"]).days, 21) if prev else 14
            f[f"last_ats_{side}"] = prev["ats"] if prev and prev["ats"] is not None else 0.0
            f[f"last_margin_{side}"] = prev["margin"] if prev else 0.0
            hist = ats_hist[t]; f[f"form_{side}"] = (sum(hist) + 2) / (len(hist) + 4) - 0.5
            loc = V.get(vid, [None] * 6)[2:4]
            f[f"travel_{side}"] = km(home_loc[t], loc) / 1000 if t in home_loc and None not in loc and None not in home_loc[t] else 0.0
        if ln and act is not None and h in home_loc and a in home_loc:
            mkt, total, opn = ln
            rows.append(dict(season=y, week=wk, home=h, away=a, neutral=bool(neu), conf=bool(conf), mkt=mkt, open=opn, total=total, act=act, **f))
        for t, sign in ((h, 1), (a, -1)):
            ats = None
            if ln and act is not None and act != ln[0]:
                ats = sign * (act - ln[0]); ats_hist[t].append(1 if ats > 0 else 0)
            last[t] = dict(day=day, ats=ats, margin=sign * act if act is not None else 0.0)
    return rows

R = [r for y in (2022, 2023, 2024, 2025, 2026) for r in season_rows(y)]
print("games with lines:", {y: sum(r["season"] == y for r in R) for y in (2022, 2023, 2024, 2025, 2026)})
TRAIN = (2022, 2023, 2024)

# Angles: each a score (positive -> take the home side); direction set by 2022-24.
ANGLES = {
    "rest advantage (days)": lambda r: r["rest_h"] - r["rest_a"],
    "home off a bye, road not": lambda r: (r["rest_h"] >= 12) - (r["rest_a"] >= 12),
    "road team travel (1000 km)": lambda r: (r["travel_a"] - r["travel_h"]),
    "fade last week's cover margin": lambda r: -(r["last_ats_h"] - r["last_ats_a"]),
    "fade last week's margin": lambda r: -(r["last_margin_h"] - r["last_margin_a"]),
    "fade season cover form": lambda r: -(r["form_h"] - r["form_a"]),
    "home underdog": lambda r: 1 if r["mkt"] < 0 and not r["neutral"] else 0,
    "big favorite (14+) covers": lambda r: (1 if r["mkt"] >= 14 else -1 if r["mkt"] <= -14 else 0),
    "underdog of 3 or 7 (+/-0.5)": lambda r: (-1 if r["mkt"] > 0 else 1) if min(abs(abs(r["mkt"]) - 3), abs(abs(r["mkt"]) - 7)) <= 0.5 else 0,
    "conference game underdog": lambda r: (-1 if r["mkt"] > 0 else 1) if r["conf"] else 0,
}
def grade(rows, score, line="mkt", top=None):
    p = [(score(r), r) for r in rows if r.get(line) is not None and r["act"] != r[line]]
    p = [(s, r) for s, r in p if s]
    if top: p = sorted(p, key=lambda x: -abs(x[0]))[:max(1, int(len(p) * top))]
    w = sum((r["act"] > r[line]) == (s > 0) for s, r in p); return w, len(p) - w
pct = lambda wl: f"{wl[0]}-{wl[1]} {wl[0]/max(1,sum(wl)):.0%}"
print("\n== single angles (direction from 2022-24; top third of each angle's strength) ==")
print(f"{'angle':<32} {'train 2022-24':>16} {'2025 close':>14} {'2025 open':>14} {'2026 close':>14}")
for name, f in ANGLES.items():
    tr = grade([r for r in R if r["season"] in TRAIN], f); sgn = 1 if tr[0] >= tr[1] else -1
    g = lambda ss, line: grade([r for r in R if r["season"] in ss], lambda r: sgn * f(r), line, top=1/3)
    print(f"{name:<32} {pct(g(TRAIN,'mkt')):>16} {pct(g((2025,),'mkt')):>14} {pct(g((2025,),'open')):>14} {pct(g((2026,),'mkt')):>14}")

# Combined logistic model on standardized angle scores
names = list(ANGLES); tr = [r for r in R if r["season"] in TRAIN and r["act"] != r["mkt"]]
Xtr = np.array([[ANGLES[n](r) for n in names] for r in tr], float); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Z = lambda rs: np.c_[np.ones(len(rs)), (np.array([[ANGLES[n](r) for n in names] for r in rs], float) - mu) / sd]
y = np.array([r["act"] > r["mkt"] for r in tr], float); b = np.zeros(len(names) + 1)
for _ in range(50):
    p = 1 / (1 + np.exp(-Z(tr) @ b)); H = (Z(tr) * (p * (1 - p))[:, None]).T @ Z(tr) + np.eye(len(b)) * 5
    b -= np.linalg.solve(H, Z(tr).T @ (p - y) + 5 * np.r_[0, b[1:]])
print("\n== combined model (trained 2022-24) ==")
for ss in ((2025,), (2026,), (2025, 2026)):
    for line in ("mkt", "open"):
        te = [r for r in R if r["season"] in ss and r.get(line) is not None and r["act"] != r[line]]
        if not te: continue
        pr = 1 / (1 + np.exp(-Z(te) @ b)); hit = (pr > .5) == np.array([r["act"] > r[line] for r in te]); conf = np.abs(pr - .5)
        top = np.argsort(-conf)[:max(1, len(te) // 10)]
        print(f"  {'+'.join(map(str,ss))} vs {'closing' if line=='mkt' else 'opening'}: all {hit.sum()}-{(~hit).sum()} {hit.mean():.1%} | top 10% {hit[top].sum()}-{(~hit[top]).sum()} {hit[top].mean():.1%}")
