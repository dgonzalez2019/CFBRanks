"""Spread and over/under picks: logistic models trained on 2022-24 only, scored once on 2025 and 2026.
Confidence = the model's probability for the side it picks. Usage: python research/picks_model.py <work dir> <repo>"""
import sys, json, numpy as np
W, ROOT = sys.argv[1], sys.argv[2]
G = json.load(open(f"{W}/picks.json"))
TRAIN, TEST = [2022, 2023, 2024], [2025, 2026]

def logit_fit(X, y, l2=1.0):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); g = X.T @ (p - y) + l2 * np.r_[0, b[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b -= np.linalg.solve(H, g)
    return b

def spread_x(g):
    e = np.clip(g["ours"] - g["mkt"], -20, 20); fav_home = 1.0 if g["mkt"] > 0 else -1.0
    move = (g["mkt"] - g["open"]) if g.get("open") is not None else 0.0
    return [1, e / 10, abs(g["mkt"]) / 20 * np.sign(e), fav_home * np.sign(e) if e else 0, 0 if g["neutral"] else 1,
            (e / 10) * (g["week"] <= 4), move / 5, (move / 5) * np.sign(e)]
def total_x(g):
    et = np.clip(g["tot_prior"] - g["total"], -25, 25)
    en = np.clip((g["tot_now"] - g["total"]) if g.get("tot_now") is not None else et, -25, 25)
    pace = ((g["pace_h"] or 0) + (g["pace_a"] or 0) - 140) / 20 if g["pace_h"] and g["pace_a"] else 0
    return [1, et / 10, en / 10, pace, g["week"] / 10, (g["total"] - 55) / 10]

SUMMARY = {}
def run(name, xf, target, valid):
    rows = [g for g in G if valid(g)]
    def XY(ss):
        r = [g for g in rows if g["season"] in ss]; return np.array([xf(g) for g in r], float), np.array([target(g) for g in r], float), r
    Xtr, ytr, _ = XY(TRAIN); b = logit_fit(Xtr, ytr)
    print(f"\n== {name}: trained on {len(ytr)} games (2022-24); in-sample hit {np.mean(((1/(1+np.exp(-Xtr@b)))>.5)==ytr):.1%}")
    out = {}
    for ss in ([2025], [2026], TEST):
        X, y, r = XY(ss)
        if not len(y): continue
        p = 1 / (1 + np.exp(-X @ b)); pick = p > .5; conf = np.where(pick, p, 1 - p); hit = pick == y
        line = f"  test {'+'.join(map(str, ss))}: all {len(y)} games {hit.mean():.1%}"
        for lo in (.52, .54, .56, .58, .60):
            sel = conf >= lo
            if sel.sum(): line += f" | conf>={lo:.0%}: {int(hit[sel].sum())}-{int((~hit[sel]).sum())} {hit[sel].mean():.0%}"
        print(line)
        top = np.argsort(-conf)[:max(1, len(conf) // 10)]
        print(f"     top 10% most confident: {hit[top].mean():.1%} of {len(top)}  (avg confidence {conf[top].mean():.1%})")
        out[tuple(ss)] = (p, y, r)
        SUMMARY.setdefault(name.split()[0].lower(), {})["+".join(map(str, ss))] = dict(
            all=[int(hit.sum()), int((~hit).sum())], top10=[int(hit[top].sum()), int(len(top) - hit[top].sum())])
    return b, out

run("SPREAD (home covers)", spread_x, lambda g: float(g["act"] > g["mkt"]), lambda g: g["act"] != g["mkt"])
run("TOTAL (over)", total_x, lambda g: float(g["act_total"] > g["total"]), lambda g: g["total"] is not None and g["act_total"] != g["total"] and g["tot_prior"] is not None)

# simple rule checks for totals, train vs test
print("\n== simple total rules (train 2022-24 | test 2025 | test 2026) ==")
for name, f in [("our total above line by 5+ -> over", lambda g: 1 if g["tot_prior"] - g["total"] >= 5 else (0 if g["tot_prior"] - g["total"] <= -5 else None)),
                ("our total above line by 8+ -> over", lambda g: 1 if g["tot_prior"] - g["total"] >= 8 else (0 if g["tot_prior"] - g["total"] <= -8 else None)),
                ("total 60+ -> under", lambda g: 0 if g["total"] >= 60 else None), ("total under 45 -> over", lambda g: 1 if g["total"] < 45 else None)]:
    cells = []
    for ss in (TRAIN, [2025], [2026]):
        r = [(f(g), g) for g in G if g["season"] in ss and g["total"] is not None and g["act_total"] != g["total"] and g["tot_prior"] is not None]
        r = [(s, g) for s, g in r if s is not None]; w = sum((g["act_total"] > g["total"]) == bool(s) for s, g in r)
        cells.append(f"{w}-{len(r)-w} {w/max(1,len(r)):.0%}")
    print(f"  {name:<36} " + " | ".join(cells))

json.dump(SUMMARY, open(f"{W}/picks_summary.json", "w"))
