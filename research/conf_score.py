"""Continuous confidence score: probability our pick covers the closing line, from a logistic model on pick-side
features (bad last week relative to opponent, efficiency model's disagreement, our gap, and both together). Trained on
2022-24, scored on 2025 and 2026; also leave-one-season-out over 2022-25. Reports top 5% / 10% hit rates and
calibration. Writes <work>/conf_coef.json. Usage: python research/conf_score.py <work> <repo>"""
import sys, json, runpy, io, contextlib, numpy as np
W, ROOT = sys.argv[1], sys.argv[2]
sys.argv = ["situational", ROOT]
with contextlib.redirect_stdout(io.StringIO()):
    S = runpy.run_path(f"{ROOT}/research/situational.py")
J = {(g["season"], g["week"], g["home"], g["away"]): g for g in json.load(open(f"{W}/joined.json"))}
PPA = {(p["season"], p["week"], p["home"], p["away"]): p["ppa_net"] for p in json.load(open(f"{W}/ppa_preds.json"))}
A_, B_ = 57.701, 2.884  # efficiency points scale fit on 2022-25 (the live constants)
R = []
for r in S["R"]:
    k = (r["season"], r["week"], r["home"], r["away"]); j = J.get(k)
    if r["week"] < 2 or not j or k not in PPA or r["act"] == r["mkt"] or j["ours"] == r["mkt"]: continue
    side = 1 if j["ours"] > r["mkt"] else -1
    ppa_m = A_ * PPA[k] + B_
    R.append(dict(season=r["season"], week=r["week"], won=((r["act"] > r["mkt"]) == (side > 0)),
                  bad=-side * (r["last_ats_h"] - r["last_ats_a"]) / 14,   # >0: our pick did worse last game than its opponent
                  dis=-side * (ppa_m - r["mkt"]) / 7,                      # >0: efficiency model on the other side of the line
                  gap=abs(j["ours"] - r["mkt"]) / 7))
FEATS = lambda r: [1, r["bad"], r["dis"], r["gap"], float(r["bad"] > 0 and r["dis"] > 0)]
def fit(rows, l2=10.0):
    X = np.array([FEATS(r) for r in rows]); y = np.array([r["won"] for r in rows], float); b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ b)); H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(b) - 1)])
        b -= np.linalg.solve(H, X.T @ (p - y) + l2 * np.r_[0, b[1:]])
    return b
prob = lambda b, rows: 1 / (1 + np.exp(-np.array([FEATS(r) for r in rows]) @ b))
def report(label, b, test):
    p = prob(b, test); won = np.array([r["won"] for r in test]); order = np.argsort(-p)
    out = []
    for frac in (0.05, 0.10):
        top = order[:max(1, int(len(test) * frac))]; out.append(f"top {int(frac*100)}%: {won[top].sum()}-{(~won[top]).sum()} ({won[top].mean():.0%}, avg score {p[top].mean():.0%})")
    rest = order[int(len(test) * .10):]
    print(f"  {label}: " + " | ".join(out) + f" | other 90%: {won[rest].mean():.1%}")
    return p, won
tr = [r for r in R if r["season"] in (2022, 2023, 2024)]; b = fit(tr)
print("trained on 2022-24, coefficients [const, bad, dis, gap, both]:", np.round(b, 3))
pall, wall = [], []
for y in (2025, 2026):
    p, w = report(f"test {y}", b, [r for r in R if r["season"] == y]); pall += list(p); wall += list(w)
print("leave one season out (train on the other 2022-25 seasons):")
for y in (2022, 2023, 2024, 2025):
    report(f"test {y}", fit([r for r in R if r["season"] in (2022, 2023, 2024, 2025) and r["season"] != y]), [r for r in R if r["season"] == y])
pall, wall = np.array(pall), np.array(wall)
print("calibration on 2025-26 (score bucket -> actual):")
for lo, hi in ((0, .50), (.50, .52), (.52, .54), (.54, .56), (.56, 1)):
    s = (pall >= lo) & (pall < hi)
    if s.sum(): print(f"  score {lo:.0%}-{hi:.0%}: {s.sum()} games, won {wall[s].mean():.1%}")
bfin = fit([r for r in R if r["season"] in (2022, 2023, 2024, 2025)])
cut = {f: float(np.quantile(prob(bfin, [r for r in R if r["season"] <= 2025]), 1 - f)) for f in (0.05, 0.10)}
json.dump(dict(coef=list(map(float, bfin)), cut_top5=cut[0.05], cut_top10=cut[0.10], scales=dict(bad=14, dis=7, gap=7)), open(f"{W}/conf_coef.json", "w"))
print("final model (2022-25) coefficients:", np.round(bfin, 3), "| score cutoffs: top 10% >=", round(cut[0.10], 4), ", top 5% >=", round(cut[0.05], 4))
