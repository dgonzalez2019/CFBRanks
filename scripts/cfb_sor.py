#!/usr/bin/env python3
"""
College Football Strength of Record (SOR) Engine
================================================

Pipeline
--------
1. POWER RATINGS  - Every team gets a rating (points vs. an average FBS team on a
   neutral field) solved from ALL games simultaneously, so a team's rating reflects
   how its opponents played against *their* opponents (and so on, network-wide).
   Home-field advantage is estimated from the data, and neutral sites get none.
   Margins are compressed (diminishing returns past ~3 scores) so blowouts of bad
   teams don't inflate ratings.

2. GAME DIFFICULTY - For every game, compute the probability that a "benchmark"
   team (default: the #25 rated team) would win THAT game, at THAT venue:
       home game:  benchmark + HFA vs opponent
       road game:  benchmark - HFA vs opponent
       neutral:    benchmark       vs opponent
   A road game at a good team is much harder than the same opponent at home.

3. STRENGTH OF RECORD - Probability a benchmark team would go at least as well as
   the team actually did against its exact schedule (Poisson-binomial DP).
   Lower probability = more impressive record. Also reported as "wins above
   benchmark" (actual wins - expected benchmark wins), which is what we rank by.

4. WIN / LOSS QUALITY - Every result is scored:
       Win value   = 1 - P(benchmark wins)   -> big for hard wins (road at a top team)
       Loss cost   = P(benchmark wins)       -> big for bad losses (home to a weak team)
   Opponents are also tiered Q1-Q4 by venue-adjusted rank (NET-style quadrants):
       Q1: home vs top 15 | neutral vs top 25 | road vs top 40
       Q2: home 16-40     | neutral 26-55     | road 41-75
       Q3: home 41-80     | neutral 56-95     | road 76-110
       Q4: everything else (incl. FCS)

Usage
-----
  # Live data (free key from collegefootballdata.com):
  export CFBD_API_KEY=your_key
  python cfb_sor.py --year 2026

  # Or a CSV with columns:
  #   week,home_team,away_team,home_points,away_points,neutral_site[,home_fbs,away_fbs]
  python cfb_sor.py --csv games.csv

  Options: --benchmark 25  --top 25  --team "Miami"  --out results.csv
"""
import argparse, csv, json, math, os, sys, urllib.request
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------- tuning knobs
MARGIN_CAP_SCALE = 21.0   # margin compression: m' = S*tanh(m/S) ~ caps near 3 scores
RIDGE = 2.0               # regularization (pulls thin-sample teams toward average)
WIN_PROB_SD = 15.0        # overwritten at fit time from model residuals
FCS_POOL = "FCS (pooled)" # all non-FBS opponents share one rating
RECENCY = 0.0             # 0 = all weeks equal; e.g. 0.03 = later weeks weigh more
# Ranking score = power rating + RESUME_WEIGHT * resume. Resume points per game by quadrant:
# Q1 wins earn the most, and a loss costs LOSS_WEIGHT of what the mirror-image win earns.
WIN_PTS = {"Q1": 4.0, "Q2": 2.0, "Q3": 1.0, "Q4": 0.5}
LOSS_PTS = {"Q1": 0.5, "Q2": 1.0, "Q3": 2.0, "Q4": 4.0}
LOSS_WEIGHT = 0.85
RESUME_WEIGHT = 0.5
# Quadrants by the opponent's power rank at any venue: Q1 = top 25, Q2 = 26-50, Q3 = 51-80, Q4 = rest + FCS.
# Power rank (not overall rank) keeps resumes from feeding back into opponents' ranks.
QUAD_CUTS = (25, 50, 80)


def compress(m):
    return MARGIN_CAP_SCALE * math.tanh(m / MARGIN_CAP_SCALE)


def win_prob(margin):
    """P(win) given expected margin, normal approximation."""
    return 0.5 * (1 + math.erf(margin / (WIN_PROB_SD * math.sqrt(2))))


# ---------------------------------------------------------------- data loading
def load_csv(path):
    games = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("home_points", "") in ("", None):
                continue  # unplayed
            games.append(dict(
                week=int(r.get("week") or 0),
                home=r["home_team"].strip(), away=r["away_team"].strip(),
                hp=int(r["home_points"]), ap=int(r["away_points"]),
                neutral=str(r.get("neutral_site", "")).lower() in ("1", "true", "yes", "y"),
                home_fbs=str(r.get("home_fbs", "true")).lower() in ("1", "true", "yes", "y"),
                away_fbs=str(r.get("away_fbs", "true")).lower() in ("1", "true", "yes", "y"),
            ))
    return games


def load_json(path):
    """The site's data/games.json, including box scores when present."""
    games = []
    for r in json.load(open(path))["games"]:
        games.append(dict(week=r[0], home=r[1], away=r[2], hp=r[3], ap=r[4], neutral=bool(r[5]),
                          home_fbs=bool(r[6]), away_fbs=bool(r[7]), box=r[8] if len(r) > 8 else None))
    return games


def load_cfbd(year, key, upcoming=None):
    """Completed games involving an FBS team. Pass a list as `upcoming` to also collect unplayed ones."""
    def get(url):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    fbs = {t["school"] for t in get(f"https://api.collegefootballdata.com/teams/fbs?year={year}")}
    games = []
    for seas in ("regular", "postseason"):
        for g in get(f"https://api.collegefootballdata.com/games?year={year}&seasonType={seas}"):
            # API field names have used both camelCase and snake_case over time
            hp = g.get("homePoints", g.get("home_points"))
            ap = g.get("awayPoints", g.get("away_points"))
            h = g.get("homeTeam", g.get("home_team")); a = g.get("awayTeam", g.get("away_team"))
            if h not in fbs and a not in fbs:
                continue  # no FBS team involved
            if hp is None or ap is None:
                if upcoming is not None:
                    upcoming.append(dict(week=g.get("week", 0) + (20 if seas == "postseason" else 0), home=h, away=a,
                                         neutral=bool(g.get("neutralSite", g.get("neutral_site"))),
                                         home_fbs=h in fbs, away_fbs=a in fbs,
                                         start=g.get("startDate", g.get("start_date"))))
                continue
            games.append(dict(week=g.get("week", 0) + (20 if seas == "postseason" else 0),
                              home=h, away=a, hp=hp, ap=ap,
                              neutral=bool(g.get("neutralSite", g.get("neutral_site"))),
                              home_fbs=h in fbs, away_fbs=a in fbs))
    return games


# ---------------------------------------------------------------- core model
def fit_ratings(games):
    """Ridge least squares: compressed_margin = HFA*home + R_home - R_away."""
    norm = lambda t, fbs: t if fbs else FCS_POOL
    teams = sorted({norm(g["home"], g["home_fbs"]) for g in games} |
                   {norm(g["away"], g["away_fbs"]) for g in games})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    max_wk = max(g["week"] for g in games) or 1

    X = np.zeros((len(games), n + 1)); y = np.zeros(len(games)); w = np.ones(len(games))
    for k, g in enumerate(games):
        h, a = norm(g["home"], g["home_fbs"]), norm(g["away"], g["away_fbs"])
        X[k, idx[h]] = 1; X[k, idx[a]] = -1
        X[k, n] = 0 if g["neutral"] else 1          # HFA column
        y[k] = compress(g["hp"] - g["ap"])
        w[k] = math.exp(-RECENCY * (max_wk - g["week"]))

    W = np.sqrt(w)[:, None]
    P = np.eye(n + 1) * RIDGE; P[n, n] = 0          # don't shrink HFA
    beta = np.linalg.solve((X * W).T @ (X * W) + P, (X * W).T @ (y * W.ravel()))
    ratings = dict(zip(teams, beta[:n])); hfa = beta[n]
    # calibrate outcome spread from residuals (same compressed units as ratings)
    global WIN_PROB_SD
    resid = y - X @ beta
    WIN_PROB_SD = float(np.sqrt(np.average(resid ** 2, weights=w)) * 1.05)

    fbs_vals = [r for t, r in ratings.items() if t != FCS_POOL]
    mean = float(np.mean(fbs_vals))                   # center: avg FBS team = 0
    ratings = {t: r - mean for t, r in ratings.items()}
    return ratings, hfa, norm


def poisson_binomial_at_least(probs, k):
    dist = np.zeros(len(probs) + 1); dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum())


def quadrant(opp_rank):
    if opp_rank is None: return "Q4"
    for q, c in zip(("Q1", "Q2", "Q3"), QUAD_CUTS):
        if opp_rank <= c: return q
    return "Q4"


def compute(games, benchmark_rank=25):
    games = [g for g in games if g["home_fbs"] or g["away_fbs"]]  # pooled-vs-pooled games carry no signal
    ratings, hfa, norm = fit_ratings(games)
    fbs_sorted = sorted((t for t in ratings if t != FCS_POOL), key=lambda t: -ratings[t])
    rank = {t: i + 1 for i, t in enumerate(fbs_sorted)}
    bench = ratings[fbs_sorted[min(benchmark_rank, len(fbs_sorted)) - 1]]
    avg_team = 0.0

    sched = defaultdict(list)
    for g in games:
        h, a = norm(g["home"], g["home_fbs"]), norm(g["away"], g["away_fbs"])
        for me, opp, pf, pa, side in ((h, a, g["hp"], g["ap"], "home"), (a, h, g["ap"], g["hp"], "road")):
            if me == FCS_POOL: continue
            venue = "neutral" if g["neutral"] else side
            ven_adj = {"home": hfa, "road": -hfa, "neutral": 0.0}[venue]
            p_bench = win_prob(bench + ven_adj - ratings[opp])
            p_avg = win_prob(avg_team + ven_adj - ratings[opp])
            won = pf > pa
            sched[me].append(dict(
                week=g["week"], opp=opp, venue=venue, pf=pf, pa=pa, won=won,
                opp_rank=rank.get(opp), quad=quadrant(rank.get(opp)),
                p_bench=p_bench,
                value=(1 - p_bench) if won else -p_bench,   # + win quality / - loss cost
            ))

    rows = []
    for t in fbs_sorted:
        gs = sched[t]
        wins = sum(x["won"] for x in gs); losses = len(gs) - wins
        pb = [x["p_bench"] for x in gs]
        exp_w = sum(pb)
        sor_prob = poisson_binomial_at_least(pb, wins)
        # SOS = benchmark's expected losses per game (higher = harder schedule)
        sos = (len(gs) - exp_w) / len(gs) if gs else 0
        q = defaultdict(lambda: [0, 0])
        for x in gs: q[x["quad"]][0 if x["won"] else 1] += 1
        W_ = sorted([x for x in gs if x["won"]], key=lambda x: -x["value"])
        L_ = sorted([x for x in gs if not x["won"]], key=lambda x: x["value"])
        resume = sum(WIN_PTS[x["quad"]] if x["won"] else -LOSS_WEIGHT * LOSS_PTS[x["quad"]] for x in gs)
        rows.append(dict(
            team=t, record=f"{wins}-{losses}", rating=ratings[t], power_rank=rank[t],
            resume=resume, score=ratings[t] + RESUME_WEIGHT * resume,
            wab=wins - exp_w, sor_prob=sor_prob, sos=sos,
            win_value=sum(x["value"] for x in W_), loss_cost=-sum(x["value"] for x in L_),
            quads={k: tuple(q[k]) for k in ("Q1", "Q2", "Q3", "Q4")},
            best_wins=W_[:3], worst_losses=L_[:3], games=sorted(gs, key=lambda x: x["week"]),
        ))
    rows.sort(key=lambda r: (-r["wab"], r["sor_prob"]))
    sos_order = sorted(rows, key=lambda r: -r["sos"])
    for i, r in enumerate(rows): r["sor_rank"] = i + 1
    for i, r in enumerate(sos_order): r["sos_rank"] = i + 1
    for i, r in enumerate(sorted(rows, key=lambda r: -r["score"])): r["rank"] = i + 1
    return rows, hfa, fbs_sorted[min(benchmark_rank, len(fbs_sorted)) - 1]


def colley(games):
    """Results-only ranking (Colley Matrix): ratings solved from wins, losses and opponents' ratings
    at once, so strength flows through the whole schedule. Margins and venue are ignored."""
    games = [g for g in games if (g["home_fbs"] or g["away_fbs"]) and g["hp"] != g["ap"]]
    norm = lambda t, fbs: t if fbs else FCS_POOL
    teams = sorted({norm(g["home"], g["home_fbs"]) for g in games} | {norm(g["away"], g["away_fbs"]) for g in games})
    idx = {t: i for i, t in enumerate(teams)}
    C = 2 * np.eye(len(teams)); b = np.ones(len(teams))
    sched = defaultdict(list)
    for g in games:
        h, a = norm(g["home"], g["home_fbs"]), norm(g["away"], g["away_fbs"])
        i, j, hw = idx[h], idx[a], g["hp"] > g["ap"]
        C[i, i] += 1; C[j, j] += 1; C[i, j] -= 1; C[j, i] -= 1
        b[i] += 0.5 if hw else -0.5; b[j] += -0.5 if hw else 0.5
        sched[h].append((a, hw)); sched[a].append((h, not hw))
    r = dict(zip(teams, np.linalg.solve(C, b)))
    fbs = sorted((t for t in teams if t != FCS_POOL), key=lambda t: -r[t])
    rank = {t: i + 1 for i, t in enumerate(fbs)}
    rows = []
    for t in fbs:
        gs = sched[t]
        wins = sorted((o for o, w in gs if w), key=lambda o: -r[o])
        losses = sorted((o for o, w in gs if not w), key=lambda o: r[o])
        rows.append(dict(team=t, rank=rank[t], rating=r[t], record=f"{len(wins)}-{len(losses)}",
                         opp_avg=sum(r[o] for o, _ in gs) / len(gs),
                         best_win=f"#{rank[wins[0]]} {wins[0]}" if wins and wins[0] in rank else (wins[0] if wins else "-"),
                         worst_loss=f"#{rank[losses[0]]} {losses[0]}" if losses and losses[0] in rank else (losses[0] if losses else "-")))
    return rows


# Matchup predictor. Separate from the ranking models, which compress margins: predictions need real points.
PRED_CAP, PRED_RIDGE, PTS_RIDGE, PRED_SD_INFLATE = 35.0, 1.0, 2.0, 1.4
# Passing and rushing efficiency, adjusted for opponents (see efficiency()); gets EFF_WEIGHT of the margin.
EFF_RIDGE, EFF_WEIGHT, EFF_MIN_GAMES = 50.0, 0.25, 50


def adj_od(games, val, ridge):
    """Weighted ridge: y = league avg + offense + opposing defense (+ home). val(g, side) -> (y, weight) or None."""
    norm = lambda t, fbs: t if fbs else FCS_POOL
    teams = sorted({norm(g["home"], g["home_fbs"]) for g in games} | {norm(g["away"], g["away_fbs"]) for g in games})
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    A = np.zeros((2 * n + 2, 2 * n + 2)); b = np.zeros(2 * n + 2)
    for g in games:
        h, a = idx[norm(g["home"], g["home_fbs"])], idx[norm(g["away"], g["away_fbs"])]
        for side in (0, 1):
            v = val(g, side)
            if v is None: continue
            y, w = v; o, d = (h, a) if side == 0 else (a, h)
            cols = [o, n + d, 2 * n] + ([2 * n + 1] if side == 0 and not g["neutral"] else [])
            for i in cols:
                b[i] += w * y
                for j in cols: A[i, j] += w
    A[np.arange(2 * n), np.arange(2 * n)] += ridge
    A[[2 * n, 2 * n + 1], [2 * n, 2 * n + 1]] += 1e-9
    x = np.linalg.solve(A, b)
    return dict(off={t: x[i] for i, t in enumerate(teams)}, defn={t: x[n + i] for i, t in enumerate(teams)},
                mu=x[2 * n], hp=x[2 * n + 1])


def exp_od(M, o, d, home):
    return M["mu"] + M["off"].get(o, 0) + M["defn"].get(d, 0) + (M["hp"] if home else 0)


def efficiency(games):
    """Opponent-adjusted yards per pass attempt and per carry, and points per YPA/YPC from this season's games."""
    bx = [g for g in games if g.get("box")]
    if len(bx) < EFF_MIN_GAMES: return None
    rate = lambda y, att: (y / att, att) if att else None
    ypa = adj_od(bx, lambda g, s: rate(g["box"][4 * s], g["box"][4 * s + 1]), EFF_RIDGE)
    ypc = adj_od(bx, lambda g, s: rate(g["box"][4 * s + 2], g["box"][4 * s + 3]), EFF_RIDGE)
    X = []; y = []
    for g in bx:
        for s in (0, 1):
            pa, ra = g["box"][4 * s + 1], g["box"][4 * s + 3]
            if pa and ra: X.append([1, g["box"][4 * s] / pa, g["box"][4 * s + 2] / ra]); y.append(g["ap"] if s else g["hp"])
    X = np.array(X); coef = np.linalg.solve(X.T @ X, X.T @ np.array(y, float))
    return dict(ypa=ypa, ypc=ypc, coef=coef)


def fit_predictor(games):
    """Margin: ridge fit on point margins capped at PRED_CAP with a home-field term. Score total: ridge fit
    on each side's points from offense and defense terms. Win chance SD = in-sample RMSE x PRED_SD_INFLATE."""
    games = [g for g in games if g["home_fbs"] or g["away_fbs"]]
    norm = lambda t, fbs: t if fbs else FCS_POOL
    teams = sorted({norm(g["home"], g["home_fbs"]) for g in games} | {norm(g["away"], g["away_fbs"]) for g in games})
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    X = np.zeros((len(games), n + 1)); y = np.zeros(len(games))
    Xp = np.zeros((2 * len(games), 2 * n + 2)); yp = np.zeros(2 * len(games))
    for k, g in enumerate(games):
        h, a = idx[norm(g["home"], g["home_fbs"])], idx[norm(g["away"], g["away_fbs"])]
        x = 0 if g["neutral"] else 1
        X[k, h] = 1; X[k, a] = -1; X[k, n] = x; y[k] = max(-PRED_CAP, min(PRED_CAP, g["hp"] - g["ap"]))
        Xp[2 * k, [h, n + a, 2 * n, 2 * n + 1]] = [1, 1, 1, x]; yp[2 * k] = g["hp"]
        Xp[2 * k + 1, [a, n + h, 2 * n]] = 1; yp[2 * k + 1] = g["ap"]
    P = np.eye(n + 1) * PRED_RIDGE; P[n, n] = 0
    m = np.linalg.solve(X.T @ X + P, X.T @ y)
    Pp = np.eye(2 * n + 2) * PTS_RIDGE; Pp[2 * n, 2 * n] = Pp[2 * n + 1, 2 * n + 1] = 0
    q = np.linalg.solve(Xp.T @ Xp + Pp, Xp.T @ yp)
    sd = float(np.sqrt(np.mean((y - X @ m) ** 2)) * PRED_SD_INFLATE)
    T = {t: dict(r=m[i], off=q[i], defn=q[n + i]) for i, t in enumerate(teams)}
    return dict(T=T, hfa=m[n], mu=q[2 * n], hp=q[2 * n + 1], sd=sd, eff=efficiency(games))


def predict(M, a, b, venue="N"):
    """venue: 'N' neutral, 'A' team a at home, 'B' team b at home."""
    A, B = M["T"][a], M["T"][b]; ha, hb = int(venue == "A"), int(venue == "B")
    margin = A["r"] - B["r"] + M["hfa"] * (ha - hb); eff = None
    if M["eff"]:
        E = M["eff"]
        def side(o, d, home):
            y, c = exp_od(E["ypa"], o, d, home), exp_od(E["ypc"], o, d, home)
            return dict(ypa=y, ypc=c, pts=E["coef"][0] + E["coef"][1] * y + E["coef"][2] * c)
        eff = dict(a=side(a, b, ha), b=side(b, a, hb))
        margin = (1 - EFF_WEIGHT) * margin + EFF_WEIGHT * (eff["a"]["pts"] - eff["b"]["pts"])
    pa = 0.5 * (1 + math.erf(margin / (M["sd"] * math.sqrt(2))))
    total = (M["mu"] + A["off"] + B["defn"] + M["hp"] * ha) + (M["mu"] + B["off"] + A["defn"] + M["hp"] * hb)
    total = max(total, abs(margin) + 7)
    sa, sb = round((total + margin) / 2), round((total - margin) / 2)
    if sa == sb:  # no ties in college football
        if margin >= 0: sa += 1
        else: sb += 1
    return dict(margin=margin, pa=pa, sa=sa, sb=sb, eff=eff)


def print_colley(rows, top):
    hdr = f"{'Rk':>3} {'Team':<22}{'Rec':>6}{'Rating':>8}{'OppAvg':>8}  Best win / Worst loss"
    print("\nResults-only ranking (Colley Matrix)\n"); print(hdr); print("-" * len(hdr))
    for r in rows[:top]:
        print(f"{r['rank']:>3} {r['team']:<22}{r['record']:>6}{r['rating']:>8.3f}{r['opp_avg']:>8.3f}  {r['best_win']} | {r['worst_loss']}")


# ---------------------------------------------------------------- output
def fmt_game(x):
    v = {"home": "vs", "road": "@", "neutral": "n-"}[x["venue"]]
    rk = f"#{x['opp_rank']} " if x["opp_rank"] else ""
    return f"{v}{rk}{x['opp']} {x['pf']}-{x['pa']} ({x['value']:+.2f})"


def print_table(rows, hfa, bench_team, top, by="power"):
    print(f"\nEstimated home-field advantage: {hfa:.2f} pts | Outcome SD: {WIN_PROB_SD:.1f} | Benchmark: {bench_team}\n")
    rows = sorted(rows, key=lambda r: r[{"score": "rank", "power": "power_rank", "sor": "sor_rank"}[by]])
    hdr = f"{'Rk':>3} {'Team':<22}{'Rec':>6}{'Score':>7}{'Pwr':>5}{'Resume':>8}{'SOR':>5}{'SOS':>5}  {'Q1':>5}{'Q2':>5}  Best win / Worst loss"
    print(hdr); print("-" * len(hdr))
    for r in rows[:top]:
        q = r["quads"]
        bw = fmt_game(r["best_wins"][0]) if r["best_wins"] else "-"
        wl = fmt_game(r["worst_losses"][0]) if r["worst_losses"] else "-"
        print(f"{r['rank']:>3} {r['team']:<22}{r['record']:>6}{r['score']:>7.1f}{r['power_rank']:>5}"
              f"{r['resume']:>+8.1f}{r['sor_rank']:>5}{r['sos_rank']:>5}  "
              f"{q['Q1'][0]}-{q['Q1'][1]:<3}{q['Q2'][0]}-{q['Q2'][1]:<3} {bw} | {wl}")


def print_team(r):
    print(f"\n{r['team']}  {r['record']}  | Rank #{r['rank']}  Power #{r['power_rank']}  Resume {r['resume']:+.1f}  SOR #{r['sor_rank']}  SOS #{r['sos_rank']}")
    print(f"Wins above benchmark {r['wab']:+.2f} | P(benchmark matches record) {r['sor_prob']*100:.1f}%")
    print(f"Total win value {r['win_value']:.2f} | total loss cost {r['loss_cost']:.2f}\n")
    for x in r["games"]:
        print(f"  Wk{x['week']:>2} {'W' if x['won'] else 'L'} {x['quad']}  {fmt_game(x):<45} bench win% {x['p_bench']*100:4.0f}")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "resume", "sor_rank", "team", "record", "wins_above_benchmark", "sor_prob", "power_rank",
                    "rating", "sos_rank", "sos", "win_value", "loss_cost", "Q1", "Q2", "Q3", "Q4",
                    "best_win", "worst_loss"])
        for r in sorted(rows, key=lambda r: r["rank"]):
            q = r["quads"]
            w.writerow([r["rank"], round(r["score"], 2), round(r["resume"], 2), r["sor_rank"], r["team"], r["record"], round(r["wab"], 3), round(r["sor_prob"], 4),
                        r["power_rank"], round(r["rating"], 2), r["sos_rank"], round(r["sos"], 3),
                        round(r["win_value"], 3), round(r["loss_cost"], 3),
                        *[f"{q[k][0]}-{q[k][1]}" for k in ("Q1", "Q2", "Q3", "Q4")],
                        fmt_game(r["best_wins"][0]) if r["best_wins"] else "",
                        fmt_game(r["worst_losses"][0]) if r["worst_losses"] else ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv"); ap.add_argument("--json", help="the site's data/games.json (includes box scores)")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--benchmark", type=int, default=25)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--rank", choices=("score", "power", "sor", "colley"), default="score",
                    help="order of the table: score (power + resume, default), power, sor, or colley (results only)")
    ap.add_argument("--team"); ap.add_argument("--out")
    ap.add_argument("--predict", nargs=2, metavar=("TEAM1", "TEAM2"), help="predict a matchup")
    ap.add_argument("--venue", choices=("neutral", "team1", "team2"), default="neutral", help="who is at home for --predict")
    a = ap.parse_args()

    if a.json:
        games = load_json(a.json)
    elif a.csv:
        games = load_csv(a.csv)
    else:
        key = os.environ.get("CFBD_API_KEY")
        if not key: sys.exit("Set CFBD_API_KEY or pass --json data/games.json or --csv games.csv")
        games = load_cfbd(a.year, key)
    if not games: sys.exit("No completed games found.")

    if a.predict:
        M = fit_predictor(games); t1, t2 = a.predict
        for t in (t1, t2):
            if t not in M["T"] or t == FCS_POOL: sys.exit(f"Team '{t}' not found.")
        p = predict(M, t1, t2, {"neutral": "N", "team1": "A", "team2": "B"}[a.venue])
        w, wp = (t1, p["pa"]) if p["pa"] >= 0.5 else (t2, 1 - p["pa"])
        where = "neutral site" if a.venue == "neutral" else f"at {t1 if a.venue == 'team1' else t2}"
        print(f"\n{t1} {p['sa']}, {t2} {p['sb']}  ({where})")
        pts = max(1, round(abs(p["margin"])))
        print(f"{w} wins {wp:.0%} of the time by about {pts} point{'' if pts == 1 else 's'}")
        if p["eff"]:
            print(f"Expected yds/att: {t1} {p['eff']['a']['ypa']:.1f}, {t2} {p['eff']['b']['ypa']:.1f} | "
                  f"yds/carry: {t1} {p['eff']['a']['ypc']:.1f}, {t2} {p['eff']['b']['ypc']:.1f}")
        return
    if a.rank == "colley":
        print_colley(colley(games), a.top); return
    rows, hfa, bench = compute(games, a.benchmark)
    print_table(rows, hfa, bench, a.top, a.rank)
    if a.team:
        m = [r for r in rows if r["team"].lower() == a.team.lower()]
        print_team(m[0]) if m else print(f"\nTeam '{a.team}' not found.")
    if a.out:
        write_csv(rows, a.out); print(f"\nFull results written to {a.out}")


if __name__ == "__main__":
    main()
