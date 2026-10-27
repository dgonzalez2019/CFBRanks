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


def load_cfbd(year, key):
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
            if hp is None or ap is None:
                continue
            h = g.get("homeTeam", g.get("home_team")); a = g.get("awayTeam", g.get("away_team"))
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


def quadrant(opp_rank, venue):
    cut = {"home": (15, 40, 80), "neutral": (25, 55, 95), "road": (40, 75, 110)}[venue]
    if opp_rank is None: return "Q4"
    for q, c in zip(("Q1", "Q2", "Q3"), cut):
        if opp_rank <= c: return q
    return "Q4"


def compute(games, benchmark_rank=25):
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
                opp_rank=rank.get(opp), quad=quadrant(rank.get(opp), venue),
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
        rows.append(dict(
            team=t, record=f"{wins}-{losses}", rating=ratings[t], power_rank=rank[t],
            wab=wins - exp_w, sor_prob=sor_prob, sos=sos,
            win_value=sum(x["value"] for x in W_), loss_cost=-sum(x["value"] for x in L_),
            quads={k: tuple(q[k]) for k in ("Q1", "Q2", "Q3", "Q4")},
            best_wins=W_[:3], worst_losses=L_[:3], games=sorted(gs, key=lambda x: x["week"]),
        ))
    rows.sort(key=lambda r: (-r["wab"], r["sor_prob"]))
    sos_order = sorted(rows, key=lambda r: -r["sos"])
    for i, r in enumerate(rows): r["sor_rank"] = i + 1
    for i, r in enumerate(sos_order): r["sos_rank"] = i + 1
    return rows, hfa, fbs_sorted[min(benchmark_rank, len(fbs_sorted)) - 1]


# ---------------------------------------------------------------- output
def fmt_game(x):
    v = {"home": "vs", "road": "@", "neutral": "n-"}[x["venue"]]
    rk = f"#{x['opp_rank']} " if x["opp_rank"] else ""
    return f"{v}{rk}{x['opp']} {x['pf']}-{x['pa']} ({x['value']:+.2f})"


def print_table(rows, hfa, bench_team, top):
    print(f"\nEstimated home-field advantage: {hfa:.2f} pts | Outcome SD: {WIN_PROB_SD:.1f} | Benchmark: {bench_team}\n")
    hdr = f"{'SOR':>3} {'Team':<22}{'Rec':>6}{'WAB':>7}{'SOR%':>8}{'Pwr':>5}{'SOS':>5}  {'Q1':>5}{'Q2':>5}  Best win / Worst loss"
    print(hdr); print("-" * len(hdr))
    for r in rows[:top]:
        q = r["quads"]
        bw = fmt_game(r["best_wins"][0]) if r["best_wins"] else "-"
        wl = fmt_game(r["worst_losses"][0]) if r["worst_losses"] else "-"
        print(f"{r['sor_rank']:>3} {r['team']:<22}{r['record']:>6}{r['wab']:>+7.2f}"
              f"{r['sor_prob']*100:>7.1f}%{r['power_rank']:>5}{r['sos_rank']:>5}  "
              f"{q['Q1'][0]}-{q['Q1'][1]:<3}{q['Q2'][0]}-{q['Q2'][1]:<3} {bw} | {wl}")


def print_team(r):
    print(f"\n{r['team']}  {r['record']}  | SOR #{r['sor_rank']}  Power #{r['power_rank']}  SOS #{r['sos_rank']}")
    print(f"Wins above benchmark {r['wab']:+.2f} | P(benchmark matches record) {r['sor_prob']*100:.1f}%")
    print(f"Total win value {r['win_value']:.2f} | total loss cost {r['loss_cost']:.2f}\n")
    for x in r["games"]:
        print(f"  Wk{x['week']:>2} {'W' if x['won'] else 'L'} {x['quad']}  {fmt_game(x):<45} bench win% {x['p_bench']*100:4.0f}")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sor_rank", "team", "record", "wins_above_benchmark", "sor_prob", "power_rank",
                    "rating", "sos_rank", "sos", "win_value", "loss_cost", "Q1", "Q2", "Q3", "Q4",
                    "best_win", "worst_loss"])
        for r in rows:
            q = r["quads"]
            w.writerow([r["sor_rank"], r["team"], r["record"], round(r["wab"], 3), round(r["sor_prob"], 4),
                        r["power_rank"], round(r["rating"], 2), r["sos_rank"], round(r["sos"], 3),
                        round(r["win_value"], 3), round(r["loss_cost"], 3),
                        *[f"{q[k][0]}-{q[k][1]}" for k in ("Q1", "Q2", "Q3", "Q4")],
                        fmt_game(r["best_wins"][0]) if r["best_wins"] else "",
                        fmt_game(r["worst_losses"][0]) if r["worst_losses"] else ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv"); ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--benchmark", type=int, default=25)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--team"); ap.add_argument("--out")
    a = ap.parse_args()

    if a.csv:
        games = load_csv(a.csv)
    else:
        key = os.environ.get("CFBD_API_KEY")
        if not key: sys.exit("Set CFBD_API_KEY or pass --csv games.csv")
        games = load_cfbd(a.year, key)
    if not games: sys.exit("No completed games found.")

    rows, hfa, bench = compute(games, a.benchmark)
    print_table(rows, hfa, bench, a.top)
    if a.team:
        m = [r for r in rows if r["team"].lower() == a.team.lower()]
        print_team(m[0]) if m else print(f"\nTeam '{a.team}' not found.")
    if a.out:
        write_csv(rows, a.out); print(f"\nFull results written to {a.out}")


if __name__ == "__main__":
    main()
