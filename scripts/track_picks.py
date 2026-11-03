#!/usr/bin/env python3
"""Live tracker: saves our pre-game prediction for each game kicking off within WINDOW_DAYS, grades it after the
final, and keeps a record for strategies fixed in advance (no betting; this is how the model earns trust).

Each run (after fetch_games.py --lines --players and fetch_injuries.py):
  - upcoming games within WINDOW_DAYS: (re)compute our margin with the Matchup tab's model (last season as a
    starting point, passing/rushing efficiency, players listed Out) and store it with the current and opening lines;
  - games that have kicked off: frozen, nothing changes;
  - finished games: graded for each strategy against the frozen lines.

Strategies (home margin convention: line = -home spread):
  model_close  our side of the last line saved before kickoff (edge = |ours - line|)
  model_open   our side of the opening line
  fav37        the favorite when the line is within 0.5 of 3 or 7 (best NIL-era angle in research, 53.9%)
  top_conf     TOP CONFIDENCE: the top 10% of each week's games by confidence score (top5 marks the top 5%)
  ou_model     our side of the closing total (over when our predicted total is above it); ou_top / ou_top5 the top
               10% / 5% of each week's games by totals confidence

Confidence score = probability our pick covers, from a logistic model (research/conf_score.py, fit on 2022-25) on:
  bad  = how much worse our pick did against the spread last game than its opponent (no lined last game = 0) / 14
  dis  = how far the play-efficiency model sits on the other side of the line from our pick / 7
  gap  = |our margin - line| / 7
  both = bad > 0 and dis > 0
Held out, the top 10% went 178-148 (54.6%) over 2022-26; games scored 54-56% won 55.1% in 2025-26.
top_conf and top5 are graded like model_close.

Totals confidence (tconf) = probability our side of the total wins, logistic (research/conf_total.py, fit on 2022-25) on
  gap  = |our total - line| / 7, last = -s * (both teams' last-game total minus its line) / 14,
  form = -s * (both teams' season over rates, shrunk by 2 games, - 1), pace = s * (avg plays per game - 67.2) / 10,
  high = -s * (line - 55) / 10, with s = +1 for the over. Held out, the top 10% went 169-138 (55.0%) over 2022-26 but
  only 47%, 47%, 72%, 59% and 44% by season: much less steady than the spread score.

data/tracker.json = {updated, model, games: [{key, week, home, away, neutral, start, ours, line, open, inj_home,
                     inj_away, first_saved, saved_at, home_pts, away_pts, results: {strategy: "W"|"L"|"P"|null},
                     edges: {model_close, model_open}}]}"""
import datetime, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cfb_sor as m
import math

# --league nfl tracks the NFL from data/nfl/ (research/nfl_conf.py: spread top 10% 134-119, 53.0%, 2016-25 held out;
# totals showed no edge, 122-128). CFB is the default.
LEAGUE = sys.argv[sys.argv.index("--league") + 1] if "--league" in sys.argv else "cfb"
m.use_league(LEAGUE)
CONF_COEF = (0.08102, 0.018616, 0.047526, -0.059625, 0.063655)  # const, bad, dis, gap, both
TOP_SHARE, TOP5_SHARE = 0.10, 0.05
TCONF_COEF = (-0.0018, 0.0605, 0.0039, 0.3163, 0.0626, 0.0974)  # const, gap, last, form, pace, high
PACE_AVG, TOTAL_MID = 67.2, 55.0
if LEAGUE == "nfl":
    CONF_COEF = (-0.0037, 0.0217, 0.2282, 0.1098, 0.0618)
    TCONF_COEF = (-0.0085, -0.1109, -0.0118, 0.2351, -0.1735, -0.0458)
    PACE_AVG, TOTAL_MID = 60.8, 45.0
    m.PPA_A, m.PPA_B = 41.734, 1.922  # points per EPA-per-play margin, fit on 2016-25


def confidence(la, lo, pm, line, ours):
    side = 1 if ours > line else -1
    bad = -((la or 0.0) - (lo or 0.0)) / 14; dis = -side * (pm - line) / 7; gap = abs(ours - line) / 7
    z = CONF_COEF[0] + CONF_COEF[1] * bad + CONF_COEF[2] * dis + CONF_COEF[3] * gap + CONF_COEF[4] * float(bad > 0 and dis > 0)
    return 1 / (1 + math.exp(-z))

def total_confidence(ours, line, last_sum, form_sum, pace):
    s = 1 if ours > line else -1
    x = (abs(ours - line) / 7, -s * last_sum / 14, -s * (form_sum - 1), s * (pace - PACE_AVG) / 10, -s * (line - TOTAL_MID) / 10)
    z = TCONF_COEF[0] + sum(c * v for c, v in zip(TCONF_COEF[1:], x))
    return 1 / (1 + math.exp(-z))

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data" / ("nfl" if LEAGUE == "nfl" else "")

# Totals confidence = an ensemble (research/ensemble.py): six one-signal models of whether the game goes over (our
# predicted total vs the line, both teams' last game and season over rates, pace, size of the total, and the
# multi-signal score above as one more member), averaged with weights from each model's log loss on every earlier
# game, research seasons plus every game graded here since (w ~ exp(-eta * (loss - best loss)); eta from research).
# Held out it beat the multi-signal score alone: NFL top 10% 118-89 (57.0%) vs 113-94, CFB 95-84 (53.1%) vs 88-91.
try: ENS = json.loads((ROOT / "research" / f"ensemble_{LEAGUE}.json").read_text())["total"]["live"]
except (OSError, ValueError, KeyError): ENS = None
# The spread models from the same research, shown per game on the site ("what each model picked"); the spread
# confidence stays the multi-signal score above, since the ensemble didn't beat it on spreads.
try: SENS = json.loads((ROOT / "research" / f"ensemble_{LEAGUE}.json").read_text())["spread"]["live"]
except (OSError, ValueError, KeyError): SENS = None


def ensemble_probs(ours_t, total, last_sum, form_sum, pace, comp_over):
    """Each totals model's chance the game goes over."""
    x = dict(gap=ours_t - total, last=last_sum, form=form_sum - 1, pace=pace - ENS["pace_avg"], high=total - ENS["mid"])
    p = {n: 1 / (1 + math.exp(-e["b"] * x[n] / e["sd"])) for n, e in ENS["experts"].items()}
    p["composite"] = comp_over
    return p


def ensemble_weights(track):
    """Weights from research losses plus every graded tracked game that stored its models' chances."""
    L = dict(ENS["loss"])
    for g in track.values():
        if not g.get("tens") or "results" not in g or g.get("total") is None: continue
        pts = g["home_pts"] + g["away_pts"]
        if pts == g["total"]: continue
        y = 1.0 if pts > g["total"] else 0.0
        for n, q in g["tens"].items():
            q = min(max(q, 1e-6), 1 - 1e-6); L[n] = L.get(n, 0.0) - (y * math.log(q) + (1 - y) * math.log(1 - q))
    lo = min(L.values()); w = {n: math.exp(-ENS["eta"] * (v - lo)) for n, v in L.items()}; s = sum(w.values())
    return {n: v / s for n, v in w.items()}
OUT = D / "tracker.json"
WINDOW_DAYS = 7
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.isoformat(timespec="seconds")
when = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def load(path, key):
    try: return json.loads(path.read_text())[key]
    except (OSError, ValueError, KeyError): return []


games = m.load_json(D / "games.json")
season = json.loads((D / "games.json").read_text())["season"]
lines = load(D / "lines.json", "games")
prev_path = D / "history" / f"{season - 1}.json"
prior = m.season_prior(m.load_json(prev_path)) if prev_path.exists() else None
M = m.fit_predictor(games, prior)
fbs = [g for g in games if g["home_fbs"] or g["away_fbs"]]
values = m.player_values(load(D / "players.json", "players"), fbs)
values = m.estimate_values(values, load(D / "injuries.json", "injuries"), load(D / "roles.json", "players"), LEAGUE)
inj = m.injury_adjustments(values, load(D / "injuries.json", "injuries"), games, lines)
fbs_teams = {g["home"] for g in games if g["home_fbs"]} | {g["away"] for g in games if g["away_fbs"]}
cur_ppa = load(D / "ppa.json", "games") or load(D / "history" / f"{season}_ppa.json", "games")
talent = dict(load(D / "history" / f"{season}_talent.json", "talent"))
P = m.ppa_model(cur_ppa, load(D / "history" / f"{season - 1}_ppa.json", "games"), talent, fbs_teams)

# Each team's result against the line in its most recent game (by week, any opponent); 0 when that game had no
# line, as in research/situational.py.
line_of = {(r[0], r[1], r[2]): r[4] for r in lines if r[4] is not None}
def last_ats(team, week):
    prev = [g for g in games if g["week"] < week and team in (g["home"], g["away"])]
    if not prev: return None
    g = max(prev, key=lambda g: g["week"]); sp = line_of.get((g["week"], g["home"], g["away"]))
    if sp is None: return 0.0
    hp, ap = g.get("official") or (g["hp"], g["ap"])  # betting results use the official score (data/overrides.json aside)
    cover = (hp - ap) + sp  # home margin minus expected home margin (-spread)
    return cover if team == g["home"] else -cover

# Totals: each team's games before a week, with the actual total minus the closing total, and plays per game.
total_of = {(r[0], r[1], r[2]): r[5] for r in lines if r[5] is not None}
def ou_form(team, week):
    prev = sorted((g for g in games if g["week"] < week and team in (g["home"], g["away"])), key=lambda g: g["week"])
    ou = [sum(g.get("official") or (g["hp"], g["ap"])) - total_of[(g["week"], g["home"], g["away"])] for g in prev if (g["week"], g["home"], g["away"]) in total_of]
    plays = [g["box"][4 * s + 1] + g["box"][4 * s + 3] for g in prev if g.get("box") for s in (0 if g["home"] == team else 1,)]
    return (ou[-1] if ou else 0.0), (sum(v > 0 for v in ou) + 1) / (len(ou) + 2), (sum(plays) / len(plays) if plays else PACE_AVG)

start_of = {(r[0], r[1], r[2]): r[8] for r in lines if r[8]}


def spread_form(team, week, start):
    """(last game's margin, season cover rate shrunk by 1/2 games, rest days) for the spread models."""
    prev = sorted((g for g in games if g["week"] < week and team in (g["home"], g["away"])), key=lambda g: g["week"])
    margin = 0.0; covers = []
    for g in prev:
        hp, ap = g["hp"], g["ap"]; mg = hp - ap if team == g["home"] else ap - hp; margin = mg
        sp = line_of.get((g["week"], g["home"], g["away"]))
        if sp is not None:
            ohp, oap = g.get("official") or (hp, ap); c = (ohp - oap) + sp
            covers.append((c if team == g["home"] else -c) > 0)
    rest = SENS["rest_default"]
    if prev and start:
        d0 = start_of.get((prev[-1]["week"], prev[-1]["home"], prev[-1]["away"]))
        if d0: rest = min((when(start) - when(d0)).days, 21)
    return margin, (sum(covers) + 1) / (len(covers) + 2), rest


def spread_probs(home, away, wk, start, line, ours, pm, opening, neutral, conf):
    """Each spread model's chance the home team covers."""
    mh, ch, rh = spread_form(home, wk, start); ma, ca, ra = spread_form(away, wk, start)
    x = dict(model=ours - line, eff=(pm - line) if pm is not None else 0.0, last_ats=(last_ats(home, wk) or 0.0) - (last_ats(away, wk) or 0.0),
             move=(line - opening) if opening is not None else 0.0, rest=rh - ra, fav=line, home=0.0 if neutral else 1.0,
             last_margin=mh - ma, cover_form=ch - ca)
    p = {n: 1 / (1 + math.exp(-e["b"] * x[n] / e["sd"])) for n, e in SENS["experts"].items()}
    if conf is not None: p["composite"] = conf if ours > line else 1 - conf
    return p


try: track = {g["key"]: g for g in json.loads(OUT.read_text())["games"]}
except (OSError, ValueError, KeyError): track = {}
EW = ensemble_weights(track) if ENS else None


def grade(margin, line):
    return None if line is None else "W" if margin > line else "L" if margin < line else "P"


for r in lines:
    wk, home, away, neutral, spread, total, hp, ap, start = r[:9]
    opening = r[10] if len(r) > 10 else None
    key = f"{wk}|{home}|{away}"; t = track.get(key); k = when(start)
    if hp is None and k and now < k <= now + datetime.timedelta(days=WINDOW_DAYS):
        if home not in M["T"] or away not in M["T"] or spread is None: continue
        ih, ia = inj.get(home, 0.0), inj.get(away, 0.0)
        pr = m.predict(M, home, away, "N" if neutral else "A", ih, ia); ours = pr["margin"]
        if t is None: t = track[key] = dict(key=key, week=wk, home=home, away=away, neutral=bool(neutral), start=start, first_saved=stamp)
        line = -spread; pick = home if ours > line else away
        opp = away if pick == home else home
        pm = m.ppa_margin(P, home, away); la, lo = last_ats(pick, wk), last_ats(opp, wk)
        conf = confidence(la, lo, pm, line, ours) if ours != line else None
        t.update(ours=round(ours, 2), line=line, open=(-opening if opening is not None else None),
                 inj_home=round(ih, 2), inj_away=round(ia, 2), ppa_margin=round(pm, 2),
                 pick_last_ats=None if la is None else round(la, 1), opp_last_ats=None if lo is None else round(lo, 1),
                 conf=None if conf is None else round(conf, 4), saved_at=stamp)
        if SENS: t["sens"] = {n: round(v, 4) for n, v in spread_probs(home, away, wk, start, line, ours, pm, t["open"], neutral, conf).items()}
        ours_t = pr["sa"] + pr["sb"]
        if total is not None and ours_t != total:
            (lh, fh, ph), (la_, fa, pa_) = ou_form(home, wk), ou_form(away, wk)
            tc = total_confidence(ours_t, total, lh + la_, fh + fa, (ph + pa_) / 2)
            t.update(total=total, ours_total=ours_t, tconf=round(tc, 4),
                     tfeat=dict(last_h=round(lh, 1), last_a=round(la_, 1), form_h=round(fh, 3), form_a=round(fa, 3), pace=round((ph + pa_) / 2, 1)))
            if ENS:   # the ensemble decides the side and the confidence; the multi-signal score is kept as tconf_model
                ps = ensemble_probs(ours_t, total, lh + la_, fh + fa, (ph + pa_) / 2, tc if ours_t > total else 1 - tc)
                pe = sum(EW.get(n, 0.0) * v for n, v in ps.items())
                t.update(tconf_model=round(tc, 4), tconf=round(max(pe, 1 - pe), 4), tside=1 if pe >= 0.5 else -1,
                         tens={n: round(v, 4) for n, v in ps.items()}, tw={n: round(EW.get(n, 0.0), 4) for n in ps})
        else: t.update(total=total, ours_total=ours_t, tconf=None)
    elif hp is not None and t is not None and "results" not in t:
        act = hp - ap
        side = lambda line: None if line is None or t["ours"] == line else (1 if t["ours"] > line else -1)
        res = {}
        for name, line in (("model_close", t["line"]), ("model_open", t["open"])):
            s = side(line); g = grade(act, line)
            res[name] = None if s is None or g is None else g if g == "P" else ("W" if (g == "W") == (s > 0) else "L")
        fav37 = None
        if t["line"] and any(abs(abs(t["line"]) - k3) <= 0.5 for k3 in (3, 7)):
            g = grade(act, t["line"]); fav37 = g if g == "P" else ("W" if (g == "W") == (t["line"] > 0) else "L")
        res["fav37"] = fav37
        res["top_conf"] = res["model_close"] if t.get("top_conf") else None
        res["top5"] = res["model_close"] if t.get("top5") else None
        tl = t.get("total")
        ou = None if tl is None or t.get("ours_total") in (None, tl) or hp + ap == tl else ("W" if (hp + ap > tl) == (t["ours_total"] > tl) else "L")
        # the flagged totals pick is the ensemble's side when there is one (older games: our model's side)
        ens = ou if t.get("tside") is None or tl is None or hp + ap == tl else ("W" if (hp + ap > tl) == (t["tside"] > 0) else "L")
        res["ou_model"] = ou; res["ou_ens"] = ens
        res["ou_top"] = ens if t.get("ttop_conf") else None; res["ou_top5"] = ens if t.get("ttop5") else None
        t.update(home_pts=hp, away_pts=ap, results=res,
                 edges=dict(model_close=round(abs(t["ours"] - t["line"]), 2),
                            model_open=round(abs(t["ours"] - t["open"]), 2) if t["open"] is not None else None))

# Each week, the top 10% (and 5%) of games by confidence. Games that have kicked off keep the flag they had; the
# week's remaining slots (its quota minus flags already locked in) go to the best-rated games not yet started. Ranking
# the whole week instead let a later game hold a slot until earlier games had locked in unflagged and then lose it,
# leaving the week with no flagged pick (NFL week 8, 2026).
def flag_week(games, score, f10, f5):
    games = [g for g in games if g.get(score) is not None]
    started = lambda g: "results" in g or (when(g["start"]) and when(g["start"]) <= now and f10 in g)
    n10 = max(1, round(len(games) * TOP_SHARE)); n5 = max(1, round(len(games) * TOP5_SHARE))
    locked = [g for g in games if started(g)]
    left10 = max(0, n10 - sum(bool(g.get(f10)) for g in locked)); left5 = max(0, n5 - sum(bool(g.get(f5)) for g in locked))
    for i, g in enumerate(sorted((g for g in games if not started(g)), key=lambda g: -g[score])):
        g[f10] = i < left10; g[f5] = i < left5


for wk in {g["week"] for g in track.values()}:
    wg = [g for g in track.values() if g["week"] == wk]
    flag_week(wg, "conf", "top_conf", "top5")
    flag_week(wg, "tconf", "ttop_conf", "ttop5")

ordered = sorted(track.values(), key=lambda g: (g["start"] or "", g["key"]))
OUT.write_text(json.dumps({"updated": stamp, "model": "Matchup model with last-season prior, passing/rushing efficiency "
                           "and players listed Out", "games": ordered}, indent=1))
done = [g for g in ordered if "results" in g]
rec = lambda s: f"{sum(g['results'].get(s) == 'W' for g in done)}-{sum(g['results'].get(s) == 'L' for g in done)}"
print(f"tracker: {len(ordered) - len(done)} open, {len(done)} graded; model vs close {rec('model_close')}, "
      f"vs open {rec('model_open')}, favorites at 3/7 {rec('fav37')}, top confidence {rec('top_conf')}, top 5% {rec('top5')}, totals {rec('ou_model')}, totals top {rec('ou_top')}; "
      f"{sum(1 for g in ordered if g.get('top_conf') and 'results' not in g)} open top-confidence picks")
