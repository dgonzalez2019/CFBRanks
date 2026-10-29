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

CONF_COEF = (0.08102, 0.018616, 0.047526, -0.059625, 0.063655)  # const, bad, dis, gap, both
TOP_SHARE, TOP5_SHARE = 0.10, 0.05
TCONF_COEF = (-0.0018, 0.0605, 0.0039, 0.3163, 0.0626, 0.0974)  # const, gap, last, form, pace, high
PACE_AVG = 67.2


def confidence(la, lo, pm, line, ours):
    side = 1 if ours > line else -1
    bad = -((la or 0.0) - (lo or 0.0)) / 14; dis = -side * (pm - line) / 7; gap = abs(ours - line) / 7
    z = CONF_COEF[0] + CONF_COEF[1] * bad + CONF_COEF[2] * dis + CONF_COEF[3] * gap + CONF_COEF[4] * float(bad > 0 and dis > 0)
    return 1 / (1 + math.exp(-z))

def total_confidence(ours, line, last_sum, form_sum, pace):
    s = 1 if ours > line else -1
    x = (abs(ours - line) / 7, -s * last_sum / 14, -s * (form_sum - 1), s * (pace - PACE_AVG) / 10, -s * (line - 55) / 10)
    z = TCONF_COEF[0] + sum(c * v for c, v in zip(TCONF_COEF[1:], x))
    return 1 / (1 + math.exp(-z))

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data"
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

try: track = {g["key"]: g for g in json.loads(OUT.read_text())["games"]}
except (OSError, ValueError, KeyError): track = {}


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
        ours_t = pr["sa"] + pr["sb"]
        if total is not None and ours_t != total:
            (lh, fh, ph), (la_, fa, pa_) = ou_form(home, wk), ou_form(away, wk)
            t.update(total=total, ours_total=ours_t, tconf=round(total_confidence(ours_t, total, lh + la_, fh + fa, (ph + pa_) / 2), 4))
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
        res["ou_model"] = ou; res["ou_top"] = ou if t.get("ttop_conf") else None; res["ou_top5"] = ou if t.get("ttop5") else None
        t.update(home_pts=hp, away_pts=ap, results=res,
                 edges=dict(model_close=round(abs(t["ours"] - t["line"]), 2),
                            model_open=round(abs(t["ours"] - t["open"]), 2) if t["open"] is not None else None))

# Each week, the top 10% (and 5%) of games by confidence. Games that have kicked off keep the flag they had.
for wk in {g["week"] for g in track.values()}:
    wg = [g for g in track.values() if g["week"] == wk and g.get("conf") is not None]
    ranked = sorted(wg, key=lambda g: -g["conf"]); n10 = max(1, round(len(wg) * TOP_SHARE)); n5 = max(1, round(len(wg) * TOP5_SHARE))
    for i, g in enumerate(ranked):
        if "results" in g or (when(g["start"]) and when(g["start"]) <= now and "top_conf" in g): continue
        g["top_conf"] = i < n10; g["top5"] = i < n5
    wt = [g for g in track.values() if g["week"] == wk and g.get("tconf") is not None]
    ranked = sorted(wt, key=lambda g: -g["tconf"]); n10 = max(1, round(len(wt) * TOP_SHARE)); n5 = max(1, round(len(wt) * TOP5_SHARE))
    for i, g in enumerate(ranked):
        if "results" in g or (when(g["start"]) and when(g["start"]) <= now and "ttop_conf" in g): continue
        g["ttop_conf"] = i < n10; g["ttop5"] = i < n5

ordered = sorted(track.values(), key=lambda g: (g["start"] or "", g["key"]))
OUT.write_text(json.dumps({"updated": stamp, "model": "Matchup model with last-season prior, passing/rushing efficiency "
                           "and players listed Out", "games": ordered}, indent=1))
done = [g for g in ordered if "results" in g]
rec = lambda s: f"{sum(g['results'].get(s) == 'W' for g in done)}-{sum(g['results'].get(s) == 'L' for g in done)}"
print(f"tracker: {len(ordered) - len(done)} open, {len(done)} graded; model vs close {rec('model_close')}, "
      f"vs open {rec('model_open')}, favorites at 3/7 {rec('fav37')}, top confidence {rec('top_conf')}, top 5% {rec('top5')}, totals {rec('ou_model')}, totals top {rec('ou_top')}; "
      f"{sum(1 for g in ordered if g.get('top_conf') and 'results' not in g)} open top-confidence picks")
