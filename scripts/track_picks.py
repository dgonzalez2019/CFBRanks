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

data/tracker.json = {updated, model, games: [{key, week, home, away, neutral, start, ours, line, open, inj_home,
                     inj_away, first_saved, saved_at, home_pts, away_pts, results: {strategy: "W"|"L"|"P"|null},
                     edges: {model_close, model_open}}]}"""
import datetime, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import cfb_sor as m

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
        ours = m.predict(M, home, away, "N" if neutral else "A", ih, ia)["margin"]
        if t is None: t = track[key] = dict(key=key, week=wk, home=home, away=away, neutral=bool(neutral), start=start, first_saved=stamp)
        t.update(ours=round(ours, 2), line=-spread, open=(-opening if opening is not None else None),
                 inj_home=round(ih, 2), inj_away=round(ia, 2), saved_at=stamp)
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
        t.update(home_pts=hp, away_pts=ap, results=res,
                 edges=dict(model_close=round(abs(t["ours"] - t["line"]), 2),
                            model_open=round(abs(t["ours"] - t["open"]), 2) if t["open"] is not None else None))

ordered = sorted(track.values(), key=lambda g: (g["start"] or "", g["key"]))
OUT.write_text(json.dumps({"updated": stamp, "model": "Matchup model with last-season prior, passing/rushing efficiency "
                           "and players listed Out", "games": ordered}, indent=1))
done = [g for g in ordered if "results" in g]
rec = lambda s: f"{sum(g['results'][s] == 'W' for g in done)}-{sum(g['results'][s] == 'L' for g in done)}"
print(f"tracker: {len(ordered) - len(done)} open, {len(done)} graded; model vs close {rec('model_close')}, "
      f"vs open {rec('model_open')}, favorites at 3/7 {rec('fav37')}")
