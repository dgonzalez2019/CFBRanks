#!/usr/bin/env python3
"""Pull completed FBS games from collegefootballdata.com into data/games.json, plus each game's
passing and rushing box score where available.
Usage: CFBD_API_KEY=... python scripts/fetch_games.py [--year 2026] [--out data/games.json] [--lines data/lines.json]"""
import argparse, json, os, sys, datetime, pathlib, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cfb_sor import load_cfbd

ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, default=datetime.date.today().year)
ap.add_argument("--out", help="output file (default: data/games.json)")
ap.add_argument("--lines", help="also write betting lines for the season's games (played and upcoming) here")
ap.add_argument("--players", help="also write player value (season PPA per play, offensive players) here")
a = ap.parse_args()
key = os.environ.get("CFBD_API_KEY") or sys.exit("CFBD_API_KEY is not set")
upcoming = []
games = load_cfbd(a.year, key, upcoming)
if not games: sys.exit("No completed games returned; leaving data/games.json unchanged.")
path = pathlib.Path(a.out) if a.out else pathlib.Path(__file__).resolve().parent.parent / "data" / "games.json"
path.parent.mkdir(parents=True, exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def num(stats, *names):
    for n in names:
        if n in stats:
            try: return float(str(stats[n]).strip())
            except ValueError: return None
    return None


def pass_att(stats):
    ca = stats.get("completionAttempts")  # "completions-attempts", e.g. "21-33"
    try: return float(str(ca).replace("/", "-").split("-")[1])
    except (IndexError, ValueError): return None


def side_stats(t):
    stats = {s.get("category"): s.get("stat") for s in t.get("stats", [])}
    vals = [num(stats, "netPassingYards"), pass_att(stats), num(stats, "rushingYards"), num(stats, "rushingAttempts")]
    return None if None in vals else [round(v) for v in vals]


# Box scores already stored are kept, so each run only asks the API for weeks that are missing some.
# Stats per game: [home pass yds, home pass att, home rush yds, home rush att, same four for away].
key_of = lambda w, h, aw: f"{w}|{h}|{aw}"
stored = {}
try:
    for g in json.loads(path.read_text()).get("games", []):
        if len(g) > 8 and g[8]: stored[key_of(g[0], g[1], g[2])] = g[8]
except (OSError, ValueError):
    pass

missing_weeks = sorted({g["week"] for g in games if key_of(g["week"], g["home"], g["away"]) not in stored})
for wk in missing_weeks:
    seas, raw = ("postseason", wk - 20) if wk > 20 else ("regular", wk)
    try:
        box = get(f"https://api.collegefootballdata.com/games/teams?year={a.year}&week={raw}&seasonType={seas}")
    except Exception as e:  # stats are optional; never block the score update
        print(f"Box scores for week {wk} unavailable: {e}"); continue
    found = 0
    for b in box:
        sides = {}
        for t in b.get("teams", []):
            sides[t.get("homeAway", t.get("home_away"))] = (t.get("team", t.get("school")), side_stats(t))
        if "home" not in sides or "away" not in sides or None in (sides["home"][1], sides["away"][1]):
            continue
        stored[key_of(wk, sides["home"][0], sides["away"][0])] = sides["home"][1] + sides["away"][1]; found += 1
    print(f"Week {wk}: {len(box)} box scores returned, {found} usable")
    if box and not found:
        print("  First box score, for debugging:", json.dumps(box[0])[:600])

rows = []
for g in games:
    row = [g["week"], g["home"], g["away"], g["hp"], g["ap"], int(g["neutral"]), int(g["home_fbs"]), int(g["away_fbs"])]
    st = stored.get(key_of(g["week"], g["home"], g["away"]))
    if st: row.append(st)
    rows.append(row)
out = {"season": a.year, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "games": rows}
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"Wrote {len(rows)} games ({sum(len(r) > 8 for r in rows)} with box scores) to {path}")


# Betting lines: one /lines call per season type covers every game. Spreads use the API's convention
# (home team's spread, negative = home favored); we store the median across sportsbooks.
if a.lines:
    def median(xs):
        xs = sorted(xs); n = len(xs)
        return None if not n else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    def num_or_none(v):
        try: return float(v)
        except (TypeError, ValueError): return None
    info = {key_of(g["week"], g["home"], g["away"]): g for g in games + upcoming}
    seasons = ["regular"] + (["postseason"] if any(g["week"] > 20 for g in games + upcoming) else [])
    out_lines = []
    for seas in seasons:
        try:
            L = get(f"https://api.collegefootballdata.com/lines?year={a.year}&seasonType={seas}")
        except Exception as e:
            print(f"Lines ({seas}) unavailable: {e}"); continue
        for item in L:
            wk = item.get("week", 0) + (20 if seas == "postseason" else 0)
            h = item.get("homeTeam", item.get("home_team")); aw = item.get("awayTeam", item.get("away_team"))
            g = info.get(key_of(wk, h, aw))
            if not g: continue
            books = item.get("lines") or []
            spread = median([x for x in (num_or_none(b.get("spread")) for b in books) if x is not None])
            total = median([x for x in (num_or_none(b.get("overUnder", b.get("over_under"))) for b in books) if x is not None])
            if spread is None: continue
            out_lines.append([wk, h, aw, int(g["neutral"]), spread, total, g.get("hp"), g.get("ap"),
                              g.get("start"), sorted({b.get("provider") for b in books if b.get("provider")})])
        print(f"Lines ({seas}): {len(L)} games returned, {len(out_lines)} matched so far")
        if L and not out_lines: print("  First line item, for debugging:", json.dumps(L[0])[:600])
    lp = pathlib.Path(a.lines); lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({"season": a.year, "updated": out["updated"], "games": out_lines}, separators=(",", ":")))
    print(f"Wrote {len(out_lines)} lines ({sum(r[6] is None for r in out_lines)} upcoming) to {lp}")


# Player value for the injury picker: one /ppa/players/season call. PPA = predicted points added per play,
# so (player's PPA - a typical backup's) x his plays per game is roughly his worth in points per game.
# Stored per player: [team, name, position, countable plays, average PPA per play (all plays)].
if a.players:
    try:
        P = get(f"https://api.collegefootballdata.com/ppa/players/season?year={a.year}&excludeGarbageTime=true")
        rows = []
        for x in P:
            avg = x.get("averagePPA", x.get("average_ppa")) or {}
            plays = x.get("countablePlays", x.get("countable_plays"))
            val = avg.get("all") if isinstance(avg, dict) else None
            if x.get("team") and x.get("name") and plays and val is not None and plays >= 10:
                rows.append([x["team"], x["name"], x.get("position") or "", int(plays), round(float(val), 4)])
        print(f"Players: {len(P)} returned, {len(rows)} kept")
        if P and not rows: print("  First player item, for debugging:", json.dumps(P[0])[:600])
        if rows:
            pp = pathlib.Path(a.players); pp.parent.mkdir(parents=True, exist_ok=True)
            pp.write_text(json.dumps({"season": a.year, "updated": out["updated"], "players": rows}, separators=(",", ":")))
    except Exception as e:
        print(f"Player values unavailable: {e}")
