#!/usr/bin/env python3
"""Pull completed FBS games from collegefootballdata.com into data/games.json, plus each game's
passing and rushing box score where available.
Usage: CFBD_API_KEY=... python scripts/fetch_games.py [--year 2026] [--out data/games.json]"""
import argparse, json, os, sys, datetime, pathlib, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cfb_sor import load_cfbd

ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, default=datetime.date.today().year)
ap.add_argument("--out", help="output file (default: data/games.json)")
a = ap.parse_args()
key = os.environ.get("CFBD_API_KEY") or sys.exit("CFBD_API_KEY is not set")
games = load_cfbd(a.year, key)
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
