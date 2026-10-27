#!/usr/bin/env python3
"""Pull completed FBS games from collegefootballdata.com into data/games.json.
Usage: CFBD_API_KEY=... python scripts/fetch_games.py [--year 2026]"""
import argparse, json, os, sys, datetime, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cfb_sor import load_cfbd

ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, default=datetime.date.today().year)
a = ap.parse_args()
key = os.environ.get("CFBD_API_KEY") or sys.exit("CFBD_API_KEY is not set")
games = load_cfbd(a.year, key)
if not games: sys.exit("No completed games returned; leaving data/games.json unchanged.")
out = {"season": a.year, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "games": [[g["week"], g["home"], g["away"], g["hp"], g["ap"], int(g["neutral"]), int(g["home_fbs"]), int(g["away_fbs"])] for g in games]}
path = pathlib.Path(__file__).resolve().parent.parent / "data" / "games.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"Wrote {len(games)} games to {path}")
