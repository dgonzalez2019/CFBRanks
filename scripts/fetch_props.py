#!/usr/bin/env python3
"""NFL player-prop lines from The Odds API (https://the-odds-api.com; secret ODDS_API_KEY, free plan = 500 credits a
month). Each game's props cost one credit per market (4: passing, rushing and receiving yards, receptions), so each
game is fetched when it kicks off within FETCH_HOURS and again within REFRESH_HOURS of kickoff only while more than
RESERVE credits remain: about 120-250 credits a week. Without the key it does nothing.

Writes data/nfl/props_lines.json = {updated, remaining, games: {event id: {start, home, away, fetched: [ISO...],
props: [[market, player, line, over price, under price, books]]}}} with each player's median line across books and
the median prices at that line (American odds). Games that kicked off more than a week ago are dropped.
Usage: python scripts/fetch_props.py"""
import datetime, json, os, pathlib, statistics, sys, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from props_model import MARKETS

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nfl" / "props_lines.json"
API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
FETCH_HOURS, REFRESH_HOURS, RESERVE = 36, 6, 200
KEYS = {v[2]: k for k, v in MARKETS.items()}  # Odds API market -> ours


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cfbranks-data"}), timeout=30) as r:
        return json.load(r), r.headers.get("x-requests-remaining")


def summarize(ev):
    """Median line per (market, player) across books, with the median over/under prices among books at that line."""
    by = {}
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            mk = KEYS.get(m.get("key"))
            if not mk: continue
            sides = {}
            for o in m.get("outcomes", []):
                if o.get("point") is None or not o.get("description"): continue
                sides.setdefault(o["description"], {})[o["name"]] = (o["point"], o["price"])
            for player, s in sides.items():
                if "Over" in s and "Under" in s and s["Over"][0] == s["Under"][0]:
                    by.setdefault((mk, player), []).append((s["Over"][0], s["Over"][1], s["Under"][1]))
    out = []
    for (mk, player), xs in sorted(by.items()):
        line = statistics.median_low([x[0] for x in xs]); at = [x for x in xs if x[0] == line]
        out.append([mk, player, line, round(statistics.median(x[1] for x in at)), round(statistics.median(x[2] for x in at)), len(xs)])
    return out


def main():
    key = os.environ.get("ODDS_API_KEY")
    if not key: print("ODDS_API_KEY is not set; skipping player props."); return
    store = json.loads(OUT.read_text()) if OUT.exists() else {"games": {}}
    now = datetime.datetime.now(datetime.timezone.utc)
    events, remaining = get(f"{API}/events?apiKey={key}")  # free: the events list costs no credits
    remaining = int(remaining) if remaining and remaining.isdigit() else None
    marks = ",".join(v[2] for v in MARKETS.values())
    n = 0
    for ev in sorted(events, key=lambda e: e["commence_time"]):
        start = datetime.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        hrs = (start - now).total_seconds() / 3600
        if hrs <= 0 or hrs > FETCH_HOURS: continue
        g = store["games"].setdefault(ev["id"], {"start": ev["commence_time"], "home": ev["home_team"], "away": ev["away_team"], "fetched": [], "props": []})
        g["start"] = ev["commence_time"]
        if g["fetched"]:
            last = datetime.datetime.fromisoformat(g["fetched"][-1])
            near = hrs <= REFRESH_HOURS and (start - last).total_seconds() / 3600 > REFRESH_HOURS
            if not near or (remaining is not None and remaining < RESERVE): continue
        try:
            data, rem = get(f"{API}/events/{ev['id']}/odds?apiKey={key}&regions=us&markets={marks}&oddsFormat=american")
        except Exception as e:
            print(f"  {ev['away_team']} at {ev['home_team']}: {e}"); continue
        if rem and rem.isdigit(): remaining = int(rem)
        props = summarize(data)
        if props: g["props"] = props
        g["fetched"].append(now.isoformat(timespec="seconds")); n += 1
        print(f"  {ev['away_team']} at {ev['home_team']}: {len(props)} props")
    week_ago = now - datetime.timedelta(days=7)
    store["games"] = {k: v for k, v in store["games"].items() if datetime.datetime.fromisoformat(v["start"].replace("Z", "+00:00")) > week_ago}
    store["updated"] = now.isoformat(timespec="seconds"); store["remaining"] = remaining
    OUT.write_text(json.dumps(store, separators=(",", ":")))
    print(f"Props: fetched {n} games; {remaining} credits left this month")


if __name__ == "__main__":
    main()
