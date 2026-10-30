#!/usr/bin/env python3
"""NFL player-prop lines from The Odds API (https://the-odds-api.com; secret ODDS_API_KEY, free plan = 500 credits a
month, reset monthly). Each game's props cost one credit per market. The four core markets (props_model.CORE) are
always fetched once a game kicks off within FETCH_HOURS; the other markets are added in priority order (EXTRA) as far
as the budget allows: credits left after reserving the core markets for every game still to come this month, split
evenly over this month's remaining games. A game is refreshed within REFRESH_HOURS of kickoff only if that spare
share covers every market. Without the key it does nothing.

Writes data/nfl/props_lines.json = {updated, remaining, games: {event id: {start, home, away, fetched: [ISO...],
props: [[market, player, line, over price, under price, books]]}}} with each player's median line across books and
the median prices at that line (American odds). Games that kicked off more than a week ago are dropped.
Usage: python scripts/fetch_props.py"""
import datetime, json, os, pathlib, statistics, sys, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from props_model import MARKETS, CORE

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nfl" / "props_lines.json"
API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
FETCH_HOURS, REFRESH_HOURS = 36, 6
EXTRA = ("anytime_td", "rush_att", "pass_tds", "pass_att", "pass_cmp", "ints")  # added in this order as budget allows
KEYS = {v[2]: k for k, v in MARKETS.items()}  # Odds API market -> ours


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cfbranks-data"}), timeout=30) as r:
        return json.load(r), r.headers.get("x-requests-remaining")


def summarize(ev):
    """Median line per (market, player) across books, with the median over/under prices among books at that line.
    Yes/no markets (anytime TD) store line 0.5 with the Yes price as "over" and the No price (or None) as "under"."""
    by, yes = {}, {}
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            mk = KEYS.get(m.get("key"))
            if not mk: continue
            sides = {}
            if MARKETS[mk][4] == "yes":
                for o in m.get("outcomes", []):
                    if o.get("description") and o.get("name") in ("Yes", "No"):
                        yes.setdefault((mk, o["description"]), {}).setdefault(o["name"], []).append(o["price"])
                continue
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
    for (mk, player), d in sorted(yes.items()):
        if d.get("Yes"):
            out.append([mk, player, 0.5, round(statistics.median(d["Yes"])), round(statistics.median(d["No"])) if d.get("No") else None, len(d["Yes"])])
    return out


def games_left_this_month(now):
    """NFL games from now to the end of the month, from data/nfl/schedule.json."""
    try: sched = json.loads((ROOT / "data" / "nfl" / "schedule.json").read_text())["games"]
    except (OSError, ValueError, KeyError): return 16
    nxt = (now.replace(day=1) + datetime.timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    n = 0
    for g in sched:
        if not g[4]: continue
        t = datetime.datetime.fromisoformat(g[4].replace("Z", "+00:00"))
        if now < t < nxt: n += 1
    return max(n, 1)


def main():
    key = os.environ.get("ODDS_API_KEY")
    if not key: print("ODDS_API_KEY is not set; skipping player props."); return
    store = json.loads(OUT.read_text()) if OUT.exists() else {"games": {}}
    now = datetime.datetime.now(datetime.timezone.utc)
    events, remaining = get(f"{API}/events?apiKey={key}")  # free: the events list costs no credits
    remaining = int(remaining) if remaining and remaining.isdigit() else None
    left = games_left_this_month(now)  # including the games fetched below
    n = 0
    for ev in sorted(events, key=lambda e: e["commence_time"]):
        start = datetime.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        hrs = (start - now).total_seconds() / 3600
        if hrs <= 0 or hrs > FETCH_HOURS: continue
        g = store["games"].setdefault(ev["id"], {"start": ev["commence_time"], "home": ev["home_team"], "away": ev["away_team"], "fetched": [], "props": []})
        g["start"] = ev["commence_time"]
        # spare credits per game this month after reserving the core markets for every game still to come
        spare = ((remaining if remaining is not None else 500) - len(CORE) * left) / left
        if g["fetched"]:
            last = datetime.datetime.fromisoformat(g["fetched"][-1])
            near = hrs <= REFRESH_HOURS and (start - last).total_seconds() / 3600 > REFRESH_HOURS
            if not near or spare < len(MARKETS): continue
        else: left = max(1, left - 1)
        want = list(CORE) + list(EXTRA[:max(0, int(spare))])
        marks = ",".join(MARKETS[m][2] for m in want)
        try:
            data, rem = get(f"{API}/events/{ev['id']}/odds?apiKey={key}&regions=us&markets={marks}&oddsFormat=american")
        except Exception as e:
            print(f"  {ev['away_team']} at {ev['home_team']}: {e}"); continue
        if rem and rem.isdigit(): remaining = int(rem)
        props = summarize(data)
        if props:
            keep = {(r[0], r[1]) for r in props}  # a refresh with fewer markets keeps the others from before
            g["props"] = props + [r for r in g.get("props", []) if (r[0], r[1]) not in keep and r[0] not in want]
        g["fetched"].append(now.isoformat(timespec="seconds")); n += 1
        print(f"  {ev['away_team']} at {ev['home_team']}: {len(props)} props ({len(want)} markets)")
    week_ago = now - datetime.timedelta(days=7)
    store["games"] = {k: v for k, v in store["games"].items() if datetime.datetime.fromisoformat(v["start"].replace("Z", "+00:00")) > week_ago}
    store["updated"] = now.isoformat(timespec="seconds"); store["remaining"] = remaining
    OUT.write_text(json.dumps(store, separators=(",", ":")))
    print(f"Props: fetched {n} games; {remaining} credits left this month")


if __name__ == "__main__":
    main()
