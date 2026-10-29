#!/usr/bin/env python3
"""Pull a past season's betting lines and per-game passers from collegefootballdata.com, for tuning.
Writes <dir>/<year>_lines.json, _qb.json, _players.json, _ppa.json, _talent.json, _returning.json and
_meta.json, plus <dir>/venues.json once; skips files that exist.
Usage: CFBD_API_KEY=... python scripts/fetch_extras.py --year 2024 [--dir data/history]"""
import argparse, json, os, sys, pathlib, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, required=True); ap.add_argument("--dir", default="data/history")
a = ap.parse_args()
key = os.environ.get("CFBD_API_KEY") or sys.exit("CFBD_API_KEY is not set")
out_dir = pathlib.Path(a.dir); out_dir.mkdir(parents=True, exist_ok=True)


def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def median(xs):
    xs = sorted(x for x in xs if x is not None); n = len(xs)
    return None if not n else xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# Lines: [week, home, away, median home spread (negative = home favored), median total, median opening spread, [books]]
import datetime
in_progress = a.year >= datetime.date.today().year  # current season: lines and players live in data/, not history

lp = out_dir / f"{a.year}_lines.json"
if not lp.exists() and not in_progress:
    rows = []
    for seas in ("regular", "postseason"):
        for it in get(f"https://api.collegefootballdata.com/lines?year={a.year}&seasonType={seas}"):
            books = it.get("lines") or []
            sp = median([num(b.get("spread")) for b in books])
            if sp is None: continue
            rows.append([it.get("week", 0) + (20 if seas == "postseason" else 0), it.get("homeTeam", it.get("home_team")),
                         it.get("awayTeam", it.get("away_team")), sp, median([num(b.get("overUnder")) for b in books]),
                         median([num(b.get("spreadOpen")) for b in books]), sorted({b.get("provider") for b in books if b.get("provider")})])
    lp.write_text(json.dumps({"season": a.year, "games": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote {len(rows)} lines")

# Passers per game: [week, home, away, [[name, attempts], ...home], [[name, attempts], ...away]]
qp = out_dir / f"{a.year}_qb.json"
if not qp.exists():
    rows = []; weeks = [("regular", w) for w in range(1, 17)] + [("postseason", 1)]
    for seas, w in weeks:
        try:
            box = get(f"https://api.collegefootballdata.com/games/players?year={a.year}&week={w}&seasonType={seas}&category=passing")
        except Exception as e:
            print(f"{a.year} {seas} week {w}: {e}"); continue
        n0 = len(rows)
        for g in box:
            sides = {}
            for t in g.get("teams", []):
                passers = []
                for cat in t.get("categories", []):
                    if cat.get("name") != "passing": continue
                    for ty in cat.get("types", []):
                        if ty.get("name") in ("C/ATT", "C/A"):
                            for ath in ty.get("athletes", []):
                                try: att = int(str(ath.get("stat")).replace("-", "/").split("/")[1])
                                except (IndexError, ValueError): continue
                                passers.append([ath.get("name"), att])
                sides[t.get("homeAway", t.get("home_away"))] = (t.get("team", t.get("school")), passers)
            if "home" in sides and "away" in sides:
                rows.append([w + (20 if seas == "postseason" else 0), sides["home"][0], sides["away"][0], sides["home"][1], sides["away"][1]])
        if box and len(rows) == n0: print("  First item, for debugging:", json.dumps(box[0])[:800])
    qp.write_text(json.dumps({"season": a.year, "games": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote passers for {len(rows)} games")

# Player values for that season, same format as data/players.json: [team, name, position, plays, avg PPA per play]
pp = out_dir / f"{a.year}_players.json"
if not pp.exists() and not in_progress:
    rows = []
    for x in get(f"https://api.collegefootballdata.com/ppa/players/season?year={a.year}&excludeGarbageTime=true"):
        avg = (x.get("averagePPA") or {}).get("all"); tot = (x.get("totalPPA") or {}).get("all")
        plays = x.get("countablePlays") or (round(tot / avg) if avg and tot is not None else None)
        if x.get("team") and x.get("name") and plays and plays >= 10 and avg is not None:
            rows.append([x["team"], x["name"], x.get("position") or "", int(plays), round(float(avg), 4)])
    pp.write_text(json.dumps({"season": a.year, "players": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote {len(rows)} player values")

# Per-game team efficiency (predicted points added per play, garbage time excluded), for opponent-adjusted
# efficiency ratings: [week, team, opponent, offense PPA/play, offense pass PPA, offense rush PPA,
#                      defense PPA/play allowed, defense pass, defense rush]
ep = out_dir / f"{a.year}_ppa.json"
if not ep.exists():
    rows = []
    for seas in ("regular", "postseason"):
        try: data = get(f"https://api.collegefootballdata.com/ppa/games?year={a.year}&seasonType={seas}&excludeGarbageTime=true")
        except Exception as e: print(f"{a.year} ppa {seas}: {e}"); continue
        for x in data:
            o, d = x.get("offense") or {}, x.get("defense") or {}
            if o.get("overall") is None or d.get("overall") is None: continue
            rows.append([x.get("week", 0) + (20 if seas == "postseason" else 0), x.get("team"), x.get("opponent"),
                         o.get("overall"), o.get("passing"), o.get("rushing"), d.get("overall"), d.get("passing"), d.get("rushing")])
        if data and not rows: print("  First ppa item, for debugging:", json.dumps(data[0])[:600])
    ep.write_text(json.dumps({"season": a.year, "games": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote {len(rows)} team-game PPA rows")

# Recruiting talent composite (247Sports) and returning production, for preseason ratings.
tp = out_dir / f"{a.year}_talent.json"
if not tp.exists():
    t = get(f"https://api.collegefootballdata.com/talent?year={a.year}")
    rows = [[x.get("team", x.get("school")), float(x.get("talent"))] for x in t if x.get("talent") is not None]
    tp.write_text(json.dumps({"season": a.year, "talent": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote talent for {len(rows)} teams")
rp = out_dir / f"{a.year}_returning.json"
if not rp.exists():
    r = get(f"https://api.collegefootballdata.com/player/returning?year={a.year}")
    rows = [[x.get("team"), x.get("percentPPA", x.get("percent_ppa")), x.get("percentPassingPPA", x.get("percent_passing_ppa")),
             x.get("percentRushingPPA", x.get("percent_rushing_ppa")), x.get("percentReceivingPPA", x.get("percent_receiving_ppa"))] for x in r]
    rp.write_text(json.dumps({"season": a.year, "returning": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote returning production for {len(rows)} teams")

# Game details for situational angles (rest, travel): [week, home, away, start, venue id, neutral, conference game,
# home conference, away conference], every game including unplayed ones.
mp = out_dir / f"{a.year}_meta.json"
if not mp.exists():
    rows = []
    for seas in ("regular", "postseason"):
        for g in get(f"https://api.collegefootballdata.com/games?year={a.year}&seasonType={seas}"):
            rows.append([g.get("week", 0) + (20 if seas == "postseason" else 0), g.get("homeTeam", g.get("home_team")),
                         g.get("awayTeam", g.get("away_team")), g.get("startDate", g.get("start_date")), g.get("venueId", g.get("venue_id")),
                         int(bool(g.get("neutralSite", g.get("neutral_site")))), int(bool(g.get("conferenceGame", g.get("conference_game")))),
                         g.get("homeConference", g.get("home_conference")), g.get("awayConference", g.get("away_conference"))])
    mp.write_text(json.dumps({"season": a.year, "games": rows}, separators=(",", ":")))
    print(f"{a.year}: wrote details for {len(rows)} games")

# Stadium locations (one call, all seasons): [venue id, name, latitude, longitude, timezone, dome]
vp = out_dir / "venues.json"
if not vp.exists():
    rows = []
    for v in get("https://api.collegefootballdata.com/venues"):
        loc = v.get("location") or {}
        lat, lon = loc.get("x", v.get("latitude")), loc.get("y", v.get("longitude"))
        rows.append([v.get("id"), v.get("name"), lat, lon, v.get("timezone"), int(bool(v.get("dome")))])
    vp.write_text(json.dumps({"venues": rows}, separators=(",", ":")))
    print(f"wrote {len(rows)} venues")
