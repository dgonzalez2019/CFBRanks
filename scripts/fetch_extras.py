#!/usr/bin/env python3
"""Pull a past season's betting lines and per-game passers from collegefootballdata.com, for tuning.
Writes <dir>/<year>_lines.json and <dir>/<year>_qb.json; skips a file that already exists.
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
lp = out_dir / f"{a.year}_lines.json"
if not lp.exists():
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
