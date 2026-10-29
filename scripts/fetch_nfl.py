#!/usr/bin/env python3
"""NFL data for the site, from nflverse (free public files on GitHub) and ESPN's public team list. No API key.

Writes, in the same shapes as the CFB files (see CLAUDE.md), under data/nfl/:
  games.json     completed games this season: [week, home, away, home_pts, away_pts, neutral, 1, 1, box]
                 box = [home pass yds, pass att, rush yds, rush att, away same four] from nflverse team stats
  lines.json     every game this season with a line: [week, home, away, neutral, home spread (negative = home
                 favored), total, home_pts|null, away_pts|null, start (UTC ISO), [source], null]
  ppa.json       per-game team efficiency (EPA per play): [week, team, opp, off all, off pass, off rush, def all,
                 def pass, def rush] (the CFB PPA contract, with EPA in place of PPA)
  players.json   [team, name, position, plays, EPA per play] for QBs, RBs, WRs, TEs
  injuries.json  this week's injury report: [team, player, pos, status, body part, date, date, note]
  teams.json     {v: 2, teams: [[name, logo, dark logo, color, alt color]]}
With --history Y1-Y2 it also writes data/nfl/history/<year>.json, <year>_lines.json and <year>_ppa.json.

Weeks are nflverse's (playoffs continue as weeks 19-22). Usage:
  python scripts/fetch_nfl.py [--season 2026] [--history 2015-2025]"""
import argparse, csv, datetime, gzip, io, json, pathlib, sys, urllib.request
from collections import defaultdict
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nfl"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
REL = "https://github.com/nflverse/nflverse-data/releases/download"
ET = ZoneInfo("America/New_York")

# nflverse team codes -> names shown on the site (the franchise's current name for every season)
TEAMS = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
         "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
         "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
         "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
         "LV": "Las Vegas Raiders", "OAK": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "SD": "Los Angeles Chargers",
         "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "STL": "Los Angeles Rams", "MIA": "Miami Dolphins",
         "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
         "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
         "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
         "WSH": "Washington Commanders"}


def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "cfbranks-data"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if url.endswith(".gz"): data = gzip.decompress(data)
    return data if binary else data.decode("utf-8")


def rows(url):
    return list(csv.DictReader(io.StringIO(get(url))))


def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def name(code):
    return TEAMS.get((code or "").strip())


def start_utc(day, time):
    """nflverse gameday (YYYY-MM-DD) and gametime (HH:MM, Eastern) -> UTC ISO."""
    try:
        d = datetime.datetime.strptime(f"{day} {time or '13:00'}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        return d.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    except ValueError:
        return None


def team_stats(season):
    """Per team-game offense from nflverse team stats: box score pieces and EPA per play. Column names have changed
    over time, so each value is looked up from a few candidates."""
    for url in (f"{REL}/stats_team/stats_team_week_{season}.csv", f"{REL}/stats_team/stats_team_week_{season}.csv.gz"):
        try: R = rows(url); break
        except Exception as e: err = e
    else:
        print(f"  team stats {season} unavailable: {err}"); return {}
    if R: print(f"  team stats {season}: {len(R)} rows; columns: {', '.join(list(R[0])[:60])}")
    pick = lambda r, *ks: next((num(r[k]) for k in ks if k in r and num(r[k]) is not None), None)
    out = {}
    for r in R:
        if (r.get("season_type") or "REG") not in ("REG", "POST"): continue
        t = name(r.get("team") or r.get("recent_team"))
        if not t: continue
        att = pick(r, "attempts", "passing_attempts"); sacks = pick(r, "sacks_suffered", "sacks") or 0
        car = pick(r, "carries", "rushing_attempts")
        out[(int(r["week"]), t)] = dict(
            pass_yds=pick(r, "passing_yards"), pass_att=att, rush_yds=pick(r, "rushing_yards"), rush_att=car,
            pass_epa=pick(r, "passing_epa"), rush_epa=pick(r, "rushing_epa"), dropbacks=(att or 0) + sacks,
            opp=name(r.get("opponent_team")))
    return out


def season_files(season, write_dir, current):
    G = [g for g in rows(GAMES_URL) if g.get("season") == str(season)]
    print(f"Season {season}: {len(G)} games in the schedule")
    if not G: return None
    ts = team_stats(season)
    games, lines, ppa = [], [], []
    for g in G:
        wk = int(g["week"]); h, a = name(g["home_team"]), name(g["away_team"])
        if not h or not a: print("  unknown team code:", g["home_team"], g["away_team"]); continue
        neutral = int((g.get("location") or "").lower() == "neutral")
        hp, ap = num(g.get("home_score")), num(g.get("away_score"))
        played = hp is not None and ap is not None
        sp, tot = num(g.get("spread_line")), num(g.get("total_line"))
        start = start_utc(g.get("gameday"), g.get("gametime"))
        if played:
            row = [wk, h, a, int(hp), int(ap), neutral, 1, 1]
            sh, sa = ts.get((wk, h)), ts.get((wk, a))
            if sh and sa and None not in (sh["pass_yds"], sh["pass_att"], sh["rush_yds"], sh["rush_att"], sa["pass_yds"], sa["pass_att"], sa["rush_yds"], sa["rush_att"]):
                row.append([round(sh["pass_yds"]), round(sh["pass_att"]), round(sh["rush_yds"]), round(sh["rush_att"]),
                            round(sa["pass_yds"]), round(sa["pass_att"]), round(sa["rush_yds"]), round(sa["rush_att"])])
            games.append(row)
            for me, op in ((h, a), (a, h)):
                s, o = ts.get((wk, me)), ts.get((wk, op))
                if not s or not o or None in (s["pass_epa"], s["rush_epa"], o["pass_epa"], o["rush_epa"]): continue
                per = lambda e, n: round(e / n, 4) if n else None
                ppa.append([wk, me, op,
                            per(s["pass_epa"] + s["rush_epa"], s["dropbacks"] + (s["rush_att"] or 0)), per(s["pass_epa"], s["dropbacks"]), per(s["rush_epa"], s["rush_att"]),
                            per(o["pass_epa"] + o["rush_epa"], o["dropbacks"] + (o["rush_att"] or 0)), per(o["pass_epa"], o["dropbacks"]), per(o["rush_epa"], o["rush_att"])])
        if sp is not None:  # nflverse spread_line: positive = home favored; ours: negative = home favored
            lines.append([wk, h, a, neutral, -sp, tot, int(hp) if played else None, int(ap) if played else None, start, ["nflverse"], None])
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    write_dir.mkdir(parents=True, exist_ok=True)
    pre = "" if current else f"{season}"
    gpath = write_dir / ("games.json" if current else f"{season}.json")
    gpath.write_text(json.dumps({"season": season, "updated": stamp, "games": games}, separators=(",", ":")))
    (write_dir / ("lines.json" if current else f"{season}_lines.json")).write_text(json.dumps({"season": season, "updated": stamp, "games": lines}, separators=(",", ":")))
    if current:  # the full schedule for the team pages: [week, home, away, neutral, start, 1, 1]
        sched = [[int(g["week"]), name(g["home_team"]), name(g["away_team"]), int((g.get("location") or "").lower() == "neutral"),
                  start_utc(g.get("gameday"), g.get("gametime")), 1, 1] for g in G if name(g["home_team"]) and name(g["away_team"])]
        (write_dir / "schedule.json").write_text(json.dumps({"season": season, "updated": stamp, "games": sched}, separators=(",", ":")))
    (write_dir / ("ppa.json" if current else f"{season}_ppa.json")).write_text(json.dumps({"season": season, "updated": stamp, "games": ppa}, separators=(",", ":")))
    print(f"  wrote {len(games)} games ({sum(len(r) > 8 for r in games)} with box scores), {len(lines)} lines "
          f"({sum(r[6] is None for r in lines)} upcoming), {len(ppa)} EPA rows")
    return stamp


def players(season):
    """EPA per play for skill players this season (nflverse player stats)."""
    for url in (f"{REL}/stats_player/stats_player_week_{season}.csv", f"{REL}/player_stats/player_stats_{season}.csv"):
        try: R = rows(url); break
        except Exception as e: err = e
    else:
        print(f"  player stats unavailable: {err}"); return
    if R: print(f"  player stats: {len(R)} rows; columns: {', '.join(list(R[0])[:50])}")
    agg = defaultdict(lambda: [0.0, 0.0, "", ""])
    for r in R:
        if (r.get("season_type") or "REG") not in ("REG", "POST"): continue
        t = name(r.get("team") or r.get("recent_team")); pos = r.get("position") or r.get("position_group") or ""
        nm = r.get("player_display_name") or r.get("player_name")
        if not t or pos not in ("QB", "RB", "WR", "TE", "FB") or not nm: continue
        plays = sum(num(r.get(k)) or 0 for k in ("attempts", "sacks_suffered", "sacks", "carries", "targets"))
        epa = sum(num(r.get(k)) or 0 for k in ("passing_epa", "rushing_epa", "receiving_epa"))
        a = agg[(t, nm)]; a[0] += plays; a[1] += epa; a[2] = "RB" if pos == "FB" else pos
    out = [[t, nm, p, int(pl), round(e / pl, 4)] for (t, nm), (pl, e, p, _) in agg.items() if pl >= 10]
    (OUT / "players.json").write_text(json.dumps({"season": season, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "players": out}, separators=(",", ":")))
    print(f"  wrote {len(out)} players")


def roles(season):
    """Each player's average snap share in the games he played (offense or defense; kickers/punters 1), for the
    injured-player estimates: [team, name, position, share]."""
    try: R = rows(f"{REL}/snap_counts/snap_counts_{season}.csv")
    except Exception as e: print(f"  snap counts unavailable: {e}"); return
    if R: print(f"  snap counts: {len(R)} rows; columns: {', '.join(list(R[0]))}")
    agg = defaultdict(list); pos_of = {}
    for r in R:
        t = name(r.get("team")); nm = r.get("player")
        if not t or not nm: continue
        f = lambda k: (lambda v: v / 100 if v and v > 1 else (v or 0.0))(num(r.get(k)))
        pos = (r.get("position") or "").upper(); pos_of[(t, nm)] = pos
        agg[(t, nm)].append(1.0 if pos in ("K", "P", "LS") and f("st_pct") > 0 else max(f("offense_pct"), f("defense_pct")))
    out = [[t, nm, pos_of[(t, nm)], round(sum(v) / len(v), 3)] for (t, nm), v in agg.items() if v]
    (OUT / "roles.json").write_text(json.dumps({"season": season, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "players": out}, separators=(",", ":")))
    print(f"  wrote {len(out)} player roles")


def injuries(season):
    try: R = rows(f"{REL}/injuries/injuries_{season}.csv")
    except Exception as e: print(f"  injuries unavailable: {e}"); return
    if not R: print("  no injury rows"); return
    print(f"  injuries: {len(R)} rows; columns: {', '.join(list(R[0]))}")
    last = max(int(r["week"]) for r in R if r.get("week"))
    first = {}
    for r in R:
        d = (r.get("date_modified") or "")[:10]; k = (name(r.get("team")), r.get("full_name"))
        if d and (k not in first or d < first[k]): first[k] = d
    out = []
    for r in R:
        if int(r.get("week") or 0) != last: continue
        t = name(r.get("team")); st = (r.get("report_status") or "").strip()
        if not t or not st: continue
        d = (r.get("date_modified") or "")[:10]; start = first.get((t, r.get("full_name")), d)
        out.append([t, r.get("full_name") or "", r.get("position") or "", st, r.get("report_primary_injury") or "", start, d,
                    r.get("practice_status") or ""])
    (OUT / "injuries.json").write_text(json.dumps({"updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "source": f"nflverse injury report, week {last}", "injuries": out}, separators=(",", ":")))
    print(f"  wrote {len(out)} injury rows (week {last})")


def teams():
    try: J = json.loads(get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"))
    except Exception as e: print(f"  ESPN teams unavailable: {e}"); return
    out = []
    for item in J.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        t = item.get("team", {}); nm = t.get("displayName")
        if nm == "Washington": nm = "Washington Commanders"
        logos = t.get("logos") or []
        light = next((l["href"] for l in logos if "dark" not in (l.get("rel") or [])), logos[0]["href"] if logos else None)
        dark = next((l["href"] for l in logos if "dark" in (l.get("rel") or [])), None)
        c = t.get("color"); ac = t.get("alternateColor")
        out.append([nm, light, dark, f"#{c}" if c else None, f"#{ac}" if ac else None])
    missing = sorted(set(TEAMS.values()) - {r[0] for r in out})
    if missing: print("  teams without an ESPN match:", missing)
    (OUT / "teams.json").write_text(json.dumps({"v": 2, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "teams": out}, separators=(",", ":")))
    print(f"  wrote {len(out)} teams")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--season", type=int)
    ap.add_argument("--history", help="also write past seasons, e.g. 2015-2025")
    a = ap.parse_args()
    today = datetime.date.today()
    season = a.season or (today.year if today.month >= 8 else today.year - 1)
    OUT.mkdir(parents=True, exist_ok=True)
    if not season_files(season, OUT, True): sys.exit(f"No {season} schedule in nflverse games.csv")
    players(season); injuries(season); roles(season); teams()
    if a.history:
        y1, y2 = (int(x) for x in a.history.split("-"))
        for y in range(y1, y2 + 1): season_files(y, OUT / "history", False)
