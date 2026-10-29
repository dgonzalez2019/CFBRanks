#!/usr/bin/env python3
"""Pull current college football injuries from SportsDataIO into data/injuries.json, with SportsDataIO teams
mapped to the team names used in data/games.json (collegefootballdata.com).
Usage: SPORTSDATAIO_API_KEY=... python scripts/fetch_injuries.py [--out data/injuries.json]"""
import argparse, datetime, json, os, pathlib, re, sys, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/injuries.json")
a = ap.parse_args()
key = os.environ.get("SPORTSDATAIO_API_KEY") or sys.exit("SPORTSDATAIO_API_KEY is not set; skipping injuries.")
root = pathlib.Path(__file__).resolve().parent.parent


def get(path):
    req = urllib.request.Request("https://api.sportsdata.io/v3/cfb/scores/json/" + path,
                                 headers={"Ocp-Apim-Subscription-Key": key, "User-Agent": "cfbranks/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


# Team names: SportsDataIO's "School" mostly matches collegefootballdata.com; ALIASES covers the rest.
ALIASES = {"Miami (FL)": "Miami", "Miami (OH)": "Miami (OH)", "Ole Miss": "Ole Miss", "Mississippi": "Ole Miss",
           "Hawaii": "Hawai'i", "San Jose State": "San José State", "UL Monroe": "UL Monroe", "Louisiana-Monroe": "UL Monroe",
           "Louisiana-Lafayette": "Louisiana", "UTSA": "UTSA", "Texas-San Antonio": "UTSA", "UTEP": "UTEP", "UConn": "UConn",
           "Connecticut": "UConn", "Massachusetts": "UMass", "Southern Miss": "Southern Miss", "Southern Mississippi": "Southern Miss",
           "App State": "App State", "Appalachian State": "App State", "Sam Houston State": "Sam Houston", "FIU": "Florida International",
           "Florida Int'l": "Florida International", "NC State": "NC State", "North Carolina State": "NC State",
           "Central Florida": "UCF", "Southern California": "USC", "Brigham Young": "BYU", "Louisiana State": "LSU",
           "Texas Christian": "TCU", "Southern Methodist": "SMU", "Nevada-Las Vegas": "UNLV", "Alabama-Birmingham": "UAB"}
ours = set()
try:
    for g in json.loads((root / "data" / "games.json").read_text())["games"]:
        if g[6]: ours.add(g[1])
        if g[7]: ours.add(g[2])
except (OSError, ValueError, KeyError):
    pass
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("state", "st").replace("&", "and"))
by_norm = {norm(t): t for t in ours}

def our_name(school):
    for cand in (ALIASES.get(school), school):
        if cand and cand in ours: return cand
        if cand and norm(cand) in by_norm: return by_norm[norm(cand)]
    return None

teams = get("Teams")
team_of, unmatched = {}, []
for t in teams:
    name = our_name(t.get("School") or "")
    if name: team_of[t["TeamID"]] = name
    elif t.get("Active") and ours: unmatched.append(t.get("School"))
inj = get("InjuredPlayers")
rows = []
for p in inj:
    team = team_of.get(p.get("TeamID"))
    if not team: continue
    rows.append([team, f"{p.get('FirstName','')} {p.get('LastName','')}".strip(), p.get("Position") or "", p.get("InjuryStatus") or "",
                 p.get("InjuryBodyPart") or "", (p.get("InjuryStartDate") or "")[:10], (p.get("Updated") or "")[:16],
                 (p.get("InjuryNotes") or "")[:300]])
out = {"updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "source": "SportsDataIO",
       "injuries": sorted(rows)}
pathlib.Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
print(f"Wrote {len(rows)} of {len(inj)} injured players to {a.out}; {len(team_of)} teams matched.")
fbs_unmatched = sorted(s for s in unmatched if s)
print(f"Active SportsDataIO teams not matched to our names ({len(fbs_unmatched)}):", fbs_unmatched[:200])
