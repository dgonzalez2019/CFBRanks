"""One-off probe run in GitHub Actions (this repo's dev environment can't reach bigballsdata.com). Round 3: what
the provider holds for college football, this key's plan and quota, and college injuries by team and as-of time.
The key is never printed."""
import os, json, urllib.request, urllib.error, urllib.parse

KEY = os.environ.get("BIGBALLS_API_KEY", ""); BASE = "https://api.bigballsdata.com"
def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}", "User-Agent": "cfbranks-probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e: return e.code, e.read().decode("utf-8", "replace")
    except Exception as e: return None, str(e)
def show(path, n=1500):
    st, body = get(path); print(f"\n== {st} GET {path}\n{body[:n]}"); return st, body

show("/v1/usage", 800)
st, body = show("/v1/coverage?sport=american_football", 400)
try:
    d = json.loads(body)["data"]
    for item in (d if isinstance(d, list) else d.get("leagues", d.get("coverage", [d]))):
        if "ncaaf" in json.dumps(item).lower(): print("\nNCAAF coverage:", json.dumps(item)[:2500])
except Exception as e: print("coverage parse:", e)
show("/v1/as-of/coverage", 2500)
st, body = show("/v1/teams?league=ncaaf&limit=5", 900)
q = urllib.parse.quote
for team in ["Ohio State", "Ohio State Buckeyes", "Georgia Bulldogs"]:
    show(f"/v1/injuries?sport=american_football&team={q(team)}", 700)
show("/v1/as-of/injuries?sport=american_football&as_of=2026-10-24T15:00:00Z&league=ncaaf", 1200)
show("/v1/as-of/injuries?sport=ncaaf&as_of=2026-10-24T15:00:00Z", 800)
