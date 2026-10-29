"""One-off probe run in GitHub Actions (this repo's dev environment can't reach bigballsdata.com). Round 2: read
the API's endpoint index and OpenAPI spec, list league ids, and try college injury routes. Key never printed."""
import os, json, urllib.request, urllib.error

KEY = os.environ.get("BIGBALLS_API_KEY", ""); BASE = "https://api.bigballsdata.com"
def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}", "User-Agent": "cfbranks-probe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e: return e.code, e.read().decode("utf-8", "replace")
    except Exception as e: return None, str(e)
def show(path, n=1500):
    st, body = get(path); print(f"\n== {st} GET {path}\n{body[:n]}"); return st, body

st, body = show("/openapi.json", 300)
try:
    spec = json.loads(body); paths = spec.get("paths", {})
    print(f"\n{len(paths)} paths in spec. Paths mentioning injur / ncaaf / college / availability:")
    for p, ops in paths.items():
        if any(k in p.lower() for k in ("injur", "ncaaf", "college", "avail", "absence", "designation")):
            for m, op in ops.items():
                params = [f"{x.get('name')}({'req' if x.get('required') else 'opt'})" for x in op.get("parameters", []) if isinstance(x, dict)]
                print(f"  {m.upper()} {p}  {op.get('summary','')[:80]}  params: {', '.join(params)}")
except Exception as e:
    print("spec not JSON:", e)
show("/v1/", 3000)
st, body = get("/v1/leagues")
try:
    L = json.loads(body)["data"]; print("\nleagues:", [(x.get("id"), x.get("sport")) for x in L])
except Exception as e: print("leagues parse failed", e, body[:300])
for p in ["/v1/injuries?league=ncaaf&limit=5", "/v1/ncaaf/teams?limit=3", "/v1/injuries?sport=american_football&league=ncaaf&limit=5",
          "/v1/ncaaf/injuries?limit=5", "/v1/injuries?competition=ncaaf&limit=5"]:
    show(p, 700)
