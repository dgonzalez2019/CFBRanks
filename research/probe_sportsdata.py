"""One-off probe run in GitHub Actions (this repo's dev environment can't reach sportsdata.io): reads the public
college football docs page, then tries likely injury and reference endpoints with SPORTSDATAIO_API_KEY and
reports what comes back, so we can see coverage and whether a trial key returns scrambled data.
The key is never printed."""
import os, re, html, json, urllib.request, urllib.error

KEY = os.environ.get("SPORTSDATAIO_API_KEY", "")
print("key present:", bool(KEY), "length:", len(KEY))
def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "cfbranks-probe/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=40) as r: return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e: return e.code, e.read().decode("utf-8", "replace")
    except Exception as e: return None, str(e)
def text(h):
    h = re.sub(r"(?s)<(script|style)[^>]*>.*?</\1>", " ", h); h = re.sub(r"<[^>]+>", " ", h)
    return re.sub(r"\s+", " ", html.unescape(h)).strip()

st, body = fetch("https://sportsdata.io/developers/api-documentation/ncaa-football")
t = text(body); print(f"\n== docs page {st}, {len(t)} chars")
for m in re.finditer(r"(?i)injur", t):
    print("  ...", t[max(0, m.start() - 200): m.start() + 250])
for m in re.finditer(r"/v3/cfb/[A-Za-z/]+(?:\{[^}]+\}[A-Za-z/]*)*", body):
    pass
paths = sorted(set(re.findall(r"/v3/cfb/[A-Za-z]+/(?:json|xml)/[A-Za-z]+", body)))
print("\nendpoint paths on the docs page:", paths[:80])

H = {"Ocp-Apim-Subscription-Key": KEY}
B = "https://api.sportsdata.io/v3/cfb"
for p in ["/scores/json/CurrentSeason", "/scores/json/CurrentWeek", "/scores/json/Teams",
          "/projections/json/InjuredPlayers", "/scores/json/Injuries", "/stats/json/Injuries",
          "/scores/json/InjuredPlayers", "/projections/json/Injuries"]:
    st, body = fetch(B + p, H)
    print(f"\n== {st} {p} ({len(body)} bytes)\n{body[:900]}")
    if st == 200 and "Injur" in p:
        try:
            rows = json.loads(body)
            print("rows:", len(rows), "| keys:", list(rows[0].keys())[:30] if rows else [])
            osu = [r for r in rows if "Ohio" in json.dumps(r)][:5]
            print("Ohio State-ish rows:", json.dumps(osu)[:1200])
        except Exception as e: print("parse:", e)
