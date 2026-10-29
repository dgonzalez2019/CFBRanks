"""One-off probe run in GitHub Actions (this repo's dev environment can't reach bigballsdata.com): prints the
text of BigBalls' public docs pages, then tries likely API addresses and key formats and reports which respond.
The key is read from BIGBALLS_API_KEY and never printed."""
import os, re, json, urllib.request, urllib.error, html

KEY = os.environ.get("BIGBALLS_API_KEY", "")
print("key present:", bool(KEY), "length:", len(KEY))

def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "cfbranks-probe/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e: return e.code, e.read().decode("utf-8", "replace")
    except Exception as e: return None, str(e)

def text(h):
    h = re.sub(r"(?s)<(script|style)[^>]*>.*?</\\1>", " ", h); h = re.sub(r"<[^>]+>", " ", h)
    return re.sub(r"\\s+", " ", html.unescape(h)).strip()

for page in ["https://bigballsdata.com/llms.txt", "https://bigballsdata.com/docs", "https://bigballsdata.com/injuries-api",
             "https://bigballsdata.com/ncaaf-api", "https://bigballsdata.com/pricing"]:
    st, body = fetch(page)
    print(f"\n===== PAGE {page} -> {st}\n{text(body)[:6000]}")

print("\n===== API PROBES")
bases = ["https://api.bigballsdata.com/v1", "https://bigballsdata.com/api/v1"]
paths = ["/injuries?league=ncaaf", "/ncaaf/injuries", "/injuries?sport=ncaaf", "/leagues"]
auths = {"bearer": ({"Authorization": f"Bearer {KEY}"}, ""), "x-api-key": ({"x-api-key": KEY}, ""),
         "query api_key": ({}, "api_key")}
seen = 0
for b in bases:
    for p in paths:
        for name, (hd, q) in auths.items():
            url = b + p + ((("&" if "?" in p else "?") + f"{q}={KEY}") if q else "")
            st, body = fetch(url, hd)
            shown = (b + p) + (f" [{name}]")
            print(f"{st} {shown}: {body[:300].replace(KEY, '***') if KEY else body[:300]}")
            seen += 1
print("requests:", seen)
