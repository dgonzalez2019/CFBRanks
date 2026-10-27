# CFB Power Rankings

College football power rankings: a live Top 25 ordered by power rating plus a résumé bonus that rewards wins (Q1 most) slightly more than it penalizes losses, with strength of record, schedule strength and Q1–Q4 records alongside, plus a team lookup that shows every opponent's rank, the venue, and how much each win or loss was worth.

## How it works
- `index.html` — the whole site (no build step). It reads `data/games.json` and runs the rating + SOR algorithm in the browser. With no data it shows a simulated sample season.
- `scripts/fetch_games.py` — pulls completed games from collegefootballdata.com into `data/games.json`.
- `scripts/cfb_sor.py` — the same algorithm in Python, for command-line rankings and CSV export.
- `.github/workflows/update-games.yml` — refreshes the data on a schedule and commits it, so the site stays current.

## Setup
1. Get a free API key at https://collegefootballdata.com/key.
2. In the repo: **Settings → Secrets and variables → Actions → New repository secret**, name `CFBD_API_KEY`.
3. **Settings → Pages**: deploy from branch `main`, folder `/ (root)`.
4. **Actions → Update game results → Run workflow** once to load this season's games.

## Run locally
```
pip install numpy
export CFBD_API_KEY=your_key
python scripts/fetch_games.py
python -m http.server 8000   # then open http://localhost:8000
python scripts/cfb_sor.py --year 2026 --team "Miami"   # --rank power or --rank sor for other orders
```
