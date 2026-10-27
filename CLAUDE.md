# Project notes for Claude Code

Static site + Python tooling for college football power rankings. Teams are ranked by power rating (who is best); strength of record (SOR) is shown alongside as a résumé check.

- **Single page:** `index.html` holds all HTML, CSS and JS. No framework, no build. Keep it that way unless asked.
- **Data contract:** `data/games.json` = `{season, updated (ISO UTC), games: [[week, home, away, home_pts, away_pts, neutral 0/1, home_fbs 0/1, away_fbs 0/1], ...]}`. Postseason weeks are offset by +20.
- **Algorithm (keep JS `compute()` and Python `compute()` in sync):**
  1. Ridge least squares on tanh-compressed margins (cap 21, ridge 2) with an estimated home-field term; neutral sites get 0; all non-FBS teams pooled as "FCS (pooled)"; ratings centered on the FBS mean.
  2. Outcome SD calibrated from residuals ×1.05; win prob = normal CDF.
  3. For each game, P(benchmark team wins at that venue); benchmark = Nth-rated team (default 25).
  4. WAB = wins − Σ benchmark win prob (SOR rank key); SOR% = Poisson-binomial P(benchmark ≥ actual wins). The main ranking is the power rating, not SOR.
  5. Win value = 1 − p; loss cost = p. Quadrants by venue-adjusted opponent rank: home 15/40/80, neutral 25/55/95, road 40/75/110.
- **Updates:** GitHub Action `update-games.yml` runs `scripts/fetch_games.py` with secret `CFBD_API_KEY` and commits `data/games.json`.
- **Test:** `python -m http.server` and open the page; `python scripts/cfb_sor.py --csv <file>` for a CLI check.
