# Project notes for Claude Code

Static site + Python tooling for college football power rankings. Teams are ranked by score = power rating + 0.5 × résumé points; strength of record (SOR) is shown on team pages as a reference.

- **Single page:** `index.html` holds all HTML, CSS and JS. No framework, no build. Keep it that way unless asked.
- **Data contract:** `data/games.json` = `{season, updated (ISO UTC), games: [[week, home, away, home_pts, away_pts, neutral 0/1, home_fbs 0/1, away_fbs 0/1], ...]}`. Postseason weeks are offset by +20.
- **Algorithm (keep JS `compute()` and Python `compute()` in sync):**
  1. Ridge least squares on tanh-compressed margins (cap 21, ridge 2) with an estimated home-field term; neutral sites get 0; all non-FBS teams pooled as "FCS (pooled)"; ratings centered on the FBS mean.
  2. Outcome SD calibrated from residuals ×1.05; win prob = normal CDF.
  3. For each game, P(benchmark team wins at that venue); benchmark = Nth-rated team (default 25).
  4. WAB = wins − Σ benchmark win prob (SOR rank key); SOR% = Poisson-binomial P(benchmark ≥ actual wins). SOR is a reference, not the main ranking.
  5. Win value = 1 − p; loss cost = p. Quadrants by opponent's power rank, any venue: Q1 ≤25, Q2 ≤50, Q3 ≤80, else Q4 (incl. FCS). Deliberately power rank, not overall rank: using overall rank creates a feedback loop that inflates whole conferences.
  6. Résumé points per game by quadrant: wins 4/2/1/0.5 (Q1–Q4), losses cost 0.85 × 0.5/1/2/4. Ranking score = rating + 0.5 × résumé (constants WIN_PTS, LOSS_PTS, LOSS_WEIGHT, RESUME_WEIGHT in both files).
- **Updates:** GitHub Action `update-games.yml` runs `scripts/fetch_games.py` with secret `CFBD_API_KEY` and commits `data/games.json`.
- **Test:** `python -m http.server` and open the page; `python scripts/cfb_sor.py --csv <file>` for a CLI check.
