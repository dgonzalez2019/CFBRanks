# Research scripts

How the model settings and the Lines tab's past-season stats were produced. They read the season files in
`data/history/` (download with the "Fetch past seasons" workflow) and write intermediate files to a work
directory. Run from the repo root, in order:

```
W=/tmp/cfb-work; mkdir -p $W
python research/predictions.py research $W/preds.json   # our week-by-week predictions, 2022-26 (uses tune.py)
python research/vs_market.py $W .                         # vs. betting lines -> $W/joined.json
python research/qb_injuries.py $W .                      # games a starting QB missed -> $W/qb_first_missed.json
python research/picks_features.py research $W .           # pre-game features for spread / total picks -> $W/picks.json
python research/picks_model.py $W .                       # picks trained on 2022-24, tested on 2025-26
python research/qb_rule.py $W .                           # QB-out betting rule with the starter named pre-game
python research/ppa_model.py research $W .               # play-efficiency (PPA) ratings with talent priors -> $W/ppa_preds.json
python research/ppa_eval.py $W                            # vs opening and closing lines, held-out seasons
python research/situational.py .                          # rest, travel, last-week and line angles; no ratings
python research/opening_2026.py $W .                      # pre-registered: does 2025's edge vs opening lines hold in 2026?
python research/confidence.py $W .                        # what makes a top-10% pick: candidate confidence measures
python research/criteria.py $W .                          # which yes/no criteria separated correct picks, 2025 <-> 2026
python research/criteria_check.py $W .                    # leak-free check of the winner on 2022-25
python research/summary.py $W .                          # -> data/research.json (shown on the Lines tab)
```

`tune.py` also holds the backtest harness used to choose the prior, margin cap and efficiency weight.

Held-out results so far (edge 3+ points vs the opening line, 2022-25 pooled): score model 51.1%, play efficiency
51.6%, both 50.9%; vs closing lines 48-50%. 2025 alone reaches 55-58%, but 2022 is 43-49%. Situational model (no ratings): 49.0% on 2025-26; the
steadiest single angle, conference-game underdogs, 53.2% over 2022-26 closing lines, best of 10 angles tried.

`data/history/2026_qb.json` (and 2026's PPA, talent and returning files) are this season's data through the week it was fetched (extras-only run), kept so
the QB rule's 2026 test can be reproduced; it does not update.

NIL era (trained on 2025 only, tested on 2026): score model vs opening lines 51.1% at 3+ points (2025 was 55.8%);
situational model 52.0%; spread picks 51.9%; totals 49.7%; score + PPA blend 46.9%. An earlier version of situational.py
scored only an arbitrary third of yes/no angles' games (ties in "strongest third"), which made underdogs at 3 or 7
look like 56% in 2025-26. Counting every game: underdogs at 3 or 7 went 171-200 (46.1%) in 2025-26 and 50% in
2022-24; favorites at 3 or 7 went 200-171 (53.9%) in 2025-26, 50% before. Nothing reached 55%.

Confidence (research/confidence.py): top 10% by gap from the opening line went 53% on 2022-24 and 57%/51% on
2025/2026. Scaling the gap by the model's typical miss, requiring the efficiency model to agree, skipping weeks 1-2
and spreads of 21+, or requiring the line to move our way did no better on 2025-26 (41-57%). The site uses a fixed
7.5-point gap (90th percentile of 2022-25): 188-175 (51.8%) over 2022-26.

Top confidence (research/criteria.py, criteria_check.py): single criteria never beat 55% in both 2025 and 2026;
the best pairs found on one season fell to 35-56% on the other. One pair found on 2026 held up elsewhere: our pick did
worse against the spread in its last game than its opponent AND the play-efficiency model is on the other side of the
line. Leak-free (efficiency scale fit on other seasons): 2022 51.9%, 2023 53.3%, 2024 59.7%, 2025 50.9%, so 238-202
(54.1%) in 2022-25, about 15% of games. This is the site's top-confidence rating, tracked live.
