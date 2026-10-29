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
python research/summary.py $W .                          # -> data/research.json (shown on the Lines tab)
```

`tune.py` also holds the backtest harness used to choose the prior, margin cap and efficiency weight.

Held-out results so far (edge 3+ points vs the opening line, 2022-25 pooled): score model 51.1%, play efficiency
51.6%, both 50.9%; vs closing lines 48-50%. 2025 alone reaches 55-58%, but 2022 is 43-49%. Situational model (no ratings): 49.0% on 2025-26; the
steadiest single angle, conference-game underdogs, 53.2% over 2022-26 closing lines, best of 10 angles tried.

`data/history/2026_qb.json` (and 2026's PPA, talent and returning files) are this season's data through the week it was fetched (extras-only run), kept so
the QB rule's 2026 test can be reproduced; it does not update.
