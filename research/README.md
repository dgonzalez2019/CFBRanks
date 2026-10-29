# Research scripts

How the model settings and the Lines tab's past-season stats were produced. They read the season files in
`data/history/` (download with the "Fetch past seasons" workflow) and write intermediate files to a work
directory. Run from the repo root, in order:

```
W=/tmp/cfb-work; mkdir -p $W
python research/predictions.py research $W/preds.json   # our week-by-week predictions, 2022-26 (uses tune.py)
python research/vs_market.py $W .                         # vs. betting lines -> $W/joined.json
python research/qb_injuries.py $W .                      # games a starting QB missed -> $W/qb_first_missed.json
python research/summary.py $W .                          # -> data/research.json (shown on the Lines tab)
```

`tune.py` also holds the backtest harness used to choose the prior, margin cap and efficiency weight.
