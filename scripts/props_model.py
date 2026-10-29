"""NFL player-prop projections, shared by research/nfl_props.py (fit and test) and scripts/track_props.py (live).

proj = c0 + c1*base + c2*base*(opp - 1) + c3*base*(imp - 1), where
  base = recency-weighted average of the player's earlier games this season (each older game counts DECAY as much
         as the next) with last season's per-game average added as PRIOR_GAMES pseudo-games
  opp  = what the opponent has allowed per game to the player's position group this season vs. the league average,
         shrunk by OPP_SHRINK games (1 = average)
  imp  = the team's implied points from the betting line (total / 2 - own spread / 2) / 22.5
P(over a line) = share of the training seasons' actual / projection spread (by projection bin) above the line."""
import csv, io, numpy as np
from collections import defaultdict

DECAY, PRIOR_GAMES, OPP_SHRINK, BINS = 0.85, 3.0, 4.0, 5
MARKETS = {  # our market -> (stat column, positions, Odds API market key, label)
    "pass_yds": ("passing_yards", ("QB",), "player_pass_yds", "passing yds"),
    "rush_yds": ("rushing_yards", ("QB", "RB"), "player_rush_yds", "rushing yds"),
    "rec_yds": ("receiving_yards", ("WR", "TE", "RB"), "player_reception_yds", "receiving yds"),
    "receptions": ("receptions", ("WR", "TE", "RB"), "player_receptions", "receptions"),
}
COLS = ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "attempts", "carries", "targets")


def load_rows(text, teams):
    """nflverse stats_player_week CSV text -> skill-player game rows (regular season + playoffs)."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("season_type") not in ("REG", "POST") or r.get("position") not in ("QB", "RB", "WR", "TE"): continue
        f = lambda k: float(r[k]) if r.get(k) not in (None, "", "NA") else 0.0
        rows.append(dict(id=r["player_id"], name=r["player_display_name"], pos=r["position"], week=int(r["week"]),
                         team=teams.get(r["team"]), opp=teams.get(r["opponent_team"]), **{k: f(k) for k in COLS}))
    return rows


def group(pos): return "WR/TE" if pos in ("WR", "TE") else pos


def eligible(pos, h):
    if len(h) < 3: return False
    m = lambda k: np.mean([g[k] for g in h])
    if pos == "QB": return m("attempts") >= 15
    if pos == "RB": return m("carries") >= 8 or m("targets") >= 3
    return m("targets") >= 3


class Season:
    """One season's games, for projections that use only games before a given week."""
    def __init__(self, rows, prev_rows=None):
        self.games = defaultdict(list)
        for g in sorted(rows, key=lambda g: g["week"]): self.games[g["id"]].append(g)
        self.allowed = defaultdict(list)   # (defense, group, col) -> [(week, value)]
        self.dweeks = defaultdict(set)     # defense -> weeks played
        self.league = defaultdict(list)    # (group, col) -> [(week, value)]
        for g in rows:
            if not g["opp"]: continue
            self.dweeks[g["opp"]].add(g["week"])
            for c in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
                self.allowed[(g["opp"], group(g["pos"]), c)].append((g["week"], g[c]))
                self.league[(group(g["pos"]), c)].append((g["week"], g[c]))
        self._cache, self._lg = {}, {}
        self.prev = defaultdict(dict)
        if prev_rows:
            by = defaultdict(list)
            for g in prev_rows: by[g["id"]].append(g)
            for pid, gs in by.items():
                for mk, (col, *_) in MARKETS.items(): self.prev[pid][mk] = float(np.mean([g[col] for g in gs]))

    def history(self, pid, wk): return [g for g in self.games[pid] if g["week"] < wk]

    def opp_factor(self, opp, grp, col, wk):
        k = (opp, grp, col, wk)
        if k not in self._cache: self._cache[k] = self._opp_factor(opp, grp, col, wk)
        return self._cache[k]

    def _league(self, grp, col, wk):
        k = (grp, col, wk)
        if k not in self._lg:
            team_games = sum(sum(1 for w in ws if w < wk) for ws in self.dweeks.values())
            self._lg[k] = sum(v for w, v in self.league[(grp, col)] if w < wk) / team_games if team_games else 0
        return self._lg[k]

    def _opp_factor(self, opp, grp, col, wk):
        lg = self._league(grp, col, wk)
        if not lg: return 1.0
        od = sum(1 for w in self.dweeks[opp] if w < wk)
        a = sum(v for w, v in self.allowed[(opp, grp, col)] if w < wk)
        return ((a + OPP_SHRINK * lg) / (od + OPP_SHRINK)) / lg

    def features(self, pid, pos, opp, wk, mk, imp):
        """[1, base, base*(opp-1), base*(imp-1)] plus the parts, or None if the player isn't prop-eligible."""
        col, poss, *_ = MARKETS[mk]
        h = self.history(pid, wk)
        if pos not in poss or not eligible(pos, h): return None
        w = [DECAY ** i for i in range(len(h))][::-1]
        s = sum(wi * g[col] for wi, g in zip(w, h)); n = sum(w)
        prior = self.prev[pid].get(mk)
        if prior is not None: s += PRIOR_GAMES * prior; n += PRIOR_GAMES
        base = s / n
        o = self.opp_factor(opp, group(pos), col, wk); i = imp / 22.5
        return dict(x=[1, base, base * (o - 1), base * (i - 1)], base=base, opp=o, imp=imp, avg=float(np.mean([g[col] for g in h])),
                    last3=float(np.mean([g[col] for g in h[-3:]])), games=len(h), prior=prior)


def implied_points(lines):
    """lines.json rows -> {(week, team): implied points}."""
    out = {}
    for r in lines:
        wk, h, a, sp, tot = r[0], r[1], r[2], r[4], r[5]
        if sp is None or tot is None: continue
        out[(wk, h)] = tot / 2 - sp / 2; out[(wk, a)] = tot / 2 + sp / 2
    return out


def project(model, x): return max(0.5, float(np.dot(model["coef"], x)))


def p_over(model, proj, line):
    i = int(np.searchsorted(model["edges"], proj)); q = np.array(model["q"][i]) * proj
    return float(1 - np.interp(line, q, np.linspace(0, 1, 101), left=0, right=1))


def fit_quantiles(projs, ys):
    p = np.array(projs); y = np.array(ys)
    edges = np.quantile(p, np.linspace(0, 1, BINS + 1))[1:-1]; b = np.searchsorted(edges, p)
    return edges.tolist(), [np.quantile(y[b == i] / p[b == i], np.linspace(0, 1, 101)).tolist() for i in range(BINS)]
