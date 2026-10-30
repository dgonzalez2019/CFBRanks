#!/usr/bin/env python3
"""Live NFL player-prop tracker. For each prop in data/nfl/props_lines.json (fetch_props.py) on a game not yet
started, projects the player's stat with scripts/props_model.py (coefficients and outcome spread from
research/nfl_props.json) and saves our side and its probability. Records freeze at kickoff and are graded from
nflverse weekly player stats once the game is final (a player who didn't play is void).

data/nfl/props.json = {updated, season, test: {market: research test numbers}, props: [{key, week, start, home, away,
  team, opp, player, pos, mk, line, over, under, books, proj, base, avg, last3, games, opp_f, imp, vac, qb_out,
  mates_out, inj_adj, p_over, side,
  conf, book_p, top10, top5, actual, won}]}
conf = our probability that our side wins; book_p = the books' no-vig probability for that side (anytime TD with only a
Yes price: its implied probability, vig included). Anytime TD: side "yes" unless we rate No higher than the books and
No is offered. top10 / top5 mark each week's most confident 10% / 5% within each market (anytime TD: by our edge over
the books), frozen at kickoff.
Usage: python scripts/track_props.py"""
import datetime, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import props_model as pm
from cfb_sor import name_key
from fetch_nfl import REL, TEAMS, get

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data" / "nfl"; OUT = D / "props.json"


def load(p, key=None, default=None):
    try:
        j = json.loads(p.read_text()); return j[key] if key else j
    except (OSError, ValueError, KeyError): return default


def implied_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def main():
    lines_store = load(D / "props_lines.json", default={"games": {}})
    model = load(ROOT / "research" / "nfl_props.json")
    gj = load(D / "games.json"); season = gj["season"]
    old = {p["key"]: p for p in load(OUT, "props", [])}
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        cur = pm.load_rows(get(f"{REL}/stats_player/stats_player_week_{season}.csv"), TEAMS)
        prev = pm.load_rows(get(f"{REL}/stats_player/stats_player_week_{season - 1}.csv"), TEAMS)
    except Exception as e:
        print(f"Player stats unavailable: {e}"); return
    S = pm.Season(cur, prev)
    lines = load(D / "lines.json", "games", [])
    imp = pm.implied_points(lines)
    week_of = {}
    for r in lines: week_of.setdefault((r[1], r[2]), []).append((r[0], r[8]))
    final = {(g[0], g[1], g[2]) for g in gj["games"]}
    inj = load(D / "injuries.json", "injuries", [])
    out_list = {(r[0], name_key(r[1])) for r in inj if r[3] == "Out"}
    missing = {(r[0], name_key(r[1])) for r in inj if r[3] in ("Out", "Doubtful")}  # teammates counted as missing
    # latest team / position / id for each player this season
    who = {}
    for g in sorted(cur, key=lambda g: g["week"]): who[(g["team"], name_key(g["name"]))] = g
    stat = {(g["id"], g["week"]): g for g in cur}
    out_ids = {}
    for (t, nk), g in who.items():
        if (t, nk) in missing: out_ids.setdefault(t, []).append(g["id"])

    props = dict(old)
    for gid, G in lines_store.get("games", {}).items():
        start = datetime.datetime.fromisoformat(G["start"].replace("Z", "+00:00"))
        cands = week_of.get((G["home"], G["away"]), [])
        if not cands: continue
        wk = min(cands, key=lambda c: abs((datetime.datetime.fromisoformat((c[1] or G["start"]).replace("Z", "+00:00")) - start).total_seconds()))[0]
        if start <= now: continue  # frozen
        for mk, player, line, over, under, books in G.get("props", []):
            key = f"{wk}|{G['home']}|{G['away']}|{mk}|{player}"
            p = who.get((G["home"], name_key(player))) or who.get((G["away"], name_key(player)))
            if not p or (p["team"], name_key(player)) in out_list: props.pop(key, None); continue
            opp = G["away"] if p["team"] == G["home"] else G["home"]
            f = S.features(p["id"], p["pos"], opp, wk, mk, imp.get((wk, p["team"]), 22.5), p["team"], out_ids.get(p["team"], []))
            if not f: props.pop(key, None); continue
            if mk not in model: continue
            dp = 2 if mk in pm.COUNT_MK else 1
            M = model[mk]; proj = pm.project(M, f["x"]); po = pm.p_over(M, proj, line)
            inj_adj = M["coef"][4] * f["x"][4] + M["coef"][5] * f["x"][5]  # points of the projection from teammates out
            if pm.MARKETS[mk][4] == "yes":
                by = implied_prob(over); bn = implied_prob(under) if under is not None else None
                byes = by / (by + bn) if bn is not None else by
                side = "no" if bn is not None and (1 - po) > (1 - byes) and po < byes else "yes"
                conf = po if side == "yes" else 1 - po; bp = byes if side == "yes" else 1 - byes
            else:
                side = "over" if po >= 0.5 else "under"; conf = po if side == "over" else 1 - po
                bo, bu = implied_prob(over), implied_prob(under); bp = (bo if side == "over" else bu) / (bo + bu)
            props[key] = dict(key=key, week=wk, start=G["start"], home=G["home"], away=G["away"], team=p["team"], opp=opp,
                              player=player, pid=p["id"], pos=p["pos"], mk=mk, line=line, over=over, under=under, books=books,
                              proj=round(proj, dp), base=round(f["base"], dp), avg=round(f["avg"], dp), last3=round(f["last3"], dp),
                              games=f["games"], prior=None if f["prior"] is None else round(f["prior"], dp),
                              opp_f=round(f["opp"], 3), imp=round(f["imp"], 1), vac=round(f["vac"], 3), qb_out=f["qb_out"],
                              mates_out=f["out"], inj_adj=round(inj_adj, 1), p_over=round(po, 4), side=side, conf=round(conf, 4),
                              book_p=round(bp, 4), actual=None, won=None)
    # weekly top 10% / 5% among props not yet started (frozen ones keep their flags)
    for wk, mk in {(p["week"], p["mk"]) for p in props.values()}:
        rank = (lambda p: p["conf"] - p["book_p"]) if pm.MARKETS[mk][4] == "yes" else (lambda p: p["conf"])
        live = [p for p in props.values() if p["week"] == wk and p["mk"] == mk and datetime.datetime.fromisoformat(p["start"].replace("Z", "+00:00")) > now]
        allw = sorted((p for p in props.values() if p["week"] == wk and p["mk"] == mk), key=lambda p: -rank(p))
        n10, n5 = max(1, round(len(allw) * .10)), max(1, round(len(allw) * .05))
        ranked = {p["key"]: i for i, p in enumerate(allw)}
        for p in live: p["top10"] = ranked[p["key"]] < n10; p["top5"] = ranked[p["key"]] < n5
    # grading: only once the team's player stats for that week are posted (they lag the final score by hours);
    # a player with no stats when his teammates have them didn't play: void
    posted = {(g["team"], g["week"]) for g in cur}
    for p in props.values():
        if p["won"] is not None or p.get("void") or (p["week"], p["home"], p["away"]) not in final: continue
        if (p["team"], p["week"]) not in posted: continue
        s = stat.get((p.get("pid"), p["week"]))
        if not s: p["void"] = True; continue
        a = s[pm.MARKETS[p["mk"]][0]]; p["actual"] = a
        p["won"] = None if a == p["line"] else (a > p["line"]) == (p["side"] in ("over", "yes"))
        if a == p["line"]: p["void"] = True
    test = {mk: model[mk].get("test") for mk in pm.MARKETS}
    OUT.write_text(json.dumps({"updated": now.isoformat(timespec="seconds"), "season": season, "remaining": lines_store.get("remaining"),
                               "test": test, "props": sorted(props.values(), key=lambda p: (p["start"], -p["conf"]))}, separators=(",", ":")))
    g = [p for p in props.values() if p["won"] is not None]
    print(f"Props tracker: {len(props)} props, {len(g)} graded ({sum(p['won'] for p in g)} won)")


if __name__ == "__main__":
    main()
