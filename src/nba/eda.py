"""Exploratory analysis behind report section 3 (port of ``Analyse_Graphique.ipynb``)."""
from __future__ import annotations

import pandas as pd

MIN_SHOTS = 2.0  # FGA per game, filters garbage-time shooting percentages


def attempts_per_team(players: pd.DataFrame) -> pd.DataFrame:
    """Season-average of each team's summed per-game 3PA and 2PA (report Fig. 3)."""
    team = players.groupby(["Season", "Team"])[["3PA", "2PA"]].sum().reset_index()
    return team.groupby("Season")[["3PA", "2PA"]].mean()


def three_point_share(players: pd.DataFrame) -> pd.Series:
    """League-average share of a player's shots that are threes (Fig. 4)."""
    shooters = players[players["FGA"] >= MIN_SHOTS]
    return (shooters["3PA"] / shooters["FGA"]).groupby(shooters["Season"]).mean()


def shooting_accuracy(players: pd.DataFrame) -> pd.DataFrame:
    """League-average 2P% and 3P% among players with >= 2 FGA (Fig. 5)."""
    shooters = players[players["FGA"] >= MIN_SHOTS]
    return shooters.groupby("Season")[["2P%", "3P%"]].mean()


def expected_points(players: pd.DataFrame, weighted: bool = False) -> pd.DataFrame:
    """Points per attempt = accuracy x shot value, averaged over teams (Fig. 6).

    The report (``weighted=False``) averages players' shooting percentages, so a player with
    no three-point attempts counts as a 0% shooter and drags 3P% down. ``weighted=True`` uses
    each team's made / attempted shots instead.
    """
    if weighted:
        team = players.groupby(["Season", "Team"])[["2P", "2PA", "3P", "3PA"]].sum()
        out = pd.DataFrame({"2PT": 2 * team["2P"] / team["2PA"],
                            "3PT": 3 * team["3P"] / team["3PA"]})
    else:
        team = players.groupby(["Season", "Team"])[["2P%", "3P%"]].mean()
        out = pd.DataFrame({"2PT": team["2P%"] * 2, "3PT": team["3P%"] * 3})
    return out.groupby("Season").mean()


def three_share_vs_wins(players: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """Team 3PA share of shots next to its win% (Fig. 7)."""
    team = players.groupby(["Season", "Team"])[["3PA", "2PA"]].mean().reset_index()
    team["3P_share"] = team["3PA"] / (team["3PA"] + team["2PA"])
    return team.merge(standings[["Team", "Season", "W%"]], on=["Team", "Season"])


CORR_STATS = ["PTS", "AST", "STL", "BLK", "TRB", "TS%", "PF", "TOV", "FT"]
CORR_AWARDS = ["MVP-1", "ALL-NBA-1", "DPOY-1"]


def team_correlations(players: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """Pearson matrix of team-average stats, lagged-award dummies and W% (Fig. 1)."""
    p = players[players["Season"] != 2005]  # no lagged awards for the first season
    agg = {c: "mean" for c in CORR_STATS} | {c: "max" for c in CORR_AWARDS}
    team = p.groupby(["Team", "Season"]).agg(agg).reset_index()
    team = team.merge(standings[["Team", "Season", "W%"]], on=["Team", "Season"])
    return team[["W%", *CORR_STATS, *CORR_AWARDS]].corr()


OFF_STATS = ["PTS", "AST", "ORB", "FT", "TS%"]
DEF_STATS = ["STL", "BLK", "DRB"]


def offense_defense_ratings(players: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """0-100 offensive / defensive composites per team-season (Fig. 8, Table 1).

    Each stat is scaled by games played / 82, averaged over the roster (>= 3 season-average
    minutes), min-max scaled within the season, then averaged into the two indices.
    """
    df = players.copy()
    df["MP_season"] = df["MP"] * df["G"] / 82
    df = df[df["MP_season"] >= 3]
    for stat in OFF_STATS + DEF_STATS:
        df[stat] = df[stat] * df["G"] / 82
    team = df.groupby(["Season", "Team"])[OFF_STATS + DEF_STATS].mean().reset_index()
    scaled = team.groupby("Season")[OFF_STATS + DEF_STATS].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    team["Offense"] = scaled[OFF_STATS].mean(axis=1) * 100
    team["Defense"] = scaled[DEF_STATS].mean(axis=1) * 100
    return team[["Season", "Team", "Offense", "Defense"]].merge(
        standings[["Team", "Season", "W%"]], on=["Team", "Season"])


def award_win_pct(players: pd.DataFrame, standings: pd.DataFrame) -> pd.Series:
    """Average W% by the best award on the roster last season (Fig. 11).

    Priority MVP > DPOY > All-NBA, so each team-season lands in exactly one bucket.
    """
    cols = ["TMVP-1", "TDPOY-1", "ALL-NBA-1"]
    team = players.groupby(["Team", "Season"])[cols].max().reset_index()
    df = standings.merge(team, on=["Team", "Season"], how="left").fillna({c: 0 for c in cols})
    df = df[df["Season"] != 2005]
    df["Roster"] = "No award winner"
    df.loc[df["ALL-NBA-1"] == 1, "Roster"] = "All-NBA"
    df.loc[df["TDPOY-1"] == 1, "Roster"] = "DPOY"
    df.loc[df["TMVP-1"] == 1, "Roster"] = "MVP"
    order = ["No award winner", "All-NBA", "DPOY", "MVP"]
    return df.groupby("Roster")["W%"].mean().reindex(order)
