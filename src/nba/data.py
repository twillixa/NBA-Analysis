"""Load, clean and merge the Basketball-Reference CSVs.

Port of the group's original Colab notebook (``Analyse_Principale.ipynb``, cells 5-12).
With the default arguments the output is identical to what the report was built on.
Two known quirks of the original code can be switched off for the audit notebook:

* ``ts_formula="as_coded"`` uses ``PTS / (2*FGA + 0.44*FTA)``; ``"standard"`` uses the
  real True Shooting formula ``PTS / (2*(FGA + 0.44*FTA))``.
* ``awards="as_coded"`` flags DPOY on any "DEF" (All-Defensive team) and finds the MVP/DPOY
  winner with a substring filter that drops SGA's 2025 MVP ("MVP-1DPOY-10" hits "DPOY-10");
  ``"parsed"`` matches the vote-finish tokens exactly.
* ``aggregates="keep"`` keeps the multi-team (``TOT``/``2TM``) and "League Average" rows with
  ``Team == 0``; they drop out of team-level models at the standings merge, but they leak into
  league-wide player averages. ``"drop"`` removes them.
* ``lag="row"`` builds the previous-season award flags with ``groupby(Player).shift(1)``
  (previous *row*, which for traded players can be the same season); ``"season"`` looks up
  the player's awards in season N-1 explicitly.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SEASONS = range(2005, 2026)

# Full names and relocated franchises -> current Basketball-Reference code
TEAM_CODES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK", "New Jersey Nets": "BRK", "NJN": "BRK",
    "Charlotte Hornets": "CHO", "Charlotte Bobcats": "CHO", "CHA": "CHO",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN", "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New Orleans Hornets": "NOP", "NOH": "NOP",
    "New Orleans/Oklahoma City Hornets": "NOP", "NOK": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Seattle SuperSonics": "OKC", "SEA": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Standings splits not used in the analysis (division, conference, month, All-Star break)
STANDINGS_DROP = ["E", "Apr", "A", "C", "SE", "NW", "P", "SW", "Post", "Pre", "Oct", "Nov",
                  "Dec", "Jan", "Feb", "Mar", "Jul", "Aug", "May"]
# Aggregate rows for players who changed team mid-season
MULTI_TEAM = ["2TM", "3TM", "TOT", "5TM", "4TM", "nan"]

NUMERIC_COLS = ["Rk", "Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P",
                "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL",
                "BLK", "TOV", "PF", "PTS"]
# Lower-ranked vote finishes that would otherwise match "MVP-1"/"DPOY-1" as a substring
_NOT_WINNER = ["MVP-10", "MVP-11", "MVP-12", "MVP-13", "MVP-14",
               "DPOY-15", "DPOY-16", "DPOY-17", "DPOY-10", "DPOY-11",
               "DPOY-12", "DPOY-13", "DPOY-14"]


def _split_record(df: pd.DataFrame, col: str, wins: str, losses: str, pct: str) -> None:
    df[[wins, losses]] = df[col].str.split("-", expand=True).astype(int)
    df[pct] = df[wins] / (df[wins] + df[losses])


def load_standings(seasons=SEASONS) -> pd.DataFrame:
    """One row per team-season with W, L, W% and home/road splits."""
    frames = []
    for season in seasons:
        df = pd.read_csv(RAW / f"st{season}.csv")
        df["Team"] = df["Team"].str.strip().replace(TEAM_CODES)
        df = df.drop(columns=STANDINGS_DROP, errors="ignore")
        _split_record(df, "Overall", "W", "L", "W%")
        _split_record(df, "Home", "HW", "HL", "HW%")
        _split_record(df, "Road", "RW", "RL", "RW%")
        df["Season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_player_season(season: int, awards: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"rs{season}.csv")
    df["Team"] = df["Team"].str.strip().replace(TEAM_CODES)
    # Masks multi-team cells to NaN (rows are kept; they drop out at the team merge)
    df = df[~df.isin(MULTI_TEAM)]
    df["MVP"] = df["Awards"].str.contains("MVP", na=False).astype(int)
    df["ALLNBA"] = df["Awards"].str.contains("NBA", na=False).astype(int)
    dpoy_pattern = "DEF|DPOY" if awards == "as_coded" else "DPOY"
    df["DPOY"] = df["Awards"].str.contains(dpoy_pattern, na=False).astype(int)
    df["Season"] = season
    return df


def _flag_award_winners(df: pd.DataFrame, awards: str) -> None:
    """TMVP / TDPOY = the actual winner (1st place), one per season."""
    df["exclude"] = df["Awards"].str.contains("|".join(_NOT_WINNER), na=False)
    df["TMVP"] = 0
    df["TDPOY"] = 0
    if awards == "parsed":
        # "-1" not followed by another digit = first place in the vote
        df["TMVP"] = df["Awards"].str.contains(r"MVP-1(?!\d)", na=False).astype(int)
        df["TDPOY"] = df["Awards"].str.contains(r"DPOY-1(?!\d)", na=False).astype(int)
        return
    for _, sub in df.groupby("Season"):
        mvp = sub[sub["Awards"].str.contains("MVP-1", na=False) & ~sub["exclude"]]
        if not mvp.empty:
            df.at[mvp.sort_values(["MVP", "PTS"], ascending=False).index[0], "TMVP"] = 1
        dpoy = sub[sub["Awards"].str.contains("DPOY-1", na=False) & ~sub["exclude"]]
        if not dpoy.empty:
            best = dpoy.sort_values(["DPOY", "STL", "BLK"], ascending=False).index[0]
            df.at[best, "TDPOY"] = 1


LAGGED = {"ALL-NBA-1": "ALLNBA", "MVP-1": "MVP", "DPOY-1": "DPOY",
          "TMVP-1": "TMVP", "TDPOY-1": "TDPOY"}


def _add_lagged_awards(df: pd.DataFrame, lag: str) -> None:
    if lag == "row":
        for new, src in LAGGED.items():
            df[new] = df.groupby("Player")[src].shift(1).fillna(0).astype(int)
    elif lag == "season":
        per_season = df.groupby(["Player", "Season"])[list(LAGGED.values())].max()
        per_season = per_season.rename(columns={v: k for k, v in LAGGED.items()}).reset_index()
        per_season["Season"] += 1
        merged = df[["Player", "Season"]].merge(per_season, on=["Player", "Season"], how="left")
        for new in LAGGED:
            df[new] = merged[new].fillna(0).astype(int).to_numpy()
    else:
        raise ValueError(f"lag must be 'row' or 'season', got {lag!r}")


def true_shooting(pts, fga, fta, formula: str = "as_coded"):
    if formula == "as_coded":
        return pts / (2 * fga + 0.44 * fta)
    if formula == "standard":
        return pts / (2 * (fga + 0.44 * fta))
    raise ValueError(f"formula must be 'as_coded' or 'standard', got {formula!r}")


def load_players(seasons=SEASONS, *, ts_formula: str = "as_coded", lag: str = "row",
                 awards: str = "as_coded", aggregates: str = "keep") -> pd.DataFrame:
    """One row per player-team-season with award dummies, lagged awards and TS%."""
    if awards not in ("as_coded", "parsed"):
        raise ValueError(f"awards must be 'as_coded' or 'parsed', got {awards!r}")
    df = pd.concat([_load_player_season(s, awards) for s in seasons], ignore_index=True)
    df = df[~df.isin(MULTI_TEAM)]
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    for col in ["ALLNBA", "MVP", "DPOY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    _flag_award_winners(df, awards)
    _add_lagged_awards(df, lag)
    df["TS%"] = true_shooting(df["PTS"], df["FGA"], df["FTA"], ts_formula)
    # As in the original: NaNs (multi-team Team, missing %s, "League Average" row) -> 0
    df = df.fillna(0)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_COLS)
    if aggregates == "drop":
        df = df[(df["Team"] != 0) & (df["Player"] != "League Average")]
    elif aggregates != "keep":
        raise ValueError(f"aggregates must be 'keep' or 'drop', got {aggregates!r}")
    return df


# Every fix found while porting (see notebooks/03_out_of_sample_audit.ipynb)
CORRECTED = dict(ts_formula="standard", lag="season", awards="parsed", aggregates="drop")


def name_key(name: str) -> str:
    """Accent-, case-, punctuation- and suffix-insensitive key for matching player names."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = re.sub(r"[.']", "", name.lower()).replace("-", " ")
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", name)
    return " ".join(name.split())


# Misspellings in rs2026.csv -> Basketball-Reference spelling
ROSTER_TYPOS = {
    "AJ Jonhson": "AJ Johnson", "Al Hortford": "Al Horford",
    "Bodgan Bogdanovic": "Bogdan Bogdanović", "Devin Vassel": "Devin Vassell",
    "E.J. Liddel": "E.J. Liddell", "Garett Temple": "Garrett Temple",
    "Joran Poole": "Jordan Poole", "Julian Philips": "Julian Phillips",
    "Kari-Anthony Towns": "Karl-Anthony Towns", "Mikai Bridges": "Mikal Bridges",
    "Normal Powell": "Norman Powell", "Russel Westbrook": "Russell Westbrook",
    "Dominik Barlow": "Dominick Barlow", "Ronald Holland II": "Ron Holland",
}


def load_roster_2026(match_names: bool = False, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """Opening-night 2025-26 rosters compiled by the group (Player, Team_2026).

    With ``match_names=True`` each name is mapped onto the spelling used in ``players``
    (whitespace, typos, accents, Jr./III suffixes); unmatched names are genuine newcomers.
    """
    df = pd.read_csv(RAW / "rs2026.csv", sep=";", encoding="utf-8-sig")
    if not match_names:
        return df
    if players is None:
        raise ValueError("match_names=True needs the historical players frame")
    known = (players.sort_values("Season").drop_duplicates("Player", keep="last")
             .assign(key=lambda d: d["Player"].map(name_key)).drop_duplicates("key", keep="last")
             .set_index("key")["Player"])
    names = df["Player"].str.strip().replace(ROSTER_TYPOS)
    df["Player"] = names.map(name_key).map(known).fillna(names)
    return df


def build_datasets(**kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (players, standings) — the notebook's ``nbas_data_full`` / ``sts_data_full``."""
    return load_players(**kwargs), load_standings()

