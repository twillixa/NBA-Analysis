"""Team win% models and the 2025-26 forecast.

Port of ``Analyse_Principale.ipynb`` cells 17-26. Player rows (>= 5 MPG) are min-max scaled,
averaged per team-season, and regressed on the team's win percentage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .data import true_shooting

AWARD_FEATURES = ["TMVP-1", "TDPOY-1", "MVP-1", "DPOY-1", "ALL-NBA-1"]
RF_FEATURES = ["FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA",
               "FT%", "ORB", "DRB", "TRB", "AST", "STL", "TS%", "BLK", "TOV", "PF", "PTS",
               *AWARD_FEATURES]
OLS_STATS = ["MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT",
             "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
OLS_AWARDS = ["MVP-1", "DPOY-1", "ALL-NBA-1", "TMVP-1", "TDPOY-1"]
MIN_MINUTES = 5
GAMES_PER_SEASON = 82
TOTAL_WINS = 1230  # 30 teams x 82 games / 2


def rf_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)


def _rotation(players: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    df["MP"] = pd.to_numeric(df["MP"], errors="coerce")
    return df[df["MP"] >= MIN_MINUTES].copy()


def _scale(df: pd.DataFrame, cols, ranges: pd.DataFrame) -> pd.DataFrame:
    for col in cols:
        mn, mx = ranges[col].min(), ranges[col].max()
        df[f"{col}_norm"] = 0.0 if mx <= mn else (df[col] - mn) / (mx - mn)
    return df


def rf_team_table(players: pd.DataFrame, standings: pd.DataFrame | None = None,
                  ranges: pd.DataFrame | None = None) -> pd.DataFrame:
    """Team-season mean of min-max scaled player stats (award flags become roster shares)."""
    df = _rotation(players)
    df = _scale(df, RF_FEATURES, df if ranges is None else ranges)
    norm = [f"{c}_norm" for c in RF_FEATURES]
    teams = df.groupby(["Team", "Season"])[norm].mean().reset_index()
    if standings is not None:
        teams = teams.merge(standings[["Team", "Season", "W%", "Rk"]], on=["Team", "Season"])
    return teams


def ols_team_table(players: pd.DataFrame, standings: pd.DataFrame) -> pd.DataFrame:
    """Same idea as the RF table, but awards enter as 0/1 'team has one' dummies."""
    df = _rotation(players)
    for col in OLS_AWARDS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = _scale(df, OLS_STATS, df)
    agg = {f"{c}_norm": "mean" for c in OLS_STATS} | {c: "max" for c in OLS_AWARDS}
    teams = df.groupby(["Team", "Season"]).agg(agg).reset_index()
    return teams.merge(standings[["Team", "Season", "W%", "Rk"]], on=["Team", "Season"])


def rf_features(table: pd.DataFrame) -> pd.DataFrame:
    return table[[f"{c}_norm" for c in RF_FEATURES]].fillna(0)


def ols_features(table: pd.DataFrame) -> pd.DataFrame:
    return table[[f"{c}_norm" for c in OLS_STATS] + OLS_AWARDS].fillna(0)


def fit_rf_holdout(table: pd.DataFrame):
    """Random 80/20 split as in the report. Returns (model, test R², importances)."""
    X, y = rf_features(table), table["W%"].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = rf_model().fit(X_train, y_train)
    importances = (pd.Series(rf.feature_importances_, index=RF_FEATURES, name="Importance")
                   .sort_values(ascending=False))
    return rf, rf.score(X_test, y_test), importances


def fit_ols(table: pd.DataFrame):
    return sm.OLS(table["W%"].fillna(0), sm.add_constant(ols_features(table))).fit()


def rank_by_season(table: pd.DataFrame, pred_col: str = "W%_pred") -> pd.DataFrame:
    """Actual vs predicted standings rank (1 = best record) for every team-season.

    Actual rank is Basketball-Reference's ``Rk``, which applies the NBA tiebreakers
    (e.g. CHI over SAS at 50-16 in 2012).
    """
    out = []
    for _, sub in table.groupby("Season"):
        sub = sub.sort_values("Rk")
        sub["Actual_rank"] = np.arange(1, len(sub) + 1)
        sub = sub.sort_values(pred_col, ascending=False, kind="stable")
        sub["Pred_rank"] = np.arange(1, len(sub) + 1)
        out.append(sub)
    ranks = pd.concat(out)
    ranks["Rank_error"] = (ranks["Pred_rank"] - ranks["Actual_rank"]).abs()
    return ranks[["Season", "Team", "W%", pred_col, "Actual_rank", "Pred_rank", "Rank_error"]]


def leader_table(ranks: pd.DataFrame, seasons=range(2006, 2026)) -> pd.DataFrame:
    """Per season: actual #1, predicted #1, and where the predicted #1 actually finished."""
    rows = []
    for season in seasons:
        sub = ranks[ranks["Season"] == season]
        actual = sub.loc[sub["Actual_rank"] == 1, "Team"].iloc[0]
        pred = sub.loc[sub["Pred_rank"] == 1].iloc[0]
        rows.append({"Season": season, "Actual_leader": actual, "Predicted_leader": pred["Team"],
                     "Predicted_leader_finished": int(pred["Actual_rank"])})
    table = pd.DataFrame(rows)
    table["Hit"] = table["Actual_leader"] == table["Predicted_leader"]
    return table


# ---------------------------------------------------------------- 2025-26 projection

COUNT_STATS = ["MP", "FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA",
               "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
# Flat stat line for players with no NBA history
ROOKIE_BASELINE = {"PTS": 10, "AST": 3, "TRB": 4, "ORB": 2, "DRB": 2, "MP": 6,
                   "3PA": 1, "2PA": 8, "FTA": 1, "BLK": 0.5, "STL": 0.5, "PF": 1,
                   "FG": 3.5, "FGA": 10, "3P": 0.3, "2P": 3.2, "FT": 0.7, "TOV": 1.5}
# Set by hand in the original: 2025 MVP / DPOY winners and the two accented-name MVP finalists
MANUAL_AWARDS_2026 = [("Shai Gilgeous-Alexander", "TMVP-1"), ("Evan Mobley", "TDPOY-1"),
                      ("Luka Dončić", "MVP-1"), ("Nikola Jokić", "MVP-1")]


def roster_2026_frame(players: pd.DataFrame, roster: pd.DataFrame,
                      carry_winners: bool = False) -> pd.DataFrame:
    """2026 player rows (stats zeroed) with 2025 award flags carried over as lagged awards.

    The original only carried nominations; ``carry_winners`` also carries the actual
    MVP / DPOY winner flags instead of relying on ``MANUAL_AWARDS_2026``.
    """
    columns = players.columns.tolist()
    out = roster[["Player"]].copy()
    out["Team_2026"] = roster["Team_2026"]
    out["Season"] = 2026
    out = out.merge(players.groupby("Player")["Age"].max().reset_index(), on="Player", how="left")
    out["Age"] = out["Age"].fillna(20).astype(int) + 1
    pos = players.groupby("Player")["Pos"].agg(lambda x: x.mode()[0] if not x.mode().empty else "SF")
    out = out.merge(pos.reset_index(), on="Player", how="left")
    out["Pos"] = out["Pos"].fillna("SF")
    for col in columns:
        if col not in ["Player", "Season", "Age", "Pos", "Team_2026"]:
            out[col] = np.nan if col in ["Rk", "Player-additional", "Awards"] else 0
    out["Team"] = out.pop("Team_2026")

    last = players[players["Season"] == 2025].groupby("Player").first()
    for col, src in [("MVP", "MVP"), ("DPOY", "DPOY"), ("ALLNBA", "ALLNBA"),
                     ("MVP-1", "MVP"), ("DPOY-1", "DPOY"), ("ALL-NBA-1", "ALLNBA"),
                     *([("TMVP-1", "TMVP"), ("TDPOY-1", "TDPOY")] if carry_winners else [])]:
        out[col] = out["Player"].map((last[src] >= 1).astype(int)).fillna(0).astype(int)
    out = out[columns]
    out.loc[out["Player"].str.contains("Jokic", case=False), "Player"] = "Nikola Jokić"
    out.loc[out["Player"].str.contains("Doncic", case=False), "Player"] = "Luka Dončić"
    return out


def _player_seasons(players: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season; traded players' team stints are games-weighted."""
    df = players[["Player", "Season", "G", *COUNT_STATS]].copy()
    w = df["G"].clip(lower=1)
    sums = df[COUNT_STATS].mul(w, axis=0).groupby([df["Player"], df["Season"]]).sum()
    return sums.div(w.groupby([df["Player"], df["Season"]]).sum(), axis=0).reset_index()


def _projection_history(players: pd.DataFrame, roster_rows: pd.DataFrame, fixes: bool):
    """Frame of (Player, Season, stats) rows to compute lags on, 2026 rows last."""
    if not fixes:
        return pd.concat([players, roster_rows], ignore_index=True)
    return pd.concat([_player_seasons(players), roster_rows[["Player", "Season", *COUNT_STATS]]],
                     ignore_index=True)


def project_2026_players(players: pd.DataFrame, roster: pd.DataFrame, fixes: bool = False,
                         ts_formula: str = "as_coded") -> pd.DataFrame:
    """Return history + 2026 rows whose box-score stats are projected from the last 3 seasons.

    ``fixes=True`` corrects three issues in the original projection: the regression no longer
    trains on the (all-zero) 2026 rows, a player's history is one row per season (not per
    team stint), and second/third-year players reuse their latest season for missing lags
    instead of zeros.
    """
    roster_rows = roster_2026_frame(players, roster, carry_winners=fixes)
    full = pd.concat([players, roster_rows], ignore_index=True)
    lagged = _projection_history(players, roster_rows, fixes)
    for col in COUNT_STATS:
        for i in (1, 2, 3):
            lagged[f"{col}_lag{i}"] = lagged.groupby("Player")[col].shift(i)

    is_2026 = full["Season"] == 2026
    lag_2026 = lagged["Season"] == 2026
    predictions: dict[str, dict[str, float]] = {}
    for stat in COUNT_STATS:
        lags = [f"{stat}_lag{i}" for i in (1, 2, 3)]
        veterans = lagged[lag_2026 & lagged[lags].notna().any(axis=1)]
        train = (lagged["Season"] >= 2009) & lagged[lags].notna().all(axis=1)
        if fixes:
            train &= ~lag_2026
        X_new = veterans[lags].ffill(axis=1) if fixes else veterans[lags].fillna(0)
        model = LinearRegression().fit(lagged.loc[train, lags], lagged.loc[train, stat])
        for player, value in zip(veterans["Player"], model.predict(X_new)):
            predictions.setdefault(player, {})[stat] = max(0, value)

    full.loc[is_2026, COUNT_STATS] = 0
    for player, stats in predictions.items():
        mask = is_2026 & (full["Player"] == player)
        for stat, value in stats.items():
            full.loc[mask, stat] = round(value, 1)
    rookies = is_2026 & (full["PTS"] == 0)
    for stat, value in ROOKIE_BASELINE.items():
        full.loc[rookies, stat] = value

    sub = full.loc[is_2026]
    fga = sub["FGA"].replace(0, np.nan)
    full.loc[is_2026, "FG%"] = sub["FG"] / fga
    full.loc[is_2026, "3P%"] = sub["3P"] / sub["3PA"].replace(0, np.nan)
    full.loc[is_2026, "2P%"] = sub["2P"] / sub["2PA"].replace(0, np.nan)
    full.loc[is_2026, "FT%"] = sub["FT"] / sub["FTA"].replace(0, np.nan)
    full.loc[is_2026, "eFG%"] = (sub["FG"] + 0.5 * sub["3P"]) / fga
    ts = true_shooting(sub["PTS"], sub["FGA"], sub["FTA"], ts_formula)
    full.loc[is_2026, "TS%"] = ts.replace([np.inf, -np.inf], np.nan)
    for col in ["FG%", "3P%", "2P%", "FT%", "eFG%", "TS%"]:
        hist_mean = full.loc[~is_2026, col].mean()
        full.loc[is_2026 & full[col].isna(), col] = hist_mean

    for player, col in MANUAL_AWARDS_2026:
        full.loc[is_2026 & (full["Player"] == player), col] = 1
    return full


def balance_wins(pred: pd.DataFrame, total: int = TOTAL_WINS) -> pd.DataFrame:
    """Round W% to wins, then nudge worst-first, one win at a time, until wins sum to 1230."""
    pred = pred.sort_values("W%_pred").reset_index(drop=True)
    pred["W_pred"] = (pred["W%_pred"] * GAMES_PER_SEASON).round(0).astype(int)
    i = 0
    while (diff := total - pred["W_pred"].sum()) != 0:
        pred.at[i % len(pred), "W_pred"] += 1 if diff > 0 else -1
        i += 1
    pred["L_pred"] = GAMES_PER_SEASON - pred["W_pred"]
    return pred.sort_values(["W_pred", "W%_pred"], ascending=False).reset_index(drop=True)


def forecast_2026(players: pd.DataFrame, standings: pd.DataFrame, roster: pd.DataFrame,
                  fixes: bool = False, ts_formula: str = "as_coded") -> pd.DataFrame:
    """Train the RF on every 2005-25 team-season and predict the 2025-26 standings."""
    history = rf_team_table(players, standings)
    rf = rf_model().fit(rf_features(history), history["W%"].fillna(0))
    full = project_2026_players(players, roster, fixes=fixes, ts_formula=ts_formula)
    # scale 2026 rows with the same (rotation-player) ranges the forest was trained on
    new = rf_team_table(full[full["Season"] == 2026], ranges=_rotation(players))
    new["W%_pred"] = rf.predict(rf_features(new))
    return balance_wins(new[["Team", "W%_pred"]])


# ---------------------------------------------------------------- honest evaluation

def walk_forward(table: pd.DataFrame, features=rf_features, make_model=rf_model,
                 first_test_season: int = 2010) -> pd.DataFrame:
    """Predict each season using only seasons before it (no look-ahead).

    Returns the team-season rows of every test season with an out-of-sample ``W%_pred``.
    """
    preds = []
    for season in sorted(s for s in table["Season"].unique() if s >= first_test_season):
        train, test = table[table["Season"] < season], table[table["Season"] == season].copy()
        model = make_model().fit(features(train), train["W%"].fillna(0))
        test["W%_pred"] = model.predict(features(test))
        preds.append(test)
    return pd.concat(preds, ignore_index=True)


def summarize(pred: pd.DataFrame) -> dict:
    """R², MAE in wins, rank MAE and leader hit-rate for a table with ``W%_pred``."""
    ranks = rank_by_season(pred)
    leaders = leader_table(ranks, seasons=sorted(pred["Season"].unique()))
    resid = pred["W%"] - pred["W%_pred"]
    r2 = 1 - (resid ** 2).sum() / ((pred["W%"] - pred["W%"].mean()) ** 2).sum()
    return {"R2": r2,
            "MAE_wins": (resid.abs() * GAMES_PER_SEASON).mean(),
            "Rank_MAE": ranks["Rank_error"].mean(),
            "Leader_hits": int(leaders["Hit"].sum()),
            "Seasons": len(leaders),
            "Leader_top3": int((leaders["Predicted_leader_finished"] <= 3).sum())}
