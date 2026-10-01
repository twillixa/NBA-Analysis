"""The cleaned datasets must match what the original notebook produced (report Appendix C)."""
import pytest

from nba.data import build_datasets, load_roster_2026, true_shooting


@pytest.fixture(scope="module")
def datasets():
    return build_datasets()


def test_shapes_match_original_notebook(datasets):
    players, standings = datasets
    assert players.shape == (13313, 45)
    assert standings.shape == (630, 17)


def test_every_season_has_30_teams(datasets):
    _, standings = datasets
    assert standings.groupby("Season")["Team"].nunique().eq(30).all()
    assert standings["Season"].between(2005, 2025).all()


def test_team_codes_match_player_codes(datasets):
    players, standings = datasets
    player_teams = set(players["Team"]) - {0}
    assert set(standings["Team"]) == player_teams


def test_records_add_up(datasets):
    _, standings = datasets
    games = standings["W"] + standings["L"]
    assert (standings["HW"] + standings["RW"] == standings["W"]).all()
    # 82 games except the 2012 lockout (66), the COVID seasons 2020 (64-75) / 2021 (72),
    # and BOS-IND 2013 (game cancelled after the Boston Marathon bombing)
    assert games[~standings["Season"].isin([2012, 2013, 2020, 2021])].eq(82).all()
    assert sorted(standings.loc[games == 81, "Team"]) == ["BOS", "IND"]
    assert games[standings["Season"] == 2012].eq(66).all()
    assert games[standings["Season"] == 2021].eq(72).all()


def test_standings_descriptives_match_report_table5(datasets):
    _, standings = datasets
    assert standings["W"].mean() == pytest.approx(40.11, abs=0.005)
    assert standings["W%"].max() == pytest.approx(0.89, abs=0.005)
    assert standings["HW%"].mean() == pytest.approx(0.58, abs=0.005)


def test_player_descriptives_match_report_table4(datasets):
    players, _ = datasets
    assert players["Rk"].mean() == pytest.approx(257.92, abs=0.005)
    assert players["Age"].mean() == pytest.approx(26.37, abs=0.005)
    assert players["FGA"].mean() == pytest.approx(6.61, abs=0.005)


def test_one_mvp_winner_per_season(datasets):
    players, _ = datasets
    per_season = players.groupby("Season")["TMVP"].sum()
    # Known quirk of the original filter: SGA's "MVP-1DPOY-10" string also matches the
    # "DPOY-10" exclusion, so 2025 has no flagged winner (fixed by awards="parsed").
    assert per_season.drop(2025).eq(1).all()
    assert per_season[2025] == 0
    winners = players[players["TMVP"] == 1].set_index("Season")["Player"]
    assert winners[2016] == "Stephen Curry"


def test_parsed_awards_find_every_winner():
    players, _ = build_datasets(awards="parsed")
    assert players.groupby("Season")["TMVP"].sum().eq(1).all()
    assert players.groupby("Season")["TDPOY"].sum().eq(1).all()
    winners = players[players["TMVP"] == 1].set_index("Season")["Player"]
    assert winners[2025] == "Shai Gilgeous-Alexander"


def test_lagged_awards_point_to_previous_season():
    players, _ = build_datasets(lag="season")
    curry_2017 = players[(players["Player"] == "Stephen Curry") & (players["Season"] == 2017)]
    assert curry_2017["TMVP-1"].iloc[0] == 1  # MVP in 2016
    assert players.loc[players["Season"] == 2005, "MVP-1"].sum() == 0


def test_true_shooting_formulas():
    # 30 pts on 20 FGA and 10 FTA
    assert true_shooting(30, 20, 10, "standard") == pytest.approx(30 / 48.8)
    assert true_shooting(30, 20, 10, "as_coded") == pytest.approx(30 / 44.4)


def test_roster_2026_has_30_teams():
    roster = load_roster_2026()
    assert list(roster.columns) == ["Player", "Team_2026"]
    assert roster["Team_2026"].nunique() == 30
