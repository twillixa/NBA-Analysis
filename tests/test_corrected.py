"""The corrected pipeline (CORRECTED options + projection fixes) and the honest evaluation."""
import pytest
from sklearn.linear_model import LinearRegression

from nba import eda
from nba import model as m
from nba.data import CORRECTED, build_datasets, load_roster_2026, name_key


@pytest.fixture(scope="module")
def corrected():
    return build_datasets(**CORRECTED)


def test_aggregate_rows_dropped(corrected):
    players, _ = corrected
    assert (players["Team"] != 0).all()
    assert "League Average" not in set(players["Player"])
    assert not players.duplicated(["Player", "Season", "Team"]).any()


def test_name_key_ignores_accents_suffixes_and_punctuation():
    assert name_key("Nikola Jokić") == name_key("Nikola Jokic ")
    assert name_key("A.J. Green") == name_key("AJ Green")
    assert name_key("Jimmy Butler III") == name_key("Jimmy Butler")


def test_roster_names_match_history(corrected):
    players, _ = corrected
    raw = load_roster_2026()
    matched = load_roster_2026(match_names=True, players=players)
    known = set(players["Player"])
    assert raw["Player"].isin(known).sum() == 404  # the report's run
    assert matched["Player"].isin(known).sum() == 440
    for name in ["Victor Wembanyama", "Karl-Anthony Towns", "Mikal Bridges", "Alperen Şengün"]:
        assert name in set(matched["Player"])


def test_projection_fixes_give_stars_real_stat_lines(corrected):
    players, _ = corrected
    roster = load_roster_2026(match_names=True, players=players)
    full = m.project_2026_players(players, roster, fixes=True, ts_formula="standard")
    new = full[full["Season"] == 2026].set_index("Player")
    assert new.loc["Victor Wembanyama", "PTS"] > 18
    assert new.loc["Shai Gilgeous-Alexander", "TMVP-1"] == 1
    rotation = new[new["MP"] >= 15]  # fringe players' independently projected stats can be odd
    assert rotation["TS%"].between(0.35, 0.75).all()


def test_corrected_forecast(corrected):
    players, standings = corrected
    roster = load_roster_2026(match_names=True, players=players)
    forecast = m.forecast_2026(players, standings, roster, fixes=True, ts_formula="standard")
    assert forecast["W_pred"].sum() == m.TOTAL_WINS
    assert forecast.iloc[0]["Team"] == "OKC" and forecast.iloc[0]["W_pred"] == 55


def test_weighted_points_per_attempt(corrected):
    ep = eda.expected_points(corrected[0], weighted=True)
    threes_ahead = ep.index[ep["3PT"] > ep["2PT"]]
    assert list(threes_ahead) == list(range(2005, 2022))
    assert ep.loc[2025, "2PT"] == pytest.approx(1.083, abs=0.001)
    assert ep.loc[2025, "3PT"] == pytest.approx(1.057, abs=0.001)


def test_corrected_eda_headlines(corrected):
    players, standings = corrected
    attempts = eda.attempts_per_team(players)
    assert attempts["3PA"].iloc[-1] / attempts["3PA"].iloc[0] - 1 == pytest.approx(1.75, abs=0.01)
    buckets = eda.award_win_pct(players, standings).round(3)
    assert buckets["MVP"] == 0.694 and buckets["No award winner"] == 0.438


@pytest.fixture(scope="module")
def report_data():
    return build_datasets()


def test_walk_forward_scores(report_data):
    rf = m.summarize(m.walk_forward(m.rf_team_table(*report_data)))
    ols = m.summarize(m.walk_forward(m.ols_team_table(*report_data), features=m.ols_features,
                                     make_model=LinearRegression))
    assert rf["R2"] == pytest.approx(0.419, abs=0.005) and rf["Leader_hits"] == 3
    assert ols["R2"] == pytest.approx(0.503, abs=0.005) and ols["Leader_hits"] == 7
    assert rf["Seasons"] == ols["Seasons"] == 16


def test_baseline_scores(report_data):
    _, standings = report_data
    base = standings[["Team", "Season", "W%", "Rk"]].copy()
    prev = base[["Team", "Season", "W%"]].assign(Season=lambda d: d.Season + 1)
    base = base.merge(prev.rename(columns={"W%": "W%_pred"}), on=["Team", "Season"])
    scores = m.summarize(base[base["Season"] >= 2010])
    assert scores["R2"] == pytest.approx(0.229, abs=0.005) and scores["Leader_hits"] == 5
