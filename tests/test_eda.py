"""EDA numbers must match the original notebook outputs (report section 3)."""
import pytest

from nba import eda
from nba.data import build_datasets


@pytest.fixture(scope="module")
def data():
    return build_datasets()


def test_three_point_volume_growth(data):
    attempts = eda.attempts_per_team(data[0])
    growth = attempts["3PA"].iloc[-1] / attempts["3PA"].iloc[0] - 1
    assert growth == pytest.approx(1.71, abs=0.01)  # report: "approximately 172%"
    share = eda.three_point_share(data[0])
    assert round(share[2005] * 100) == 20 and round(share[2025] * 100) == 42


def test_two_pointers_stay_worth_more(data):
    ep = eda.expected_points(data[0])
    assert (ep["2PT"] > ep["3PT"]).all()
    assert ep.loc[2025, "2PT"] == pytest.approx(1.038, abs=0.001)
    assert ep.loc[2025, "3PT"] == pytest.approx(0.888, abs=0.001)


def test_correlations_match_heatmap(data):
    corr = eda.team_correlations(*data)["W%"].round(2)
    assert corr["TOV"] == -0.45
    assert corr["PF"] == -0.35
    assert corr["TS%"] == 0.25
    assert corr["ALL-NBA-1"] == 0.53
    assert corr["MVP-1"] == 0.48


def test_three_point_share_barely_tracks_wins(data):
    team = eda.three_share_vs_wins(*data)
    assert team["W%"].corr(team["3P_share"]) == pytest.approx(0.119, abs=0.001)


def test_offense_defense_correlations(data):
    ratings = eda.offense_defense_ratings(*data)
    assert ratings["Offense"].corr(ratings["W%"]) == pytest.approx(0.547, abs=0.001)
    assert ratings["Defense"].corr(ratings["W%"]) == pytest.approx(0.571, abs=0.001)


def test_award_buckets_match_figure11(data):
    buckets = eda.award_win_pct(*data).round(3)
    assert buckets.to_dict() == {"No award winner": 0.437, "All-NBA": 0.588,
                                 "DPOY": 0.625, "MVP": 0.693}
