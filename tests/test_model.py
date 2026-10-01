"""Model results must reproduce the report (within library-version noise)."""
import pytest

from nba import model as m
from nba.data import build_datasets, load_roster_2026


@pytest.fixture(scope="module")
def data():
    return build_datasets()


@pytest.fixture(scope="module")
def rf_table(data):
    return m.rf_team_table(*data)


def test_team_table_has_every_team_season(rf_table):
    assert len(rf_table) == 630
    assert rf_table.groupby("Season")["Team"].nunique().eq(30).all()


def test_random_forest_holdout_matches_report(rf_table):
    _, r2, importances = m.fit_rf_holdout(rf_table)
    assert r2 == pytest.approx(0.4389, abs=0.005)  # report §4.1.4
    assert importances.index[:3].tolist() == ["ALL-NBA-1", "TOV", "FG%"]
    assert importances["ALL-NBA-1"] == pytest.approx(0.277, abs=0.002)


def test_in_sample_leader_table_matches_report(rf_table):
    table = rf_table.copy()
    rf = m.rf_model().fit(m.rf_features(table), table["W%"])
    table["W%_pred"] = rf.predict(m.rf_features(table))
    leaders = m.leader_table(m.rank_by_season(table))
    assert leaders["Hit"].sum() == 13  # report Table 2
    assert (leaders["Predicted_leader_finished"] - 1).mean() == pytest.approx(0.5, abs=0.06)


def test_ols_matches_report(data):
    table = m.ols_team_table(*data)
    ols = m.fit_ols(table)
    assert ols.rsquared == pytest.approx(0.589, abs=0.0005)
    table["W%_pred"] = ols.predict(m.sm.add_constant(m.ols_features(table)))
    leaders = m.leader_table(m.rank_by_season(table))
    assert (leaders["Predicted_leader_finished"] - 1).mean() == pytest.approx(1.45)  # Table 3
    assert leaders["Hit"].sum() == 9  # Table 3 rows (the report text says 8)


def test_forecast_2026_matches_report_figure14(data):
    forecast = m.forecast_2026(*data, load_roster_2026())
    assert forecast["W_pred"].sum() == m.TOTAL_WINS
    assert len(forecast) == 30
    wins = forecast.set_index("Team")["W_pred"]
    assert wins["CLE"] == wins["OKC"] == 54
    assert wins["MIN"] == 52
    assert wins.idxmin() == "ORL" and wins.min() == 28


def test_walk_forward_never_trains_on_the_test_season(rf_table):
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(len(X))
            return self

        def predict(self, X):
            return [0.5] * len(X)

    pred = m.walk_forward(rf_table, make_model=Spy, first_test_season=2024)
    assert sorted(pred["Season"].unique()) == [2024, 2025]
    assert seen == [19 * 30, 20 * 30]  # 2005-2023, then 2005-2024


def test_balance_wins_hits_league_total():
    import pandas as pd
    pred = pd.DataFrame({"Team": list("ABCD"), "W%_pred": [0.9, 0.8, 0.7, 0.6]})
    out = m.balance_wins(pred, total=164)
    assert out["W_pred"].sum() == 164
    assert (out["W_pred"] + out["L_pred"]).eq(82).all()
