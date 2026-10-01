"""Generate the three project notebooks.

Run from the repo root, then execute them:
    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
REPO = "twillixa/NBA-Analysis"

SETUP = '''import sys, os
if "google.colab" in sys.modules:  # running on Colab: fetch the repo first
    !git clone -q https://github.com/twillixa/NBA-Analysis
    %cd NBA-Analysis/notebooks
sys.path.insert(0, os.path.abspath("../src"))

import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # spurious on macOS Accelerate
import pandas as pd
from nba import plots
from nba.data import build_datasets, CORRECTED
plots.use_style()
pd.set_option("display.precision", 3)'''


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


def notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in cells]
    nbf.write(nb, OUT / name)


md, code = "md", "code"

notebook("01_data_and_eda.ipynb", [
    (md, f"""# 1 · Data and exploratory analysis
{badge("01_data_and_eda.ipynb")}

Twenty-one regular seasons (2004-05 to 2024-25) of [Basketball-Reference](https://www.basketball-reference.com)
player per-game stats and team standings. A season is named by the year it **ends** (2025 = 2024-25).

The questions from the report:
1. Which team statistics move with winning?
2. Are three-pointers worth more than twos?
3. Does offense or defense matter more?
4. How much does award-winning talent help?

This notebook uses the **corrected** cleaning (`CORRECTED` options, see notebook 3's audit log) and prints the
report's original value next to each result so every difference is visible."""),
    (code, SETUP + "\nfrom nba import eda"),
    (md, """## Load and clean
`build_datasets()` (in `src/nba/data.py`) does what the report's section 2 describes:
- maps every franchise to its current code (NJN → BRK, SEA → OKC, CHA → CHO, …)
- turns the `Awards` string (`"MVP-1DPOY-10ASNBA1"`) into dummies and identifies the actual MVP/DPOY winner
- adds **previous-season** award flags (`MVP-1`, `ALL-NBA-1`, …), the only award information a forecaster has before the season
- parses `"64-18"` records into W, L, W%

With `CORRECTED` it also drops traded players' multi-team total rows and the "League Average" row, uses the
standard True Shooting formula, and looks lagged awards up by season."""),
    (code, '''players, standings = build_datasets(**CORRECTED)
report_players, _ = build_datasets()  # the report's original cleaning, for comparison
print(f"{players.groupby(['Player', 'Season']).ngroups:,} player-seasons "
      f"({len(players):,} team stints) · {len(standings)} team-seasons · "
      f"{standings.Season.min()}-{standings.Season.max()}")
players[["Player", "Team", "Season", "PTS", "TRB", "AST", "TS%", "Awards", "ALL-NBA-1"]].head()'''),
    (code, '''standings[["Season", "Team", "W", "L", "W%", "HW%", "RW%"]].describe().round(3)'''),
    (md, "## 1 · Which statistics move with winning?\nTeam averages of player stats, correlated with team win %. Award flags are from the **previous** season, so they are known before tip-off."),
    (code, '''corr = eda.team_correlations(players, standings)
fig = plots.win_correlations(corr["W%"]); plots.save(fig, "win_correlations")
pd.DataFrame({"corrected": corr["W%"],
              "report": eda.team_correlations(report_players, standings)["W%"]}).drop("W%").round(2)'''),
    (md, """Raw volume (points, assists, rebounds) correlates *negatively*: scoring rose league-wide over these 20 years,
so 100 points meant something different in 2005 than in 2025. Turnovers (−0.45) and fouls (−0.35), which both give the
ball away, and prior-season All-NBA / MVP players are the clearest signals."""),
    (md, "## 2 · Are three-pointers worth more than twos?"),
    (code, '''attempts = eda.attempts_per_team(players)
fig = plots.attempts(attempts); plots.save(fig, "three_point_volume")
share = eda.three_point_share(players)
print(f"3PA per team: {attempts['3PA'].iloc[0]:.0f} → {attempts['3PA'].iloc[-1]:.0f} "
      f"(+{attempts['3PA'].iloc[-1] / attempts['3PA'].iloc[0] - 1:.0%}; report: +171%)")
print("3PA share of shots (%):", (share.loc[[2005, 2010, 2015, 2020, 2025]] * 100).round().astype(int).to_dict())'''),
    (md, """**Points per attempt.** The report multiplied the *average of players' shooting percentages* by the shot value.
A player who never shoots threes enters that average as a 0% shooter, which drags 3P% down and made twos look
better every season. Dividing a team's made shots by its attempts instead:"""),
    (code, '''ep = eda.expected_points(players, weighted=True)
fig = plots.expected_points(ep); plots.save(fig, "expected_points")
comparison = ep.join(eda.expected_points(report_players), rsuffix=" (report method)")
comparison["three worth more"] = comparison["3PT"] > comparison["2PT"]
comparison.round(3)'''),
    (code, '''team_3p = eda.three_share_vs_wins(players, standings)
r = team_3p["W%"].corr(team_3p["3P_share"])
print(f"Correlation between a team's 3PA share and its win %: r = {r:.3f} (n = {len(team_3p)})")'''),
    (md, """Threes nearly tripled, and for most of the period that was simply rational: from 2005 to 2021 a three returned
**more** points per attempt than a two. Since 2022 twos have edged ahead again (better rim finishing, and defenses
running shooters off the line). Still, a team's three-point share barely tracks its record (r ≈ 0.12): *how often* a
team shoots threes says little about how good it is. This reverses the report's conclusion that twos were worth more throughout."""),
    (md, "## 3 · Offense or defense?\nComposite 0-100 ratings: offense = points, assists, offensive rebounds, free throws, TS%; defense = steals, blocks, defensive rebounds (each min-max scaled within the season)."),
    (code, '''ratings = eda.offense_defense_ratings(players, standings)
fig = plots.offense_defense(ratings); plots.save(fig, "offense_defense")
print(f"r(offense, W%) = {ratings.Offense.corr(ratings['W%']):.3f} · "
      f"r(defense, W%) = {ratings.Defense.corr(ratings['W%']):.3f}")
ratings.nlargest(10, "W%").round(3)'''),
    (md, "Both indices correlate with winning about equally (defense slightly more; the report had the two numbers swapped). The best records sit in the top-right. A few defense-first teams (2016 Spurs, 2025 Thunder) got there with a merely average offense."),
    (md, "## 4 · Award-winning talent"),
    (code, '''buckets = eda.award_win_pct(players, standings)
fig = plots.awards(buckets); plots.save(fig, "awards")
pd.DataFrame({"corrected": buckets, "report": eda.award_win_pct(report_players, standings)}).round(3)'''),
    (md, "Teams with last season's MVP averaged a .694 win %, versus .438 for teams with no prior-season award winner. That gap is why lagged awards are model features (notebook 2)."),
])

notebook("02_model_and_forecast.ipynb", [
    (md, f"""# 2 · Models and the 2025-26 forecast
{badge("02_model_and_forecast.ipynb")}

**Part A** reproduces the report's section 4 exactly: a random forest and an OLS regression that map a roster's
average (min-max scaled) player stats and prior-season awards to the team's win %.
**Part B** re-runs the 2025-26 forecast with the projection fixes from notebook 3 and compares it with the report's."""),
    (code, SETUP + "\nfrom nba import model as m\nfrom nba.data import load_roster_2026\nplayers, standings = build_datasets()  # report's cleaning"),
    (md, "# Part A · The report's models\n## Team-season feature table\nPlayers averaging ≥ 5 minutes, each stat min-max scaled across all player-seasons, averaged per team-season → 630 team-seasons × 28 features."),
    (code, '''table = m.rf_team_table(players, standings)
print(table.shape)
table.head()'''),
    (md, "## Random forest (report §4.1.2)\n100 trees, depth ≤ 10, random 80/20 split."),
    (code, '''rf, r2, importances = m.fit_rf_holdout(table)
print(f"Test R² = {r2:.3f}   (report: 0.439)")
fig = plots.importances(importances); plots.save(fig, "feature_importance")
importances.head(10).round(3)'''),
    (md, "**Ranking accuracy.** The report scores its leader table with a forest fit on *all* seasons, i.e. on data it has already seen. Notebook 3 re-scores it on unseen seasons."),
    (code, '''table_rf = table.copy()
rf_all = m.rf_model().fit(m.rf_features(table_rf), table_rf["W%"])
table_rf["W%_pred"] = rf_all.predict(m.rf_features(table_rf))
leaders_rf = m.leader_table(m.rank_by_season(table_rf))
print(f"Leader correctly identified: {leaders_rf.Hit.sum()} / {len(leaders_rf)} seasons "
      f"(report: 13/20, in-sample)")
leaders_rf'''),
    (md, "## OLS regression (report §4.1.6)"),
    (code, '''ols_table = m.ols_team_table(players, standings)
ols = m.fit_ols(ols_table)
ols_table["W%_pred"] = ols.predict(m.sm.add_constant(m.ols_features(ols_table)))
ranks_ols = m.rank_by_season(ols_table)
leaders_ols = m.leader_table(ranks_ols)
print(f"R² = {ols.rsquared:.3f} · rank MAE = {ranks_ols.Rank_error.mean():.2f} · "
      f"leader hits = {leaders_ols.Hit.sum()}/20 · leader rank error = "
      f"{(leaders_ols.Predicted_leader_finished - 1).mean():.2f}")
significant = ols.pvalues[ols.pvalues < 0.05].drop("const", errors="ignore")
ols.params[significant.index].round(2).sort_values()'''),
    (md, "Most OLS coefficients are not significant: the features are heavily collinear (points, attempts and makes move together), which is also why the forest's importances and the OLS signs don't always agree."),
    (md, """# Part B · Forecasting 2025-26
1. Start from the opening-night rosters the group compiled (`data/raw/rs2026.csv`).
2. For each box-score stat, fit a linear regression on a player's previous three seasons (2009-2025) and project 2026. Rookies get a flat baseline line.
3. Carry 2025 awards over as the lagged award flags.
4. Aggregate to teams, predict win % with a forest trained on all 630 team-seasons, convert to wins and rebalance so the league sums to 1,230.

**The report's run** (reproduced first) had three problems in steps 1-2:
- 118 roster names didn't match the stats history (typos such as "Kari-Anthony Towns" and "Mikai Bridges", missing accents such as Şengün and Porziņģis, and a trailing space after "Victor Wembanyama"). All of them got the 10-point rookie line.
- The projection regressions were trained on the 2026 rows themselves, whose stats are all zero.
- Second- and third-year players had their missing seasons filled with zeros."""),
    (code, '''report_forecast = m.forecast_2026(players, standings, load_roster_2026())
report_forecast.round(3).to_csv("../reports/predictions_2026_report.csv", index=False)
print("Report forecast reproduced: " + ", ".join(f"{t} {w}" for t, w in report_forecast[["Team", "W_pred"]].head(5).values))'''),
    (code, '''fixed_players, fixed_standings = build_datasets(**CORRECTED)
roster = load_roster_2026(match_names=True, players=fixed_players)
known = set(fixed_players.Player)
print(f"Roster names matched to a stats history: {roster.Player.isin(known).sum()} of {len(roster)} "
      f"(report: {load_roster_2026().Player.isin(set(players.Player)).sum()}). Unmatched = 2025 draftees and other newcomers.")
forecast = m.forecast_2026(fixed_players, fixed_standings, roster, fixes=True, ts_formula="standard")
forecast.round(3).to_csv("../reports/predictions_2026.csv", index=False)
m.rf_team_table(fixed_players, fixed_standings).to_csv("../data/processed/team_seasons.csv", index=False)
fig = plots.forecast(forecast, compare=report_forecast,
                     subtitle="Bars: corrected projections · ticks: the report's forecast (Dec 2025)")
plots.save(fig, "forecast_2026")
both = forecast[["Team", "W_pred"]].merge(report_forecast[["Team", "W_pred"]], on="Team", suffixes=("", "_report"))
both.assign(change=both.W_pred - both.W_pred_report).head(12)'''),
    (code, '''projected = m.project_2026_players(fixed_players, roster, fixes=True, ts_formula="standard")
projected[projected.Season == 2026].nlargest(10, "PTS")[["Player", "Team", "PTS", "TRB", "AST", "MP"]]'''),
    (md, "Oklahoma City stays on top. Cleveland drops from co-favourite to the 50-win pack, and Phoenix (−7) and Philadelphia (+6) move the most once their stars get real projections instead of rookie lines."),
])

notebook("03_out_of_sample_audit.ipynb", [
    (md, f"""# 3 · Out-of-sample check and audit
{badge("03_out_of_sample_audit.ipynb")}

The report's headline (*the random forest picks the league leader in 13 of 20 seasons*) is computed with a model
trained on those same 20 seasons. This notebook asks the question a forecaster cares about:
**how well does the method do on a season it has never seen?**

- **Walk-forward backtest**: for each season 2010-2025, train only on earlier seasons, then predict that season.
- **Baseline**: "every team repeats last season's win %".
- **Corrected data**: re-run everything with the cleaning fixes from the audit log at the bottom."""),
    (code, SETUP + "\nfrom sklearn.linear_model import LinearRegression\nfrom nba import model as m"),
    (code, '''FIRST = 2010
rows = {}
for label, options in [("report cleaning", {}), ("corrected", CORRECTED)]:
    players, standings = build_datasets(**options)
    rf_table = m.rf_team_table(players, standings)
    in_sample = rf_table.copy()
    in_sample["W%_pred"] = m.rf_model().fit(m.rf_features(rf_table), rf_table["W%"]).predict(m.rf_features(rf_table))
    rows[f"Random forest, in-sample ({label})"] = m.summarize(in_sample[in_sample.Season >= FIRST])
    rows[f"Random forest, walk-forward ({label})"] = m.summarize(m.walk_forward(rf_table, first_test_season=FIRST))
    ols_table = m.ols_team_table(players, standings)
    rows[f"OLS, walk-forward ({label})"] = m.summarize(
        m.walk_forward(ols_table, features=m.ols_features, make_model=LinearRegression, first_test_season=FIRST))

base = standings[["Team", "Season", "W%", "Rk"]].copy()
prev = base[["Team", "Season", "W%"]].assign(Season=lambda d: d.Season + 1).rename(columns={"W%": "W%_pred"})
base = base.merge(prev, on=["Team", "Season"])
rows["Baseline: repeat last season"] = m.summarize(base[base.Season >= FIRST])

results = pd.DataFrame(rows).T
results.round(3).to_csv("../reports/evaluation.csv")
results.round(3)'''),
    (code, '''chart = results.loc[["Random forest, in-sample (report cleaning)", "Random forest, walk-forward (report cleaning)",
                     "OLS, walk-forward (report cleaning)", "Baseline: repeat last season"]]
chart.index = ["Random forest, in-sample (report)", "Random forest, walk-forward",
               "OLS, walk-forward", "Baseline: repeat last season"]
fig = plots.evaluation(chart, seasons=f"{FIRST}-2025"); plots.save(fig, "evaluation");'''),
    (md, """**Reading the table**
- In-sample, the forest explains ~89% of win % and picks 11 of 16 leaders. It is largely recalling seasons it trained on.
- Walk-forward, it explains ~42% and picks 3 of 16, fewer than the naive baseline's 5. The simpler **OLS generalises better** (R² ≈ 0.50, 7 of 16).
- On R², win error and rank error, both models beat the baseline (R² ≈ 0.23): box scores and prior awards carry real signal, just less than the report's headline suggests.
- The cleaning fixes move these results by less than the season-to-season noise.

This flips one of the report's conclusions (random forest > OLS). Choosing the model with a time-based backtest is the first thing to change in a v2."""),
    (md, """## Audit log: issues found while porting the original notebooks

| Issue | Effect | Fix |
|---|---|---|
| 2026 roster names didn't match the history (whitespace, typos, accents) | 118 of 522 players, including Wembanyama, Towns and Bridges, projected with the rookie line | `load_roster_2026(match_names=True)` |
| Projection regressions trained on the all-zero 2026 rows | projected stats biased toward 0 | `forecast_2026(fixes=True)` |
| Missing lag seasons filled with 0 | 2nd/3rd-year players under-projected | `fixes=True` (reuse latest season) |
| "Points per attempt" averaged players' shooting % | non-shooters count as 0% from three; twos looked better every season | `expected_points(weighted=True)` |
| Multi-team total rows and "League Average" kept as a pseudo-team `0` | league-wide averages double-count traded players (+171% vs +175% 3PA growth) | `aggregates="drop"` |
| TS% coded as `PTS / (2·FGA + 0.44·FTA)` | TS% inflated (should be `PTS / (2·(FGA + 0.44·FTA))`) | `ts_formula="standard"` |
| Lagged awards via `groupby(Player).shift(1)` | for traded players the "previous season" can be the same season | `lag="season"` |
| Winner filter excludes strings containing `DPOY-10` | 2025 MVP (SGA, `MVP-1DPOY-10…`) not flagged; patched by hand in the forecast | `awards="parsed"` |
| `DPOY` dummy also matches `DEF` | All-Defensive selections counted as DPOY votes | `awards="parsed"` |
| Ties in W% ranked by sort order | 2012 CHI/SAS (50-16) could swap | ranks use Basketball-Reference `Rk` (always on) |
| 21 names in `rs2024.csv` have `?` for accented letters | those players lose their 2024 → 2025 award lag | not fixed (source file); `name_key` matching would be the fix |
| Players matched by name only | two players with the same name would merge | not fixed; Basketball-Reference IDs (`Player-additional`) would be the fix |"""),
])
print(f"wrote 3 notebooks to {OUT}")
