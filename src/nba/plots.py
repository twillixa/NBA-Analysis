"""README / notebook figures in one consistent style."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"  # sequential step 250: de-emphasised bars
RED = "#e34948"
NEUTRAL = "#b8b7b1"
BLUE_RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 28, "axes.axisbelow": True, "font.size": 10.5, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "legend.frameon": False, "figure.dpi": 110,
    })


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    return path


def _end_label(ax, x, y, text, color, dy=0):
    ax.plot([x], [y], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center",
                color=INK, fontsize=10)


def forecast(pred: pd.DataFrame, compare: pd.DataFrame | None = None, highlight: int = 1,
             title=None, subtitle=None, compare_label="original report"):
    """Horizontal bars of predicted wins; optional ticks for a second forecast."""
    pred = pred.sort_values(["W_pred", "W%_pred"])
    fig, ax = plt.subplots(figsize=(7.5, 8.4))
    colors = [BLUE if i >= len(pred) - highlight else BLUE_LIGHT for i in range(len(pred))]
    ax.barh(pred["Team"], pred["W_pred"], height=0.62, color=colors)
    label_x = pred["W_pred"].copy()
    if compare is not None:
        other = pred["Team"].map(compare.set_index("Team")["W_pred"])
        ax.scatter(other, range(len(pred)), marker="|", s=170, linewidths=2.2, color=INK,
                   label=compare_label, zorder=3)
        label_x = pd.concat([label_x, other], axis=1).max(axis=1)
        ax.legend(loc="lower right", handletextpad=0.3)
    for y, (w, x) in enumerate(zip(pred["W_pred"], label_x)):
        ax.text(x + 0.8, y, str(w), va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, 62)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Predicted wins (82 games)")
    top = pred.iloc[-1]
    ax.set_title(title or f"2025-26 forecast: {top['Team']} on top with {top['W_pred']} wins")
    _subtitle(ax, subtitle or "Random forest on projected rosters · league total balanced to 1,230 wins")
    return fig


def attempts(attempts_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    s = attempts_df.index
    ax.plot(s, attempts_df["2PA"], color=ORANGE)
    ax.plot(s, attempts_df["3PA"], color=BLUE)
    _end_label(ax, s[-1], attempts_df["2PA"].iloc[-1], f"2-pt attempts  {attempts_df['2PA'].iloc[-1]:.0f}", ORANGE)
    _end_label(ax, s[-1], attempts_df["3PA"].iloc[-1], f"3-pt attempts  {attempts_df['3PA'].iloc[-1]:.0f}", BLUE)
    ax.plot([s[0]], [attempts_df["3PA"].iloc[0]], "o", color=BLUE, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"{attempts_df['3PA'].iloc[0]:.0f}", (s[0], attempts_df["3PA"].iloc[0]),
                xytext=(0, -16), textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_xlim(s[0] - 0.5, s[-1] + 4.5)
    ax.set_xticks(range(2005, 2026, 5))
    ax.set_ylabel("Attempts per game, summed over roster")
    growth = attempts_df["3PA"].iloc[-1] / attempts_df["3PA"].iloc[0] - 1
    ax.set_title(f"Threes per team are up {growth:.0%} since {s[0]}")
    _subtitle(ax, "Two-point volume barely moved")
    return fig


def expected_points(ep: pd.DataFrame, title=None, subtitle=None):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    s = ep.index
    ax.plot(s, ep["2PT"], color=ORANGE)
    ax.plot(s, ep["3PT"], color=BLUE)
    two, three = ep["2PT"].iloc[-1], ep["3PT"].iloc[-1]
    span = ep.max().max() - ep.min().min()
    nudge = 7 if abs(two - three) < 0.12 * span else 0  # keep close end labels apart
    _end_label(ax, s[-1], two, f"2-pointer  {two:.2f}", ORANGE, dy=nudge if two >= three else -nudge)
    _end_label(ax, s[-1], three, f"3-pointer  {three:.2f}", BLUE, dy=nudge if three > two else -nudge)
    ax.set_ylim(ep.min().min() - 0.1 * span, ep.max().max() + 0.1 * span)
    ax.set_xlim(s[0] - 0.5, s[-1] + 4.5)
    ax.set_xticks(range(2005, 2026, 5))
    ax.set_ylabel("Points per attempt")
    threes_ahead = ep.index[ep["3PT"] > ep["2PT"]]
    if title is None and len(threes_ahead) and len(threes_ahead) < len(ep):
        title = f"A three out-earned a two every season until {threes_ahead.max() + 1}"
    ax.set_title(title or "Points per shot attempt")
    _subtitle(ax, subtitle or "Made shots × value ÷ attempts, per team, averaged across the league")
    return fig


LABELS = {"ALL-NBA-1": "All-NBA player last season", "MVP-1": "MVP candidate last season",
          "DPOY-1": "Defensive honours last season", "TS%": "True shooting %",
          "BLK": "Blocks", "AST": "Assists", "PTS": "Points", "FT": "Free throws",
          "STL": "Steals", "TRB": "Rebounds", "PF": "Personal fouls", "TOV": "Turnovers"}


def win_correlations(corr: pd.Series):
    corr = corr.drop("W%").sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = [BLUE if v > 0 else RED for v in corr]
    ax.barh([LABELS.get(c, c) for c in corr.index], corr, height=0.6, color=colors)
    for y, v in enumerate(corr):
        ax.text(v + (0.015 if v > 0 else -0.015), y, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlim(-0.6, 0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Pearson correlation with team win %")
    ax.set_title("Star power and ball security track winning")
    _subtitle(ax, "Team averages, 2006-2025 · raw volume stats are confounded by era")
    return fig


def offense_defense(ratings: pd.DataFrame, labels=(("GSW", 2016), ("OKC", 2025), ("SAS", 2016))):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    r = ratings.sort_values("W%")
    pts = ax.scatter(r["Offense"], r["Defense"], c=r["W%"], cmap=BLUE_RAMP, s=34,
                     edgecolors=SURFACE, linewidths=0.8)
    for team, season in labels:
        match = r[(r["Team"] == team) & (r["Season"] == season)]
        if match.empty:
            continue
        row = match.iloc[0]
        ax.annotate(f"{season} {team} ({row['W%']:.0%})", (row["Offense"], row["Defense"]),
                    xytext=(-10, 10), textcoords="offset points", ha="right", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    cbar = fig.colorbar(pts, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Win %", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_xlabel("Offensive rating (0-100)")
    ax.set_ylabel("Defensive rating (0-100)")
    ax.set_title("Elite records need both ends of the floor")
    _subtitle(ax, "Each dot is a team-season, 2005-2025 · darker = more wins")
    return fig


def awards(buckets: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [NEUTRAL] + [BLUE] * (len(buckets) - 1)
    ax.bar(buckets.index, buckets.values, width=0.55, color=colors)
    for x, v in enumerate(buckets.values):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.axhline(0.5, color=INK_2, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(len(buckets) - 0.5, 0.5, " .500", va="center", color=INK_2, fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Average win %")
    ax.set_title("Last season's award winner on the roster lifts win %")
    _subtitle(ax, "Team-seasons 2006-2025, grouped by the best prior-season award on the roster")
    return fig


def importances(imp: pd.Series, top: int = 10):
    imp = imp.head(top).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = [BLUE if i == len(imp) - 1 else BLUE_LIGHT for i in range(len(imp))]
    ax.barh([LABELS.get(c, c) for c in imp.index], imp.values, height=0.6, color=colors)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Random forest feature importance")
    ax.set_title("Having an All-NBA player is the strongest signal")
    _subtitle(ax, "Top 10 of 28 features")
    return fig


def evaluation(summary: pd.DataFrame, seasons: str = "2010-2025"):
    """Small multiples: R² and leader hits per evaluation setup (rows of ``summary``)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    order = summary.index[::-1]
    colors = [NEUTRAL if "in-sample" in i.lower() else BLUE for i in order]
    for ax, col, fmt, title in [(axes[0], "R2", "{:.2f}", "R² (variance explained)"),
                                (axes[1], "Leader_hits", "{:.0f}", "Correct #1 seed")]:
        vals = summary.loc[order, col]
        ax.barh(order, vals, height=0.55, color=colors)
        for y, v in enumerate(vals):
            label = fmt.format(v) + (f" / {int(summary['Seasons'].iloc[0])}" if col == "Leader_hits" else "")
            ax.text(v, y, "  " + label, va="center", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(min(0, vals.min() * 1.35), max(vals.max(), 0.01) * 1.35)
    fig.suptitle("On seasons it never saw, the forest's fit drops sharply",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, 0.9, f"Seasons {seasons} · grey = model evaluated on seasons it was trained on",
             color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
