from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pupil_on_pole.analysis import bootstrap_ci

# One colour per method in every figure
COLORS = {"dagger": "#1f77b4", "rlpd": "#d62728", "team2": "#555555"}
LABELS = {"dagger": "DAgger", "rlpd": "RLPD", "team2": "team2 (expert)"}
NOTE_COLOR = "#444444"


def new_figure(width=6.0, height=3.8):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
    })
    return plt.subplots(figsize=(width, height), layout="constrained")


def track_label(track, train_track=None):
    name = track.capitalize()
    return f"{name}\n(training)" if track == train_track else name


def ordered_tracks(tracks, train_track=None):
    rest = sorted(t for t in set(tracks) if t != train_track)
    return [train_track, *rest] if train_track in set(tracks) else rest


def seed_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return values[0], values[0], values[0]
    low, high = bootstrap_ci(values)
    return values.mean(), low, high


def seed_series(method, values):
    stats = [seed_interval(v) for v in values]
    return {
        "key": method,
        "label": LABELS.get(method, method),
        "means": [s[0] for s in stats],
        "low": [s[1] for s in stats],
        "high": [s[2] for s in stats],
        "points": [np.asarray(v, dtype=float) for v in values],
    }


def steps_axis(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))
    ax.set_xlabel("Online environment steps")


def set_title(ax, title, note=None):
    ax.set_title(title, pad=18 if note else 6)
    if note:
        ax.text(0.5, 1.01, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=NOTE_COLOR)


def legend_below(fig, ax, ncol=4):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(ncol, len(handles)))


def grouped_bars(ax, categories, series):
    width = 0.8 / len(series)
    x = np.arange(len(categories))
    for i, s in enumerate(series):
        offset = x - 0.4 + width * (i + 0.5)
        color = COLORS.get(s["key"])
        ax.bar(offset, s["means"], width * 0.92, color=color, alpha=0.45, label=s["label"])
        if s.get("low") is not None:
            means = np.asarray(s["means"], dtype=float)
            err = np.vstack([means - np.asarray(s["low"]), np.asarray(s["high"]) - means])
            ax.errorbar(offset, means, yerr=err, fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=3)
        for j, points in enumerate(s.get("points") or []):
            ax.scatter(np.full(len(points), offset[j]), points, color=color, s=14, zorder=4, edgecolors="white",
                       linewidths=0.5)
    ax.set_xticks(x, categories)
    return x


def save_figure(fig, stem):
    """Write stem.png and stem.pdf, return both paths"""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths
