import numpy as np

from pupil_on_pole.plotting.style import COLORS, LABELS, new_figure, save_figure


def plot_held_out(held_out, expert, metric, ylabel, stem):
    """Grouped bars per held-out track, bar is the mean over seeds and dots are seeds"""
    tracks = sorted(held_out["track"].unique())
    methods = sorted(held_out["method"].unique())
    groups = [*methods, "team2"] if not expert.empty else methods
    width = 0.8 / len(groups)
    fig, ax = new_figure(width=7.0)
    x = np.arange(len(tracks))
    for i, method in enumerate(groups):
        offset = x - 0.4 + width * (i + 0.5)
        if method == "team2":
            values = [expert.loc[expert["track"] == t, metric].iloc[0] for t in tracks]
            ax.bar(offset, values, width, color=COLORS["team2"], alpha=0.6, label=LABELS["team2"])
            continue
        data = held_out[held_out["method"] == method]
        means = [data.loc[data["track"] == t, metric].mean() for t in tracks]
        ax.bar(offset, means, width, color=COLORS.get(method), alpha=0.6, label=LABELS.get(method, method))
        for j, track in enumerate(tracks):
            seeds = data.loc[data["track"] == track, metric]
            ax.scatter(np.full(len(seeds), offset[j]), seeds, color=COLORS.get(method), s=12, zorder=3)
    ax.set_xticks(x, tracks)
    ax.set_xlabel("Held-out track")
    ax.set_ylabel(ylabel)
    ax.set_title("Final checkpoint, zero-shot, dots are seeds")
    ax.legend(frameon=False, fontsize=8)
    return save_figure(fig, stem)


def plot_generalization(runs, train_track, stem):
    """Lap time over team2's on the same track, training track first, 1 is team2"""
    tracks = [train_track, *sorted(t for t in runs["track"].unique() if t != train_track)]
    methods = sorted(runs["method"].unique())
    width = 0.8 / len(methods)
    x = np.arange(len(tracks))
    fig, ax = new_figure(width=7.0)
    for i, method in enumerate(methods):
        data = runs[runs["method"] == method]
        offset = x - 0.4 + width * (i + 0.5)
        means = [data.loc[data["track"] == t, "vs_expert"].mean() for t in tracks]
        ax.bar(offset, means, width, color=COLORS.get(method), alpha=0.6, label=LABELS.get(method, method))
        for j, track in enumerate(tracks):
            seeds = data.loc[data["track"] == track, "vs_expert"]
            ax.scatter(np.full(len(seeds), offset[j]), seeds, color=COLORS.get(method), s=12, zorder=3)
    ax.axhline(1.0, color=COLORS["team2"], ls="--", lw=1.2, label=LABELS["team2"])
    ax.set_xticks(x, [f"{t}\n(train)" if t == train_track else t for t in tracks])
    ax.set_ylabel("Lap time / team2 lap time")
    ax.set_title("Generalization, lower is better, dots are seeds")
    ax.legend(frameon=False, fontsize=8)
    return save_figure(fig, stem)
