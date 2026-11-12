import numpy as np

from pupil_on_pole.plotting.style import (
    COLORS,
    LABELS,
    legend_below,
    new_figure,
    save_figure,
    seed_interval,
    set_title,
    steps_axis,
    track_label,
)


def plot_curve(curves, expert, metric, ylabel, train_track, stem, title=None, note=None, cap=None):
    """Metric against online steps, thin line per seed, thick mean, expert as a horizontal line"""
    fig, ax = new_figure()
    for method, group in curves.groupby("method"):
        color = COLORS.get(method)
        for _, seed in group.groupby("seed"):
            seed = seed.sort_values("online_step")
            ax.plot(seed["online_step"], seed[metric], color=color, lw=0.7, alpha=0.35)
        steps, stats = [], []
        for step, at_step in group.groupby("online_step"):
            steps.append(step)
            stats.append(seed_interval(at_step[metric]))
        mean, low, high = np.array(stats).T
        ax.fill_between(steps, low, high, color=color, alpha=0.15, lw=0)
        ax.plot(steps, mean, color=color, lw=2.0, label=f"{LABELS.get(method, method)}, mean of seeds")
        if method == "dagger" and 0 in steps:
            ax.scatter([0], [mean[steps.index(0)]], color=color, marker="o", s=30, zorder=4,
                       label="BC (DAgger at step 0)")
    if not expert.empty:
        row = expert[expert["track"] == train_track]
        if not row.empty:
            value = row[metric].iloc[0]
            ax.axhline(value, color=COLORS["team2"], ls="--", lw=1.2, label=LABELS["team2"])
            low, high = ax.get_ylim()
            ax.set_ylim(min(low, value - 0.05 * (high - low)), max(high, value + 0.05 * (high - low)))
    if cap is not None:
        ax.axhline(cap, color=COLORS["team2"], ls=":", lw=1.0, label="Time limit")
    steps_axis(ax)
    ax.set_ylabel(ylabel)
    set_title(ax, title or f"{ylabel} on {track_label(train_track)} during training", note)
    legend_below(fig, ax, ncol=3)
    return save_figure(fig, stem)
