import numpy as np

from pupil_on_pole.plotting.style import COLORS, LABELS, new_figure, save_figure

ACTIONS = ("brake", "drift", "nitro", "fire")


def plot_action_use(final, expert, train_track, stem):
    """Fraction of steps each binary action is on at the final checkpoint, per method and for the expert"""
    rows = {m: [g[f"use_{a}"].mean() for a in ACTIONS] for m, g in final.groupby("method")}
    if not expert.empty:
        row = expert[expert["track"] == train_track]
        if not row.empty:
            rows["team2"] = [row[f"use_{a}"].iloc[0] for a in ACTIONS]
    width = 0.8 / max(len(rows), 1)
    x = np.arange(len(ACTIONS))
    fig, ax = new_figure()
    for i, (method, values) in enumerate(rows.items()):
        ax.bar(x - 0.4 + width * (i + 0.5), values, width, color=COLORS.get(method), label=LABELS.get(method, method))
    ax.set_xticks(x, ACTIONS)
    ax.set_ylabel("Fraction of steps on")
    ax.set_title(f"Action use on {train_track}, mean over seeds")
    ax.legend(frameon=False, fontsize=8)
    return save_figure(fig, stem)


def plot_fire(runs, expert, train_track, stem):
    """Fire presses per lap on every track, final checkpoint, per method and for team2"""
    tracks = [train_track, *sorted(t for t in runs["track"].unique() if t != train_track)]
    groups = {m: [g.loc[g["track"] == t, "fire_presses"].mean() for t in tracks] for m, g in runs.groupby("method")}
    if not expert.empty:
        groups["team2"] = [expert.loc[expert["track"] == t, "fire_presses"].mean() for t in tracks]
    width = 0.8 / len(groups)
    x = np.arange(len(tracks))
    fig, ax = new_figure(width=7.0)
    for i, (method, values) in enumerate(groups.items()):
        ax.bar(x - 0.4 + width * (i + 0.5), values, width, color=COLORS.get(method), label=LABELS.get(method, method))
    ax.set_xticks(x, [f"{t}\n(train)" if t == train_track else t for t in tracks])
    ax.set_ylabel("Fire presses per lap")
    ax.set_title("Item use, mean over seeds and episodes")
    ax.legend(frameon=False, fontsize=8)
    return save_figure(fig, stem)
