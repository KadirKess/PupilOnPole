from pupil_on_pole.plotting.style import (
    COLORS,
    LABELS,
    grouped_bars,
    legend_below,
    new_figure,
    ordered_tracks,
    save_figure,
    seed_series,
    set_title,
    track_label,
)


def plot_held_out(held_out, expert, metric, ylabel, stem, title=None, cap=None):
    """Grouped bars per held-out track, bar is the mean over seeds and dots are seeds"""
    tracks = ordered_tracks(held_out["track"])
    series = [seed_series(m, [g.loc[g["track"] == t, metric] for t in tracks]) for m, g in held_out.groupby("method")]
    if not expert.empty:
        series.append({
            "key": "team2",
            "label": LABELS["team2"],
            "means": [expert.loc[expert["track"] == t, metric].iloc[0] for t in tracks],
        })
    fig, ax = new_figure(width=6.5)
    grouped_bars(ax, [track_label(t) for t in tracks], series)
    if cap is not None:
        ax.axhline(cap, color=COLORS["team2"], ls=":", lw=1.0, label="Time limit")
    ax.set_xlabel("Held-out track")
    ax.set_ylabel(ylabel)
    set_title(ax, title or f"{ylabel} on held-out tracks")
    legend_below(fig, ax)
    return save_figure(fig, stem)


def plot_generalization(runs, train_track, stem, title=None):
    """Lap time over team2's on the same track, training track first, 1 is team2"""
    tracks = ordered_tracks(runs["track"], train_track)
    series = [seed_series(m, [g.loc[g["track"] == t, "vs_expert"] for t in tracks]) for m, g in runs.groupby("method")]
    fig, ax = new_figure(width=6.5)
    ax.axhline(1.0, color=COLORS["team2"], ls="--", lw=1.2, label=LABELS["team2"])
    grouped_bars(ax, [track_label(t, train_track) for t in tracks], series)
    ax.set_xlabel("Track")
    ax.set_ylabel("Lap time relative to team2")
    set_title(ax, title or "Lap time relative to team2 on each track")
    legend_below(fig, ax)
    return save_figure(fig, stem)
