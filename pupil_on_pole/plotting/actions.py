from pupil_on_pole.plotting.style import (
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

ACTIONS = ("brake", "drift", "nitro", "fire")


def plot_action_use(final, expert, train_track, stem, title=None, note=None):
    """Fraction of steps each binary action is on at the final checkpoint, per method and for the expert"""
    series = [seed_series(m, [g[f"use_{a}"] for a in ACTIONS]) for m, g in final.groupby("method")]
    if not expert.empty:
        row = expert[expert["track"] == train_track]
        if not row.empty:
            series.append({"key": "team2", "label": LABELS["team2"],
                           "means": [row[f"use_{a}"].iloc[0] for a in ACTIONS]})
    fig, ax = new_figure()
    grouped_bars(ax, [a.capitalize() for a in ACTIONS], series)
    ax.set_xlabel("Action")
    ax.set_ylabel("Fraction of steps pressed")
    set_title(ax, title or f"Use of binary actions on {track_label(train_track)}", note)
    legend_below(fig, ax)
    return save_figure(fig, stem)


def plot_fire(runs, expert, train_track, stem, title=None, note=None):
    """Fire presses per lap on every track, final checkpoint, per method and for team2"""
    tracks = ordered_tracks(runs["track"], train_track)
    series = [seed_series(m, [g.loc[g["track"] == t, "fire_presses"] for t in tracks])
              for m, g in runs.groupby("method")]
    if not expert.empty:
        series.append({"key": "team2", "label": LABELS["team2"],
                       "means": [expert.loc[expert["track"] == t, "fire_presses"].mean() for t in tracks]})
    fig, ax = new_figure(width=6.5)
    grouped_bars(ax, [track_label(t, train_track) for t in tracks], series)
    ax.set_xlabel("Track")
    ax.set_ylabel("Item uses per lap")
    set_title(ax, title or "Item use per lap on each track", note)
    legend_below(fig, ax)
    return save_figure(fig, stem)
