from pupil_on_pole.plotting.style import COLORS, LABELS, new_figure, save_figure


def plot_curve(curves, expert, metric, ylabel, train_track, stem):
    """Metric against online steps, thin line per seed, thick mean, expert as a horizontal line"""
    fig, ax = new_figure()
    for method, group in curves.groupby("method"):
        color = COLORS.get(method)
        for _, seed in group.groupby("seed"):
            seed = seed.sort_values("online_step")
            ax.plot(seed["online_step"], seed[metric], color=color, lw=0.8, alpha=0.4)
        mean = group.groupby("online_step")[metric].mean()
        ax.plot(mean.index, mean.to_numpy(), color=color, lw=2.2, label=f"{LABELS.get(method, method)} (mean)")
        if method == "dagger" and 0 in mean.index:
            ax.scatter([0], [mean.loc[0]], color=color, marker="o", zorder=3, label="BC (DAgger step 0)")
    if not expert.empty:
        row = expert[expert["track"] == train_track]
        if not row.empty:
            ax.axhline(row[metric].iloc[0], color=COLORS["team2"], ls="--", lw=1.2, label=LABELS["team2"])
    ax.set_xlabel("Online environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{train_track}, thin lines are seeds")
    ax.legend(frameon=False, fontsize=8)
    return save_figure(fig, stem)
