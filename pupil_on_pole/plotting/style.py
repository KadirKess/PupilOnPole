from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# One colour per method in every figure
COLORS = {"dagger": "#1f77b4", "rlpd": "#d62728", "team2": "#555555"}
LABELS = {"dagger": "DAgger", "rlpd": "RLPD", "team2": "team2 (expert)"}


def new_figure(width=6.0, height=3.6):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt.subplots(figsize=(width, height), constrained_layout=True)


def save_figure(fig, stem):
    """Write stem.png and stem.pdf, return both paths"""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths
