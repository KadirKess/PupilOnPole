"""Read an experiment folder and print what the lap-time table hides, nothing is re-run

uv run python -m scripts.diagnose outputs/20261111-225022-pilot
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pupil_on_pole.env import ACTION_NAMES, STEP_SECONDS

CURVE_COLUMNS = ["online_step", "finish_rate", "stuck_rate", "progress", "lap_s", "mean_speed",
                 "use_brake", "use_drift", "use_nitro", "use_fire"]
TB_TAGS = ["episode/progress", "episode/finished", "episode/steps", "episode/return",
           "q/mean", "loss/critic", "policy/entropy", "policy/temperature", "loss/bc"]


def summary_row(summary):
    row = {name: summary.get(name) for name in ("finish_rate", "stuck_rate", "progress", "mean_speed")}
    # Lap time of the finished laps only, the table mixes it with the 150 s cap
    row["lap_s"] = summary["lap_steps"] * STEP_SECONDS if summary.get("lap_steps") else None
    row.update({f"use_{k}": v for k, v in summary["action_use"].items()})
    return row


def outcome(episode):
    if episode["finished"]:
        return f"F{episode['steps'] * STEP_SECONDS:.0f}"
    return f"{'S' if episode['stuck'] else 'T'}{100 * episode['progress']:.0f}%"


def print_expert(experiment):
    path = experiment / "expert" / "evaluation.json"
    if not path.exists():
        print("no expert/evaluation.json")
        return None
    data = json.loads(path.read_text())
    rows = [{"track": t, **summary_row(r["summary"])} for t, r in data["tracks"].items()]
    print("\n== Expert ==")
    print(pd.DataFrame(rows).to_string(index=False))
    train = data["tracks"][data["train_track"]]["episodes"]
    return {e["seed"]: outcome(e) for e in train}


def print_run(run, expert_outcomes):
    path = run / "evaluation.json"
    if not path.exists():
        print(f"\nno {path}")
        return
    data = json.loads(path.read_text())
    curve = data["curve"]
    rows = [{"online_step": p["online_step"], **summary_row(p["summary"])} for p in curve]
    print(f"\n== {run.parent.name}/{run.name}, training track ==")
    print(pd.DataFrame(rows)[CURVE_COLUMNS].to_string(index=False))

    # One column per eval seed: F<lap s> finished, S<progress> stuck, T<progress> timeout
    seeds = [e["seed"] for e in curve[0]["episodes"]]
    grid = {p["online_step"]: [outcome(e) for e in p["episodes"]] for p in curve}
    table = pd.DataFrame(grid, index=seeds).T
    if expert_outcomes:
        table.loc["expert"] = [expert_outcomes.get(s, "?") for s in seeds]
    print("\nPer eval seed (F = finished in s, S = stuck at progress, T = timeout at progress):")
    print(table.to_string())
    stuck = [e for p in curve for e in p["episodes"] if not e["finished"]]
    if stuck:
        share = np.mean([e["stuck"] for e in stuck])
        print(f"Unfinished episodes: {len(stuck)}, {100 * share:.0f}% of them ended stuck, "
              f"median progress {100 * np.median([e['progress'] for e in stuck]):.0f}%")


def print_dataset(experiment):
    path = experiment / "data" / "d0.npz"
    if not path.exists():
        return
    with np.load(path) as d:
        actions, episode, rewards = d["actions"], d["episode"], d["rewards"]
    on = actions > 0
    print("\n== D0 ==")
    print(f"{len(actions)} transitions, {len(np.unique(episode))} episodes, "
          f"reward per step mean {rewards.mean():.3f}, min {rewards.min():.3f}, max {rewards.max():.3f}")
    brake = (actions[:, 0] < 0).astype(int)
    print(f"     brake on {100 * brake.mean():5.2f}% of steps")
    for i, name in enumerate(ACTION_NAMES):
        if i >= 2:
            print(f"  {name:>8} on {100 * on[:, i].mean():5.2f}% of steps")
    # The expert brakes only inside its reverse manoeuvre, count those manoeuvres
    starts = int(((np.diff(brake, prepend=0) == 1)).sum())
    print(f"  reverse manoeuvres (brake switching on): {starts}")


def print_tensorboard(run, window=10_000):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard is not installed, skipping training logs")
        return
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    tags = [t for t in TB_TAGS if t in acc.Tags()["scalars"]]
    if not tags:
        print(f"\nno TensorBoard scalars in {run}")
        return
    frames = []
    for tag in tags:
        events = acc.Scalars(tag)
        s = pd.Series([e.value for e in events], index=[e.step for e in events])
        frames.append(s.groupby((s.index - 1) // window * window + window).mean().rename(tag))
    print(f"\n== {run.parent.name}/{run.name}, training logs, mean per {window} env steps ==")
    print(pd.concat(frames, axis=1).to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    experiment = Path(sys.argv[1]).resolve()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    print_dataset(experiment)
    expert_outcomes = print_expert(experiment)
    for run in sorted(experiment.glob("*/seed-*")):
        print_run(run, expert_outcomes)
        print_tensorboard(run)


if __name__ == "__main__":
    main()