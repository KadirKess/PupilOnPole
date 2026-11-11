import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = (
    "lap_time_s",
    "finish_rate",
    "lap_steps",
    "lap_time_std_s",
    "best_lap_time_s",
    "stuck_rate",
    "progress",
    "return",
    "mean_speed",
    "mean_center_distance",
    "fire_presses",
)


def _row(summary):
    row = {name: summary.get(name) for name in METRICS}
    row.update({f"use_{name}": value for name, value in summary["action_use"].items()})
    return row


def run_tables(evaluation, method, seed):
    """Learning curve and held-out rows of one run's evaluation.json"""
    curve = [
        {"method": method, "seed": seed, "online_step": point["online_step"], **_row(point["summary"]),
         **{f"extra_{k}": v for k, v in point["extra"].items()}}
        for point in evaluation["curve"]
    ]
    held_out = [
        {"method": method, "seed": seed, "track": track, **_row(result["summary"])}
        for track, result in evaluation["held_out"].items()
    ]
    return curve, held_out


def expert_table(evaluation):
    return pd.DataFrame(
        [{"method": evaluation["method"], "track": track, **_row(result["summary"])}
         for track, result in evaluation["tracks"].items()]
    )


def load_results(experiment_dir):
    """Every evaluation.json of an experiment as DataFrames"""
    experiment_dir = Path(experiment_dir)
    manifest = json.loads((experiment_dir / "experiment.json").read_text())
    curves, held_out = [], []
    for method in manifest["methods"]:
        for seed in manifest["seeds"]:
            path = experiment_dir / method / f"seed-{seed}" / "evaluation.json"
            if path.exists():
                c, h = run_tables(json.loads(path.read_text()), method, seed)
                curves += c
                held_out += h
    expert_path = experiment_dir / "expert" / "evaluation.json"
    expert = expert_table(json.loads(expert_path.read_text())) if expert_path.exists() else pd.DataFrame()
    return {
        "manifest": manifest,
        "curves": pd.DataFrame(curves),
        "held_out": pd.DataFrame(held_out),
        "expert": expert,
    }


def bootstrap_ci(values, n_resamples=10_000, confidence=0.95, seed=0):
    """Percentile bootstrap interval of the mean"""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def welch(a, b):
    """Welch t test, nan when both samples are constant"""
    result = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False)
    return float(result.statistic), float(result.pvalue)


def _summary(df, metric, reference=None):
    rows = []
    for method, group in df.groupby("method"):
        values = group.sort_values("seed")[metric].to_numpy(dtype=float)
        low, high = bootstrap_ci(values)
        row = {
            "method": method,
            "seeds": group["seed"].nunique(),
            **{f"seed_{s}": v for s, v in zip(group.sort_values("seed")["seed"], values)},
            "mean": values.mean(),
            "ci_low": low,
            "ci_high": high,
            "finish_rate": group["finish_rate"].mean(),
        }
        if reference is not None:
            row["diff_expert"] = row["mean"] - reference
            row["diff_expert_pct"] = 100 * row["diff_expert"] / reference if reference else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def endpoint(curves, budget=None):
    """Rows at the fixed endpoint, the last common online step by default"""
    if budget is None:
        budget = curves.groupby(["method", "seed"])["online_step"].max().min()
    return curves[curves["online_step"] == budget], int(budget)


def generalization(final, held_out, expert, train_track, metric="lap_time_s"):
    """Per seed and track, the metric over team2's on the same track, and its gap to the training track

    vs_expert = 1 means as fast as team2, above 1 is slower. gap = vs_expert on a
    held-out track minus vs_expert on the training track, 0 means the policy
    transfers as well as the expert does.
    """
    rows = pd.concat([final.assign(track=train_track), held_out], ignore_index=True)
    reference = expert.set_index("track")[metric]
    rows = rows.assign(vs_expert=rows[metric] / rows["track"].map(reference))
    train = rows[rows["track"] == train_track].set_index(["method", "seed"])["vs_expert"]
    rows["gap"] = rows["vs_expert"] - rows.set_index(["method", "seed"]).index.map(train)
    return rows[["method", "seed", "track", metric, "finish_rate", "fire_presses", "vs_expert", "gap"]]


def generalization_summary(rows):
    out = []
    for (method, track), group in rows.groupby(["method", "track"]):
        low, high = bootstrap_ci(group["vs_expert"])
        out.append({
            "method": method,
            "track": track,
            "vs_expert": group["vs_expert"].mean(),
            "ci_low": low,
            "ci_high": high,
            "gap": group["gap"].mean(),
            "finish_rate": group["finish_rate"].mean(),
            "fire_presses": group["fire_presses"].mean(),
        })
    return pd.DataFrame(out)


def efficiency(curves, reference, tolerance=0.05, metric="lap_time_s"):
    """Per run, the first online step within tolerance of team2 and the mean metric over training

    steps_to_expert is NaN when the run never gets there. mean_over_training is
    the area under the curve divided by the budget, lower is better for lap time.
    """
    rows = []
    for (method, seed), group in curves.groupby(["method", "seed"]):
        group = group.sort_values("online_step")
        steps, values = group["online_step"].to_numpy(float), group[metric].to_numpy(float)
        reached = steps[values <= reference * (1 + tolerance)] if reference is not None else []
        span = steps[-1] - steps[0]
        rows.append({
            "method": method,
            "seed": seed,
            "steps_to_expert": float(reached[0]) if len(reached) else np.nan,
            "mean_over_training": float(np.trapezoid(values, steps) / span) if span > 0 else float(values[0]),
        })
    return pd.DataFrame(rows)


def summarize_results(results, budget=None, metric="lap_time_s", methods=("rlpd", "dagger")):
    """Endpoint summary on the training track, Welch tests, held-out, generalization and efficiency"""
    curves, expert = results["curves"], results["expert"]
    train_track = results["manifest"]["env_config"]["track"]

    def reference(track, name=metric):
        if expert.empty:
            return None
        return float(expert.loc[expert["track"] == track, name].iloc[0])

    final, budget = endpoint(curves, budget)
    main = pd.concat(
        [_summary(final, name, reference(train_track, name)).assign(metric=name)
         for name in (metric, "finish_rate", "fire_presses")],
        ignore_index=True,
    )
    tests = []
    for name in (metric, "finish_rate", "fire_presses"):
        a, b = (final.loc[final["method"] == m, name] for m in methods)
        t, p = welch(a, b)
        tests.append({"metric": name, "a": methods[0], "b": methods[1], "t": t, "p": p, "n_a": len(a), "n_b": len(b)})
    held_out = results["held_out"]
    summary = {
        "budget": budget,
        "main": main,
        "welch": pd.DataFrame(tests),
        "held_out": pd.concat(
            [_summary(group, metric, reference(track)).assign(track=track) for track, group in held_out.groupby("track")],
            ignore_index=True,
        ) if not held_out.empty else pd.DataFrame(),
        "efficiency": efficiency(curves, reference(train_track)),
    }
    if not held_out.empty and not expert.empty:
        summary["generalization_runs"] = generalization(final, held_out, expert, train_track, metric)
        summary["generalization"] = generalization_summary(summary["generalization_runs"])
    return summary


def save_tables(summary, results, experiment_dir):
    out = Path(experiment_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    tables = {name: summary[name] for name in
              ("main", "welch", "held_out", "efficiency", "generalization", "generalization_runs") if name in summary}
    tables.update(curves=results["curves"], expert=results["expert"])
    for name, table in tables.items():
        paths[name] = out / f"{name}.csv"
        table.to_csv(paths[name], index=False)
    return paths
