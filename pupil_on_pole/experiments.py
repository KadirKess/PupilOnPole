import json
import multiprocessing
import subprocess
import time
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields, replace
from importlib.metadata import version
from pathlib import Path

from pupil_on_pole.config import DaggerConfig, DatasetConfig, EnvConfig, EvalConfig, RLPDConfig

PYSTK_PROJECT_COMMIT = "ddf2cb6"
METHODS = ("dagger", "rlpd")
CONFIGS = {
    "env_config": EnvConfig,
    "dataset_config": DatasetConfig,
    "dagger_config": DaggerConfig,
    "rlpd_config": RLPDConfig,
    "eval_config": EvalConfig,
}


def _from_dict(cls, values):
    """Rebuild a config from JSON, where tuples came back as lists"""
    hints = typing.get_type_hints(cls)
    values = dict(values)
    for field in fields(cls):
        if typing.get_origin(hints[field.name]) is tuple and values.get(field.name) is not None:
            values[field.name] = tuple(values[field.name])
    return cls(**values)


def _pystk_commit():
    """Commit of the pystk_project clone, or None when it cannot be read"""
    from pupil_on_pole.expert import pystk_project_src

    try:
        src = pystk_project_src()
        out = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (RuntimeError, OSError, subprocess.CalledProcessError):
        return None


def _versions():
    return {name: version(name) for name in ("pystk2-gymnasium", "pysupertuxkart2", "torch")}


def create_experiment(
    name,
    methods=METHODS,
    seeds=(1, 2, 3),
    env_cfg=EnvConfig(),
    dataset_cfg=DatasetConfig(),
    dagger_cfg=DaggerConfig(),
    rlpd_cfg=RLPDConfig(),
    eval_cfg=EvalConfig(),
    workers=4,
    torch_threads=2,
    root="outputs",
):
    """Create outputs/YYYYMMDD-HHMMSS-<name>/experiment.json and return the folder"""
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
    if dagger_cfg.online_steps != rlpd_cfg.online_steps:
        raise ValueError("Both methods must get the same online step budget")
    # Absolute, the expert changes the working directory of the processes that load it
    experiment_dir = Path(root).resolve() / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    experiment_dir.mkdir(parents=True)
    manifest = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods": list(methods),
        "seeds": [int(s) for s in seeds],
        "workers": int(workers),
        "torch_threads": int(torch_threads),
        "env_config": asdict(env_cfg),
        "dataset_config": asdict(dataset_cfg),
        "dagger_config": asdict(dagger_cfg),
        "rlpd_config": asdict(rlpd_cfg),
        "eval_config": asdict(eval_cfg),
        "versions": _versions(),
        "pystk_project_commit": _pystk_commit(),
        "expected_pystk_project_commit": PYSTK_PROJECT_COMMIT,
    }
    (experiment_dir / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return experiment_dir


def load_manifest(experiment_dir):
    manifest = json.loads((Path(experiment_dir).resolve() / "experiment.json").read_text())
    configs = {key: _from_dict(cls, manifest[key]) for key, cls in CONFIGS.items()}
    return manifest, configs


def run_dir(experiment_dir, method, seed):
    return Path(experiment_dir).resolve() / method / f"seed-{seed}"


def dataset_path(experiment_dir):
    return Path(experiment_dir).resolve() / "data" / "d0.npz"


# Worker functions, top level so that spawn can pickle them. Each one runs in a fresh process with its own STK race.


def _init_worker(torch_threads):
    import torch

    torch.set_num_threads(int(torch_threads))


def _dataset_worker(env_cfg, dataset_cfg, path):
    from pupil_on_pole import data
    from pupil_on_pole.env import SoloKartEnv
    from pupil_on_pole.expert import Expert

    with SoloKartEnv(env_cfg) as env:
        expert = Expert(env, team=dataset_cfg.expert_team)
        arrays = data.generate_dataset(env, expert, dataset_cfg)
        meta = data.dataset_meta(env, dataset_cfg, _pystk_commit())
    data.save_dataset(path, arrays, meta)
    return data.episode_summary(arrays)


def _train_worker(method, cfg, env_cfg, dataset, out, log_training):
    if method == "dagger":
        from pupil_on_pole.dagger import run_dagger as run
    else:
        from pupil_on_pole.rlpd import run_rlpd as run
    return str(run(cfg, env_cfg, dataset, out, log_training=log_training))


def _evaluate_worker(out, eval_cfg, env_cfg, held_out):
    from pupil_on_pole.evaluation import evaluate_run

    return str(evaluate_run(out, eval_cfg, env_cfg, held_out))


def _expert_worker(out, eval_cfg, env_cfg, team):
    from pupil_on_pole.evaluation import evaluate_expert_reference

    return str(evaluate_expert_reference(out, eval_cfg, env_cfg, team))


def _pool(workers, torch_threads):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
        initializer=_init_worker,
        initargs=(torch_threads,),
    )


def _run_all(tasks, workers, torch_threads):
    """Run (name, fn, args) tasks in worker processes, raise at the end if any failed"""
    results, errors = {}, {}
    with _pool(workers, torch_threads) as pool:
        futures = {pool.submit(fn, *args): name for name, fn, args in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"done {name}")
            except Exception as error:  # noqa: BLE001, reported below
                errors[name] = error
                print(f"failed {name}: {error!r}")
    if errors:
        raise RuntimeError(f"{len(errors)} task(s) failed: {sorted(errors)}") from next(iter(errors.values()))
    return results


def generate_dataset(experiment_dir):
    """Record D0 once per experiment in a worker process, so the caller keeps no STK race"""
    manifest, configs = load_manifest(experiment_dir)
    path = dataset_path(experiment_dir)
    if path.exists():
        raise ValueError(f"{path} already exists")
    summary = _run_all(
        [("d0", _dataset_worker, (configs["env_config"], configs["dataset_config"], path))], 1, manifest["torch_threads"]
    )["d0"]
    finished = sum(e["finished"] for e in summary)
    print(f"D0: {len(summary)} episodes, {finished} finished, {sum(e['steps'] for e in summary)} transitions")
    return path


def train_experiment(experiment_dir, log_training=True):
    """Train every method and seed in parallel, runs that already have checkpoints are skipped"""
    manifest, configs = load_manifest(experiment_dir)
    dataset = dataset_path(experiment_dir)
    if not dataset.exists():
        raise ValueError("Generate the dataset first")
    tasks = []
    for method in manifest["methods"]:
        for seed in manifest["seeds"]:
            out = run_dir(experiment_dir, method, seed)
            if (out / "checkpoints").exists():
                print(f"skip {method} seed {seed}, it already has checkpoints")
                continue
            cfg = replace(configs[f"{method}_config"], seed=seed)
            tasks.append((f"{method}/seed-{seed}", _train_worker,
                          (method, cfg, configs["env_config"], dataset, out, log_training)))
    return _run_all(tasks, manifest["workers"], manifest["torch_threads"])


def evaluate_experiment(experiment_dir, held_out=True):
    """Expert reference plus evaluate_run for every run, in parallel"""
    manifest, configs = load_manifest(experiment_dir)
    env_cfg, eval_cfg = configs["env_config"], configs["eval_config"]
    tasks = [("expert", _expert_worker,
              (Path(experiment_dir) / "expert", eval_cfg, env_cfg, configs["dataset_config"].expert_team))]
    for method in manifest["methods"]:
        for seed in manifest["seeds"]:
            out = run_dir(experiment_dir, method, seed)
            tasks.append((f"{method}/seed-{seed}", _evaluate_worker, (out, eval_cfg, env_cfg, held_out)))
    return _run_all(tasks, manifest["workers"], manifest["torch_threads"])


def load_experiment(experiment_dir, budget=None):
    """Evaluation results as tables, also saved as CSV in tables/"""
    from pupil_on_pole.analysis import load_results, save_tables, summarize_results

    results = load_results(experiment_dir)
    results["summary"] = summarize_results(results, budget)
    results["tables"] = save_tables(results["summary"], results, experiment_dir)
    return results


def plot_experiment(results, experiment_dir):
    """The four figures of the study in figures/, as PNG and PDF"""
    from pupil_on_pole.analysis import endpoint
    from pupil_on_pole.plotting import plot_action_use, plot_curve, plot_fire, plot_generalization, plot_held_out

    out = Path(experiment_dir) / "figures"
    curves, expert = results["curves"], results["expert"]
    track = results["manifest"]["env_config"]["track"]
    figures = {
        "lap_time": plot_curve(curves, expert, "lap_time_s", "Lap time (s), unfinished = cap", track, out / "lap_time"),
        "finish_rate": plot_curve(curves, expert, "finish_rate", "Finish rate", track, out / "finish_rate"),
        "actions": plot_action_use(endpoint(curves, results["summary"]["budget"])[0], expert, track, out / "actions"),
    }
    if not results["held_out"].empty:
        figures["held_out"] = plot_held_out(results["held_out"], expert, "lap_time_s", "Lap time (s)", out / "held_out")
        figures["held_out_finish"] = plot_held_out(results["held_out"], expert, "finish_rate", "Finish rate",
                                                   out / "held_out_finish")
    runs = results["summary"].get("generalization_runs")
    if runs is not None:
        figures["generalization"] = plot_generalization(runs, track, out / "generalization")
        figures["fire"] = plot_fire(runs, expert, track, out / "fire")
    return figures
