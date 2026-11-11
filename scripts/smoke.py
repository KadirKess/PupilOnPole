"""Mini training of both methods in a few minutes, then evaluation, tables and figures

Run from the repository root, as a module so that pupil_on_pole is importable:
uv run python -m scripts.smoke
"""

import pandas as pd

from pupil_on_pole import experiments as ex
from pupil_on_pole.config import DaggerConfig, DatasetConfig, EvalConfig, RLPDConfig

STEPS = 5_000
SHOWN = ["method", "seed", "online_step", "lap_time_s", "finish_rate", "stuck_rate", "progress", "fire_presses",
         "use_brake", "use_drift", "use_nitro", "use_fire"]


def main():
    out = ex.create_experiment(
        "smoke",
        seeds=(1,),
        dataset_cfg=DatasetConfig(n_episodes=5),
        dagger_cfg=DaggerConfig(online_steps=STEPS, checkpoint_every=1_000, grad_steps=5_000),
        rlpd_cfg=RLPDConfig(online_steps=STEPS, checkpoint_every=1_000, learning_starts=1_000),
        eval_cfg=EvalConfig(n_episodes=3),
        workers=3,
        torch_threads=2,
    )
    print(f"Experiment folder: {out}")
    ex.generate_dataset(out)
    ex.train_experiment(out)
    ex.evaluate_experiment(out)
    results = ex.load_experiment(out)
    figures = ex.plot_experiment(results, out)

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 3)
    summary = results["summary"]
    print("\n== Expert (team2) ==\n", results["expert"][["track", "lap_time_s", "finish_rate", "fire_presses"]])
    print("\n== Learning curves on the training track ==\n", results["curves"][SHOWN])
    print(f"\n== Endpoint at {summary['budget']} steps ==\n", summary["main"])
    print("\n== Welch tests (one seed, only checks the code) ==\n", summary["welch"])
    print("\n== Efficiency ==\n", summary["efficiency"])
    if "generalization" in summary:
        print("\n== Generalization (lap time / team2) ==\n", summary["generalization"])
    print(f"\nTables in {out / 'tables'}, figures in {out / 'figures'}: {sorted(figures)}")


if __name__ == "__main__":
    main()
