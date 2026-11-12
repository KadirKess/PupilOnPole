"""Main study on a reduced budget, 3 seeds of both methods in parallel, then evaluation, tables and figures

uv run python -m scripts.main
"""

from pupil_on_pole import experiments as ex
from pupil_on_pole.config import DaggerConfig, EvalConfig, RLPDConfig

# The pilot's budget, RLPD is the limit at about 80 ms per step with utd 6
BUDGET = 100_000


def main():
    out = ex.create_experiment(
        "main",
        seeds=(1, 2, 3),
        # About 45 retrainings of 13k steps, so DAgger gets about as many gradient updates as RLPD's critic,
        # (BUDGET - learning_starts) * utd = 594k
        dagger_cfg=DaggerConfig(online_steps=BUDGET, episodes_per_iter=4, grad_steps=13_000),
        # With D0 in the buffer, a long uniform random phase only drives backwards
        rlpd_cfg=RLPDConfig(online_steps=BUDGET, learning_starts=1_000),
        eval_cfg=EvalConfig(n_episodes=30),
        # 6 SuperTuxKart processes do not fit in 16 GB next to the desktop, 4 leave a margin
        workers=4,
        torch_threads=2,
    )
    print(f"Experiment folder: {out}")
    ex.generate_dataset(out)
    ex.train_experiment(out)
    ex.evaluate_experiment(out)
    results = ex.load_experiment(out, budget=BUDGET)
    ex.plot_experiment(results, out)
    summary = results["summary"]
    for name in ("main", "welch", "held_out"):
        if name in summary:
            print(f"\n== {name} ==\n{summary[name]}")
    print(f"\nTables in {out / 'tables'}, figures in {out / 'figures'}")


if __name__ == "__main__":
    main()
