"""Short RLPD run on an existing D0, to check that it learns before the main study

uv run python -m scripts.check_rlpd outputs/<experiment with data/d0.npz> [steps]
"""

import sys
import time
from pathlib import Path

from pupil_on_pole.config import EnvConfig, EvalConfig, RLPDConfig
from pupil_on_pole.evaluation import evaluate_run
from pupil_on_pole.experiments import dataset_path
from pupil_on_pole.rlpd import run_rlpd
from scripts.diagnose import print_run, print_tensorboard


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    experiment = Path(sys.argv[1]).resolve()
    steps = int(sys.argv[2]) if len(sys.argv) == 3 else 15_000
    out = experiment / "rlpd-check" / f"seed-1-{time.strftime('%H%M%S')}"
    cfg = RLPDConfig(online_steps=steps, learning_starts=1_000, checkpoint_every=2_500)
    run_rlpd(cfg, EnvConfig(), dataset_path(experiment), out, log_training=True)
    evaluate_run(out, EvalConfig(n_episodes=5), EnvConfig(), held_out=False)
    print_run(out, None)
    print_tensorboard(out, window=2_500)
    print(f"\nRun in {out}")


if __name__ == "__main__":
    main()
