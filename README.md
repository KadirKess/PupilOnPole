# Pupil on Pole

We compare DAgger and RLPD-style symmetric sampling on a single kart driving one lap of a SuperTuxKart track as fast as possible, at an equal budget of environment steps.

The expert policies are the hand-coded agents from [osigaud/pystk_project](https://github.com/osigaud/pystk_project/tree/master/src/agents).

## Study

Both methods start from the same dataset D0, 20 noisy laps of team2 on scotland, and then get the same number of online environment steps.

- **DAgger** (Ross et al. 2011): iteration 0 is behavioural cloning on D0, then the learner drives alone (beta = 0) and the expert labels every visited state. Each iteration collects 2 episodes and retrains from scratch on all the data.
- **RLPD** (Ball et al. 2023): SAC on the race reward times 10, each batch half D0 and half online replay. It uses 3 LayerNorm critics, a target that is the minimum over a random 2 of them, and 6 critic updates per environment step.

The environment changes pystk2 in four ways, for every driver:

- Each step, with probability 0.02, the steering is replaced by a uniform value for 5 steps. The noise depends only on the episode seed.
- An episode is truncated after 150 steps without 1 m of progress.
- Rescue is always off.
- The expert never fires.

Every checkpoint is evaluated on scotland, and the last one zero-shot on lighthouse, olivermath and fortmagma. Evaluation uses the deterministic policy and the same 10 noisy seeds for every driver. The primary metric is lap time, with an unfinished lap counted as the 150 s cap, reported next to the finish rate.

## Setup

```sh
uv sync
```

The expert is team2's agent from that repository at commit `ddf2cb6`. Clone it anywhere and point `PYSTK_PROJECT_SRC` to its `src` folder.

```sh
git clone https://github.com/osigaud/pystk_project.git
git -C pystk_project checkout ddf2cb6
```

Then set the variable, in bash or zsh with `export PYSTK_PROJECT_SRC=$PWD/pystk_project/src`, or in PowerShell with `$env:PYSTK_PROJECT_SRC = "$PWD\pystk_project\src"`.

The first environment launch downloads the SuperTuxKart game data (about 800 MB) into the user cache. SuperTuxKart allows only one race per process, so parallel runs use separate processes.

## Reproducing the study

Run `experiments.ipynb` from top to bottom. Each stage is a function of `pupil_on_pole.experiments` and runs SuperTuxKart in worker processes:

1. `create_experiment` writes `outputs/YYYYMMDD-HHMMSS-<name>/experiment.json` with every config.
2. `generate_dataset` records D0 to `data/d0.npz`.
3. `train_experiment` trains every method and seed in parallel into `<method>/seed-<n>/` (checkpoints and TensorBoard logs).
4. `evaluate_experiment` evaluates team2 into `expert/` and each run into `<method>/seed-<n>/evaluation.json`.
5. `load_experiment` and `plot_experiment` write `tables/` (CSV) and `figures/` (PNG and PDF).

The order is smoke test, then pilot, which fixes the budget, then the main study with seeds 1, 2 and 3. Scripts that call these functions outside a notebook need an `if __name__ == "__main__":` guard, because workers are started with spawn.

## Credits

The SAC code follows [CleanRL](https://github.com/vwxyzjn/cleanrl)'s single-file SAC (MIT licence). DAgger is from Ross, Gordon and Bagnell (2011), and RLPD from Ball, Smith, Kostrikov and Levine (2023).

## Tests

```sh
uv run python -m unittest discover -s tests -t .
```

Tests that need the expert are skipped when `PYSTK_PROJECT_SRC` is not set.
