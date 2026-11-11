# Pupil on Pole

We compare DAgger and RLPD-style symmetric sampling on a single kart driving one lap of a SuperTuxKart track as fast as possible, at an equal budget of environment steps.

The expert policies are the hand-coded agents from [osigaud/pystk_project](https://github.com/osigaud/pystk_project/tree/master/src/agents).

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

## Tests

```sh
uv run python -m unittest discover -s tests -t .
```

Tests that need the expert are skipped when `PYSTK_PROJECT_SRC` is not set.
