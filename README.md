# Pupil on Pole

We compare DAgger and RLPD-style symmetric sampling on a single kart driving one lap of a SuperTuxKart track as fast as possible, at an equal budget of environment steps.

The expert policies are the hand-coded agents from [osigaud/pystk_project](https://github.com/osigaud/pystk_project/tree/master/src/agents).

## Setup

```sh
uv sync
```

The first environment launch downloads the SuperTuxKart game data (about 800 MB) into the user cache. On macOS, scripts that create the environment need an `if __name__ == "__main__":` guard, because the simulator runs in a spawned subprocess by default.
