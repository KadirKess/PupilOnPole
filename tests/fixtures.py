import os
import unittest
from dataclasses import asdict

import numpy as np

from pupil_on_pole.env import ACTION_DIM, feature_names
from pupil_on_pole.expert import PYSTK_SRC_VARIABLE

needs_expert = unittest.skipUnless(
    os.environ.get(PYSTK_SRC_VARIABLE), f"{PYSTK_SRC_VARIABLE} is not set"
)


def synthetic_dataset(env_cfg, n=64, seed=0):
    """Random transitions shaped like a recorded dataset, with metadata matching env_cfg"""
    rng = np.random.default_rng(seed)
    names = feature_names(env_cfg.n_paths, env_cfg.n_items)
    obs = rng.normal(size=(n, len(names))).astype(np.float32)
    terminated = np.zeros(n, dtype=bool)
    terminated[-1] = True
    arrays = {
        "obs": obs,
        "actions": rng.uniform(-1, 1, size=(n, ACTION_DIM)).astype(np.float32),
        "rewards": rng.normal(size=n).astype(np.float32),
        "next_obs": np.roll(obs, -1, axis=0),
        "terminated": terminated,
        "truncated": np.zeros(n, dtype=bool),
        "perturbed": np.zeros(n, dtype=bool),
        "episode": np.zeros(n, dtype=np.int32),
    }
    meta = {"env_config": asdict(env_cfg), "feature_names": names}
    return arrays, meta
