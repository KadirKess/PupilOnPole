from pupil_on_pole.config import EnvConfig
from pupil_on_pole.env import ACTION_DIM, decode_action, feature_names

import numpy as np


class FakeEnv:
    """Stand-in for SoloKartEnv without SuperTuxKart, episodes finish after episode_steps steps"""

    def __init__(self, cfg=EnvConfig(), episode_steps=7, finish=True):
        self.cfg = cfg
        self.episode_steps = episode_steps
        self.finish = finish
        self.feature_names = feature_names(cfg.n_paths, cfg.n_items)
        self.observation_dim = len(self.feature_names)
        self.action_dim = ACTION_DIM
        self.track = cfg.track
        self.resets = []

    def set_track(self, track):
        self.track = track

    def _obs(self):
        return np.full(self.observation_dim, self._steps / self.episode_steps, dtype=np.float32)

    def reset(self, seed):
        self.resets.append((self.track, int(seed)))
        self._steps = 0
        return self._obs(), {"raw_obs": {"steps": 0}, "steps": 0}

    def step(self, action):
        return self.step_dict(decode_action(action))

    def step_dict(self, action):
        executed = {"acceleration": float(action.get("acceleration", 0.0)), "steer": float(action.get("steer", 0.0))}
        for name in ("brake", "drift", "nitro", "fire"):
            executed[name] = bool(action.get(name, False))
        executed["rescue"] = False
        self._steps += 1
        terminated = self.finish and self._steps >= self.episode_steps
        truncated = not terminated and self._steps >= self.episode_steps
        info = {
            "raw_obs": {"steps": self._steps, "velocity": [3.0, 0.0, 4.0], "center_path_distance": -2.0},
            "distance": float(self._steps),
            "progress": self._steps / self.episode_steps,
            "finished": terminated,
            "stuck": truncated,
            "steps": self._steps,
            "perturbed": False,
            "executed": executed,
        }
        return self._obs(), 1.0, terminated, truncated, info

    def close(self):
        pass


class FakeExpert:
    """Always the same action, brake on, counts its calls"""

    ACTION = {"acceleration": 1.0, "steer": 0.5, "brake": True, "drift": False, "nitro": False, "fire": False}

    def __init__(self, fire_every=None):
        # With fire_every, fire is on for one step out of fire_every
        self.fire_every = fire_every
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, raw_obs):
        self.calls += 1
        action = dict(self.ACTION)
        if self.fire_every:
            action["fire"] = self.calls % self.fire_every == 0
        return action
