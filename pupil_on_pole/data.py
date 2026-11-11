import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

FIELDS = ("obs", "actions", "rewards", "next_obs", "terminated", "truncated", "perturbed", "episode")


def generate_dataset(env, expert, cfg):
    """Record expert episodes, the stored action is both what was sent and the DAgger label"""
    from pupil_on_pole.env import encode_action

    rows = {name: [] for name in FIELDS}
    for episode in tqdm(range(cfg.n_episodes), desc="expert episodes"):
        obs, info = env.reset(cfg.first_seed + episode)
        expert.reset()
        done = False
        while not done:
            action = encode_action(expert.act(info["raw_obs"]))
            next_obs, reward, terminated, truncated, info = env.step(action)
            for name, value in (
                ("obs", obs),
                ("actions", action),
                ("rewards", reward),
                ("next_obs", next_obs),
                ("terminated", terminated),
                ("truncated", truncated),
                ("perturbed", info["perturbed"]),
                ("episode", episode),
            ):
                rows[name].append(value)
            obs, done = next_obs, terminated or truncated
    dtypes = {"terminated": bool, "truncated": bool, "perturbed": bool, "episode": np.int32}
    return {name: np.asarray(values, dtype=dtypes.get(name, np.float32)) for name, values in rows.items()}


def dataset_meta(env, dataset_cfg, pystk_commit=None):
    return {
        "env_config": asdict(env.cfg),
        "dataset_config": asdict(dataset_cfg),
        "feature_names": env.feature_names,
        "episode_seeds": [dataset_cfg.first_seed + i for i in range(dataset_cfg.n_episodes)],
        "pystk_project_commit": pystk_commit,
        "versions": {name: version(name) for name in ("pystk2-gymnasium", "pysupertuxkart2", "torch")},
    }


def check_dataset(meta, env_cfg):
    """Refuse a dataset recorded with other environment settings"""
    if meta["env_config"] != asdict(env_cfg):
        raise ValueError("The dataset was recorded with different environment settings")


def save_dataset(path, arrays, meta):
    """Write arrays as .npz and metadata as JSON next to it"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_dataset(path):
    path = Path(path)
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return arrays, meta


def episode_summary(arrays):
    summary = []
    for episode in np.unique(arrays["episode"]):
        mask = arrays["episode"] == episode
        summary.append(
            {
                "episode": int(episode),
                "steps": int(mask.sum()),
                "finished": bool(arrays["terminated"][mask].any()),
                "return": float(arrays["rewards"][mask].sum()),
            }
        )
    return summary


class Normalizer:
    """Standardize observations with statistics frozen from the expert data"""

    def __init__(self, mean, std):
        std = np.where(np.asarray(std) < 1e-6, 1.0, std)
        self.mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        self.std = torch.as_tensor(np.asarray(std), dtype=torch.float32)

    @classmethod
    def from_observations(cls, obs):
        obs = np.asarray(obs, dtype=np.float64)
        return cls(obs.mean(axis=0), obs.std(axis=0))

    def __call__(self, x):
        return torch.clamp((x - self.mean) / self.std, -10.0, 10.0)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"].numpy(), state["std"].numpy())


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.index = 0

    @classmethod
    def from_dataset(cls, arrays):
        buffer = cls(arrays["obs"].shape[1], arrays["actions"].shape[1], len(arrays["obs"]))
        for name in ("obs", "actions", "rewards", "next_obs"):
            getattr(buffer, name)[:] = arrays[name]
        buffer.terminated[:] = arrays["terminated"]
        buffer.size = buffer.capacity
        return buffer

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.index
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n, rng):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        idx = rng.integers(0, self.size, size=int(n))
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "terminated": self.terminated[idx],
        }

    def __len__(self):
        return self.size
