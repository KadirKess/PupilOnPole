from dataclasses import asdict
from pathlib import Path

import torch

from pupil_on_pole.config import DaggerConfig, EnvConfig, RLPDConfig
from pupil_on_pole.data import Normalizer
from pupil_on_pole.networks import DeterministicPolicy, GaussianPolicy

METHODS = {
    "dagger": (DaggerConfig, DeterministicPolicy),
    "rlpd": (RLPDConfig, GaussianPolicy),
}


def checkpoint_path(run_dir, online_step):
    return Path(run_dir) / "checkpoints" / f"step-{online_step:09d}.pt"


def save_checkpoint(run_dir, method, cfg, env_cfg, online_step, policy, normalizer, extra=None, critic=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    path = checkpoint_path(run_dir, online_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "config": asdict(cfg),
            "env_config": asdict(env_cfg),
            "online_step": int(online_step),
            "obs_dim": policy.obs_dim,
            "action_dim": policy.action_dim,
            "policy": policy.state_dict(),
            "normalizer": normalizer.state_dict(),
            "extra": dict(extra or {}),
            "critic": None if critic is None else critic.state_dict(),
        },
        path,
    )
    return path


def list_checkpoints(run_dir):
    return sorted((Path(run_dir) / "checkpoints").glob("step-*.pt"))


def load_checkpoint(path):
    """Return the payload, method config, env config, policy and normalizer on the CPU"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    Config, Policy = METHODS[payload["method"]]
    config = dict(payload["config"])
    config["hidden"] = tuple(config["hidden"])
    cfg = Config(**config)
    env_cfg = EnvConfig(**payload["env_config"])
    policy = Policy(payload["obs_dim"], payload["action_dim"], cfg.hidden)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    normalizer = Normalizer.from_state_dict(payload["normalizer"])
    return payload, cfg, env_cfg, policy, normalizer
