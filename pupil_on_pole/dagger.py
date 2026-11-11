import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from pupil_on_pole.checkpoints import save_checkpoint
from pupil_on_pole.data import Normalizer, check_dataset, load_dataset
from pupil_on_pole.env import encode_action
from pupil_on_pole.networks import DeterministicPolicy, policy_action

EPISODE_SEED_STRIDE = 1_000_000


def train_bc(obs, actions, normalizer, cfg, seed):
    """Train a fresh DeterministicPolicy by MSE on the labels, return it and the mean loss of the last 100 steps"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = DeterministicPolicy(obs.shape[1], actions.shape[1], cfg.hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    # Normalized once, the normalizer is frozen
    x = normalizer(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
    y = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    tail = []
    for step in range(cfg.grad_steps):
        idx = torch.as_tensor(rng.integers(0, len(x), size=cfg.batch_size))
        loss = F.mse_loss(policy(x[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= cfg.grad_steps - 100:
            tail.append(loss.item())
    policy.eval()
    return policy, float(np.mean(tail))


def dagger_loop(env, expert, cfg, arrays, normalizer, save, writer=None):
    """BC on the expert data, then DAgger with beta 0 until cfg.online_steps online steps

    save(step, policy, counters) is called at step 0, at every multiple of
    cfg.checkpoint_every and at the last step, with the latest trained policy,
    the one driving. That policy was trained at the end of the previous
    iteration, counters["trained_on_steps"] records how many online steps of
    labels it saw, at most the checkpoint step.
    """
    obs_data, act_data = [arrays["obs"]], [np.asarray(arrays["actions"], dtype=np.float32)]
    counters = {
        "iterations": 0,
        "episodes": 0,
        "expert_queries": 0,
        "grad_updates": 0,
        # Size of the latest policy's training set, and the online steps in it
        "dataset_size": len(arrays["obs"]),
        "trained_on_steps": 0,
    }

    def retrain(step):
        obs, actions = np.concatenate(obs_data), np.concatenate(act_data)
        policy, loss = train_bc(obs, actions, normalizer, cfg, seed=cfg.seed * 1_000 + counters["iterations"])
        counters["grad_updates"] += cfg.grad_steps
        counters["dataset_size"] = len(obs)
        counters["trained_on_steps"] = step
        if writer is not None:
            writer.add_scalar("loss/bc", loss, step)
            writer.add_scalar("dagger/dataset_size", len(obs), step)
            writer.add_scalar("dagger/expert_queries", counters["expert_queries"], step)
        return policy

    # Iteration 0, behavioural cloning on D0 (beta = 1)
    policy = retrain(0)
    save(0, policy, counters)

    step = 0
    progress = tqdm(total=cfg.online_steps, desc=f"dagger seed {cfg.seed}")
    while step < cfg.online_steps:
        counters["iterations"] += 1
        for _ in range(cfg.episodes_per_iter):
            if step >= cfg.online_steps:
                break
            obs, info = env.reset(cfg.seed * EPISODE_SEED_STRIDE + counters["episodes"])
            expert.reset()
            counters["episodes"] += 1
            rewards, done = [], False
            while not done and step < cfg.online_steps:
                # The expert labels the state the learner is in, and keeps its memory along the learner's path
                obs_data.append(obs[None])
                act_data.append(encode_action(expert.act(info["raw_obs"]))[None])
                counters["expert_queries"] += 1
                # beta = 0, only the learner drives
                obs, reward, terminated, truncated, info = env.step(policy_action(policy, normalizer, obs))
                rewards.append(reward)
                step += 1
                progress.update(1)
                done = terminated or truncated
                if step % cfg.checkpoint_every == 0 or step == cfg.online_steps:
                    save(step, policy, counters)
            if done and writer is not None:
                for name, value in (
                    ("return", sum(rewards)),
                    ("steps", info["steps"]),
                    ("finished", float(terminated)),
                    ("progress", info["progress"]),
                    ("max_reward", max(rewards)),
                    ("min_reward", min(rewards)),
                ):
                    writer.add_scalar(f"episode/{name}", value, step)
        if step >= cfg.online_steps:
            # The budget is spent, a retraining here would not be evaluated
            break
        # From scratch on the whole aggregated dataset, as in Ross et al. 2011
        policy = retrain(step)
    progress.close()
    return policy, counters


def run_dagger(cfg, env_cfg, dataset_path, run_dir, log_training=False):
    """Train BC then DAgger from the shared expert dataset, saving checkpoints every cfg.checkpoint_every online steps"""
    from pupil_on_pole.env import SoloKartEnv
    from pupil_on_pole.expert import Expert

    # Absolute, because the expert changes the working directory while TensorBoard writes from a thread
    run_dir, dataset_path = Path(run_dir).resolve(), Path(dataset_path).resolve()
    if (run_dir / "checkpoints").exists():
        raise ValueError(f"{run_dir} already has checkpoints")
    arrays, meta = load_dataset(dataset_path)
    check_dataset(meta, env_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"dagger": asdict(cfg), "env": asdict(env_cfg), "dataset": str(dataset_path)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    normalizer = Normalizer.from_observations(arrays["obs"])
    team = meta.get("dataset_config", {}).get("expert_team", 2)

    writer = SummaryWriter(run_dir) if log_training else None
    start = time.time()

    def save(step, policy, counters):
        extra = {**counters, "wall_time_s": time.time() - start}
        save_checkpoint(run_dir, "dagger", cfg, env_cfg, step, policy, normalizer, extra)

    env = SoloKartEnv(env_cfg)
    try:
        expert = Expert(env, team=team)
        dagger_loop(env, expert, cfg, arrays, normalizer, save, writer)
    finally:
        if writer is not None:
            writer.close()
        env.close()
    return run_dir
