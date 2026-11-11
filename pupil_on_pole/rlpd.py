import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from pupil_on_pole.checkpoints import save_checkpoint
from pupil_on_pole.data import Normalizer, ReplayBuffer, check_dataset, load_dataset
from pupil_on_pole.env import SoloKartEnv
from pupil_on_pole.networks import EnsembleCritic, GaussianPolicy


class RLPDAgent:
    def __init__(self, obs_dim, action_dim, cfg, normalizer):
        self.cfg = cfg
        self.normalizer = normalizer
        self.actor = GaussianPolicy(obs_dim, action_dim, cfg.hidden)
        self.critic = EnsembleCritic(obs_dim, action_dim, cfg.hidden, cfg.n_critics, cfg.critic_layer_norm)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature = torch.tensor(np.log(cfg.init_temperature), dtype=torch.float32, requires_grad=True)
        self.target_entropy = -float(action_dim) if cfg.target_entropy is None else float(cfg.target_entropy)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=cfg.lr_temperature)

    @property
    def temperature(self):
        return self.log_temperature.exp().detach()

    def _tensors(self, batch):
        t = {name: torch.as_tensor(value) for name, value in batch.items()}
        t["obs"] = self.normalizer(t["obs"])
        t["next_obs"] = self.normalizer(t["next_obs"])
        t["rewards"] = t["rewards"] * self.cfg.reward_scale
        return t

    @torch.no_grad()
    def act(self, obs):
        x = self.normalizer(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return self.actor.sample(x)[0][0].numpy()

    def update_critic(self, batch):
        b = self._tensors(batch)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(b["next_obs"])
            target_q = self.target_critic(b["next_obs"], next_action)
            subset = torch.randperm(self.cfg.n_critics)[: self.cfg.n_target_subset]
            target_q = target_q[subset].min(0).values
            if self.cfg.backup_entropy:
                target_q = target_q - self.temperature * next_log_prob
            # Truncations are not terminal, so they bootstrap like any other step
            target = b["rewards"] + self.cfg.gamma * (1.0 - b["terminated"]) * target_q
        q = self.critic(b["obs"], b["actions"])
        loss = ((q - target.unsqueeze(0)) ** 2).mean(1).sum()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.lerp_(param, self.cfg.tau)
        return loss.item(), q.mean().item()

    def update_actor(self, batch):
        b = self._tensors(batch)
        action, log_prob = self.actor.sample(b["obs"])
        # The actor maximizes the mean over the ensemble, as in RLPD
        q = self.critic(b["obs"], action).mean(0)
        actor_loss = (self.temperature * log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        temperature_loss = -(self.log_temperature * (log_prob.detach() + self.target_entropy)).mean()
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        return actor_loss.item(), -log_prob.mean().item()


def mixed_batch(offline, online, batch_size, offline_ratio, rng):
    """Symmetric sampling, a fixed share of each batch from the offline data and the rest online"""
    n_offline = int(round(batch_size * offline_ratio))
    parts = []
    if n_offline > 0:
        parts.append(offline.sample(n_offline, rng))
    if batch_size - n_offline > 0:
        parts.append(online.sample(batch_size - n_offline, rng))
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def run_rlpd(cfg, env_cfg, dataset_path, run_dir, log_training=False):
    """Train RLPD from the shared expert dataset, saving checkpoints every cfg.checkpoint_every online steps"""
    run_dir = Path(run_dir)
    if (run_dir / "checkpoints").exists():
        raise ValueError(f"{run_dir} already has checkpoints")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"rlpd": asdict(cfg), "env": asdict(env_cfg), "dataset": str(dataset_path)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    arrays, meta = load_dataset(dataset_path)
    check_dataset(meta, env_cfg)
    normalizer = Normalizer.from_observations(arrays["obs"])
    offline = ReplayBuffer.from_dataset(arrays)

    env = SoloKartEnv(env_cfg)
    online = ReplayBuffer(env.observation_dim, env.action_dim, cfg.online_steps)
    agent = RLPDAgent(env.observation_dim, env.action_dim, cfg, normalizer)
    writer = SummaryWriter(run_dir) if log_training else None
    start = time.time()
    episodes, grad_updates = 0, 0

    def save(step):
        extra = {"episodes": episodes, "grad_updates": grad_updates, "wall_time_s": time.time() - start}
        save_checkpoint(run_dir, "rlpd", cfg, env_cfg, step, agent.actor, normalizer, extra, critic=agent.critic)

    try:
        save(0)
        obs, _ = env.reset(cfg.seed * 1_000_000)
        rewards = []
        for step in tqdm(range(1, cfg.online_steps + 1), desc=f"rlpd seed {cfg.seed}"):
            if step <= cfg.learning_starts:
                action = rng.uniform(-1.0, 1.0, env.action_dim).astype(np.float32)
            else:
                action = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Rewards are stored raw and scaled when sampled, like the offline data
            online.add(obs, action, reward, next_obs, terminated)
            rewards.append(reward)
            obs = next_obs
            if terminated or truncated:
                if writer is not None:
                    for name, value in (
                        ("return", sum(rewards)),
                        ("steps", info["steps"]),
                        ("finished", float(terminated)),
                        ("progress", info["progress"]),
                        ("max_reward", max(rewards)),
                        ("min_reward", min(rewards)),
                    ):
                        writer.add_scalar(f"episode/{name}", value, step)
                episodes += 1
                obs, _ = env.reset(cfg.seed * 1_000_000 + episodes)
                rewards = []

            if step >= cfg.learning_starts and len(online) >= cfg.batch_size:
                for _ in range(cfg.utd):
                    batch = mixed_batch(offline, online, cfg.batch_size, cfg.offline_ratio, rng)
                    critic_loss, q_mean = agent.update_critic(batch)
                    grad_updates += 1
                batch = mixed_batch(offline, online, cfg.batch_size, cfg.offline_ratio, rng)
                actor_loss, entropy = agent.update_actor(batch)
                if writer is not None:
                    writer.add_scalar("loss/critic", critic_loss, step)
                    writer.add_scalar("loss/actor", actor_loss, step)
                    writer.add_scalar("q/mean", q_mean, step)
                    writer.add_scalar("policy/entropy", entropy, step)
                    writer.add_scalar("policy/temperature", agent.temperature.item(), step)

            if step % cfg.checkpoint_every == 0 or step == cfg.online_steps:
                save(step)
    finally:
        if writer is not None:
            writer.close()
        env.close()
    return run_dir
