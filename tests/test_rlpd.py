import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pupil_on_pole.checkpoints import list_checkpoints, load_checkpoint
from pupil_on_pole.config import EnvConfig, RLPDConfig
from pupil_on_pole.data import Normalizer, ReplayBuffer, save_dataset
from pupil_on_pole.rlpd import RLPDAgent, mixed_batch, run_rlpd
from tests.fixtures import synthetic_dataset

SMALL = dict(hidden=(16, 16), batch_size=8)


def small_agent(**changes):
    torch.manual_seed(0)
    cfg = RLPDConfig(**{**SMALL, **changes})
    return RLPDAgent(109, 7, cfg, Normalizer(np.zeros(109), np.ones(109)))


class MixedBatchTests(unittest.TestCase):
    def test_symmetric_split(self):
        arrays, _ = synthetic_dataset(EnvConfig(), n=20)
        offline = ReplayBuffer.from_dataset(arrays)
        online = ReplayBuffer(109, 7, 10)
        for _ in range(10):
            online.add(np.full(109, 7.0), np.zeros(7), 0.0, np.zeros(109), False)
        batch = mixed_batch(offline, online, 8, 0.5, np.random.default_rng(0))
        self.assertEqual(batch["obs"].shape, (8, 109))
        self.assertTrue(np.all(batch["obs"][4:] == 7.0))
        self.assertFalse(np.any(np.all(batch["obs"][:4] == 7.0, axis=1)))

    def test_ratio_one_never_samples_online(self):
        arrays, _ = synthetic_dataset(EnvConfig(), n=20)
        offline = ReplayBuffer.from_dataset(arrays)
        batch = mixed_batch(offline, ReplayBuffer(109, 7, 1), 8, 1.0, np.random.default_rng(0))
        self.assertEqual(len(batch["rewards"]), 8)


class AgentUpdateTests(unittest.TestCase):
    def batch(self, terminated=0.0, rewards=1.0):
        rng = np.random.default_rng(0)
        return {
            "obs": rng.normal(size=(8, 109)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(8, 7)).astype(np.float32),
            "rewards": np.full(8, rewards, dtype=np.float32),
            "next_obs": rng.normal(size=(8, 109)).astype(np.float32),
            "terminated": np.full(8, terminated, dtype=np.float32),
        }

    def test_critic_update_moves_critic_and_target(self):
        agent = small_agent()
        before = [p.clone() for p in agent.critic.parameters()]
        target_before = [p.clone() for p in agent.target_critic.parameters()]
        loss, q_mean = agent.update_critic(self.batch())
        self.assertTrue(np.isfinite(loss) and np.isfinite(q_mean))
        for old, new in zip(before, agent.critic.parameters()):
            self.assertFalse(torch.equal(old, new))
        for old, new, online in zip(target_before, agent.target_critic.parameters(), agent.critic.parameters()):
            torch.testing.assert_close(new, old + agent.cfg.tau * (online.detach() - old))

    def test_terminal_target_is_scaled_reward(self):
        agent = small_agent(reward_scale=10.0, lr_critic=1e-2)
        batch = self.batch(terminated=1.0, rewards=2.0)
        for _ in range(500):
            agent.update_critic(batch)
        b = agent._tensors(batch)
        q = agent.critic(b["obs"], b["actions"])
        torch.testing.assert_close(q, torch.full_like(q, 20.0), atol=0.5, rtol=0)

    def test_actor_and_temperature_update(self):
        agent = small_agent()
        actor_before = [p.clone() for p in agent.actor.parameters()]
        temperature = agent.temperature.item()
        actor_loss, entropy = agent.update_actor(self.batch())
        self.assertTrue(np.isfinite(actor_loss) and np.isfinite(entropy))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())))
        self.assertNotEqual(agent.temperature.item(), temperature)
        self.assertEqual(agent.target_entropy, -7.0)

    def test_act_in_range(self):
        action = small_agent().act(np.zeros(109, dtype=np.float32))
        self.assertEqual(action.shape, (7,))
        self.assertTrue(np.all(np.abs(action) <= 1))


class RunRLPDTests(unittest.TestCase):
    def test_short_run(self):
        env_cfg = EnvConfig(max_steps=40)
        cfg = RLPDConfig(
            **SMALL, online_steps=60, learning_starts=20, utd=2, checkpoint_every=25
        )
        arrays, meta = synthetic_dataset(env_cfg)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "d0.npz"
            save_dataset(dataset, arrays, meta)
            run_dir = run_rlpd(cfg, env_cfg, dataset, Path(directory) / "run", log_training=True)
            steps = [load_checkpoint(p)[0]["online_step"] for p in list_checkpoints(run_dir)]
            self.assertEqual(steps, [0, 25, 50, 60])
            payload = load_checkpoint(list_checkpoints(run_dir)[-1])[0]
            self.assertIsNotNone(payload["critic"])
            # Updates start at learning_starts, utd critic updates per step
            self.assertEqual(payload["extra"]["grad_updates"], 2 * (60 - 20 + 1))
            self.assertEqual(payload["extra"]["episodes"], 1)
            self.assertEqual(len(list(run_dir.glob("events.out.tfevents.*"))), 1)
            self.assertTrue((run_dir / "config.json").exists())
            with self.assertRaises(ValueError):
                run_rlpd(cfg, env_cfg, dataset, run_dir)

    def test_refuses_mismatched_dataset(self):
        arrays, meta = synthetic_dataset(EnvConfig(noise_prob=0.0))
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "d0.npz"
            save_dataset(dataset, arrays, meta)
            with self.assertRaises(ValueError):
                run_rlpd(RLPDConfig(**SMALL), EnvConfig(), dataset, Path(directory) / "run")
