import tempfile
import unittest
from pathlib import Path

import numpy as np

from pupil_on_pole.checkpoints import list_checkpoints, load_checkpoint
from pupil_on_pole.config import DaggerConfig, EnvConfig
from pupil_on_pole.dagger import dagger_loop, run_dagger, train_bc
from pupil_on_pole.data import Normalizer, save_dataset
from pupil_on_pole.env import encode_action
from tests.fakes import FakeEnv, FakeExpert
from tests.fixtures import needs_expert, synthetic_dataset

SMALL = dict(hidden=(16, 16), batch_size=8, grad_steps=50)


class TrainBCTests(unittest.TestCase):
    def test_fits_constant_labels(self):
        arrays, _ = synthetic_dataset(EnvConfig(), n=64)
        target = np.tile(encode_action(FakeExpert.ACTION), (64, 1))
        normalizer = Normalizer.from_observations(arrays["obs"])
        _, short = train_bc(arrays["obs"], target, normalizer, DaggerConfig(**{**SMALL, "grad_steps": 100}), seed=0)
        _, long = train_bc(arrays["obs"], target, normalizer, DaggerConfig(**{**SMALL, "grad_steps": 1000}), seed=0)
        self.assertLess(long, short)

    def test_same_seed_same_policy(self):
        arrays, _ = synthetic_dataset(EnvConfig(), n=32)
        normalizer = Normalizer.from_observations(arrays["obs"])
        a, _ = train_bc(arrays["obs"], arrays["actions"], normalizer, DaggerConfig(**SMALL), seed=3)
        b, _ = train_bc(arrays["obs"], arrays["actions"], normalizer, DaggerConfig(**SMALL), seed=3)
        for x, y in zip(a.parameters(), b.parameters()):
            self.assertTrue((x == y).all())


class DaggerLoopTests(unittest.TestCase):
    def run_loop(self, online_steps, checkpoint_every, episode_steps=7, episodes_per_iter=2):
        cfg = DaggerConfig(**SMALL, online_steps=online_steps, checkpoint_every=checkpoint_every,
                           episodes_per_iter=episodes_per_iter)
        arrays, _ = synthetic_dataset(EnvConfig(), n=40)
        env, expert = FakeEnv(episode_steps=episode_steps), FakeExpert()
        saved = []

        def save(step, policy, counters):
            saved.append((step, dict(counters)))

        dagger_loop(env, expert, cfg, arrays, Normalizer.from_observations(arrays["obs"]), save)
        return saved, env, expert, cfg

    def test_checkpoint_steps_and_budget(self):
        saved, env, expert, _ = self.run_loop(online_steps=30, checkpoint_every=10)
        self.assertEqual([s for s, _ in saved], [0, 10, 20, 30])
        # One expert query per online step, the budget cuts the last episode
        self.assertEqual(expert.calls, 30)
        self.assertEqual(saved[-1][1]["expert_queries"], 30)
        self.assertEqual(len(env.resets), 5)

    def test_episode_seeds_and_expert_resets(self):
        _, env, expert, cfg = self.run_loop(online_steps=21, checkpoint_every=100)
        seeds = [seed for _, seed in env.resets]
        self.assertEqual(seeds, [cfg.seed * 1_000_000 + i for i in range(3)])
        self.assertEqual(expert.resets, 3)

    def test_retrains_from_scratch_each_iteration(self):
        saved, *_ = self.run_loop(online_steps=28, checkpoint_every=100, episodes_per_iter=2)
        # 28 steps is 2 iterations of 2 episodes, BC then one retraining between them
        final = saved[-1][1]
        self.assertEqual(final["iterations"], 2)
        # No extra training once the budget is spent
        self.assertEqual(final["grad_updates"], 2 * SMALL["grad_steps"])
        self.assertEqual([s for s, _ in saved], [0, 28])

    def test_checkpoints_hold_the_latest_trained_policy(self):
        saved, *_ = self.run_loop(online_steps=30, checkpoint_every=10)
        # Iterations end at 14 and 28 steps, D0 has 40 transitions
        self.assertEqual([c["trained_on_steps"] for _, c in saved], [0, 0, 14, 28])
        self.assertEqual([c["dataset_size"] for _, c in saved], [40, 40, 54, 68])


class RunDaggerTests(unittest.TestCase):
    def test_refuses_mismatched_dataset(self):
        arrays, meta = synthetic_dataset(EnvConfig(noise_prob=0.0))
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "d0.npz"
            save_dataset(dataset, arrays, meta)
            with self.assertRaises(ValueError):
                run_dagger(DaggerConfig(**SMALL), EnvConfig(), dataset, Path(directory) / "run")

    @needs_expert
    def test_short_run(self):
        env_cfg = EnvConfig(max_steps=40)
        cfg = DaggerConfig(**SMALL, online_steps=60, checkpoint_every=25, episodes_per_iter=1)
        arrays, meta = synthetic_dataset(env_cfg)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "d0.npz"
            save_dataset(dataset, arrays, meta)
            out = run_dagger(cfg, env_cfg, dataset, Path(directory) / "run", log_training=True)
            paths = list_checkpoints(out)
            self.assertEqual([load_checkpoint(p)[0]["online_step"] for p in paths], [0, 25, 50, 60])
            payload = load_checkpoint(paths[-1])[0]
            self.assertEqual(payload["method"], "dagger")
            self.assertEqual(payload["extra"]["expert_queries"], 60)
            self.assertIsNone(payload["critic"])
            self.assertEqual(len(list(out.glob("events.out.tfevents.*"))), 1)
            with self.assertRaises(ValueError):
                run_dagger(cfg, env_cfg, dataset, out)
