import tempfile
import unittest
from pathlib import Path

import numpy as np

from pupil_on_pole.checkpoints import list_checkpoints, save_checkpoint
from pupil_on_pole.config import DaggerConfig, EnvConfig, EvalConfig
from pupil_on_pole.data import Normalizer
from pupil_on_pole.evaluation import evaluate_checkpoints, evaluate_expert, evaluate_policy, summarize
from pupil_on_pole.networks import DeterministicPolicy
from pupil_on_pole.env import ACTION_DIM
from tests.fakes import FakeEnv, FakeExpert


def episode(finished, steps, brake=0, fire_presses=0):
    return {"finished": finished, "stuck": not finished and steps < 1500, "steps": steps,
            "progress": 1.0 if finished else 0.5, "return": 1.0, "fire_presses": fire_presses,
            "mean_speed": 10.0, "mean_center_distance": 1.0,
            "action_steps": {"brake": brake, "drift": 0, "nitro": 0, "rescue": 0, "fire": fire_presses}}


class SummarizeTests(unittest.TestCase):
    def test_unfinished_counts_as_cap(self):
        # A stuck episode ends early but counts as the 1,500-step cap
        summary = summarize([episode(True, 600, brake=60), episode(False, 150)], max_steps=1500)
        self.assertEqual(summary["finish_rate"], 0.5)
        self.assertEqual(summary["lap_steps"], 600)
        self.assertAlmostEqual(summary["lap_time_s"], (600 + 1500) / 2 * 0.1)
        self.assertAlmostEqual(summary["action_use"]["brake"], 60 / 750)
        self.assertEqual(summary["stuck_rate"], 0.5)
        self.assertAlmostEqual(summary["best_lap_time_s"], 60.0)
        self.assertAlmostEqual(summary["lap_time_std_s"], 45.0)

    def test_fire_presses(self):
        summary = summarize([episode(True, 600, fire_presses=4), episode(True, 600)], max_steps=1500)
        self.assertEqual(summary["fire_presses"], 2.0)

    def test_no_finished_lap(self):
        summary = summarize([episode(False, 1500)], max_steps=1500)
        self.assertIsNone(summary["lap_steps"])
        self.assertEqual(summary["finish_rate"], 0.0)


class EvaluateTests(unittest.TestCase):
    def test_expert_on_fixed_seeds(self):
        env, expert = FakeEnv(episode_steps=5), FakeExpert()
        result = evaluate_expert(env, expert, "lighthouse", n_episodes=3, first_seed=100)
        self.assertEqual(env.resets, [("lighthouse", 100), ("lighthouse", 101), ("lighthouse", 102)])
        self.assertEqual(expert.resets, 3)
        self.assertEqual(result["summary"]["finish_rate"], 1.0)
        self.assertAlmostEqual(result["summary"]["lap_time_s"], 0.5)
        self.assertEqual(result["summary"]["action_use"]["brake"], 1.0)
        self.assertAlmostEqual(result["summary"]["mean_speed"], 5.0)
        self.assertAlmostEqual(result["summary"]["mean_center_distance"], 2.0)
        self.assertEqual(result["summary"]["fire_presses"], 0.0)

    def test_fire_counts_presses_not_steps(self):
        # Fire on every second step of a 6-step lap is 3 presses
        result = evaluate_expert(FakeEnv(episode_steps=6), FakeExpert(fire_every=2), "scotland", 1, 0)
        self.assertEqual(result["episodes"][0]["fire_presses"], 3)
        self.assertAlmostEqual(result["summary"]["action_use"]["fire"], 0.5)

    def test_policy_same_seeds_for_every_policy(self):
        env = FakeEnv(episode_steps=4)
        policy = DeterministicPolicy(env.observation_dim, ACTION_DIM, (8,))
        normalizer = Normalizer(np.zeros(env.observation_dim), np.ones(env.observation_dim))
        evaluate_policy(env, policy, normalizer, "scotland", 2, 100_000)
        evaluate_policy(env, policy, normalizer, "scotland", 2, 100_000)
        self.assertEqual([s for _, s in env.resets], [100_000, 100_001] * 2)

    def test_checkpoints_curve_and_held_out(self):
        env = FakeEnv(episode_steps=3)
        cfg = DaggerConfig(hidden=(8,))
        normalizer = Normalizer(np.zeros(env.observation_dim), np.ones(env.observation_dim))
        eval_cfg = EvalConfig(n_episodes=2)
        with tempfile.TemporaryDirectory() as directory:
            for step in (0, 10, 20):
                policy = DeterministicPolicy(env.observation_dim, ACTION_DIM, cfg.hidden)
                save_checkpoint(directory, "dagger", cfg, EnvConfig(), step, policy, normalizer, {"episodes": step})
            result = evaluate_checkpoints(env, list_checkpoints(Path(directory)), eval_cfg)
        self.assertEqual([p["online_step"] for p in result["curve"]], [0, 10, 20])
        self.assertEqual(sorted(result["held_out"]), sorted(eval_cfg.held_out_tracks))
        self.assertEqual(result["method"], "dagger")
        self.assertEqual(env.track, "scotland")
