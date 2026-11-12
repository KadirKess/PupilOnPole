import unittest
from dataclasses import replace

import numpy as np

from pupil_on_pole.config import EnvConfig
from pupil_on_pole.env import (
    ACTION_DIM,
    SoloKartEnv,
    decode_action,
    encode_action,
    feature_names,
    progress_reward,
    step_distance,
)
from tests.fixtures import needs_expert


class ActionTests(unittest.TestCase):
    def test_round_trip(self):
        action = {
            "acceleration": 0.25,
            "steer": -0.5,
            "brake": True,
            "drift": False,
            "nitro": True,
            "rescue": False,
            "fire": False,
        }
        vector = encode_action(action)
        self.assertEqual(vector.shape, (ACTION_DIM,))
        self.assertTrue(np.all(np.abs(vector) <= 1))
        self.assertEqual(decode_action(vector), action)

    def test_missing_keys_are_off(self):
        decoded = decode_action(encode_action({}))
        self.assertEqual(decoded["acceleration"], 0.0)
        self.assertFalse(any(decoded[name] for name in ("brake", "drift", "nitro", "rescue", "fire")))

    def test_decode_clips_and_thresholds(self):
        decoded = decode_action(np.array([3.0, -2.0, 0.1, -0.1, 0.0, 1.0, -1.0]))
        self.assertEqual(decoded["acceleration"], 1.0)
        self.assertEqual(decoded["steer"], -1.0)
        self.assertTrue(decoded["brake"])
        self.assertFalse(decoded["drift"])
        self.assertFalse(decoded["nitro"])

    def test_feature_count(self):
        self.assertEqual(len(feature_names(5, 5)), 109)

class RewardTests(unittest.TestCase):
    def test_track_jumps_are_ignored(self):
        # Jump sizes seen in D0 on scotland, none of them is the track length
        steps = [(0.0, 5.0), (5.0, 1173.5), (1173.5, 1176.0), (1176.0, 23.0)]
        deltas = [step_distance(a, b) for a, b in steps]
        self.assertEqual(deltas, [5.0, 0.0, 2.5, 0.0])

    def test_reversing_counts_as_negative_progress(self):
        self.assertEqual(step_distance(100.0, 97.5), -2.5)

    def test_reward_matches_pystk2_without_wrap(self):
        self.assertAlmostEqual(progress_reward(2.0, False), 0.1)
        self.assertAlmostEqual(progress_reward(2.0, True), 10.1)


class SoloKartEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = SoloKartEnv(EnvConfig())

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def use(self, **changes):
        self.env.cfg = replace(EnvConfig(), **changes)

    def run_episode(self, action, seed=0, limit=2000):
        obs, _ = self.env.reset(seed)
        flags = []
        for _ in range(limit):
            obs, reward, terminated, truncated, info = self.env.step(action)
            flags.append(info["perturbed"])
            if terminated or truncated:
                return terminated, truncated, info, flags
        self.fail("episode did not end")

    def test_observation_shape(self):
        self.use()
        obs, info = self.env.reset(0)
        self.assertEqual(obs.shape, (self.env.observation_dim,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIn("raw_obs", info)

    def test_only_one_race_per_process(self):
        with self.assertRaises(RuntimeError):
            SoloKartEnv(EnvConfig())

    def test_timeout_truncates(self):
        self.use(max_steps=30)
        full_throttle = encode_action({"acceleration": 1.0})
        terminated, truncated, info, _ = self.run_episode(full_throttle)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["stuck"])
        self.assertEqual(info["steps"], 30)

    def test_stuck_truncates(self):
        self.use(stuck_steps=20, noise_prob=0.0)
        terminated, truncated, info, _ = self.run_episode(encode_action({}))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["stuck"])
        self.assertEqual(info["steps"], 20)

    def test_noise_depends_only_on_seed(self):
        self.use(max_steps=200, noise_prob=0.1)
        slow = encode_action({"acceleration": 0.3})
        fast = encode_action({"acceleration": 1.0})
        first = self.run_episode(slow, seed=7)[3]
        second = self.run_episode(fast, seed=7)[3]
        other = self.run_episode(slow, seed=8)[3]
        self.assertTrue(any(first))
        n = min(len(first), len(second))
        self.assertEqual(first[:n], second[:n])
        self.assertNotEqual(first, other)

    def test_perturbation_replaces_steer(self):
        self.use(max_steps=200, noise_prob=0.1)
        self.env.reset(7)
        straight = encode_action({"acceleration": 1.0, "steer": 0.0})
        for _ in range(200):
            *_, terminated, truncated, info = self.env.step(straight)
            if info["perturbed"]:
                self.assertNotEqual(info["executed"]["steer"], 0.0)
                return
        self.fail("no perturbation in 200 steps")

    def test_reversing_earns_nothing(self):
            self.use(noise_prob=0.0, max_steps=100)
            self.env.reset(0)
            forward, backward = encode_action({"acceleration": 1.0}), encode_action({"brake": True})
            rewards = []
            for action in [forward] * 20 + [backward] * 80:
                _, reward, terminated, truncated, info = self.env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break
            self.assertLess(max(rewards), 1.0)
            self.assertLess(info["progress"], 0.5)

    def test_rescue_is_ignored(self):
        self.use(noise_prob=0.0, max_steps=60)
        self.env.reset(0)
        for _ in range(60):
            *_, info = self.env.step(encode_action({"rescue": True}))
            self.assertFalse(info["executed"]["rescue"])
        self.assertLess(info["distance"], 50.0)

    @needs_expert
    def test_expert_clean_lap(self):
        from pupil_on_pole.expert import Expert

        self.use(noise_prob=0.0)
        expert = Expert(self.env)
        _, info = self.env.reset(0)
        expert.reset()
        for _ in range(self.env.cfg.max_steps):
            action = expert.act(info["raw_obs"])
            self.assertFalse(action["fire"])
            _, _, terminated, truncated, info = self.env.step(encode_action(action))
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertEqual(info["steps"], 543)
