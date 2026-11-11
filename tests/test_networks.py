import tempfile
import unittest

import numpy as np
import torch

from pupil_on_pole.checkpoints import list_checkpoints, load_checkpoint, save_checkpoint
from pupil_on_pole.config import DaggerConfig, EnvConfig, RLPDConfig
from pupil_on_pole.data import Normalizer
from pupil_on_pole.networks import (
    DeterministicPolicy,
    EnsembleCritic,
    GaussianPolicy,
    policy_action,
)


class PolicyTests(unittest.TestCase):
    def test_deterministic_policy_range(self):
        policy = DeterministicPolicy(10, 7, (16,))
        out = policy(100 * torch.randn(32, 10))
        self.assertEqual(out.shape, (32, 7))
        self.assertTrue(torch.all(out.abs() <= 1))

    def test_gaussian_log_prob_matches_torch(self):
        torch.manual_seed(0)
        policy = GaussianPolicy(10, 7, (16,))
        obs = torch.randn(64, 10)
        torch.manual_seed(1)
        action, log_prob = policy.sample(obs)
        self.assertEqual(action.shape, (64, 7))
        self.assertEqual(log_prob.shape, (64,))
        mean, log_std = policy._mean_log_std(obs)
        torch.manual_seed(1)
        pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        base = torch.distributions.Normal(mean, log_std.exp())
        expected = base.log_prob(pre_tanh).sum(-1) - torch.log(1 - torch.tanh(pre_tanh) ** 2).sum(-1)
        torch.testing.assert_close(log_prob, expected, atol=1e-4, rtol=1e-4)

    def test_policy_action_uses_mean(self):
        policy = GaussianPolicy(10, 7, (16,))
        normalizer = Normalizer(np.zeros(10), np.ones(10))
        obs = np.random.default_rng(0).normal(size=10).astype(np.float32)
        a = policy_action(policy, normalizer, obs)
        b = policy_action(policy, normalizer, obs)
        self.assertEqual(a.shape, (7,))
        np.testing.assert_array_equal(a, b)


class EnsembleCriticTests(unittest.TestCase):
    def test_shape_and_independent_members(self):
        critic = EnsembleCritic(10, 7, (16, 16), n_critics=3)
        q = critic(torch.randn(5, 10), torch.randn(5, 7))
        self.assertEqual(q.shape, (3, 5))
        self.assertFalse(torch.allclose(q[0], q[1]))

    def test_member_gradients_are_separate(self):
        critic = EnsembleCritic(10, 7, (16,), n_critics=3)
        q = critic(torch.randn(5, 10), torch.randn(5, 7))
        q[0].sum().backward()
        weight = critic.model[0].weight.grad
        self.assertGreater(weight[0].abs().sum().item(), 0)
        self.assertEqual(weight[1:].abs().sum().item(), 0)

    def test_layer_norm_flag(self):
        with_norm = EnsembleCritic(10, 7, (16, 16), 2, layer_norm=True)
        without = EnsembleCritic(10, 7, (16, 16), 2, layer_norm=False)
        self.assertEqual(len(with_norm.model), len(without.model) + 2)


class CheckpointTests(unittest.TestCase):
    def test_round_trip(self):
        normalizer = Normalizer(np.arange(109.0), np.full(109, 2.0))
        obs = torch.randn(4, 109)
        for method, Config, Policy in (
            ("dagger", DaggerConfig, DeterministicPolicy),
            ("rlpd", RLPDConfig, GaussianPolicy),
        ):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                cfg = Config(hidden=(8, 8))
                policy = Policy(109, 7, cfg.hidden)
                critic = EnsembleCritic(109, 7, cfg.hidden, 3) if method == "rlpd" else None
                for step in (0, 5000):
                    save_checkpoint(directory, method, cfg, EnvConfig(), step, policy, normalizer, {"episodes": 3}, critic)
                paths = list_checkpoints(directory)
                self.assertEqual([p.name for p in paths], ["step-000000000.pt", "step-000005000.pt"])
                payload, loaded_cfg, env_cfg, loaded, loaded_norm = load_checkpoint(paths[1])
                self.assertEqual(loaded_cfg, cfg)
                self.assertEqual(env_cfg, EnvConfig())
                self.assertEqual(payload["online_step"], 5000)
                self.assertEqual(payload["extra"], {"episodes": 3})
                self.assertEqual(payload["critic"] is None, critic is None)
                x = normalizer(obs)
                if method == "rlpd":
                    torch.testing.assert_close(loaded.deterministic(loaded_norm(obs)), policy.deterministic(x))
                else:
                    torch.testing.assert_close(loaded(loaded_norm(obs)), policy(x))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            save_checkpoint("unused", "bc", DaggerConfig(), EnvConfig(), 0, None, None)
