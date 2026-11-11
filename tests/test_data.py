import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pupil_on_pole.config import DatasetConfig, EnvConfig
from pupil_on_pole.data import (
    FIELDS,
    Normalizer,
    ReplayBuffer,
    check_dataset,
    episode_summary,
    load_dataset,
    save_dataset,
)
from tests.fixtures import needs_expert, synthetic_dataset


class DatasetFileTests(unittest.TestCase):
    def test_save_load_round_trip(self):
        arrays, meta = synthetic_dataset(EnvConfig())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d0.npz"
            save_dataset(path, arrays, meta)
            self.assertTrue(path.with_suffix(".json").exists())
            loaded, loaded_meta = load_dataset(path)
        self.assertEqual(loaded_meta, meta)
        for name in FIELDS:
            np.testing.assert_array_equal(loaded[name], arrays[name])

    def test_check_dataset(self):
        _, meta = synthetic_dataset(EnvConfig())
        check_dataset(meta, EnvConfig())
        with self.assertRaises(ValueError):
            check_dataset(meta, EnvConfig(noise_prob=0.0))

    def test_episode_summary(self):
        arrays, _ = synthetic_dataset(EnvConfig(), n=10)
        arrays["episode"][5:] = 1
        arrays["terminated"][:] = False
        arrays["terminated"][9] = True
        summary = episode_summary(arrays)
        self.assertEqual([s["steps"] for s in summary], [5, 5])
        self.assertEqual([s["finished"] for s in summary], [False, True])


class NormalizerTests(unittest.TestCase):
    def test_standardizes_and_keeps_constant_columns(self):
        rng = np.random.default_rng(0)
        obs = rng.normal(3.0, 2.0, size=(1000, 4))
        obs[:, 2] = 5.0
        normalizer = Normalizer.from_observations(obs)
        out = normalizer(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        np.testing.assert_allclose(out[:, [0, 1, 3]].mean(0), 0.0, atol=1e-5)
        np.testing.assert_allclose(out[:, [0, 1, 3]].std(0), 1.0, atol=1e-4)
        np.testing.assert_array_equal(out[:, 2], 0.0)

    def test_clips_and_round_trips(self):
        normalizer = Normalizer(np.zeros(2), np.ones(2))
        out = normalizer(torch.tensor([[100.0, -100.0]]))
        torch.testing.assert_close(out, torch.tensor([[10.0, -10.0]]))
        copy = Normalizer.from_state_dict(normalizer.state_dict())
        torch.testing.assert_close(copy.mean, normalizer.mean)
        torch.testing.assert_close(copy.std, normalizer.std)


class ReplayBufferTests(unittest.TestCase):
    def test_wraps_around(self):
        buffer = ReplayBuffer(2, 7, capacity=3)
        for i in range(5):
            buffer.add(np.full(2, i), np.zeros(7), float(i), np.full(2, i + 1), i == 4)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(sorted(buffer.rewards.tolist()), [2.0, 3.0, 4.0])
        batch = buffer.sample(50, np.random.default_rng(0))
        self.assertEqual(batch["obs"].shape, (50, 2))
        np.testing.assert_array_equal(batch["terminated"], batch["rewards"] == 4.0)

    def test_from_dataset(self):
        arrays, _ = synthetic_dataset(EnvConfig(), n=20)
        buffer = ReplayBuffer.from_dataset(arrays)
        self.assertEqual(len(buffer), 20)
        np.testing.assert_array_equal(buffer.terminated, arrays["terminated"].astype(np.float32))

    def test_empty_buffer_refuses_to_sample(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(2, 7, 3).sample(1, np.random.default_rng(0))


@needs_expert
class GenerateDatasetTests(unittest.TestCase):
    def test_short_episodes(self):
        from pupil_on_pole.data import generate_dataset
        from pupil_on_pole.env import SoloKartEnv
        from pupil_on_pole.expert import Expert

        cfg = replace(EnvConfig(), max_steps=40)
        with SoloKartEnv(cfg) as env:
            arrays = generate_dataset(env, Expert(env), DatasetConfig(n_episodes=2))
        self.assertEqual(set(arrays), set(FIELDS))
        self.assertEqual(len(arrays["obs"]), 80)
        self.assertEqual(arrays["obs"].shape[1], 109)
        self.assertEqual(arrays["truncated"].sum(), 2)
        self.assertFalse(arrays["terminated"].any())
        self.assertTrue(np.all(np.abs(arrays["actions"]) <= 1))
        self.assertTrue(np.all(arrays["actions"][:, 6] == -1.0))
        np.testing.assert_array_equal(arrays["obs"][1:40], arrays["next_obs"][:39])
