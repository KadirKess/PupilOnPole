import tempfile
import unittest

from pupil_on_pole.config import DaggerConfig, DatasetConfig, EnvConfig, EvalConfig, RLPDConfig
from pupil_on_pole.experiments import (
    create_experiment,
    evaluate_experiment,
    generate_dataset,
    load_experiment,
    load_manifest,
    plot_experiment,
    train_experiment,
)
from tests.fixtures import needs_expert


class ManifestTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            out = create_experiment("unit", seeds=(1, 2), dagger_cfg=DaggerConfig(hidden=(8, 8)), root=directory)
            manifest, configs = load_manifest(out)
            self.assertEqual(manifest["seeds"], [1, 2])
            self.assertEqual(configs["dagger_config"].hidden, (8, 8))
            self.assertEqual(configs["eval_config"], EvalConfig())

    def test_refuses_unequal_budgets(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            create_experiment("unit", dagger_cfg=DaggerConfig(online_steps=10), root=directory)


@needs_expert
class SmokeTests(unittest.TestCase):
    def test_every_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            out = create_experiment(
                "smoke",
                seeds=(1,),
                env_cfg=EnvConfig(max_steps=60),
                dataset_cfg=DatasetConfig(n_episodes=2),
                dagger_cfg=DaggerConfig(online_steps=80, checkpoint_every=40, grad_steps=20, hidden=(16,),
                                        episodes_per_iter=1),
                rlpd_cfg=RLPDConfig(online_steps=80, checkpoint_every=40, learning_starts=20, utd=1, hidden=(16,)),
                eval_cfg=EvalConfig(n_episodes=1, held_out_tracks=("lighthouse",)),
                workers=2,
                torch_threads=1,
                root=directory,
            )
            generate_dataset(out)
            train_experiment(out)
            evaluate_experiment(out)
            results = load_experiment(out)
            self.assertEqual(results["summary"]["budget"], 80)
            figures = plot_experiment(results, out)
            self.assertIn("held_out", figures)
            self.assertTrue((out / "tables" / "main.csv").exists())
