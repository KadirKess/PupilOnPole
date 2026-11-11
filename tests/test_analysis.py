import unittest

import pandas as pd
from scipy import stats

import numpy as np

from pupil_on_pole.analysis import bootstrap_ci, efficiency, endpoint, summarize_results, welch


def curves():
    rows = []
    for method, base in (("dagger", 70.0), ("rlpd", 60.0)):
        for seed in (1, 2, 3):
            for step in (0, 5, 10):
                rows.append({"method": method, "seed": seed, "online_step": step,
                             "lap_time_s": base + seed - step / 10, "finish_rate": 1.0, "fire_presses": 0.0})
    return pd.DataFrame(rows)


def held_out():
    # Every run is twice as slow as on the training track, team2 is 1.5 times slower there
    final = curves()[curves()["online_step"] == 10]
    return final.assign(track="lighthouse", lap_time_s=final["lap_time_s"] * 2).drop(columns="online_step")


def expert():
    return pd.DataFrame([
        {"method": "team2", "track": "scotland", "lap_time_s": 63.0, "finish_rate": 1.0, "fire_presses": 0.0},
        {"method": "team2", "track": "lighthouse", "lap_time_s": 94.5, "finish_rate": 1.0, "fire_presses": 0.0},
    ])


class StatsTests(unittest.TestCase):
    def test_bootstrap_constant(self):
        self.assertEqual(bootstrap_ci([3.0, 3.0, 3.0]), (3.0, 3.0))

    def test_bootstrap_contains_mean(self):
        low, high = bootstrap_ci([1.0, 2.0, 6.0])
        self.assertLessEqual(low, 3.0)
        self.assertGreaterEqual(high, 3.0)
        self.assertGreaterEqual(low, 1.0)
        self.assertLessEqual(high, 6.0)

    def test_welch_matches_scipy(self):
        a, b = [1.0, 2.0, 4.0], [3.0, 5.0, 9.0]
        expected = stats.ttest_ind(a, b, equal_var=False)
        t, p = welch(a, b)
        self.assertAlmostEqual(t, expected.statistic)
        self.assertAlmostEqual(p, expected.pvalue)


class SummaryTests(unittest.TestCase):
    def test_endpoint_defaults_to_last_common_step(self):
        df = curves()
        df = df[~((df["method"] == "rlpd") & (df["seed"] == 3) & (df["online_step"] == 10))]
        _, budget = endpoint(df)
        self.assertEqual(budget, 5)

    def summary(self):
        results = {
            "manifest": {"env_config": {"track": "scotland"}},
            "curves": curves(),
            "held_out": held_out(),
            "expert": expert(),
        }
        return summarize_results(results)

    def test_summary_against_expert(self):
        summary = self.summary()
        self.assertEqual(summary["budget"], 10)
        main = summary["main"][summary["main"]["metric"] == "lap_time_s"].set_index("method")
        self.assertAlmostEqual(main.loc["rlpd", "mean"], 61.0)
        self.assertAlmostEqual(main.loc["dagger", "diff_expert"], 71.0 - 63.0)
        welch_rows = summary["welch"].set_index("metric")
        self.assertLess(welch_rows.loc["lap_time_s", "p"], 0.05)
        self.assertIn("fire_presses", welch_rows.index)

    def test_generalization_ratio_and_gap(self):
        runs = self.summary()["generalization_runs"]
        rlpd = runs[(runs["method"] == "rlpd") & (runs["seed"] == 1)].set_index("track")
        # Seed 1 of RLPD ends at 60 s on scotland and 120 s on lighthouse
        self.assertAlmostEqual(rlpd.loc["scotland", "vs_expert"], 60.0 / 63.0)
        self.assertAlmostEqual(rlpd.loc["lighthouse", "vs_expert"], 120.0 / 94.5)
        self.assertAlmostEqual(rlpd.loc["lighthouse", "gap"], 120.0 / 94.5 - 60.0 / 63.0)
        self.assertAlmostEqual(rlpd.loc["scotland", "gap"], 0.0)


class EfficiencyTests(unittest.TestCase):
    def test_steps_to_expert_and_mean(self):
        df = pd.DataFrame({"method": "rlpd", "seed": 1, "online_step": [0, 5, 10], "lap_time_s": [100.0, 64.0, 60.0]})
        row = efficiency(df, reference=62.0).iloc[0]
        # 64 is within 5 % of 62, the mean over training is the trapezoid area over 10 steps
        self.assertEqual(row["steps_to_expert"], 5.0)
        self.assertAlmostEqual(row["mean_over_training"], ((100 + 64) / 2 * 5 + (64 + 60) / 2 * 5) / 10)

    def test_never_reaches_expert(self):
        df = pd.DataFrame({"method": "dagger", "seed": 1, "online_step": [0, 5], "lap_time_s": [100.0, 90.0]})
        self.assertTrue(np.isnan(efficiency(df, reference=60.0).iloc[0]["steps_to_expert"]))
