import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pupil_on_pole.plotting import plot_action_use, plot_curve, plot_fire, plot_generalization, plot_held_out

USE = {"use_brake": 0.1, "use_drift": 0.0, "use_nitro": 0.2, "use_fire": 0.0}


class PlottingTests(unittest.TestCase):
    def test_figures_are_written(self):
        curves = pd.DataFrame([
            {"method": m, "seed": s, "online_step": t, "lap_time_s": 60 + s + t, "finish_rate": 1.0, **USE}
            for m in ("dagger", "rlpd") for s in (1, 2) for t in (0, 5)
        ])
        held_out = pd.DataFrame([
            {"method": m, "seed": s, "track": t, "lap_time_s": 50.0 + s}
            for m in ("dagger", "rlpd") for s in (1, 2) for t in ("lighthouse", "fortmagma")
        ])
        expert = pd.DataFrame([
            {"method": "team2", "track": t, "lap_time_s": 55.0, "finish_rate": 1.0, **USE}
            for t in ("scotland", "lighthouse", "fortmagma")
        ])
        expert = expert.assign(fire_presses=0.0)
        runs = pd.DataFrame([
            {"method": m, "seed": s, "track": t, "vs_expert": 1.0 + s / 10, "fire_presses": float(m == "rlpd")}
            for m in ("dagger", "rlpd") for s in (1, 2) for t in ("scotland", "lighthouse", "fortmagma")
        ])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            paths = [
                *plot_curve(curves, expert, "lap_time_s", "Lap time (s)", "scotland", out / "lap"),
                *plot_held_out(held_out, expert, "lap_time_s", "Lap time (s)", out / "held"),
                *plot_action_use(curves[curves["online_step"] == 5], expert, "scotland", out / "actions"),
                *plot_generalization(runs, "scotland", out / "generalization"),
                *plot_fire(runs, expert, "scotland", out / "fire"),
            ]
            for path in paths:
                self.assertTrue(path.exists() and path.stat().st_size > 0, path)
            self.assertEqual({p.suffix for p in paths}, {".png", ".pdf"})
