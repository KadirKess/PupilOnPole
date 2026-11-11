import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

PYSTK_SRC_VARIABLE = "PYSTK_PROJECT_SRC"


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def pystk_project_src(path=None):
    """Return the src folder of the pystk_project clone, from the argument or PYSTK_PROJECT_SRC"""
    path = path or os.environ.get(PYSTK_SRC_VARIABLE)
    if not path:
        raise RuntimeError(
            f"Set {PYSTK_SRC_VARIABLE} to the src folder of a pystk_project clone"
        )
    path = Path(path).expanduser().resolve()
    if not (path / "agents" / "kart_agent.py").exists():
        raise RuntimeError(f"{path} does not look like pystk_project/src")
    return path


class Expert:
    """Hand-coded team agent from pystk_project, queried on raw pystk observations"""

    def __init__(self, env, team=2, pystk_src=None):
        self.env = env
        self.team = int(team)
        self.src = pystk_project_src(pystk_src)
        # Appended so the installed pystk2_gymnasium wins over the copy in src
        if str(self.src) not in sys.path:
            sys.path.append(str(self.src))
        # Team2 reads its config with a path relative to src/main when imported
        with _working_directory(self.src / "main"):
            module = importlib.import_module(f"agents.team{self.team}.agent{self.team}")
        self._agent_class = getattr(module, f"Agent{self.team}")
        self.agent = None

    def reset(self):
        """Build a fresh agent so its internal counters start from zero, call after env.reset"""
        with _working_directory(self.src / "main"):
            self.agent = self._agent_class(self.env.race, path_lookahead=3)

    def act(self, raw_obs):
        if self.agent is None:
            self.reset()
        action = dict(self.agent.choose_action(raw_obs))
        # Items aimed at opponents are out of scope, so the expert never fires
        action["fire"] = False
        action.setdefault("rescue", False)
        return action
