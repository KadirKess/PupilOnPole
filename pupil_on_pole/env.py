import numpy as np

from pupil_on_pole.config import EnvConfig

STEP_SECONDS = 0.1
ACTION_NAMES = ("acceleration", "steer", "brake", "drift", "nitro", "rescue", "fire")
BINARY_ACTIONS = ACTION_NAMES[2:]
ACTION_DIM = len(ACTION_NAMES)
N_ITEM_TYPES = 7

KART_FEATURES = (
    *(f"velocity_{axis}" for axis in "xyz"),
    *(f"front_{axis}" for axis in "xyz"),
    *(f"center_path_{axis}" for axis in "xyz"),
    "center_path_distance",
    "max_steer_angle",
    "energy",
    "skeed_factor",
    "jumping",
)


def _scalar(value):
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])

# 10 m in one 0.1 s step is 360 km/h, no kart gets there
MAX_STEP_DISTANCE = 10.0


def step_distance(previous, current):
    """Distance driven in one step, 0 when STK jumps to another track sector"""
    delta = float(current) - float(previous)
    return 0.0 if abs(delta) > MAX_STEP_DISTANCE else delta


def progress_reward(delta, finished):
    """pystk2's reward for a single kart, computed on the jump-free distance"""
    return delta / 10.0 - 0.1 + (10.0 if finished else 0.0)



def encode_action(action):
    """Map a pystk action dict to a float32 vector in [-1, 1], binary actions as +1 or -1"""
    vector = np.empty(ACTION_DIM, dtype=np.float32)
    vector[0] = 2.0 * np.clip(_scalar(action.get("acceleration", 0.0)), 0.0, 1.0) - 1.0
    vector[1] = np.clip(_scalar(action.get("steer", 0.0)), -1.0, 1.0)
    for index, name in enumerate(BINARY_ACTIONS, start=2):
        vector[index] = 1.0 if bool(action.get(name, False)) else -1.0
    return vector


def decode_action(vector):
    """Map a policy output in [-1, 1] to a pystk action dict, binary actions on above 0"""
    vector = np.clip(np.asarray(vector, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"Expected {ACTION_DIM} action values, got {vector.shape}")
    action = {
        "acceleration": float((vector[0] + 1.0) / 2.0),
        "steer": float(vector[1]),
    }
    for index, name in enumerate(BINARY_ACTIONS, start=2):
        action[name] = bool(vector[index] > 0.0)
    return action


def feature_names(n_paths, n_items):
    names = list(KART_FEATURES)
    for i in range(n_paths):
        names += [f"path{i}_start_{axis}" for axis in "xyz"]
        names += [f"path{i}_end_{axis}" for axis in "xyz"]
        names += [f"path{i}_width", f"path{i}_valid"]
    for i in range(n_items):
        names += [f"item{i}_{axis}" for axis in "xyz"]
        names += [f"item{i}_valid"]
        names += [f"item{i}_type{t}" for t in range(N_ITEM_TYPES)]
    return names


def featurize(raw, n_paths, n_items):
    """Turn the variable-size pystk observation into a fixed vector in the kart frame"""
    kart = np.concatenate(
        [
            np.asarray(raw["velocity"], dtype=np.float32).reshape(3),
            np.asarray(raw["front"], dtype=np.float32).reshape(3),
            np.asarray(raw["center_path"], dtype=np.float32).reshape(3),
            [
                _scalar(raw["center_path_distance"]),
                _scalar(raw["max_steer_angle"]),
                _scalar(raw["energy"]),
                _scalar(raw["skeed_factor"]),
                float(raw["jumping"]),
            ],
        ]
    )
    paths = np.zeros((n_paths, 8), dtype=np.float32)
    starts, ends, widths = raw["paths_start"], raw["paths_end"], raw["paths_width"]
    for i in range(min(n_paths, len(starts))):
        paths[i, 0:3] = starts[i]
        paths[i, 3:6] = ends[i]
        paths[i, 6] = _scalar(widths[i])
        paths[i, 7] = 1.0
    items = np.zeros((n_items, 4 + N_ITEM_TYPES), dtype=np.float32)
    positions, types = raw["items_position"], raw["items_type"]
    # pystk already sorts items by distance to the kart
    for i in range(min(n_items, len(positions))):
        items[i, 0:3] = positions[i]
        items[i, 3] = 1.0
        items[i, 4 + int(types[i])] = 1.0
    return np.concatenate([kart, paths.ravel(), items.ravel()]).astype(np.float32)


_live_env = None


class SoloKartEnv:
    """Single kart, single lap, with steering perturbations and stuck truncation

    Only one SuperTuxKart race can run per process, so a second env can only be
    created after the first one is closed.
    """

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        global _live_env
        if _live_env is not None:
            raise RuntimeError("Close the existing SoloKartEnv first, one race per process")
        if cfg.reward_version != 2:
            raise ValueError("Only reward_version 2 is implemented, rerun with the current EnvConfig")
        from pystk2_gymnasium import AgentSpec
        from pystk2_gymnasium.envs import STKRaceEnv

        self.cfg = cfg
        self.race = STKRaceEnv(
            agent=AgentSpec(rank_start=0, use_ai=False),
            track=cfg.track,
            num_kart=1,
            laps=1,
            use_subprocess=False,
        )
        self.feature_names = feature_names(cfg.n_paths, cfg.n_items)
        self.observation_dim = len(self.feature_names)
        self.action_dim = ACTION_DIM
        self._closed = False
        _live_env = self

    @property
    def track(self):
        return self.race.default_track

    def set_track(self, track):
        """Use another track from the next reset on"""
        self.race.default_track = track

    @property
    def track_length(self):
        return float(self.race.path_cache.track_length)

    def reset(self, seed):
        raw, _ = self.race.reset(seed=int(seed))
        # The noise depends only on the episode seed, so every driver sees the same perturbation draws
        self._rng = np.random.default_rng(int(seed))
        self._noise_left = 0
        self._noise_steer = 0.0
        self._steps = 0
        world = self.race.world
        self._stk_distance = max(0.0, float(world.karts[self.race.kart_ix].overall_distance))
        self._distance = 0.0
        self._best_distance = 0.0
        self._last_progress = 0
        return self._features(raw), {"raw_obs": raw, "steps": 0}

    def step(self, action):
        """Step with a policy vector in [-1, 1]^7"""
        return self.step_dict(decode_action(action))

    def step_dict(self, action):
        """Step with a pystk action dict, missing keys count as off"""
        executed = {
            "acceleration": float(np.clip(_scalar(action.get("acceleration", 0.0)), 0.0, 1.0)),
            "steer": float(np.clip(_scalar(action.get("steer", 0.0)), -1.0, 1.0)),
        }
        for name in BINARY_ACTIONS:
            executed[name] = bool(action.get(name, False))
        # The expert never rescues, and a rescue near the start pays a full lap, so no driver gets to use it
        executed["rescue"] = False

        if self._noise_left == 0 and self._rng.random() < self.cfg.noise_prob:
            self._noise_left = self.cfg.noise_steps
            self._noise_steer = float(self._rng.uniform(-1.0, 1.0))
        perturbed = self._noise_left > 0
        if perturbed:
            self._noise_left -= 1
            executed["steer"] = self._noise_steer
        raw, stk_reward, terminated, _, info = self.race.step(executed)
        self._steps += 1
        terminated=bool(terminated)
        delta = step_distance(self._stk_distance, info["distance"])
        # Follow STK even after a jump, so later steps and the jump back are measured correctly
        self._stk_distance = float(info["distance"])
        self._distance += delta
        distance = self._distance
        if distance >= self._best_distance + self.cfg.stuck_min_progress:
            self._best_distance = distance
            self._last_progress = self._steps
        stuck = self._steps - self._last_progress >= self.cfg.stuck_steps
        timeout = self._steps >= self.cfg.max_steps
        truncated = not terminated and (stuck or timeout)
        out = {
            "raw_obs": raw,
            "distance": distance,
            "stk_reward": float(stk_reward),
            "progress": float(np.clip(distance / self.track_length, 0.0, 1.0)),
            "finished": terminated,
            "stuck": not terminated and stuck,
            "steps": self._steps,
            "perturbed": perturbed,
            "executed": executed,
        }
        return self._features(raw), progress_reward(delta, terminated), terminated, truncated, out

    def _features(self, raw):
        return featurize(raw, self.cfg.n_paths, self.cfg.n_items)

    def close(self):
        global _live_env
        if self._closed:
            return
        self.race.close()
        self._closed = True
        if _live_env is self:
            _live_env = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
