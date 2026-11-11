from dataclasses import dataclass

TRAIN_TRACK = "scotland"
HELD_OUT_TRACKS = ("lighthouse", "olivermath", "fortmagma")


def _check_positive(cfg, names):
    for name in names:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EnvConfig:
    """One kart alone on one lap, with steering perturbations shared by every driver"""

    track: str = TRAIN_TRACK
    max_steps: int = 1500
    # Each step starts a perturbation with this probability, which then lasts noise_steps
    noise_prob: float = 0.02
    noise_steps: int = 5
    # Truncate after stuck_steps without stuck_min_progress metres of progress
    stuck_steps: int = 150
    stuck_min_progress: float = 1.0
    n_paths: int = 5
    n_items: int = 5

    def __post_init__(self):
        _check_positive(self, ("max_steps", "noise_steps", "stuck_steps", "n_paths", "n_items"))
        if not 0 <= self.noise_prob <= 1:
            raise ValueError("noise_prob must be in [0, 1]")
        if self.stuck_min_progress < 0:
            raise ValueError("stuck_min_progress must be nonnegative")


@dataclass(frozen=True)
class DatasetConfig:
    """Expert episodes recorded once and shared by DAgger and RLPD"""

    n_episodes: int = 20
    first_seed: int = 10_000
    expert_team: int = 2

    def __post_init__(self):
        _check_positive(self, ("n_episodes",))


@dataclass(frozen=True)
class DaggerConfig:
    """BC on the expert data, then DAgger with beta 0, step counts in online env steps"""

    seed: int = 1
    online_steps: int = 100_000
    episodes_per_iter: int = 2
    # Each training restarts from scratch on the aggregated data, as in Ross et al. 2011
    grad_steps: int = 20_000
    batch_size: int = 256
    lr: float = 3e-4
    hidden: tuple[int, ...] = (256, 256)
    checkpoint_every: int = 5_000

    def __post_init__(self):
        _check_positive(
            self,
            ("online_steps", "episodes_per_iter", "grad_steps", "batch_size", "lr", "checkpoint_every"),
        )


@dataclass(frozen=True)
class RLPDConfig:
    """SAC with symmetric sampling, LayerNorm critics and a small ensemble, Ball et al. 2023"""

    seed: int = 1
    online_steps: int = 100_000
    n_critics: int = 3
    n_target_subset: int = 2
    # Critic updates per env step, the actor is updated once per env step
    utd: int = 6
    offline_ratio: float = 0.5
    batch_size: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_temperature: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_temperature: float = 0.1
    # None means minus the action dimension
    target_entropy: float | None = None
    backup_entropy: bool = True
    critic_layer_norm: bool = True
    reward_scale: float = 10.0
    # Uniform random actions before the first update
    learning_starts: int = 5_000
    hidden: tuple[int, ...] = (256, 256)
    checkpoint_every: int = 5_000

    def __post_init__(self):
        _check_positive(
            self,
            (
                "online_steps",
                "n_critics",
                "n_target_subset",
                "utd",
                "batch_size",
                "lr_actor",
                "lr_critic",
                "lr_temperature",
                "init_temperature",
                "reward_scale",
                "checkpoint_every",
            ),
        )
        if self.n_target_subset > self.n_critics:
            raise ValueError("n_target_subset cannot exceed n_critics")
        if not 0 <= self.offline_ratio <= 1:
            raise ValueError("offline_ratio must be in [0, 1]")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be nonnegative")


@dataclass(frozen=True)
class EvalConfig:
    """Deterministic policies on fixed noisy episodes, the same seeds for every run"""

    n_episodes: int = 10
    first_seed: int = 100_000
    held_out_tracks: tuple[str, ...] = HELD_OUT_TRACKS

    def __post_init__(self):
        _check_positive(self, ("n_episodes",))
