import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pupil_on_pole.checkpoints import list_checkpoints, load_checkpoint
from pupil_on_pole.env import BINARY_ACTIONS, STEP_SECONDS, decode_action
from pupil_on_pole.networks import policy_action


def run_episode(env, seed, act, on_reset=None):
    """Drive one episode, act(obs, info) returns a pystk action dict"""
    obs, info = env.reset(seed)
    if on_reset is not None:
        on_reset()
    total, perturbed = 0.0, 0
    action_steps = dict.fromkeys(BINARY_ACTIONS, 0)
    speeds, center_distances = [], []
    fire_presses, firing = 0, False
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step_dict(act(obs, info))
        total += reward
        perturbed += int(info["perturbed"])
        # Counted on what was actually sent, rescue is always off
        for name in BINARY_ACTIONS:
            action_steps[name] += int(info["executed"][name])
        # A press is an off to on switch, each one uses the item the kart holds
        fire_presses += int(info["executed"]["fire"] and not firing)
        firing = info["executed"]["fire"]
        raw = info["raw_obs"]
        speeds.append(float(np.linalg.norm(np.asarray(raw["velocity"], dtype=np.float32))))
        center_distances.append(abs(float(np.asarray(raw["center_path_distance"], dtype=np.float32).reshape(-1)[0])))
        done = terminated or truncated
    return {
        "seed": int(seed),
        "finished": bool(terminated),
        "stuck": bool(info["stuck"]),
        "steps": int(info["steps"]),
        "progress": float(info["progress"]),
        "return": float(total),
        "perturbed_steps": perturbed,
        "action_steps": action_steps,
        "fire_presses": fire_presses,
        "mean_speed": float(np.mean(speeds)),
        "mean_center_distance": float(np.mean(center_distances)),
    }


def summarize(episodes, max_steps):
    """Metrics over episodes, an unfinished episode counts as max_steps in lap_time_s"""
    finished = [e for e in episodes if e["finished"]]
    total_steps = sum(e["steps"] for e in episodes)
    # Stuck truncations end early, they still count as the full cap
    penalized = [e["steps"] if e["finished"] else max_steps for e in episodes]
    return {
        "n_episodes": len(episodes),
        # Time and completion
        "finish_rate": len(finished) / len(episodes),
        "lap_steps": float(np.mean([e["steps"] for e in finished])) if finished else None,
        "lap_time_s": float(np.mean(penalized)) * STEP_SECONDS,
        "lap_time_std_s": float(np.std(penalized)) * STEP_SECONDS,
        "best_lap_time_s": float(min(e["steps"] for e in finished)) * STEP_SECONDS if finished else None,
        "stuck_rate": float(np.mean([e["stuck"] for e in episodes])),
        "progress": float(np.mean([e["progress"] for e in episodes])),
        "return": float(np.mean([e["return"] for e in episodes])),
        # Driving style
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
        "mean_center_distance": float(np.mean([e["mean_center_distance"] for e in episodes])),
        # Fire, as presses per lap and through action_use["fire"] as a fraction of steps
        "fire_presses": float(np.mean([e["fire_presses"] for e in episodes])),
        "action_use": {
            name: sum(e["action_steps"][name] for e in episodes) / total_steps for name in BINARY_ACTIONS
        },
    }


def _evaluate(env, track, n_episodes, first_seed, act, on_reset=None):
    env.set_track(track)
    episodes = [run_episode(env, first_seed + i, act, on_reset) for i in range(n_episodes)]
    return {"track": track, "summary": summarize(episodes, env.cfg.max_steps), "episodes": episodes}


def evaluate_policy(env, policy, normalizer, track, n_episodes, first_seed):
    """Deterministic policy on fixed noisy episodes"""

    def act(obs, info):
        return decode_action(policy_action(policy, normalizer, obs))

    return _evaluate(env, track, n_episodes, first_seed, act)


def evaluate_expert(env, expert, track, n_episodes, first_seed):
    """The expert on the same episodes, reset with the environment"""
    return _evaluate(env, track, n_episodes, first_seed, lambda obs, info: expert.act(info["raw_obs"]), expert.reset)


def evaluate_checkpoints(env, paths, eval_cfg, held_out=True):
    """Every checkpoint on the training track, and the last one on each held-out track"""
    train_track = env.cfg.track
    result = {"eval_config": asdict(eval_cfg), "train_track": train_track, "curve": [], "held_out": {}}
    for path in paths:
        payload, _, _, policy, normalizer = load_checkpoint(path)
        evaluation = evaluate_policy(env, policy, normalizer, train_track, eval_cfg.n_episodes, eval_cfg.first_seed)
        result["method"] = payload["method"]
        result["curve"].append(
            {"online_step": payload["online_step"], "extra": payload["extra"], **evaluation}
        )
    if held_out and paths:
        for track in eval_cfg.held_out_tracks:
            result["held_out"][track] = evaluate_policy(
                env, policy, normalizer, track, eval_cfg.n_episodes, eval_cfg.first_seed
            )
    env.set_track(train_track)
    return result


def evaluate_run(run_dir, eval_cfg, env_cfg, held_out=True):
    """Evaluate a training run in one environment and write run_dir/evaluation.json"""
    from pupil_on_pole.env import SoloKartEnv

    run_dir = Path(run_dir).resolve()
    paths = list_checkpoints(run_dir)
    if not paths:
        raise ValueError(f"{run_dir} has no checkpoints")
    if load_checkpoint(paths[0])[2] != env_cfg:
        raise ValueError("The run was trained with different environment settings")
    with SoloKartEnv(env_cfg) as env:
        result = evaluate_checkpoints(env, paths, eval_cfg, held_out)
    path = run_dir / "evaluation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def evaluate_expert_reference(out_dir, eval_cfg, env_cfg, team=2):
    """Expert on the training track and every held-out track, written to out_dir/evaluation.json"""
    from pupil_on_pole.env import SoloKartEnv
    from pupil_on_pole.expert import Expert

    # Absolute, the expert changes the working directory when it loads
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = (env_cfg.track, *eval_cfg.held_out_tracks)
    with SoloKartEnv(env_cfg) as env:
        expert = Expert(env, team=team)
        results = {
            track: evaluate_expert(env, expert, track, eval_cfg.n_episodes, eval_cfg.first_seed) for track in tracks
        }
    result = {"method": f"team{team}", "eval_config": asdict(eval_cfg), "train_track": env_cfg.track, "tracks": results}
    path = out_dir / "evaluation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path
