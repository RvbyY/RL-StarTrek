import gymnasium as gym
import numpy as np
from typing import Optional


def get_termination_reason(obs: np.ndarray, terminated: bool, truncated: bool, info: dict) -> str:
    if truncated:
        return "sleep"

    if not terminated:
        return "running"

    x, y, vx, vy, theta, theta_dot, leg_l, leg_r = obs
    if abs(x) > 1.0 or y > 1.5 or y < -0.1:
        return "out_of_view"

    if "landed" in info:
        return "landing" if info["landed"] else "crash"

    both_legs = (leg_l > 0.5) and (leg_r > 0.5)
    slow      = abs(vx) < 0.3 and abs(vy) < 0.3
    level     = abs(theta) < 0.3

    if both_legs and slow and level:
        return "landing"

    return "crash"


def make_env(
    seed: int = 0,
    render_mode: Optional[str] = None,
    continuous: bool = False,
    record_video: bool = False,
    video_folder: Optional[str] = None,
) -> gym.Env:
    _render_mode = render_mode
    if record_video and render_mode is None:
        _render_mode = "rgb_array"

    env = gym.make("LunarLander-v3", render_mode=_render_mode, continuous=continuous)

    if record_video:
        if video_folder is None:
            raise ValueError("video_folder quand record_video=True")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda ep_id: True,
            name_prefix=f"seed{seed}",
        )

    env.reset(seed=seed)
    return env