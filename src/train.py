import argparse
import sys
import numpy as np
import os
import gymnasium as gym
import torch
import imageio
import yaml
from pathlib import Path

sys.path.insert(0, ".")
from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger
from src.policies import RandomPolicy, HeuristicPolicy
from src.lunarAI import ANN, ReplayMemory, Agent
from collections import deque

VIDEO_PATTERN = "video_*.mp4"

class Training():

    def __init__(self, cfg, cfg_path):
        self.run_name = cfg.get("run_name", Path(cfg_path).stem)
        self.algo = cfg.get("algo", "random")
        self.n_ep = cfg['n_episodes']
        self.seed = cfg['seed']
        base_dir = cfg.get("log_dir", "logs")
        self.epsilon = cfg.get('epsilon_start', 1.0)
        self.epsilon_end = cfg.get('epsilon_end', 0.01)
        self.epsilon_decay = cfg.get('epsilon_decay', 0.995)
        self.max_time_step = cfg.get('max_time_steps', 1000)
        self.env = make_env(self.seed, render_mode="rgb_array")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.run_folder = f"{base_dir}/train/{self.run_name}"

        # Define model path
        os.makedirs(f"{base_dir}/brain", exist_ok=True)
        os.makedirs(self.run_folder, exist_ok=True)
        self.model_path = f"{base_dir}/brain/{self.run_name}_model.pth"
        self.logger = EpisodeLogger(f"{self.run_folder}/{self.run_name}.csv", run_name=self.run_name, verbose=True)
        self.frames = deque(maxlen=5)

def video_index(path):
    suffix = path.stem.split("_")[-1]
    return int(suffix) if suffix.isdigit() else -1

def show_video_of_model(agent, env_name, output_dir="videos", keep_last=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = sorted(output_dir.glob(VIDEO_PATTERN), key=video_index)
    next_index = 1
    if existing_videos:
        next_index = video_index(existing_videos[-1]) + 1

    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.get_action(state, epsilon=1)
        state, _, done, _, _ = env.step(action.item())
    env.close()

    video_path = output_dir / f"video_{next_index}.mp4"
    imageio.mimsave(video_path, frames, fps=30)

    if keep_last > 0:
        videos_to_remove = sorted(output_dir.glob(VIDEO_PATTERN), key=video_index)[:-keep_last]
        for old_video in videos_to_remove:
            old_video.unlink(missing_ok=True)


def save_recent_training_videos(train, output_dir="videos"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def next_video_index(existing_paths):
        if not existing_paths:
            return 1
        return max(video_index(path) for path in existing_paths) + 1

    existing_videos = sorted(output_dir.glob(VIDEO_PATTERN), key=video_index)
    next_index = next_video_index(existing_videos)

    for episode_frames in train.frames:
        if not episode_frames:
            continue

        video_path = output_dir / f"video_{next_index}.mp4"
        imageio.mimsave(video_path, episode_frames, fps=30)
        next_index += 1

    videos_to_remove = sorted(output_dir.glob(VIDEO_PATTERN), key=video_index)[:-5]
    for old_video in videos_to_remove:
        old_video.unlink(missing_ok=True)


def run(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train = Training(cfg, cfg_path)
    if train.algo == "random":
        agent = RandomPolicy(train.env.action_space, train.seed)
    elif train.algo == "heuristic":
        agent = HeuristicPolicy()
    elif train.algo == "dqn":
        try:
            agent = Agent(train.state_size, train.action_size, cfg=cfg)
            if os.path.exists(train.model_path):
                agent = torch.load(train.model_path, weights_only=False)
        except ImportError:
            print("QQN Failed import.")
            sys.exit(1)
    else:
        print(f"Unknown algo : {train.algo}")
        sys.exit(1)

    learn_loop(agent, train)
    train.env.close()
    save_recent_training_videos(train)

    # GIF script temporarily deactivated for perf
    # video_dir = "./videos"
    # os.makedirs(video_dir, exist_ok=True)
    # existing_tests = [f for f in os.listdir(video_dir) if f.startswith("lunar_lander_test_") and f.endswith(".gif")]
    # test_number = len(existing_tests) + 1
    # filename = f"{video_dir}/lunar_lander_test_{test_number}.gif"
    # if train.frames:
    #     max_frames = 1000
    #     frames_to_write = train.frames[-max_frames:]
    #     if len(train.frames) > max_frames:
    #         print(f"Captured {len(train.frames)} frames; exporting last {max_frames} frames to GIF.")
    #     try:
    #         imageio.mimwrite(filename, frames_to_write, fps=20)
    #         print(f"Saved GIF → {filename}")
    #     except Exception as e:
    #         print(f"Failed to write GIF: {e}")
    #     train.frames.clear()
    # else:
    #     print("No valid frames captured; skipping GIF export.")

    return 0

def time_step_loop(agent, epsilon, state, score, train, length):
    episode_frames = []
    for _ in range(0, train.max_time_step):
        if hasattr(agent, "get_action"):
            action = agent.get_action(state, epsilon)
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = train.env.step(action)
        done = terminated or truncated

        if hasattr(agent, "step"):
            agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        frame = train.env.render()
        if frame is not None:
            episode_frames.append(frame)

        length += 1

        if done:
            train.frames.append(episode_frames)
            reason = get_termination_reason(next_state, terminated, truncated, info)
            train.logger.log_episode(score=score, length=length,
                               terminated=terminated, truncated=truncated,
                               reason=reason, extra={"epsilon": epsilon})
            break
    return score, length

def learn_loop(agent, train):
    scores_100_episodes = deque(maxlen=100)
    try:
        for episode in range(0, train.n_ep):
            state, _ = train.env.reset(seed=train.seed + episode)
            score = 0
            length = 0
            score, length = time_step_loop(agent, train.epsilon, state, score, train, length)
            scores_100_episodes.append(score)
            train.epsilon = max(train.epsilon_end, train.epsilon * train.epsilon_decay)
            if episode % 10 == 0:
                print('Episode {} Avg Score: {:.2f}'.format(episode, np.mean(scores_100_episodes)))
            if np.mean(scores_100_episodes) >= 200:
                print('Congratulation, Solved in {:d} episodes \t Avg Score {:.2f}'.format(episode, np.mean(scores_100_episodes)))
                break
    except KeyboardInterrupt:
        print("\nKeyboard interruption. Save in progress...")
    finally:
        if train.algo == "dqn":
            torch.save(agent, train.model_path)
        train.env.close()
        train.logger.print_summary()
        if hasattr(train.logger, 'generate_plots'):
            train.logger.generate_plots()
        print(f"Done. CSV/Plots → {train.run_folder}/{train.run_name}.csv")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML")
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()