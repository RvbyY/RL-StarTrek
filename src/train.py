import argparse
import sys
import numpy as np
import os
import gymnasium as gym
import torch
import imageio
from pathlib import Path

sys.path.insert(0, ".")
from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger
from src.policies import RandomPolicy, HeuristicPolicy
from src.lunarAI import ANN, ReplayMemory, Agent
from collections import deque
FILE_PATH = "logs/saved.txt"

class Training():

    def __init__(self, cfg, cfg_path):
        self.run_name = cfg.get("run_name", Path(cfg_path).stem)
        self.algo = cfg.get("algo", "random")
        self.n_ep = cfg['n_episodes']
        self.seed = cfg['seed']
        self.log_dir = cfg.get("log_dir", "logs")
        self.epsilon = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.max_time_step = cfg['max_time_steps']
        self.env = make_env(self.seed, render_mode="rgb_array")
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.logger = EpisodeLogger(f"{self.log_dir}/{self.run_name}.csv", run_name=self.run_name, verbose=True)
        self.frames = []

def run(cfg_path: str):
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train = Training(cfg, cfg_path)
    if train.algo == "random":
        agent = RandomPolicy(train.env.action_space, train.seed)
    elif train.algo == "heuristic":
        agent = HeuristicPolicy()
    elif train.algo == "dqn":
        try:
            agent = Agent(train.state_size, train.action_size)
            if os.path.exists(FILE_PATH):
                agent = torch.load(FILE_PATH, weights_only=False)
        except ImportError:
            print("QQN Failed import.")
            sys.exit(1)
    else:
        print(f"Unknown algo : {train.algo}")
        sys.exit(1)

    learn_loop(agent, train)
    train.env.close()
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)
    existing_tests = [f for f in os.listdir(video_dir) if f.startswith("lunar_lander_test_") and f.endswith(".gif")]
    test_number = len(existing_tests) + 1
    filename = f"{video_dir}/lunar_lander_test_{test_number}.gif"
    if train.frames:
        max_frames = 1000
        frames_to_write = train.frames[-max_frames:]
        if len(train.frames) > max_frames:
            print(f"Captured {len(train.frames)} frames; exporting last {max_frames} frames to GIF.")
        try:
            imageio.mimwrite(filename, frames_to_write, fps=20)
            print(f"Saved GIF → {filename}")
        except Exception as e:
            print(f"Failed to write GIF: {e}")
        train.frames.clear()
    else:
        print("No valid frames captured; skipping GIF export.")

    return 0

def time_step_loop(agent, epsilon, state, score, train, length):
    for _ in range(0, train.max_time_step):
        action = agent.get_action(state, epsilon)
        next_state, reward, terminated, truncated, info = train.env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        frame = train.env.render()
        if frame is not None:
            train.frames.append(frame)

        length += 1

        if done:
            reason = get_termination_reason(next_state, terminated, truncated, info)
            train.logger.log_episode(score=score, length=length,
                               terminated=terminated, truncated=truncated,
                               reason=reason)
            break
    return score, length

def learn_loop(agent, train):
    scores_100_episodes = deque(maxlen=100)
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

    torch.save(agent, FILE_PATH)
    train.env.close()
    train.logger.print_summary()
    print(f"Done. CSV → {train.log_dir}/{train.run_name}.csv")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML")
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()