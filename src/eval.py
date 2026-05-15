import argparse
import sys
import numpy as np
import os
import torch
import yaml
from pathlib import Path

sys.path.insert(0, ".")
from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger
from src.lunarAI import Agent
from collections import deque

from src.policies import RandomPolicy, HeuristicPolicy

class Eval():

    def __init__(self, cfg, cfg_path):
        self.run_name = cfg.get("run_name", Path(cfg_path).stem)
        self.algo = cfg.get("algo", "dqn")
        self.n_ep = cfg['n_episodes']
        self.seed = cfg['seed']
        base_dir = cfg.get("log_dir", "logs")
        self.run_folder = f"{base_dir}/eval/{self.run_name}"
        self.model_path = f"{base_dir}/brain/{self.run_name}_model.pth"
        self.max_time_step = cfg.get('max_time_steps', 1000)
        self.env = make_env(self.seed)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.logger = EpisodeLogger(f"{self.run_folder}/{self.run_name}_eval.csv", run_name=self.run_name, verbose=False)

def run(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    eval_ctx = Eval(cfg, cfg_path)

    if eval_ctx.algo == "random":
        agent = RandomPolicy(eval_ctx.env.action_space, eval_ctx.seed)
    elif eval_ctx.algo == "heuristic":
        agent = HeuristicPolicy()
    elif eval_ctx.algo == "dqn":
        if not os.path.exists(eval_ctx.model_path):
            print(f"Model not found: {eval_ctx.model_path}")
            sys.exit(1)

        loaded = torch.load(eval_ctx.model_path, weights_only=False, map_location=torch.device('cpu'))
        if isinstance(loaded, Agent):
            agent = loaded
        else:
            agent = Agent(eval_ctx.state_size, eval_ctx.action_size, cfg=cfg)
            try:
                agent.local_qnetwork.load_state_dict(loaded)
                agent.target_qnetwork.load_state_dict(agent.local_qnetwork.state_dict())
            except Exception:
                print("Failed to load model")
                sys.exit(1)
    else:
        print(f"Unknown algo : {eval_ctx.algo}")
        sys.exit(1)

    eval_loop(agent, eval_ctx)
    return 0

def time_step_loop(agent, state, eval_ctx):
    score = 0.0
    length = 0

    for _ in range(0, eval_ctx.max_time_step):
        if hasattr(agent, "get_action"):
            action = agent.get_action(state, 0.0)  # exploitation only
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = eval_ctx.env.step(action)
        state = next_state
        score += reward
        length += 1

        if terminated or truncated:
            reason = get_termination_reason(next_state, terminated, truncated, info)
            eval_ctx.logger.log_episode(score=score, length=length, terminated=terminated, truncated=truncated, reason=reason)
            return score, length, terminated, truncated, reason

    return score, length, False, True, "sleep"

def eval_loop(agent, eval_ctx):
    scores_100_episodes = deque(maxlen=100)

    for episode in range(1, 100):
        state, _ = eval_ctx.env.reset(seed=eval_ctx.seed + episode)
        score, length, terminated, truncated, reason = time_step_loop(agent, state, eval_ctx)
        scores_100_episodes.append(score)

        if episode % 10 == 0:
            print('Episode {} Avg Score: {:.2f}'.format(episode, np.mean(scores_100_episodes)))

    eval_ctx.env.close()
    eval_ctx.logger.print_summary()
    eval_ctx.logger.generate_plots()
    print(f"Done. CSV/Plots → {eval_ctx.run_folder}/{eval_ctx.run_name}_eval.csv")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML")
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()