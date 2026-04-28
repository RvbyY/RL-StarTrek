import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger
from src.policies import RandomPolicy, HeuristicPolicy
from src.lunarAI import ANN, ReplayMemory, Agent
from collections import deque


def run(cfg_path: str):
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg.get("run_name", Path(cfg_path).stem)
    algo = cfg.get("algo", "random")
    n_ep = cfg['n_episodes']
    seed = cfg['seed']
    log_dir  = cfg.get("log_dir", "logs")
    epsilon = cfg['epsilon_start']
    epsilon_end = cfg['epsilon_end']
    epsilon_decay = cfg['epsilon_decay']
    max_time_step = cfg['max_time_step']

    env = make_env(seed=seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    logger = EpisodeLogger(f"{log_dir}/{run_name}.csv", run_name=run_name, verbose=True)

    if algo == "random":
        agent = RandomPolicy(env.action_space, seed=seed)
    elif algo == "heuristic":
        agent = HeuristicPolicy()
    elif algo == "dqn":
        try:
            agent = Agent(state_size, action_size)
        except ImportError:
            print("DQN pas encore implémenté.")
            sys.exit(1)
    else:
        print(f"Algo inconnu : {algo}")
        sys.exit(1)

    learn_loop(agent, n_ep, epsilon, epsilon_end, epsilon_decay, env, logger, max_time_step,
               seed, log_dir, run_name)
    return 0

def time_step_loop(agent, epsilon, state, score, max_time_step, env, logger, length):
    for _ in range(0, max_time_step):
        action = agent.get_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            reason = get_termination_reason(next_state, terminated, truncated, info)
            logger.log_episode(score=score, length=length,
                               terminated=terminated, truncated=truncated,
                               reason=reason)
            break
    return score

def learn_loop(agent, n_ep, epsilon, epsilon_end, epsilon_decay, env, logger, max_time_step, seed, log_dir, run_name):
    scores_100_episodes = deque(maxlen=100)
    length = 0
    for episode in range(0, n_ep):
        state, _ = env.reset(seed)
        score = 0
        score = time_step_loop(agent, epsilon, state, score, max_time_step, env, logger, length)
        scores_100_episodes.append(score)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if episode % 10 == 0:
            print('Episode {} Avg Score: {:.2f}'.format(episode, np.mean(scores_100_episodes)))
        if np.mean(scores_100_episodes) >= 200:
            print('Congratulation, Solved in {:d} episodes \t Avg Score {:.2f}'.format(episode, np.mean(scores_100_episodes)))
            break

    env.close()
    logger.print_summary()
    print(f"Done. CSV → {log_dir}/{run_name}.csv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML")
    args = parser.parse_args()
    run(args.config)