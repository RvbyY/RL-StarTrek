import argparseimport sys
from pathlib import Path

sys.path.insert(0, ".")

from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger
from src.policies import RandomPolicy, HeuristicPolicy


def run(cfg_path: str):
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg.get("run_name", Path(cfg_path).stem)
    algo = cfg.get("algo", "random")
    n_ep = cfg.get("n_episodes", 200)
    seed = cfg.get("seed", 0)
    log_dir  = cfg.get("log_dir", "logs")

    env = make_env(seed=seed)
    logger = EpisodeLogger(f"{log_dir}/{run_name}.csv", run_name=run_name, verbose=True)

    if algo == "random":
        policy = RandomPolicy(env.action_space, seed=seed)
    elif algo == "heuristic":
        policy = HeuristicPolicy()
    elif algo == "dqn":
        try:
            from src.agent import DQNAgent
            policy = DQNAgent(cfg)
        except ImportError:
            print("DQN pas encore implémenté.")
            sys.exit(1)
    else:
        print(f"Algo inconnu : {algo}")
        sys.exit(1)

    obs, info = env.reset(seed=seed)
    score, length = 0.0, 0
    episode = 0

    while episode < n_ep:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score  += reward
        length += 1

        if terminated or truncated:
            reason = get_termination_reason(obs, terminated, truncated, info)
            logger.log_episode(score=score, length=length,
                               terminated=terminated, truncated=truncated,
                               reason=reason)
            score, length = 0.0, 0
            episode += 1
            policy.reset()
            obs, info = env.reset()

    env.close()
    logger.print_summary()
    print(f"Done. CSV → {log_dir}/{run_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin vers le fichier YAML")
    args = parser.parse_args()
    run(args.config)