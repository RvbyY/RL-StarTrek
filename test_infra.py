import sys
sys.path.insert(0, ".")

from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger
from src.policies import RandomPolicy, HeuristicPolicy

SEED = 0
N_EPISODES = 10


def run(policy, name):
    env    = make_env(seed=SEED)
    logger = EpisodeLogger(f"logs/{name}.csv", run_name=name, verbose=True)

    obs, info = env.reset(seed=SEED)
    score, length = 0.0, 0
    episode = 0

    while episode < N_EPISODES:
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


if __name__ == "__main__":
    env_tmp = make_env(seed=SEED)

    print("=== Random ===")
    run(RandomPolicy(env_tmp.action_space, seed=SEED), "random")

    print("=== Heuristic ===")
    run(HeuristicPolicy(), "heuristic")

    env_tmp.close()