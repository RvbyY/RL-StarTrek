import sys
import numpy as np
sys.path.insert(0, ".")

from src.env_utils import make_env, get_termination_reason
from src.logger import EpisodeLogger

SEED = 0
N_EPISODES = 5

def main():
    print("=== Test infra — LunarLander-v3 ===\n")

    env    = make_env(seed=SEED)
    logger = EpisodeLogger("logs/test_infra.csv", run_name="test", verbose=True)

    obs, info = env.reset(seed=SEED)
    score, length = 0.0, 0

    episode = 0
    while episode < N_EPISODES:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score  += reward
        length += 1

        if terminated or truncated:
            reason = get_termination_reason(obs, terminated, truncated, info)
            logger.log_episode(
                score=score, length=length,
                terminated=terminated, truncated=truncated,
                reason=reason,
            )
            score, length = 0.0, 0
            episode += 1
            obs, info = env.reset()

    env.close()
    logger.print_summary()
    print(f"CSV sauvegardé → logs/test_infra.csv")

if __name__ == "__main__":
    main()