"""
Reproducibility script: Train and evaluate DQN agent over 5 seeds.
Generates configs, trains, evaluates, and reports statistics.
"""

import os
import sys
import yaml
import subprocess
import numpy as np
from pathlib import Path

# Configuration
BASE_CONFIG = "configs/base_config.yaml"
N_SEEDS = 5
LOG_DIR = "logs"
CONFIGS_DIR = "configs"

def create_seed_configs(startSeed):
    print(f"\n{'='*60}")
    print("Step 1: Generating {n} seed configs".format(n=N_SEEDS))
    print(f"{'='*60}\n")

    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)

    config_paths = []
    for seed_idx in range(N_SEEDS):
        cfg = base_cfg.copy()
        cfg['seed'] = startSeed + seed_idx
        cfg['run_name'] = f"dqn_seed{seed_idx}"

        # Create new config file
        config_file = Path(CONFIGS_DIR) / f"dqn_seed{seed_idx}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(cfg, f)
        config_paths.append(str(config_file))
        print(f"  ✓ {config_file} (seed={startSeed + seed_idx})")

    return config_paths

"""Train agent for each seed"""
def run_training(config_paths):
    print(f"\n{'='*60}")
    print("Step 2: Training on {n} seeds".format(n=N_SEEDS))
    print(f"{'='*60}\n")

    for i, config_path in enumerate(config_paths):
        print(f"\n[{i+1}/{N_SEEDS}] Training with {config_path}...")
        cmd = ["python3", "src/train.py", "--config", config_path]
        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            print(f"ERROR: Training failed for {config_path}")
            sys.exit(1)

def cleanup_eval_csv():
    for seed_idx in range(N_SEEDS):
        csv_file = Path(LOG_DIR) / f"dqn_seed{seed_idx}_eval.csv"
        if csv_file.exists():
            csv_file.unlink()

"""Evaluate agent for each seed"""
def run_evaluation(config_paths):
    print(f"\n{'='*60}")
    print("Step 3: Evaluating on {n} seeds".format(n=N_SEEDS))
    print(f"{'='*60}\n")

    cleanup_eval_csv()

    eval_scores = []

    for i, config_path in enumerate(config_paths):
        print(f"\n[{i+1}/{N_SEEDS}] Evaluating with {config_path}...")
        cmd = ["python3", "src/eval.py", "--config", config_path]
        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            print(f"ERROR: Evaluation failed for {config_path}")
            sys.exit(1)

    return eval_scores

"""Read evaluation scores from CSV files"""
def read_eval_scores():
    print(f"\n{'='*60}")
    print("Step 4: Collecting results")
    print(f"{'='*60}\n")

    all_scores = []

    for seed_idx in range(N_SEEDS):
        csv_file = Path(LOG_DIR) / f"dqn_seed{seed_idx}_eval.csv"
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found")
            continue

        scores = []
        with open(csv_file) as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    try:
                        score = float(parts[1])
                        scores.append(score)
                    except ValueError:
                        pass

        if scores:
            mean_score = float(np.mean(scores[-100:]))
            all_scores.append(mean_score)
            print(f"  Seed {seed_idx}: mean={mean_score:.2f} ± std={np.std(scores[-100:]):.2f}")

    return all_scores

def report_statistics(scores):
    print(f"\n{'='*60}")
    print("REPRODUCIBILITY REPORT")
    print(f"{'='*60}\n")

    if not scores:
        print("ERROR: No scores collected!")
        return

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    ci_95 = 1.96 * (std / np.sqrt(len(scores))) if len(scores) > 1 else 0.0

    print(f"Seeds tested: {len(scores)}")
    print(f"Scores: {[f'{s:.2f}' for s in scores]}")
    print(f"\nMean ± 95% CI: {mean:.2f} ± {ci_95:.2f}")
    print(f"Confidence Interval: [{mean - ci_95:.2f}, {mean + ci_95:.2f}]")
    print(f"Std Dev: {std:.2f}")

    if ci_95 < 15:
        verdict = "✓ REPRODUCIBLE (stable results, low variance)"
    elif ci_95 < 30:
        verdict = "~ MODERATELY REPRODUCIBLE"
    else:
        verdict = "✗ NOT REPRODUCIBLE (high variance)"

    print(f"\nVerdict: {verdict}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 reproduce.py <startFromSeed>")
        return
    try:
        startSeed = int(sys.argv[1])
    except ValueError:
        print("Argument must be int")

    print("REPRODUCIBILITY PIPELINE (5-Seed Evaluation) started from seed " + str(startSeed))

    # Step 1: Generate configs
    config_paths = create_seed_configs(startSeed)

    # Step 2: Train on all seeds
    run_training(config_paths)

    # Step 3: Evaluate on all seeds
    run_evaluation(config_paths)

    # Step 4: Collect and report
    scores = read_eval_scores()
    report_statistics(scores)

    print("Reproducibility pipeline complete")

if __name__ == "__main__":
    main()
