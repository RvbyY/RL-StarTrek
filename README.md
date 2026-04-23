<div align="center">

```
    ██╗     ██╗   ██╗███╗   ██╗ █████╗ ██████╗     ██╗      █████╗ ███╗   ██╗██████╗ ███████╗██████╗ 
    ██║     ██║   ██║████╗  ██║██╔══██╗██╔══██╗    ██║     ██╔══██╗████╗  ██║██╔══██╗██╔════╝██╔══██╗
    ██║     ██║   ██║██╔██╗ ██║███████║██████╔╝    ██║     ███████║██╔██╗ ██║██║  ██║█████╗  ██████╔╝
    ██║     ██║   ██║██║╚██╗██║██╔══██║██╔══██╗    ██║     ██╔══██║██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗
    ███████╗╚██████╔╝██║ ╚████║██║  ██║██║  ██║    ███████╗██║  ██║██║ ╚████║██████╔╝███████╗██║  ██║
    ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

### `< Reinforcement Learning — Epitech G-AIA-401 />`

**Train an autonomous agent to land a lunar module on the Moon.**  
**No pilot. No mercy. Just math.**

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.3-FF6B6B?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</div>

---

## 🌕 The Mission

You are part of one of the first teams exploring our solar system. **Destination: Mars.**

Before sending humans across space, a relay base must be established on the Moon.  
You can't afford to put a pilot in every lunar module — so you'll train one to land itself.

The agent learns by **trial and error** from sparse rewards.  
It will crash. A lot. Then one day, it won't.

---

## ⚡ Quick Start

```bash
git clone <repo-url>
cd <repo>

# Setup
python3 -m venv venv
source venv/bin/activate.fish    # fish shell
# source venv/bin/activate       # bash / zsh

pip install -r requirements.txt

# Test that everything works
python test_infra.py

# Run baselines
python train.py --config configs/baseline_random.yaml
python train.py --config configs/baseline_heuristic.yaml
```

---

## 📁 Project Structure

```
.
├── 🚀 train.py                     # Training entry point
├── 🎯 eval.py                      # Evaluation entry point
├── 🧪 test_infra.py                # Quick sanity check
│
├── configs/
│   ├── baseline_random.yaml        # Random policy
│   ├── baseline_heuristic.yaml     # Heuristic policy
│   └── dqn.yaml                    # DQN agent
│
├── src/
│   ├── env_utils.py                # Gymnasium wrapper + termination detection
│   ├── logger.py                   # Episode logger → CSV + live terminal output
│   ├── policies.py                 # Random & Heuristic baselines
│   └── agent.py                    # DQN agent (PyTorch) ← WIP
│
├── logs/                           # CSV logs per run (auto-created)
├── results/plots/                  # PNG figures (auto-created)
└── videos/                         # Recorded episodes (auto-created)
```

---

## 🌍 Environment — LunarLander-v3

> A 2D rocket landing task built on Box2D physics.

### State — 8D observation vector

| # | Variable | Description |
|---|---|---|
| 0 | `x` | Horizontal position (0 = center) |
| 1 | `y` | Vertical position (0 = ground) |
| 2 | `vx` | Horizontal velocity |
| 3 | `vy` | Vertical velocity (negative = falling) |
| 4 | `theta` | Lander angle (0 = upright) |
| 5 | `theta_dot` | Angular velocity |
| 6 | `leg_left` | Left leg ground contact (0 or 1) |
| 7 | `leg_right` | Right leg ground contact (0 or 1) |

### Actions — Discrete

| Action | Effect |
|---|---|
| `0` | Do nothing |
| `1` | Fire left thruster |
| `2` | Fire main engine (thrust up) |
| `3` | Fire right thruster |

### Reward Shaping

| Event | Reward |
|---|---|
| Close to landing pad | `+` |
| Moving slowly | `+` |
| Staying level | `+` |
| Both legs touching | `+` |
| Engine firing | small `-` per step |
| Safe landing | **+100** |
| Crash | **-100** |

> ✅ **Solved** when mean score ≥ 200 over 100 consecutive episodes.

---

## 🧠 Architecture

```
train.py  ──────────────────────────────────────────────────────────
  │
  ├── make_env(cfg)               # env_utils.py
  │     └── LunarLander-v3       # Gymnasium + Box2D
  │
  ├── EpisodeLogger(csv_path)     # logger.py
  │     └── logs every episode → CSV + terminal
  │
  └── policy.select_action(obs)   # policies.py / agent.py
        │
        └── env.step(action)
              │
              └── get_termination_reason()   # env_utils.py
                    └── log_episode(reason)
```

---

## 📊 Termination Reasons

Every episode is logged with a cause — critical for debugging the agent.

| Reason | Icon | Meaning |
|---|---|---|
| `landing` | ✅ | Both legs down, low speed, level angle |
| `crash` | 💥 | Hull hit the ground |
| `out_of_view` | 🚀 | Lander flew off-screen |
| `sleep` | 💤 | Step limit reached (truncated) |

---

## 📈 Baselines

| Policy | Mean Score | Notes |
|---|---|---|
| Random | ~-165 | Pure chaos, crashes every time |
| Heuristic | ~TBD | Rule-based, no learning |
| DQN | ≥ 200 | Target — solves the environment |

---

## 🔁 Reproducibility

All experiments run on **5 seeds (0–4)**. Results reported as **mean ± 95% CI**.

```bash
chmod +x repro.sh
./repro.sh
# Regenerates all logs, plots and CSVs from scratch
```

---

## 📦 Requirements

```
gymnasium[box2d] >= 0.29.0
torch            >= 2.0.0
numpy            >= 1.24.0
matplotlib       >= 3.7.0
pyyaml           >= 6.0
```

---

## 👥 Team

| Role | Scope |
|---|---|
| Infra & Baselines | `env_utils`, `logger`, `policies`, `train`, `eval`, configs, README |
| DQN Agent | `agent.py`, replay buffer, target network, epsilon decay, ablations |

---

<div align="center">

```
   [ EPITECH ]  ·  G-AIA-401  ·  2025
   Shoot for the moon. Land on the moon.
```

</div>