# CartPole with Deep Q-Learning (DQN)

This repository contains a **fully functioning, well-documented** PyTorch implementation of a Deep Q-Network (DQN) agent for solving **CartPole-v1** from Gymnasium.

## What is included

- Complete DQN training loop.
- Experience replay.
- Target network updates.
- Epsilon-greedy exploration with decay.
- Model checkpoint save/load.
- Separate train and evaluation modes.
- Reproducibility controls (seeds).

## Requirements

- Python 3.9+
- See `requirements.txt`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

### 1) Train a model

```bash
python cartpole_dqn.py --train --episodes 400
```

The script saves a model by default at:

- `checkpoints/cartpole_dqn.pt`

### 2) Evaluate the model

```bash
python cartpole_dqn.py --eval --model-path checkpoints/cartpole_dqn.pt
```

### 3) Evaluate with rendering

```bash
python cartpole_dqn.py --eval --render --model-path checkpoints/cartpole_dqn.pt
```

## Useful CLI options

- `--learning-rate` (default `1e-3`)
- `--gamma` (default `0.99`)
- `--batch-size` (default `64`)
- `--buffer-size` (default `50000`)
- `--epsilon-start` / `--epsilon-end` / `--epsilon-decay-steps`
- `--target-update-interval` (default `200` steps)
- `--solve-threshold` (default `475` average over 100 episodes)
- `--seed` for reproducibility
- `--cpu` to force CPU execution

See all options:

```bash
python cartpole_dqn.py --help
```

## Notes on expected performance

- CartPole-v1 is usually solved quickly by DQN.
- Exact training speed and final reward depend on hardware and random seed.
- A moving average reward near or above `475` over the last 100 episodes is considered solved.
