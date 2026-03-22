"""
CartPole solved with Deep Q-Network (DQN).

This script provides a complete, well-documented implementation of reinforcement
learning for the CartPole-v1 environment using PyTorch and Gymnasium.

Features
--------
- Reproducible training via random seeds.
- Experience replay buffer.
- Target network for stable learning.
- Epsilon-greedy exploration with exponential decay.
- Configurable hyperparameters from the command line.
- Checkpoint save/load for trained models.
- Training and evaluation modes.

Usage
-----
Train:
    python cartpole_dqn.py --train --episodes 400

Evaluate a saved model:
    python cartpole_dqn.py --eval --model-path checkpoints/cartpole_dqn.pt

Render while evaluating:
    python cartpole_dqn.py --eval --render --model-path checkpoints/cartpole_dqn.pt
"""

from __future__ import annotations

import argparse
import collections
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Transition:
    """A single transition tuple for replay memory."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy DQN updates."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random minibatch from memory."""
        batch = random.sample(self.buffer, batch_size)
        states = np.asarray([t.state for t in batch], dtype=np.float32)
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.asarray([t.next_state for t in batch], dtype=np.float32)
        dones = np.asarray([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP approximator for Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNAgent:
    """DQN agent with online and target networks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        lr: float,
        gamma: float,
        hidden_dim: int,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.action_dim = action_dim

        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: Tuple[np.ndarray, ...]) -> float:
        """Perform one gradient update using a sampled transition batch."""
        states, actions, rewards, next_states, dones = batch

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.online_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        """Hard update target network weights."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: Path) -> None:
        """Save online network parameters."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load model parameters to online and target nets."""
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, start: float, end: float, decay_steps: int) -> float:
    """Exponentially decay epsilon over training steps."""
    if decay_steps <= 0:
        return end
    decay = np.exp(-step / decay_steps)
    return end + (start - end) * decay


def train(args: argparse.Namespace) -> Path:
    """Train a DQN agent on CartPole-v1 and return the model path."""
    set_seed(args.seed)

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
    )
    replay = ReplayBuffer(args.buffer_size)

    total_steps = 0
    rewards_window: Deque[float] = collections.deque(maxlen=100)
    losses: List[float] = []

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0

        for _ in range(args.max_steps_per_episode):
            epsilon = epsilon_by_step(total_steps, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps)
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(Transition(state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay) >= args.batch_size:
                loss = agent.update(replay.sample(args.batch_size))
                losses.append(loss)

            if total_steps % args.target_update_interval == 0:
                agent.update_target()

            if done:
                break

        rewards_window.append(episode_reward)
        avg_reward = float(np.mean(rewards_window))

        if episode % args.log_interval == 0 or episode == 1:
            avg_loss = float(np.mean(losses[-100:])) if losses else 0.0
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Avg(100): {avg_reward:6.2f} | "
                f"Epsilon: {epsilon:5.3f} | "
                f"Loss(100): {avg_loss:7.4f}"
            )

        # CartPole-v1 is considered solved at average reward >= 475 over 100 episodes.
        if len(rewards_window) == 100 and avg_reward >= args.solve_threshold:
            print(f"Solved at episode {episode} with Avg(100) reward {avg_reward:.2f}")
            break

    model_path = Path(args.model_path)
    agent.save(model_path)
    print(f"Saved model to {model_path}")

    env.close()
    return model_path


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation episodes with a trained model."""
    set_seed(args.seed)

    render_mode = "human" if args.render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        hidden_dim=args.hidden_dim,
    )

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    agent.load(model_path)

    episode_rewards: List[float] = []
    for ep in range(1, args.eval_episodes + 1):
        state, _ = env.reset(seed=args.seed + 10_000 + ep)
        total_reward = 0.0

        for _ in range(args.max_steps_per_episode):
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Eval episode {ep:3d}: reward={total_reward:.1f}")

    print(f"Average evaluation reward: {np.mean(episode_rewards):.2f}")
    env.close()


def parse_args() -> argparse.Namespace:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train/evaluate DQN on CartPole-v1")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train a new model")
    mode.add_argument("--eval", action="store_true", help="Evaluate a saved model")

    parser.add_argument("--model-path", type=str, default="checkpoints/cartpole_dqn.pt", help="Path to save/load the model")
    parser.add_argument("--episodes", type=int, default=400, help="Max training episodes")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps-per-episode", type=int, default=500, help="Max steps before truncation")

    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer capacity")
    parser.add_argument("--hidden-dim", type=int, default=128, help="MLP hidden layer size")

    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000, help="Exploration decay constant")

    parser.add_argument("--target-update-interval", type=int, default=200, help="Target net sync interval (steps)")
    parser.add_argument("--solve-threshold", type=float, default=475.0, help="Avg reward threshold to stop early")
    parser.add_argument("--log-interval", type=int, default=10, help="Episodes between log lines")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")

    return parser.parse_args()


def main() -> None:
    """Script entrypoint."""
    args = parse_args()

    # Make matrix multiplications deterministic where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)


if __name__ == "__main__":
    main()
