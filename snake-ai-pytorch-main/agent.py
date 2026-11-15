import argparse
import os
import random
from collections import deque

import numpy as np
import torch

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

CHECKPOINT_PATH = os.path.join("model", "training_checkpoint.pth")
BEST_MODEL_PATH = os.path.join("model", "model.pth")


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if not self.memory:
            return
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move


def save_checkpoint(agent, record, plot_scores, plot_mean_scores):
    """Save resumable training state after a completed episode."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.trainer.optimizer.state_dict(),
        "n_games": agent.n_games,
        "record": record,
        "plot_scores": plot_scores,
        "plot_mean_scores": plot_mean_scores,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


def load_checkpoint(agent):
    """Load the latest full checkpoint; fall back to weights-only model.pth."""
    if os.path.isfile(CHECKPOINT_PATH):
        try:
            # weights_only=False is needed because the checkpoint also stores
            # Python/NumPy RNG state and ordinary training metadata.
            try:
                checkpoint = torch.load(
                    CHECKPOINT_PATH, map_location="cpu", weights_only=False
                )
            except TypeError:  # Compatibility with older PyTorch versions
                checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

            agent.model.load_state_dict(checkpoint["model_state_dict"])
            agent.trainer.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            agent.n_games = int(checkpoint.get("n_games", 0))

            if "python_rng_state" in checkpoint:
                random.setstate(checkpoint["python_rng_state"])
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])

            record = int(checkpoint.get("record", 0))
            plot_scores = list(checkpoint.get("plot_scores", []))
            plot_mean_scores = list(checkpoint.get("plot_mean_scores", []))

            print(
                f"Resumed checkpoint: {agent.n_games} completed games; "
                f"best score = {record}"
            )
            print(
                "Note: replay memory is not stored in this lightweight "
                "checkpoint and will refill during resumed training."
            )
            return record, plot_scores, plot_mean_scores

        except Exception as exc:
            raise RuntimeError(
                f"Could not load {CHECKPOINT_PATH}: {exc}. "
                "Do not delete the checkpoint; send this error for diagnosis."
            ) from exc

    if os.path.isfile(BEST_MODEL_PATH):
        try:
            try:
                state_dict = torch.load(
                    BEST_MODEL_PATH, map_location="cpu", weights_only=True
                )
            except TypeError:
                state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
            agent.model.load_state_dict(state_dict)
            print(
                "Loaded model.pth weights only. Optimizer, episode count and "
                "replay memory from the old session cannot be recovered."
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load existing best model {BEST_MODEL_PATH}: {exc}"
            ) from exc

    return 0, [], []


def train(games=500, show_plot=True, resume=True):
    if games <= 0:
        raise ValueError("--games must be a positive integer.")

    agent = Agent()
    if resume:
        record, plot_scores, plot_mean_scores = load_checkpoint(agent)
    else:
        record, plot_scores, plot_mean_scores = 0, [], []
        print("Starting a fresh training run; existing files are not deleted.")

    total_score = sum(plot_scores)
    games_this_run = 0
    game = SnakeGameAI()

    print(f"Training target: {games} additional completed games.")
    print(f"Starting from episode: {agent.n_games}")

    while games_this_run < games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(
            state_old, final_move, reward, state_new, done
        )
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            games_this_run += 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / len(plot_scores)
            plot_mean_scores.append(mean_score)

            # Save after every completed game so an interruption loses
            # at most the currently unfinished game.
            save_checkpoint(agent, record, plot_scores, plot_mean_scores)

            print(
                f"Game {agent.n_games} | Score: {score} | "
                f"Record: {record} | "
                f"Run progress: {games_this_run}/{games}"
            )

            if show_plot:
                plot(plot_scores, plot_mean_scores)

    print(f"Training segment complete: {games_this_run} games completed.")
    print(f"Total completed episodes: {agent.n_games}")
    print(f"Best score: {record}")
    print(f"Latest resumable checkpoint: {CHECKPOINT_PATH}")
    print(f"Best model weights: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or resume a DQN agent to play Snake."
    )
    parser.add_argument(
        "--games", type=int, default=500,
        help="Number of additional complete games to train (default: 500)."
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable the live training chart."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints and start with new weights."
    )
    args = parser.parse_args()

    try:
        train(
            games=args.games,
            show_plot=not args.no_plot,
            resume=not args.fresh
        )
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. The latest completed-game checkpoint "
            "is preserved. Run the same command again to resume."
        )
