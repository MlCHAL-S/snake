import torch
import random
import numpy as np
from collections import deque
from Snake.snake import SnakeGame
from Snake.direction import Direction
from Snake.constants import Point, BLOCK_SIZE
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self) -> None:
        """
        Initialize the Agent with necessary attributes and neural network model.
        """
        self.number_of_games: int = 0
        self.epsilon: int = 0
        self.gamma: float = 0.9  # Default gamma value for discount factor
        self.memory: deque = deque(maxlen=MAX_MEMORY)  # Memory buffer for experience replay
        self.model: LinearQNet = LinearQNet(11, 256, 3)
        self.trainer: QTrainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    @staticmethod
    def get_state(game: SnakeGame) -> np.ndarray:
        """
        Get the current state of the game.

        Parameters:
        game (SnakeGame): The game instance.

        Returns:
        np.ndarray: The current state represented as a numpy array.
        """
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state: np.ndarray, action: list[int], reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store the experience in memory.

        Parameters:
        state (np.ndarray): The current state.
        action (np.ndarray): The action taken.
        reward (float): The reward received.
        next_state (np.ndarray): The next state.
        done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """
        Train the model using long-term memory (experience replay).
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)  # Convert deque to list

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: list[int], reward: int, next_state: np.ndarray,
                           done: bool) -> None:
        """
        Train the model using a single step of experience.

        Parameters:
        state (np.ndarray): The current state.
        action (np.ndarray): The action taken.
        reward (float): The reward received.
        next_state (np.ndarray): The next state.
        done (bool): Whether the episode is done.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> list[int]:
        """
        Get the action to be taken based on the current state.

        Parameters:
        state (np.ndarray): The current state.

        Returns:
        list[int]: The action to be taken as a one-hot encoded list.
        """
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train() -> None:
    """
    Train the agent to play the Snake game.
    """
    plot_scores = list()
    plot_mean_scores = list()
    total_score: int = 0
    record: int = 0
    agent: Agent = Agent()
    game: SnakeGame = SnakeGame()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get the move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game: {agent.number_of_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
