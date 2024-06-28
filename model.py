import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialize the neural network with an input layer, one hidden layer, and an output layer.

        Parameters:
        input_size (int): The size of the input layer.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer.
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the neural network.

        Parameters:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after passing through the network.
        """
        x = f.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        """
        Save the model to a file.

        Parameters:
        file_name (str): The name of the file to save the model. Defaults to 'model.pth'.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model: nn.Module, learning_rate: float, gamma: float) -> None:
        """
        Initialize the Q-learning trainer with a model, learning rate, and discount factor.

        Parameters:
        model (nn.Module): The neural network model to be trained.
        learning_rate (float): The learning rate for the optimizer.
        gamma (float): The discount factor for future rewards.
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state: np.ndarray, action: list[int], reward: int, next_state: np.ndarray,
                   done: bool) -> None:
        """
        Perform a single training step.

        Parameters:
        state (np.ndarray): The current state.
        action (np.ndarray): The action taken.
        reward (np.ndarray): The reward received.
        next_state (np.ndarray): The next state.
        done (np.ndarray): A flag indicating if the episode is done.
        """
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.uint8)

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        for index in range(len(done)):
            q_new = reward[index]
            if not done[index]:
                q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
