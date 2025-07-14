import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DeepQNetwork, self).__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU())
            in_dim = h
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="human")
    policy_net = DeepQNetwork(input_dim=8, hidden_dims=[128, 128], output_dim=4)

    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close() 