import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

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
    target_net = DeepQNetwork(input_dim=8, hidden_dims=[128, 128], output_dim=4)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 1000
    max_steps = 1000
    loss_list = []

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    action = policy_net(state_tensor).argmax().item()

            next_state, reward, done, truncated, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            total_reward += float(reward)

            if len(replay_buffer) > batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
                state_batch = torch.tensor(state_batch, dtype=torch.float32)
                action_batch = torch.tensor(action_batch, dtype=torch.long)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
                done_batch = torch.tensor(done_batch, dtype=torch.float32)

                current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

                loss = criterion(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            if done or truncated:
                print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                break

            state = next_state

        if (episode + 1) % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close() 

    # Calculate moving average for smoothing
    window_size = 100
    moving_avg = np.convolve(loss_list, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, alpha=0.3, color='blue', label='Raw Loss')
    plt.plot(range(window_size-1, len(loss_list)), moving_avg, color='red', label='Moving Average')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()