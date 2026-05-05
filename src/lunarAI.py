import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import gymnasium as gym
from torch.autograd import Variable
from collections import deque, namedtuple

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
with open('configs/dqn.yaml') as f:
    cfg = yaml.safe_load(f)
learning_rate = float(cfg.get('learning_rate', 5e-4))
minibatch = int(cfg.get('minibatch', 64))
gamma = float(cfg.get('gamma', 0.99))
interpolation_parameter = float(cfg.get('interpolation_parameter', 1e-3))
replay_buffer_size = int(cfg.get('replay_buffer_size', 100000))


class ANN(nn.Module):

    def __init__(self, state_size, action_size, seed = 42):
        super(ANN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

class Agent():

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = ANN(state_size, action_size)
        self.target_qnetwork = ANN(state_size, action_size)
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())
        self.optimizer = torch.optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > minibatch:
            experiences = self.memory.sample(minibatch)
            self.learn(experiences, gamma)

    def get_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_qnetwork, target_qnetwork, interpolation_parameter):
        tau = float(interpolation_parameter)
        for target_params, local_params in zip(target_qnetwork.parameters(), local_qnetwork.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)
