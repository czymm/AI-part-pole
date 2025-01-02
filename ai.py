import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 定义Q网络（神经网络）
class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义Replay Buffer（经验回放）
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_space, action_space, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001, batch_size=64, buffer_size=10000):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.q_network = DQN(state_space, action_space)
        self.target_network = DQN(state_space, action_space)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_space))  # 探索
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()  # 利用

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            target_q_value = rewards + self.gamma * torch.max(target_q_values, dim=1)[0] * (1 - dones)
        
        # 计算损失
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 训练DQN智能体
def train_dqn():
    env = gym.make('CartPole-v1')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = DQNAgent(state_space, action_space)
    
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
        
        agent.update_target_network()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    env.close()

# 执行训练
train_dqn()
