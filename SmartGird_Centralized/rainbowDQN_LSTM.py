
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import VariableInitialization as v
from collections import deque, namedtuple

import os

from buffer import ReplayBuffer
import random

# 定义 Experience 结构
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.35):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha

    def add(self, experience, priority):
        self.buffer.append(experience)
        self.priorities.append(float(priority))

    def sample(self, batch_size, beta=0.5):
        indices = self._get_sample_indices(batch_size)
        experiences = [self.buffer[i] for i in indices]
        weights = self._calculate_weights(indices, beta)
        return experiences, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def _get_sample_indices(self, batch_size):
        probabilities = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return indices

    def _calculate_weights(self, indices, beta):
        priorities = np.array(self.priorities, dtype=np.float64)
        probabilities = priorities[indices] ** self.alpha
        probabilities /= probabilities.sum()
        weights = (len(self.buffer) * probabilities) ** (-beta)
        weights /= weights.max()
        return torch.tensor(weights, dtype=torch.float32)


class FeatureExtractor(nn.Module):
    def __init__(self, state_dim, hiddens_dim):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hiddens_dim
        self.lstm = nn.LSTM(state_dim, hiddens_dim)
        self.fc = nn.Linear(hiddens_dim, hiddens_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 经过LSTM层
        lstm_out, hidden = self.lstm(x)  # [b,n_states]-->[b,n_hiddens]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = lstm_out
        out = self.relu(out)
        out = self.fc(out)  # [b, n_actions]
        out = self.softmax(out)  # [b, n_actions]  计算每个动作的概率
        return out


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, hiddens_dim, action_dim):
        super().__init__()
        self.hidden = hiddens_dim
        self.feature_extractor = FeatureExtractor(state_dim, hiddens_dim)
        self.value_stream = nn.Linear(hiddens_dim, 1)
        self.advantage_stream = nn.Linear(hiddens_dim, action_dim)

    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.value_stream(features).expand(-1, self.advantage_stream.out_features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class RainbowDQN:
    def __init__(self, num_updates_per_step, state_dim, action_dim, user_num, hidden_dim, batch_size, ckpt_dir, gamma=0.99, lr=0.0005, buffer_size=10000,
                 epsilon=1, eps_end=0.2, eps_dec=0.9995):
        self.loss = None
        self.num_updates_per_step = 10
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir
        self.hidden = hidden_dim
        self.action_dim = action_dim
        self.user_num = user_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_network = DuelingQNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

    def save_models(self, episode):
        self.online_network.save_checkpoint(self.checkpoint_dir + 'Q_eval/RainBow_DQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.target_network.save_checkpoint(
            self.checkpoint_dir + 'Q_target/RainBow_DQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.online_network.load_checkpoint(self.checkpoint_dir + 'Q_eval/RainBow_DQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.target_network.load_checkpoint(
            self.checkpoint_dir + 'Q_target/RainBow_DQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')

    def choose_action(self, state, isTrain=False):
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                q_values = self.online_network(state)
                q_values, actions = q_values.max(dim=1)
                q_values = q_values.reshape(v.n)
                actions = actions.reshape(v.n)
                actions = actions.cpu().numpy() if isinstance(actions,torch.Tensor) else actions
        else:
            actions = np.random.randint(0, self.online_network.advantage_stream.out_features, self.user_num)
        return actions

    def remember(self, state, action, reward, next_state, done, task):
        if not isinstance(reward, (list, np.ndarray)):
            reward = [reward]
        if not isinstance(done, (list, np.ndarray)):
            done = [done]
        # 如果是GPU上的张量，先移到CPU，再转换为numpy
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        next_state = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
        done = done.cpu().numpy() if isinstance(done, torch.Tensor) else done

        # 然后再把数据转回GPU
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        experience = Experience(state, action, reward, next_state, done)
        if reward.mean()>0:
            priority = abs(reward.mean()-150)*0.0001 + 0.2
        else:
            priority = 0.2
        priority = torch.tensor(priority, dtype=torch.float32).to(self.device)
        priority = torch.clamp(priority, min=0.0, max=1.0)  # 用torch的方法来替代np.clip

        self.replay_buffer.add(experience, priority)

    def learn(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return 0
        if self.epsilon > self.eps_end:

            self.epsilon *= self.eps_dec
        else:
            self.epsilon = self.eps_end
        # print("epsilon ", self.epsilon)
        #  从优先经验回放中采样
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 确保在转化之前将张量从 GPU 转到 CPU
        states = torch.tensor(np.array([state.cpu().numpy() for state in states]),
                              dtype=torch.float32).cpu()  # 确保 states 在 CPU 上
        actions = torch.tensor(np.array([action.cpu().numpy() for action in actions]),
                               dtype=torch.long).cpu().unsqueeze(-1)
        rewards = torch.tensor(np.array([reward.cpu().numpy() for reward in rewards]),
                               dtype=torch.float32).cpu().unsqueeze(-1)
        next_states = torch.tensor(np.array([next_state.cpu().numpy() for next_state in next_states]),
                                   dtype=torch.float32).cpu()
        dones = torch.tensor(np.array([done.cpu().numpy() for done in dones]), dtype=torch.float32).cpu().unsqueeze(-1)
        weights = weights.cpu()  # 确保权重在 CPU 上

        # 将所有张量转回到 GPU
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # 计算目标Q值
        target_q_values = self.target_network(next_states).detach()
        max_next_q_values, _ = target_q_values.max(dim=1)
        max_next_q_values = max_next_q_values.reshape(-1, self.user_num, 1)
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        # actions = actions.permute(0, 2, 1)

        # 计算在线网络的Q值
        actions = actions.long()
        online_q_values = self.online_network(states).reshape(-1, self.user_num,  self.action_dim)
        online_q_values = online_q_values.gather(2, actions)
        # 计算TD误差
        td_errors = target_q_values - online_q_values

        # 计算带有优先经验回放权重的损失
        self.loss = 0.5 * (weights * td_errors.pow(2)).mean()
        # 进行梯度下降
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # 更新优先经验回放池的权重
        self.replay_buffer.update_priorities(indices, td_errors.mean(dim=1).squeeze().abs().detach().cpu().numpy())
        return 1

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
