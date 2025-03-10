import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
import VariableInitialization as v

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)


    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DuelingDQN:
    def __init__(self, alpha, state_dim, action_dim, user_num, batch_size, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, tau=0.005, epsilon=1.0, eps_end=0.01, eps_dec=0.999,
                 max_size=1000000):
        self.loss = None
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir
        self.user_num = user_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size, user_num=user_num)
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params, in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, next_state, done, task):
        # 存储转换
        # 如果是GPU上的张量，先移到CPU，再转换为numpy
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        reward = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
        next_state = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state
        done = done.cpu().numpy() if isinstance(done, torch.Tensor) else done

        # 然后将数据转换为GPU可用的张量
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)  # 假设action是整数类型
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)  # 假设done是浮点类型

        # 现在state, action, reward, next_state, done都是GPU上的张量

        self.memory.store_transition(state, action, reward, next_state, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, isTrain=True):
        if np.random.random() > self.epsilon:
            with T.no_grad():
                state = T.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
                _, A = self.q_eval.forward(state)
                _, actions = A.max(dim=2)
                # q_values = q_values.reshape(v.n)
                # print("actions_Q: ", actions)
                actions = actions.reshape(v.n)
                actions = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        else:
            actions = np.random.randint(0, self.action_dim, v.n)
        return actions

    def learn(self):
        if not self.memory.ready():
            return 0

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()

        # 将状态和其他张量移动到 GPU
        states_tensor = states.to(device)
        actions_tensor = actions.to(device).to(torch.int64)  # 确保 actions_tensor 是 int64 类型
        rewards_tensor = rewards.to(device)
        next_states_tensor = next_states.to(device)
        terminals_tensor = terminals.to(device)
        states_tensor = states_tensor.view(-1, self.state_dim)
        next_states_tensor = next_states_tensor.view(-1, self.state_dim)

        with T.no_grad():
            # 计算目标网络的 Q 值
            V_, A_ = self.q_target.forward(next_states_tensor)
            V_ = V_.expand(-1, self.action_dim)  # 扩展 V_ 以适应动作维度
            q_ = V_ + A_ - T.mean(A_, dim=-1, keepdim=True)
            q_ = q_.reshape(-1, self.user_num, self.action_dim)
            q_[terminals_tensor] = 0.0  # 处理终止状态
            target = rewards_tensor + self.gamma * T.max(q_, dim=-1)[0]  # 目标 Q 值

        # 计算当前评估网络的 Q 值
        V, A = self.q_eval.forward(states_tensor)
        V = V.expand(-1, self.action_dim)  # 扩展 V 以适应动作维度
        actions_tensor = actions_tensor.unsqueeze(-1)  # 扩展动作张量的维度
        # 计算 Q 值
        q = (V + A - T.mean(A, dim=-1, keepdim=True)).reshape(-1, self.user_num, self.action_dim)
        q = q.gather(2, actions_tensor)
        # 确保 target 和 q 的维度匹配
        target = target.unsqueeze(-1)  # 增加维度，使 target 与 q 对齐

        # 计算损失
        self.loss = F.mse_loss(q, target.detach())  # 使用 target 的 detach 版本
        self.q_eval.optimizer.zero_grad()  # 清空梯度
        self.loss.backward()  # 反向传播
        self.q_eval.optimizer.step()  # 更新网络参数

        # 更新目标网络参数
        self.update_network_parameters()
        self.decrement_epsilon()  # 减小 epsilon（探索率）

        return 1

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DuelingDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DuelingDQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DuelingDQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DuelingDQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
