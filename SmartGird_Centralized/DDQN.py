import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
import VariableInitialization as v

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(device)


class DoubleDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DoubleDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))

        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class DDQN:
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
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]

        self.q_eval = DoubleDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DoubleDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                           fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=max_size, batch_size=batch_size, user_num=user_num)
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params, in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def remember(self, state, action, reward, stata_, done, task):
        self.memory.store_transition(state, action, reward, stata_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, isTrain=True):
        if np.random.random() > self.epsilon:
            with T.no_grad():
                state = T.from_numpy(np.array(observation)).float().unsqueeze(0).to(device)
                q = self.q_eval.forward(state)
                _, actions = q.max(dim=2)
                print("actions_Q: ", actions)
                print(actions.shape)
                actions = actions.reshape(self.user_num)
                actions = actions.cpu().numpy() if isinstance(actions, T.Tensor) else actions
        else:
            actions = np.random.randint(0, self.action_dim, self.user_num)
        return actions

    def learn(self):
        if not self.memory.ready():
            return 0

        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals, dtype=T.bool).to(device)
        batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)

        with T.no_grad():
            q_ = self.q_target.forward(next_states_tensor)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * T.max(q_, dim=-1)[0]
        q_eval = self.q_eval.forward(states_tensor)
        actions_tensor = actions_tensor.unsqueeze(-1)
        q = q_eval.gather(2, actions_tensor)
        target = target.unsqueeze(-1)

        self.loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        self.loss.backward()
        self.q_eval.optimizer.step()

        self.update_network_parameters()
        self.decrement_epsilon()
        return 1

    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')

    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DDQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DDQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
