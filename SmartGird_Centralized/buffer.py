import numpy as np
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size, user_num):
        self.mem_size = max_size
        self.mem_cnt = 0
        self.batch_size = batch_size
        self.user_num = user_num
        self.state_memory = torch.zeros((self.mem_size, self.user_num, state_dim), dtype=torch.float32)
        self.action_memory = torch.zeros((self.mem_size, self.user_num), dtype=torch.float32)
        self.reward_memory = torch.zeros((self.mem_size, self.user_num), dtype=torch.float32)
        self.next_state_memory = torch.zeros((self.mem_size, self.user_num, state_dim), dtype=torch.float32)
        self.terminal_memory = torch.zeros((self.mem_size, self.user_num), dtype=torch.bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        self.state_memory[mem_idx] = torch.tensor(state).to(device)
        self.action_memory[mem_idx] = torch.tensor(action).to(device)
        self.reward_memory[mem_idx] = torch.tensor(reward).to(device)
        self.next_state_memory[mem_idx] = torch.tensor(state_).to(device)
        self.terminal_memory[mem_idx] = torch.tensor(done).to(device)

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)

        # 生成一个随机索引 tensor
        batch = torch.randint(0, mem_len, (self.batch_size,), dtype=torch.long)

        # 使用 tensor 索引从各个 memory 中提取数据
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals

    def ready(self):
        return self.mem_cnt > self.batch_size