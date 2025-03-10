# If this comment is removed the program will blow up
# 如果删了此处注释程序就炸了
# This function has been here since 1987. DON'T FXXKING TOUCH IT
# 这函数1987年就这在了，别他娘动它
import os

import torch

import VariableInitialization as v
import CalculateElectricity as c
import utils
from environment import Environment_Distributed
from rainbowDQN_LSTM import RainbowDQN
from DuelingDQN import DuelingDQN
from DDQN import DDQN
from DQN import DQN

from matplotlib import pyplot as plt
import numpy as np
import argparse
import collections
import random

random.seed(20)  # 重新设置随机种子，使用系统时间或其他源生成种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def budda_bless():
    print("""

                       _ooOoo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      O\\  =  /O
                   ____/`---'\\____
                 .'  \\\\|     |//  `.
                /  \\\\|||  :  |||//  \\
               /  _||||| -:- |||||-  \\
               |   | \\\\\  -  /// |   |
               | \_|  ''\\---/''  |   |
               \\  .-\\__  `-`  ___/-. /
             ___`. .'  /--.--\\  `. . __
          ."" '<  `.___\\_<|>_/___.'  >'"".
         | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
         \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /
    ======`-.____`-.___\\_____/___.-`____.-'======
                       `=---='
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                佛祖保佑       永无BUG
    """)


parser = argparse.ArgumentParser()

parser.add_argument('--max_episodes', type=int, default=5000)
parser.add_argument('--RainBow_ckpt_dir1', type=str, default='./checkpoints/RainBowDQN1/')
parser.add_argument('--RainBow_time_cost_path1', type=str, default='./output_images/RainBow_time_cost1.png')
parser.add_argument('--RainBow_epsilon_path1', type=str, default='./output_images/RainBow_epsilon1.png')
parser.add_argument('--RainBow_cost_path1', type=str, default='./output_images/RainBow_cost1.png')
parser.add_argument('--RainBow_ckpt_dir2', type=str, default='./checkpoints/RainBowDQN2/')
parser.add_argument('--RainBow_time_cost_path2', type=str, default='./output_images/RainBow_time_cost2.png')
parser.add_argument('--RainBow_epsilon_path2', type=str, default='./output_images/RainBow_epsilon2.png')
parser.add_argument('--RainBow_cost_path2', type=str, default='./output_images/RainBow_cost2.png')
parser.add_argument('--RainBow_ckpt_dir3', type=str, default='./checkpoints/RainBowDQN3/')
parser.add_argument('--RainBow_time_cost_path3', type=str, default='./output_images/RainBow_time_cost3.png')
parser.add_argument('--RainBow_epsilon_path3', type=str, default='./output_images/RainBow_epsilon3.png')
parser.add_argument('--RainBow_cost_path3', type=str, default='./output_images/RainBow_cost3.png')
parser.add_argument('--RainBow_ckpt_dir4', type=str, default='./checkpoints/RainBowDQN4/')
parser.add_argument('--RainBow_time_cost_path4', type=str, default='./output_images/RainBow_time_cost4.png')
parser.add_argument('--RainBow_epsilon_path4', type=str, default='./output_images/RainBow_epsilon4.png')
parser.add_argument('--RainBow_cost_path4', type=str, default='./output_images/RainBow_cost4.png')

parser.add_argument('--dueling_ckpt_dir1', type=str, default='./checkpoints/DuelingDQN1/')
parser.add_argument('--dueling_time_cost_path1', type=str, default='./output_images/dueling_time_cost1.png')
parser.add_argument('--dueling_epsilon_path1', type=str, default='./output_images/dueling_epsilon1.png')
parser.add_argument('--dueling_cost_path1', type=str, default='./output_images/dueling_cost1.png')
parser.add_argument('--dueling_ckpt_dir2', type=str, default='./checkpoints/DuelingDQN2/')
parser.add_argument('--dueling_time_cost_path2', type=str, default='./output_images/dueling_time_cost2.png')
parser.add_argument('--dueling_epsilon_path2', type=str, default='./output_images/dueling_epsilon2.png')
parser.add_argument('--dueling_cost_path2', type=str, default='./output_images/dueling_cost2.png')
parser.add_argument('--dueling_ckpt_dir3', type=str, default='./checkpoints/DuelingDQN3/')
parser.add_argument('--dueling_time_cost_path3', type=str, default='./output_images/dueling_time_cost3.png')
parser.add_argument('--dueling_epsilon_path3', type=str, default='./output_images/dueling_epsilon3.png')
parser.add_argument('--dueling_cost_path3', type=str, default='./output_images/dueling_cost3.png')
parser.add_argument('--dueling_ckpt_dir4', type=str, default='./checkpoints/DuelingDQN4/')
parser.add_argument('--dueling_time_cost_path4', type=str, default='./output_images/dueling_time_cost4.png')
parser.add_argument('--dueling_epsilon_path4', type=str, default='./output_images/dueling_epsilon4.png')
parser.add_argument('--dueling_cost_path4', type=str, default='./output_images/dueling_cost4.png')

parser.add_argument('--ddqn_ckpt_dir1', type=str, default='./checkpoints/DDQN1/')
parser.add_argument('--ddqn_time_cost_path1', type=str, default='./output_images/ddqn_time_cost1.png')
parser.add_argument('--ddqn_epsilon_path1', type=str, default='./output_images/ddqn_epsilon1.png')
parser.add_argument('--ddqn_cost_path1', type=str, default='./output_images/ddqn_cost1.png')
parser.add_argument('--ddqn_ckpt_dir2', type=str, default='./checkpoints/DDQN2/')
parser.add_argument('--ddqn_time_cost_path2', type=str, default='./output_images/ddqn_time_cost2.png')
parser.add_argument('--ddqn_epsilon_path2', type=str, default='./output_images/ddqn_epsilon2.png')
parser.add_argument('--ddqn_cost_path2', type=str, default='./output_images/ddqn_cost2.png')
parser.add_argument('--ddqn_ckpt_dir3', type=str, default='./checkpoints/DDQN3/')
parser.add_argument('--ddqn_time_cost_path3', type=str, default='./output_images/ddqn_time_cost3.png')
parser.add_argument('--ddqn_epsilon_path3', type=str, default='./output_images/ddqn_epsilon3.png')
parser.add_argument('--ddqn_cost_path3', type=str, default='./output_images/ddqn_cost3.png')
parser.add_argument('--ddqn_ckpt_dir4', type=str, default='./checkpoints/DDQN4/')
parser.add_argument('--ddqn_time_cost_path4', type=str, default='./output_images/ddqn_time_cost4.png')
parser.add_argument('--ddqn_epsilon_path4', type=str, default='./output_images/ddqn_epsilon4.png')
parser.add_argument('--ddqn_cost_path4', type=str, default='./output_images/ddqn_cost4.png')

parser.add_argument('--dqn_ckpt_dir1', type=str, default='./checkpoints/DQN1/')
parser.add_argument('--dqn_time_cost_path1', type=str, default='./output_images/dqn_time_cost1.png')
parser.add_argument('--dqn_epsilon_path1', type=str, default='./output_images/dqn_epsilon1.png')
parser.add_argument('--dqn_cost_path1', type=str, default='./output_images/dqn_cost1.png')
parser.add_argument('--dqn_ckpt_dir2', type=str, default='./checkpoints/DQN2/')
parser.add_argument('--dqn_time_cost_path2', type=str, default='./output_images/dqn_time_cost2.png')
parser.add_argument('--dqn_epsilon_path2', type=str, default='./output_images/dqn_epsilon2.png')
parser.add_argument('--dqn_cost_path2', type=str, default='./output_images/dqn_cost2.png')
parser.add_argument('--dqn_ckpt_dir3', type=str, default='./checkpoints/DQN3/')
parser.add_argument('--dqn_time_cost_path3', type=str, default='./output_images/dqn_time_cost3.png')
parser.add_argument('--dqn_epsilon_path3', type=str, default='./output_images/dqn_epsilon3.png')
parser.add_argument('--dqn_cost_path3', type=str, default='./output_images/dqn_cost3.png')
parser.add_argument('--dqn_ckpt_dir4', type=str, default='./checkpoints/DQN4/')
parser.add_argument('--dqn_time_cost_path4', type=str, default='./output_images/dqn_time_cost4.png')
parser.add_argument('--dqn_epsilon_path4', type=str, default='./output_images/dqn_epsilon4.png')
parser.add_argument('--dqn_cost_path4', type=str, default='./output_images/dqn_cost4.png')

args = parser.parse_args()


def test():
    pre_train('rainbow_dqn')
    off_on_line_train('rainbow_dqn')


def pre_train(net_type):
    # 规定迭代轮次
    episodes = 500
    epis = [i for i in range(episodes)]
    # 选择环境
    env = Environment_Distributed()
    envtype = "./Centralized_Model"
    step = "pre_train"
    # 选择网络
    if net_type == 'rainbow_dqn':
        agent1 = RainbowDQN(num_updates_per_step=10, state_dim=env.feature, action_dim=4, user_num=v.n, hidden_dim=10,
                            batch_size=128, ckpt_dir=args.RainBow_ckpt_dir1, epsilon=1, eps_end=0.01, eps_dec=0.999)
        agent2 = RainbowDQN(num_updates_per_step=10, state_dim=env.feature, action_dim=4, user_num=v.n, hidden_dim=10,
                            batch_size=64, ckpt_dir=args.RainBow_ckpt_dir1, epsilon=1, eps_end=0.2, eps_dec=0.9992)
        agent3 = RainbowDQN(num_updates_per_step=10, state_dim=env.feature, action_dim=4, user_num=v.n, hidden_dim=10,
                            batch_size=128, ckpt_dir=args.RainBow_ckpt_dir1, epsilon=1, eps_end=0.1, eps_dec=0.999)
        agent4 = RainbowDQN(num_updates_per_step=10, state_dim=env.feature, action_dim=4, user_num=v.n, hidden_dim=10,
                            batch_size=128, ckpt_dir=args.RainBow_ckpt_dir1, epsilon=1, eps_end=0.2, eps_dec=0.9995)
        # 定义保存的文件路径
        nettype = "RainbowDQN"
    elif net_type == 'dueling_dqn':
        agent1 = DuelingDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                            fc1_dim=64, fc2_dim=64, ckpt_dir=args.dueling_ckpt_dir2, gamma=0.99, tau=0.005,
                            epsilon=1.0, eps_end=0.01, eps_dec=0.999, max_size=1000000)
        agent2 = DuelingDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=64,
                            fc1_dim=64, fc2_dim=64, ckpt_dir=args.dueling_ckpt_dir2, gamma=0.99, tau=0.005,
                            epsilon=1.0, eps_end=0.2, eps_dec=0.9992, max_size=1000000)
        agent3 = DuelingDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                            fc1_dim=64, fc2_dim=64, ckpt_dir=args.dueling_ckpt_dir2, gamma=0.99, tau=0.005,
                            epsilon=1.0, eps_end=0.1, eps_dec=0.999, max_size=1000000)
        agent4 = DuelingDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                            fc1_dim=64, fc2_dim=64, ckpt_dir=args.dueling_ckpt_dir2, gamma=0.99, tau=0.005,
                            epsilon=1.0, eps_end=0.2, eps_dec=0.9995, max_size=1000000)
        # 定义保存的文件路径
        nettype = "DuelingDQN"
    elif net_type == 'ddqn':
        agent1 = DDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                      fc1_dim=64, fc2_dim=64, ckpt_dir=args.ddqn_ckpt_dir1, gamma=0.99, tau=0.005,
                      epsilon=1.0, eps_end=0.01, eps_dec=0.999, max_size=1000000)
        agent2 = DDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=64,
                      fc1_dim=64, fc2_dim=64, ckpt_dir=args.ddqn_ckpt_dir1, gamma=0.99, tau=0.005,
                      epsilon=1.0, eps_end=0.2, eps_dec=0.9992, max_size=1000000)
        agent3 = DDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                      fc1_dim=64, fc2_dim=64, ckpt_dir=args.ddqn_ckpt_dir1, gamma=0.99, tau=0.005,
                      epsilon=1.0, eps_end=0.1, eps_dec=0.999, max_size=1000000)
        agent4 = DDQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                      fc1_dim=64, fc2_dim=64, ckpt_dir=args.ddqn_ckpt_dir1, gamma=0.99, tau=0.005,
                      epsilon=1.0, eps_end=0.2, eps_dec=0.9995, max_size=1000000)
        # 定义保存的文件路径
        nettype = "DDQN"
    elif net_type == 'dqn':
        agent1 = DQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                     fc1_dim=64, fc2_dim=64, ckpt_dir=args.dqn_ckpt_dir4, gamma=0.99, tau=0.005,
                     epsilon=1.0, eps_end=0.01, eps_dec=0.999, max_size=1000000)
        agent2 = DQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=64,
                     fc1_dim=64, fc2_dim=64, ckpt_dir=args.dqn_ckpt_dir4, gamma=0.99, tau=0.005,
                     epsilon=1.0, eps_end=0.2, eps_dec=0.9992, max_size=1000000)
        agent3 = DQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                     fc1_dim=64, fc2_dim=64, ckpt_dir=args.dqn_ckpt_dir4, gamma=0.99, tau=0.005,
                     epsilon=1.0, eps_end=0.1, eps_dec=0.999, max_size=1000000)
        agent4 = DQN(alpha=0.0005, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                     fc1_dim=64, fc2_dim=64, ckpt_dir=args.dqn_ckpt_dir4, gamma=0.99, tau=0.005,
                     epsilon=1.0, eps_end=0.2, eps_dec=0.9995, max_size=1000000)
        # 定义保存的文件路径
        nettype = "DQN"
    else:
        raise ValueError("Net type not supported")

    states, states_, actions, rewards, dones, processes = [], [], [], [], [], []
    states1, states1_, actions1, rewards1, dones1, processes1 = [], [], [], [], [], []
    states2, states2_, actions2, rewards2, dones2, processes2 = [], [], [], [], [], []
    states3, states3_, actions3, rewards3, dones3, processes3 = [], [], [], [], [], []
    states4, states4_, actions4, rewards4, dones4, processes4 = [], [], [], [], [], []

    total_rewards, avg_reward, avg_rewards, eps_history = [], [], [], []
    total_Gs, avg_G, avg_Gs = [], [], []
    total_costs1, avg_costs1, eps_history1 = [], [], []
    total_costs2, avg_costs2, eps_history2 = [], [], []
    total_costs3, avg_costs3, eps_history3 = [], [], []
    total_costs4, avg_costs4, eps_history4 = [], [], []
    process = np.zeros(episodes, dtype=int)

    for i in range(episodes):
        state, reward, done = env.reset()
        j = 0
        total_reward = np.zeros(v.n)
        total_G = 0
        count = 1
        total_cost1, total_cost2, total_cost3, total_cost4 = None, None, None, None
        while True:
            print("任务刻:",j)
            E_l_need = sum([v.W[m] - v.U * v.I_g_m for m in range(v.k, v.k + v.num_ones)])
            pre_process = int(E_l_need // sum(v.P_i)) + 1
            if j < v.k - pre_process:
                action = agent1.choose_action(state)
                state_, reward, done = env.step(j, action)
                if done[0] == 0 or j == v.m - 1 or count % 2 == 0:
                    if done[0] == 0 or j == v.m - 1:
                        total_reward += reward
                        total_G += c.G[j]
                    agent1.remember(state, action, reward, state_, done, j)
                    cost_cpt1 = agent1.learn()
                    if cost_cpt1:
                        if total_cost1 is None:
                            total_cost1 = agent1.loss.item()
                        else:
                            total_cost1 += agent1.loss.item()
                    states1.append(state)
                    states1_.append(state_)
                    actions1.append(action)
                    rewards1.append(reward)
                    dones1.append(done)
                    processes1.append(j)
            elif j < v.k:
                action = agent2.choose_action(state)
                state_, reward, done = env.step(j, action)
                if done[0] == 0 or j == v.m - 1 or count % 2 == 0:
                    if done[0] == 0 or j == v.m - 1:
                        total_reward += reward
                        total_G += c.G[j]
                    agent2.remember(state, action, reward, state_, done, j)
                    if done[0] == 0 and j >= v.k - pre_process:
                        agent2.remember(state, action, reward, state_, done, j)
                        agent2.remember(state, action, reward, state_, done, j)
                        agent2.remember(state, action, reward, state_, done, j)
                        agent2.remember(state, action, reward, state_, done, j)
                        agent2.remember(state, action, reward, state_, done, j)
                        agent2.remember(state, action, reward, state_, done, j)
                        agent2.remember(state, action, reward, state_, done, j)
                    agent2.learn()
                    agent2.learn()
                    agent2.learn()
                    agent2.learn()
                    cost_cpt2 = agent2.learn()
                    if cost_cpt2:
                        if total_cost2 is None:
                            total_cost2 = agent2.loss.item()
                        else:
                            total_cost2 += agent2.loss.item()
                    states2.append(state)
                    states2_.append(state_)
                    actions2.append(action)
                    rewards2.append(reward)
                    dones2.append(done)
                    processes2.append(j)
            elif v.k <= j < v.k + v.num_ones:
                action = agent3.choose_action(state)
                state_, reward, done = env.step(j, action)
                if done[0] == 0 or j == v.m - 1 or count % 2 == 0:
                    if done[0] == 0 or j == v.m - 1:
                        total_reward += reward
                        total_G += c.G[j]
                    agent3.remember(state, action, reward, state_, done, j)
                    if done[0] == 0:
                        agent3.remember(state, action, reward, state_, done, j)
                        agent3.remember(state, action, reward, state_, done, j)
                        agent3.remember(state, action, reward, state_, done, j)
                        agent3.remember(state, action, reward, state_, done, j)
                    agent3.learn()
                    agent3.learn()
                    agent3.learn()
                    agent3.learn()
                    cost_cpt3 = agent3.learn()
                    if cost_cpt3:
                        if total_cost3 is None:
                            total_cost3 = agent3.loss.item()
                        else:
                            total_cost3 += agent3.loss.item()
                    states3.append(state)
                    states3_.append(state_)
                    actions3.append(action)
                    rewards3.append(reward)
                    dones3.append(done)
                    processes3.append(j)

            elif j >= v.k + v.num_ones:
                action = agent4.choose_action(state)
                state_, reward, done = env.step(j, action)
                if done[0] == 0 or j == v.m - 1 or count % 2 == 0:
                    if done[0] == 0 or j == v.m - 1:
                        total_reward += reward
                        total_G += c.G[j]
                    agent4.remember(state, action, reward, state_, done, j)
                    cost_cpt4 = agent4.learn()
                    if cost_cpt4:
                        if total_cost4 is None:
                            total_cost4 = agent4.loss.item()
                        else:
                            total_cost4 += agent4.loss.item()
                    states4.append(state)
                    states4_.append(state_)
                    actions4.append(action)
                    rewards4.append(reward)
                    dones4.append(done)
                    processes4.append(j)

            if done[0] == 0 or j == v.m - 1:
                # agent.remember(state, action, reward, state_, done, j)
                states.append(state)
                states_.append(state_)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                processes.append(j)
                # print("action:", action)
                # print("reward:", reward)
                # # print("G_s:", c.G_s)
                # # print("G_p:", c.G_p)
                # print("state:\n", state)
                # print("I_b:\n", v.I_b_ij[:, j])
                # print("E_l:\n", c.E_l[:, j])
                # print("fi[j]:", v.fi[j])
                # print("state_:\n", state_)
                # print("done:", done)
                state = state_
                process[i] = j
                j = j + 1
                if j >= v.m or done[0] == 1:
                    break

        if total_cost1:
            total_costs1.append(total_cost1)
            avg_cost1 = np.mean(total_costs1[-100:])
            avg_costs1.append(avg_cost1)
            eps_history1.append(agent1.epsilon)
        if total_cost2:
            total_costs2.append(total_cost2)
            avg_cost2 = np.mean(total_costs2[-100:])
            avg_costs2.append(avg_cost2)
            eps_history2.append(agent2.epsilon)
        if total_cost3:
            total_costs3.append(total_cost3)
            avg_cost3 = np.mean(total_costs3[-100:])
            avg_costs3.append(avg_cost3)
            eps_history3.append(agent3.epsilon)
        if total_cost3:
            total_costs3.append(total_cost3)
            avg_cost3 = np.mean(total_costs3[-100:])
            avg_costs3.append(avg_cost3)
            eps_history3.append(agent3.epsilon)
        if total_cost4:
            total_costs4.append(total_cost4)
            avg_cost4 = np.mean(total_costs4[-100:])
            avg_costs4.append(avg_cost4)
            eps_history4.append(agent4.epsilon)
        total_Gs.append(total_G)
        avg_G = np.mean(total_Gs[-100:])
        avg_Gs.append(avg_G)
        total_rewards.append(total_reward[0])
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        print(f"第{i}轮:total_rewards:", total_rewards[i])
        print(f"第{i}轮:avg_rewards:", avg_rewards[i])
        print(f"第{i}轮:total_Gs:", total_Gs[i])
        print(f"第{i}轮:avg_Gs:", avg_Gs[i])
        print("a_s:\n", v.a_s)
        print("a_p:\n", v.a_p)
    # print(collections.Counter(process))
    # 保存训练数据
    dir_path = os.path.join(envtype, nettype, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 保存各种统计数据
    np.save(os.path.join(dir_path, "rewards.npy"),
            avg_rewards.cpu().numpy() if isinstance(avg_rewards, torch.Tensor) else avg_rewards)
    np.save(os.path.join(dir_path, "G.npy"), avg_Gs.cpu().numpy() if isinstance(avg_Gs, torch.Tensor) else avg_Gs)
    np.save(os.path.join(dir_path, "cost1.npy"),
            avg_costs1.cpu().numpy() if isinstance(avg_costs1, torch.Tensor) else avg_costs1)
    np.save(os.path.join(dir_path, "cost2.npy"),
            avg_costs2.cpu().numpy() if isinstance(avg_costs2, torch.Tensor) else avg_costs2)
    np.save(os.path.join(dir_path, "cost3.npy"),
            avg_costs3.cpu().numpy() if isinstance(avg_costs3, torch.Tensor) else avg_costs3)
    np.save(os.path.join(dir_path, "cost4.npy"),
            avg_costs4.cpu().numpy() if isinstance(avg_costs4, torch.Tensor) else avg_costs4)
    np.save(os.path.join(dir_path, "processes.npy"),
            process.cpu().numpy() if isinstance(process, torch.Tensor) else process)

    # 保存经验池数据
    np.save("./experience/states.npy", states.cpu().numpy() if isinstance(states, torch.Tensor) else states)
    np.save("./experience/states_.npy", states_.cpu().numpy() if isinstance(states_, torch.Tensor) else states_)
    np.save('./experience/actions.npy', actions)
    np.save("./experience/rewards.npy", rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards)
    np.save("./experience/dones.npy", dones.cpu().numpy() if isinstance(dones, torch.Tensor) else dones)
    np.save("./experience/processes.npy", processes.cpu().numpy() if isinstance(processes, torch.Tensor) else processes)

    # # ... 重复上述模式以处理其他变量
    # np.save("./experience/states1.npy", states1.cpu().numpy() if isinstance(states1, torch.Tensor) else states1)
    # np.save("./experience/states1_.npy", states1_.cpu().numpy() if isinstance(states1_, torch.Tensor) else states1_)
    # np.save("./experience/actions1.npy", actions1.cpu().numpy() if isinstance(actions1, torch.Tensor) else actions1)
    # np.save("./experience/rewards1.npy", rewards1.cpu().numpy() if isinstance(rewards1, torch.Tensor) else rewards1)
    # np.save("./experience/dones1.npy", dones1.cpu().numpy() if isinstance(dones1, torch.Tensor) else dones1)
    # np.save("./experience/processes1.npy", processes1.cpu().numpy() if isinstance(processes1, torch.Tensor) else processes1)
    #
    # # ... 以此类推，直到处理完所有变量
    # np.save("./experience/states2.npy", states2.cpu().numpy() if isinstance(states2, torch.Tensor) else states2)
    # np.save("./experience/states2_.npy", states2_.cpu().numpy() if isinstance(states2_, torch.Tensor) else states2_)
    # np.save("./experience/actions2.npy", actions2.cpu().numpy() if isinstance(actions2, torch.Tensor) else actions2)
    # np.save("./experience/rewards2.npy", rewards2.cpu().numpy() if isinstance(rewards2, torch.Tensor) else rewards2)
    # np.save("./experience/dones2.npy", dones2.cpu().numpy() if isinstance(dones2, torch.Tensor) else dones2)
    # np.save("./experience/processes2.npy", processes2.cpu().numpy() if isinstance(processes2, torch.Tensor) else processes2)
    #
    # np.save("./experience/states3.npy", states3.cpu().numpy() if isinstance(states3, torch.Tensor) else states3)
    # np.save("./experience/states3_.npy", states3_.cpu().numpy() if isinstance(states3_, torch.Tensor) else states3_)
    # np.save("./experience/actions3.npy", actions3.cpu().numpy() if isinstance(actions3, torch.Tensor) else actions3)
    # np.save("./experience/rewards3.npy", rewards3.cpu().numpy() if isinstance(rewards3, torch.Tensor) else rewards3)
    # np.save("./experience/dones3.npy", dones3.cpu().numpy() if isinstance(dones3, torch.Tensor) else dones3)
    # np.save("./experience/processes3.npy", processes3.cpu().numpy() if isinstance(processes3, torch.Tensor) else processes3)
    #
    # np.save("./experience/states4.npy", states4.cpu().numpy() if isinstance(states4, torch.Tensor) else states4)
    # np.save("./experience/states4_.npy", states4_.cpu().numpy() if isinstance(states4_, torch.Tensor) else states4_)
    # print(actions)
    # print(rewards)
    # print(dones)
    # print(states)
    # print(states_)

    # 创建NumPy数组
    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    eps4 = np.arange(1, len(avg_costs4) + 1)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(epis, process)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Pre_train Process')
    plt.xlabel('Episodes')
    plt.ylabel('Process')
    plt.savefig(dir_path + '/Pre_train Process.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='DQN3', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.plot(eps4, avg_costs4, label='DQN4', color=(0.8, 0.2, 0.2), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Pre_train Average cost')
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.savefig(dir_path + '/Pre_train Average cost.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epis, avg_Gs, label='avg_G', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Pre_train Average G')
    plt.xlabel('Episodes')
    plt.ylabel('G')
    plt.savefig(dir_path + '/Pre_train Average G.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(epis, total_Gs, label='G', color=(0.8, 0.6, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Pre_train Total G')
    plt.xlabel('Episodes')
    plt.ylabel('G')
    plt.savefig(dir_path + '/Pre_train Total G.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epis, avg_rewards, label='avg_reward', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Pre_train Average reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(dir_path + '/Pre_train Average reward.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(epis, total_rewards, label='reward', color=(0.8, 0.6, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Pre_train Total reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(dir_path + '/Pre_train Total reward.png', dpi=300, bbox_inches='tight')
    plt.show()


def off_on_line_train(net_type):
    env = Environment_Distributed()
    if net_type == 'rainbow_dqn':
        agent = RainbowDQN(num_updates_per_step=10, state_dim=env.feature, action_dim=4, user_num=v.n, hidden_dim=10,
                           batch_size=128, ckpt_dir=args.RainBow_ckpt_dir1, epsilon=1, eps_end=0.01, eps_dec=0.999)
        nettype = "RainbowDQN"
    elif net_type == 'dueling_dqn':
        agent = DuelingDQN(alpha=0.0001, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                           fc1_dim=64, fc2_dim=64, ckpt_dir=args.dueling_ckpt_dir2, gamma=0.99, tau=0.005,
                           epsilon=1.0, eps_end=0.01, eps_dec=0.999, max_size=1000000)
        # 定义保存的文件路径
        nettype = "DuelingDQN"
    elif net_type == 'ddqn':
        agent = DDQN(alpha=0.0001, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                     fc1_dim=64, fc2_dim=64, ckpt_dir=args.ddqn_ckpt_dir1, gamma=0.99, tau=0.005,
                     epsilon=1.0, eps_end=0.01, eps_dec=0.999, max_size=1000000)
        # 定义保存的文件路径
        nettype = "DDQN"
    elif net_type == 'dqn':
        agent = DQN(alpha=0.0001, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=128,
                    fc1_dim=64, fc2_dim=64, ckpt_dir=args.dqn_ckpt_dir4, gamma=0.99, tau=0.005,
                    epsilon=1.0, eps_end=0.01, eps_dec=0.999, max_size=1000000)
        # 定义保存的文件路径
        nettype = "DQN"
    else:
        raise ValueError("Net type not supported")

    actions = np.load("./experience/actions.npy", allow_pickle=True)
    rewards = np.load("./experience/rewards.npy", allow_pickle=True)
    dones = np.load("./experience/dones.npy", allow_pickle=True)
    states = np.load("./experience/states.npy", allow_pickle=True)
    states_ = np.load("./experience/states_.npy", allow_pickle=True)
    processes = np.load("./experience/processes.npy", allow_pickle=True)

    # global states, states_, actions, rewards, dones, processes
    # states_np = [states[i] for i in range(len(states))]
    # states__np = [states_[i] for i in range(len(states_))]
    # actions_np = [actions[i] for i in range(len(actions))]
    # rewards_np = [rewards[i] for i in range(len(rewards))]
    # dones_np = [dones[i] for i in range(len(dones))]
    # processes_np = [processes[i] for i in range(len(processes))]

    offline_buffer = 5000
    actions = actions[-offline_buffer:]
    rewards = rewards[-offline_buffer:]
    dones = dones[-offline_buffer:]
    states = states[-offline_buffer:]
    states_ = states_[-offline_buffer:]
    processes = processes[-offline_buffer:]

    # 离线训练
    print("开始离线训练")
    episodes = len(actions)

    total_costs, avg_costs, eps_history = [], [], []

    for i in range(episodes):
        total_cost = None
        agent.remember(states[i, :], actions[i], rewards[i, :], states_[i, :], dones[i, :], processes[i])
        cost_cpt = agent.learn()
        if cost_cpt:
            if total_cost is None:
                total_cost = agent.loss.item()
            else:
                total_cost += agent.loss.item()
        if total_cost:
            total_costs.append(total_cost)
            avg_cost = np.mean(total_costs[-100:])
            avg_costs.append(avg_cost)
            eps_history.append(i)

    # 定义保存的文件路径
    envtype = "./Centralized_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, nettype, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(os.path.join(dir_path, "cost.npy"),
            avg_costs.cpu().numpy() if isinstance(avg_costs, torch.Tensor) else avg_costs)

    epis = np.arange(1, len(avg_costs) + 1)
    plt.plot(epis, avg_costs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Offline_train Average cost')
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.savefig(dir_path + '/Offline_train Average cost.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    print("开始在线训练")
    online_episodes = 300
    epis = [i for i in range(online_episodes)]
    total_rewards, avg_reward, avg_rewards, eps_history = [], [], [], []
    total_Gs, avg_G, avg_Gs = [], [], []
    total_costs, avg_costs, eps_history = [], [], []
    process = np.zeros(online_episodes, dtype=int)
    for i in range(online_episodes):
        state, reward, done = env.reset()
        j = 0
        total_reward = np.zeros(v.n)
        total_G = 0
        total_cost = None
        while True:
            print("任务刻:",j)
            action = agent.choose_action(state)
            state_, reward, done = env.step(j, action)
            if done[0] == 0 or j == v.m - 1:
                total_reward += reward
                total_G += c.G[j]
                agent.remember(state, action, reward, state_, done, j)
                cost_cpt = agent.learn()
                if cost_cpt:
                    if total_cost is None:
                        total_cost = agent.loss.item()
                    else:
                        total_cost += agent.loss.item()
                # print("action:\n", action)
                # print("reward:\n", reward)
                # print("state:\n", state)
                # print("I_b:\n", v.I_b_ij[:, j])
                # print("E_l:\n", c.E_l[:, j])
                # print("fi[j]:\n", v.fi[j])
                # print("state_:\n", state_)
                # print("done:\n", done)
                state = state_
                process[i] = j
                j = j + 1
                if j >= v.m or done[0] == 1:
                    break
        print("a_s:\n", v.a_s)
        print("a_p:\n", v.a_p)
        total_Gs.append(total_G)
        avg_G = np.mean(total_Gs[-100:])
        avg_Gs.append(avg_G)
        total_rewards.append(total_reward[0])
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        if total_cost:
            total_costs.append(total_cost)
            avg_cost = np.mean(total_costs[-100:])
            avg_costs.append(avg_cost)
            eps_history.append(agent.epsilon)
        print(f"第{i}轮:total_costs:", total_costs[i])
        print(f"第{i}轮:avg_costs:", avg_costs[i])
        print(f"第{i}轮:total_rewards:", total_rewards[i])
        print(f"第{i}轮:avg_rewards:", avg_rewards[i])
        print(f"第{i}轮:total_Gs:", total_Gs[i])
        print(f"第{i}轮:avg_Gs:", avg_Gs[i])
    # print(collections.Counter(process))

    # 定义保存的文件路径
    envtype = "./Centralized_Model"
    step = "online_train"
    dir_path = os.path.join(envtype, nettype, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.save(os.path.join(dir_path, "rewards.npy"),
            avg_rewards.cpu().numpy() if isinstance(avg_rewards, torch.Tensor) else avg_rewards)
    np.save(os.path.join(dir_path, "cost.npy"),
            avg_costs.cpu().numpy() if isinstance(avg_costs, torch.Tensor) else avg_costs)
    np.save(os.path.join(dir_path, "G.npy"), avg_Gs.cpu().numpy() if isinstance(avg_Gs, torch.Tensor) else avg_Gs)
    np.save(os.path.join(dir_path, "processes.npy"),
            process.cpu().numpy() if isinstance(process, torch.Tensor) else process)
    plt.figure(figsize=(10, 6))
    plt.plot(epis, process)
    plt.title('Online_train Process')
    plt.xlabel('Episodes')
    plt.ylabel('Process')
    plt.savefig(dir_path + '/Online_train Process.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(epis, avg_rewards, label='avg_reward', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Online_train Average reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(dir_path + '/Online_train Average reward.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(epis, total_rewards, label='reward', color=(0.8, 0.6, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Online_train Total reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(dir_path + '/Online_train Total reward.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(epis, avg_costs, label='cost', color=(0.8, 0.6, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Online_train Average Cost')
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.savefig(dir_path + '/Online_train Average Cost.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epis, avg_Gs, label='avg_G', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Online_train Average G')
    plt.xlabel('Episodes')
    plt.ylabel('G')
    plt.savefig(dir_path + '/Online_train Average G.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(epis, total_Gs, label='G', color=(0.8, 0.6, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Online_train Total G')
    plt.xlabel('Episodes')
    plt.ylabel('G')
    plt.savefig(dir_path + '/Online_train Total G.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_train():
    episodes = 20
    epis = [i for i in range(episodes)]
    env = Environment_Distributed()
    # agent = RainbowDQN(num_updates_per_step=10, state_dim=env.feature, action_dim=4, user_num=v.n,
    #                    hidden_dim=10,
    #                    batch_size=32, ckpt_dir=args.RainBow_ckpt_dir1)
    agent = DuelingDQN(alpha=0.0001, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=10,
                       fc1_dim=64, fc2_dim=64, ckpt_dir=args.dueling_ckpt_dir2, gamma=0.99, tau=0.005,
                       epsilon=1.0, eps_end=0.05, eps_dec=5e-4, max_size=1000000)
    agent = DDQN(alpha=0.0001, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=10,
                 fc1_dim=64, fc2_dim=64, ckpt_dir=args.ddqn_ckpt_dir1, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.05, eps_dec=5e-4, max_size=1000000)
    agent = DQN(alpha=0.0001, state_dim=env.feature, action_dim=4, user_num=v.n, batch_size=10,
                fc1_dim=64, fc2_dim=64, ckpt_dir=args.dqn_ckpt_dir4, gamma=0.99, tau=0.005,
                epsilon=1.0, eps_end=0.05, eps_dec=5e-4, max_size=1000000)
    total_rewards, avg_reward, avg_rewards, eps_history = [], [], [], []
    process = np.zeros(episodes, dtype=int)
    for i in range(episodes):
        print(f"\n第{i}轮:\n")
        state, reward, done = env.reset()
        # print("state:\n", state)
        # print("reward:\n", reward)
        # print("done:\n", done)
        j = 0
        total_reward = np.zeros(v.n)
        while True:
            print("任务刻:", j)
            print()
            action = agent.choose_action(state)
            state_, reward, done = env.step(j, action)
            if done == 0 and j != v.m - 1:
                total_reward += reward
                agent.remember(state, action, reward, state_, done, j)
                agent.learn()
                # print("action:\n", action)
                # print("reward:\n", reward)
                # print("state:\n", state)
                # print("I_b:\n", v.I_b_ij[:, j])
                # print("E_l:\n", c.E_l[:, j])
                # print("fi[j]:\n", v.fi[j])
                # print("state_:\n", state_)
                # print("done:\n", done)
                state = state_
                process[i] = j
                j = j + 1
                if j >= v.m or done[0] == 1:
                    break
        print("a_s:\n", v.a_s)
        print("a_p:\n", v.a_p)
        total_rewards.append(total_reward[0])
        avg_reward = np.mean(total_rewards[-10:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
    # print(collections.Counter(process))
    eps_total = [i for i in range(len(total_rewards))]
    eps_avg = [i for i in range(len(avg_rewards))]
    plt.plot(epis, process)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(eps_total, avg_rewards, label='avg_reward', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Average reward all')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(eps_avg, total_rewards, label='reward', color=(0.8, 0.6, 0.0), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Total reward all')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

def show_data():
    actions = np.load("./experience/actions.npy", allow_pickle=True)
    rewards = np.load("./experience/rewards.npy", allow_pickle=True)
    dones = np.load("./experience/dones.npy", allow_pickle=True)
    states = np.load("./experience/states.npy", allow_pickle=True)
    states_ = np.load("./experience/states_.npy", allow_pickle=True)
    processes = np.load("./experience/processes.npy", allow_pickle=True)
    actions1 = np.load("./experience/actions1.npy", allow_pickle=True)
    rewards1 = np.load("./experience/rewards1.npy", allow_pickle=True)
    dones1 = np.load("./experience/dones1.npy", allow_pickle=True)
    states1 = np.load("./experience/states1.npy", allow_pickle=True)
    states1_ = np.load("./experience/states1_.npy", allow_pickle=True)
    processes1 = np.load("./experience/processes1.npy", allow_pickle=True)
    actions2 = np.load("./experience/actions2.npy", allow_pickle=True)
    rewards2 = np.load("./experience/rewards2.npy", allow_pickle=True)
    dones2 = np.load("./experience/dones2.npy", allow_pickle=True)
    states2 = np.load("./experience/states2.npy", allow_pickle=True)
    states2_ = np.load("./experience/states2_.npy", allow_pickle=True)
    processes2 = np.load("./experience/processes2.npy", allow_pickle=True)
    actions3 = np.load("./experience/actions3.npy", allow_pickle=True)
    rewards3 = np.load("./experience/rewards3.npy", allow_pickle=True)
    dones3 = np.load("./experience/dones3.npy", allow_pickle=True)
    states3 = np.load("./experience/states3.npy", allow_pickle=True)
    states3_ = np.load("./experience/states3_.npy", allow_pickle=True)
    processes3 = np.load("./experience/processes4.npy", allow_pickle=True)
    actions4 = np.load("./experience/actions4.npy", allow_pickle=True)
    rewards4 = np.load("./experience/rewards4.npy", allow_pickle=True)
    dones4 = np.load("./experience/dones4.npy", allow_pickle=True)
    states4 = np.load("./experience/states4.npy", allow_pickle=True)
    states4_ = np.load("./experience/states4_.npy", allow_pickle=True)
    processes4 = np.load("./experience/processes4.npy", allow_pickle=True)

    episodes = len(processes2)
    epis1 = [i for i in range(len(processes1))]
    epis2 = [i for i in range(len(processes2))]
    epis3 = [i for i in range(len(processes3))]
    epis4 = [i for i in range(len(processes4))]

    print("总经验数:", len(processes))
    # print("total_precoess:\n", processes)
    print("DQN1经验数:", len(processes1))
    print("total_precoess1:\n", processes1)
    print("DQN2经验数:", len(processes2))
    print("total_precoess2:\n", processes2)
    print("DQN3经验数:", len(processes3))
    print("total_precoess3:\n", processes3)
    print("DQN4经验数:", len(processes4))
    print("total_precoess4:\n", processes4)

    print()
    print("action:\n", actions2)
    print("reward:\n", rewards2)
    print("state:\n", states2)
    print("state_:\n", states2_)
    print("done:\n", dones2)



if __name__ == '__main__':
    test()

