import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_DQNs():
    envtype = "./Distributed_Model"
    Parameter = "DQNs"
    # 预训练
    step = "pre_train"
    print(step+'\n')
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(2, 1)
    type = ['Reward', 'G']
    filename = ['rewards.npy', 'G.npy']
    nettype = ['RainBowDQN', 'DuelingDQN', 'DDQN', 'DQN']
    for i in range(2):
        avg_value1 = np.load(os.path.join(envtype, 'RainBowDQN', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'DuelingDQN', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'DDQN', step, filename[i]))
        avg_value4 = np.load(os.path.join(envtype, 'DQN', step, filename[i]))
        avg_values = []
        avg_values.extend([avg_value1, avg_value2, avg_value3, avg_value4])
        for j in range(4):
            check_converged(avg_values[j], nettype[j])

        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        avg_result4 = avg_value4[-result_episodes:].mean()

        print(type[i]+"\n")
        print("差值：", f"{avg_result1-avg_result2:.2f}", "下降百分比：", f"{(avg_result1-avg_result2)/avg_result2*100:.2f}%")
        print("差值：", f"{avg_result1-avg_result3:.2f}", "下降百分比：", f"{(avg_result1-avg_result3)/avg_result3*100:.2f}%")
        print("差值：", f"{avg_result1-avg_result4:.2f}", "下降百分比：", f"{(avg_result1-avg_result4)/avg_result4*100:.2f}%")

        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        eps4 = np.arange(1, len(avg_value4) + 1)
        axs[i].plot(eps1, avg_value1, label='RainBowDQN', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='DuelingDQN', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='DoubleDQN', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_value4, label='DQN', color=(0.8, 0.2, 0.2), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Reward_G.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Loss
    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(2, 2)
    nettype = ['RainBowDQN', 'DuelingDQN', 'DDQN', 'DQN']
    for i in range(2):
        for j in range(2):
            avg_costs1 = np.load(os.path.join(envtype, nettype[i*2+j], step, 'cost1.npy'))
            avg_costs2 = np.load(os.path.join(envtype, nettype[i*2+j], step, 'cost2.npy'))
            avg_costs3 = np.load(os.path.join(envtype, nettype[i*2+j], step, 'cost3.npy'))
            avg_costs4 = np.load(os.path.join(envtype, nettype[i*2+j], step, 'cost4.npy'))
            eps1 = np.arange(1, len(avg_costs1) + 1)
            eps2 = np.arange(1, len(avg_costs2) + 1)
            eps3 = np.arange(1, len(avg_costs3) + 1)
            eps4 = np.arange(1, len(avg_costs4) + 1)
            axs[i, j].plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
            # axs[i, j].plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
            axs[i, j].plot(eps3, avg_costs3, label='DQN2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
            axs[i, j].plot(eps4, avg_costs4, label='DQN3', color=(0.0, 0.4, 0.8), linestyle='-', linewidth=2)
            axs[i, j].set_title('Pre_train' + ' ' + nettype[i*2+j] + ' ' + 'Loss')
            axs[i, j].set_xlabel('Episodes')
            axs[i, j].set_ylabel('Loss')
            axs[i, j].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Loss.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 离线训练
    envtype = "./Distributed_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    avg_costs1 = np.load(os.path.join(envtype, 'RainBowDQN', step, 'cost.npy'))
    avg_costs2 = np.load(os.path.join(envtype, 'DuelingDQN', step, 'cost.npy'))
    avg_costs3 = np.load(os.path.join(envtype, 'DDQN', step, 'cost.npy'))
    avg_costs4 = np.load(os.path.join(envtype, 'DQN', step, 'cost.npy'))

    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    eps4 = np.arange(1, len(avg_costs4) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='RainBowDQN', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='DuelingDQN', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='DoubleDQN', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.plot(eps4, avg_costs4, label='DQN', color=(0.8, 0.2, 0.2), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Offline_train Loss', fontsize=18)
    plt.xlabel('Episodes', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/Offline_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    envtype = "./Distributed_Model"
    step = "online_train"
    print(step+'\n')
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['Reward', 'G', 'Loss']
    filename = ['rewards.npy', 'G.npy', 'cost.npy']
    nettype = ['RainBowDQN', 'DuelingDQN', 'DDQN', 'DQN']

    for i in range(3):
        avg_value1 = np.load(os.path.join(envtype, 'RainBowDQN', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'DuelingDQN', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'DDQN', step, filename[i]))
        avg_value4 = np.load(os.path.join(envtype, 'DQN', step, filename[i]))
        avg_values = []
        avg_values.extend([avg_value1, avg_value2, avg_value3, avg_value4])
        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        avg_result4 = avg_value4[-result_episodes:].mean()
        for j in range(4):
            check_converged(avg_values[j], nettype[j])
        print(type[i] + "\n")
        print("差值：", f"{avg_result1-avg_result2:.2f}", "下降百分比：",
              f"{(avg_result1 - avg_result2) / avg_result2 * 100:.2f}%")
        print("差值：", f"{avg_result1-avg_result3:.2f}", "下降百分比：",
              f"{(avg_result1 - avg_result3) / avg_result3 * 100:.2f}%")
        print("差值：", f"{avg_result1-avg_result4:.2f}", "下降百分比：",
              f"{(avg_result1 - avg_result4) / avg_result4 * 100:.2f}%")
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        eps4 = np.arange(1, len(avg_value4) + 1)
        axs[i].plot(eps1, avg_value1, label='RainBowDQN', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='DuelingDQN', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='DoubleDQN', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_value4, label='DQN', color=(0.8, 0.2, 0.2), linestyle='-', linewidth=2)
        axs[i].set_title('Online_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Online_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_process():
    envtype = "./Distributed_Model"
    Parameter = "process"
    # 预训练
    step = "pre_train"
    print(step+'\n')
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(2, 1)
    type = ['Reward', 'G']
    filename = ['rewards.npy', 'G.npy']
    parametertype = ['process_12', 'process_18', 'process_24']
    for i in range(2):
        avg_value1 = np.load(os.path.join(envtype, 'process_12', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'process_18', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'process_24', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        avg_values = []
        avg_values.extend([avg_value1, avg_value2, avg_value3])
        for j in range(len(parametertype)):
            check_converged(avg_values[j], parametertype[j])

        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        print(type[i] + "\n")
        print('具体值', f"{avg_result1:.2f}")
        print('具体值', f"{avg_result2:.2f}")
        print('具体值', f"{avg_result3:.2f}")

        axs[i].plot(eps1, avg_value1, label='process_12', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='process_18', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='process_24', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Reward_G.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Loss
    plt.figure(figsize=(100, 30))
    type = ['process_12', 'process_18', 'process_24']
    fig, axs = plt.subplots(len(type), 1)
    for i in range(len(type)):
        avg_costs1 = np.load(os.path.join(envtype, type[i], step, 'cost1.npy'))
        avg_costs2 = np.load(os.path.join(envtype, type[i], step, 'cost2.npy'))
        avg_costs3 = np.load(os.path.join(envtype, type[i], step, 'cost3.npy'))
        avg_costs4 = np.load(os.path.join(envtype, type[i], step, 'cost4.npy'))
        eps1 = np.arange(1, len(avg_costs1) + 1)
        eps2 = np.arange(1, len(avg_costs2) + 1)
        eps3 = np.arange(1, len(avg_costs3) + 1)
        eps4 = np.arange(1, len(avg_costs4) + 1)
        axs[i].plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        # axs[i].plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_costs3, label='DQN2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_costs4, label='DQN3', color=(0.0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i] + ' ' + 'Loss')
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Loss.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 离线训练
    envtype = "./Distributed_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    avg_costs1 = np.load(os.path.join(envtype, 'process_12', step, 'cost.npy'))
    avg_costs2 = np.load(os.path.join(envtype, 'process_18', step, 'cost.npy'))
    avg_costs3 = np.load(os.path.join(envtype, 'process_24', step, 'cost.npy'))
    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='process_12', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='process_18', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='process_24', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Offline_train Loss', fontsize=18)
    plt.xlabel('Episodes', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/Offline_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    envtype = "./Distributed_Model"
    step = "online_train"
    print(step+"\n")
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['Reward', 'G', 'Loss']
    filename = ['rewards.npy', 'G.npy', 'cost.npy']
    parametertype = ['process_12', 'process_18', 'process_24']
    for i in range(3):
        avg_value1 = np.load(os.path.join(envtype, 'process_12', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'process_18', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'process_24', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        avg_values = []
        avg_values.extend([avg_value1, avg_value2, avg_value3])
        for j in range(len(parametertype)):
            check_converged(avg_values[j], parametertype[j])

        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        print(type[i] + "\n")
        print('具体值', f"{avg_result1:.2f}")
        print('具体值', f"{avg_result2:.2f}")
        print('具体值', f"{avg_result3:.2f}")
        axs[i].plot(eps1, avg_value1, label='process_12', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='process_18', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='process_24', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Online_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Online_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_C():
    envtype = "./Distributed_Model"
    Parameter = "C"
    # 预训练
    step = "pre_train"
    print(step+'\n')
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(2, 1)
    type = ['Reward', 'G']
    filename = ['rewards.npy', 'G.npy']
    for i in range(2):
        avg_value1 = np.load(os.path.join(envtype, 'C_3000', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'C_3500', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'C_4000', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)

        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        print(type[i] + "\n")
        print('具体值', f"{avg_result1:.2f}")
        print('具体值', f"{avg_result2:.2f}")
        print('具体值', f"{avg_result3:.2f}")

        axs[i].plot(eps1, avg_value1, label='C_3000', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='C_3500', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='C_4000', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Reward_G.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Loss
    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['C_3000', 'C_3500', 'C_4000']
    for i in range(3):
        avg_costs1 = np.load(os.path.join(envtype, type[i], step, 'cost1.npy'))
        avg_costs2 = np.load(os.path.join(envtype, type[i], step, 'cost2.npy'))
        avg_costs3 = np.load(os.path.join(envtype, type[i], step, 'cost3.npy'))
        avg_costs4 = np.load(os.path.join(envtype, type[i], step, 'cost4.npy'))
        eps1 = np.arange(1, len(avg_costs1) + 1)
        eps2 = np.arange(1, len(avg_costs2) + 1)
        eps3 = np.arange(1, len(avg_costs3) + 1)
        eps4 = np.arange(1, len(avg_costs4) + 1)
        axs[i].plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        # axs[i].plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_costs3, label='DQN2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_costs4, label='DQN3', color=(0.0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i] + ' ' + 'Loss', fontsize=12)
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Loss.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 离线训练
    envtype = "./Distributed_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    avg_costs1 = np.load(os.path.join(envtype, 'C_3000', step, 'cost.npy'))
    avg_costs2 = np.load(os.path.join(envtype, 'C_3500', step, 'cost.npy'))
    avg_costs3 = np.load(os.path.join(envtype, 'C_4000', step, 'cost.npy'))
    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='C_3000', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='C_3500', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='C_4000', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Offline_train Loss', fontsize=18)
    plt.xlabel('Episodes', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/Offline_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    envtype = "./Distributed_Model"
    step = "online_train"
    print(step+'\n')
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['Reward', 'G', 'Loss']
    filename = ['rewards.npy', 'G.npy', 'cost.npy']
    parametertype = ['C_3000', 'C_3500', 'C_4000']
    for i in range(3):
        avg_value1 = np.load(os.path.join(envtype, 'C_3000', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'C_3500', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'C_4000', step, filename[i]))
        # avg_value1 = avg_value1[:200]
        # avg_value2 = avg_value2[:200]
        # avg_value3 = avg_value3[:200]
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        avg_values = []
        avg_values.extend([avg_value1, avg_value2, avg_value3])
        for j in range(len(parametertype)):
            check_converged(avg_values[j], parametertype[j])

        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        print(type[i] + "\n")
        print('具体值', f"{avg_result1:.2f}")
        print('具体值', f"{avg_result2:.2f}")
        print('具体值', f"{avg_result3:.2f}")

        axs[i].plot(eps1, avg_value1, label='C_3000', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='C_3500', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='C_4000', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Online_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Online_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_peak():
    envtype = "./Distributed_Model"
    Parameter = "peak"
    # 预训练
    step = "pre_train"
    print(step+'\n')
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(2, 1)
    type = ['Reward', 'G']
    filename = ['rewards.npy', 'G.npy']
    for i in range(2):
        avg_value1 = np.load(os.path.join(envtype, 'peak_0', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'peak_1', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'peak_2', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        print(type[i] + "\n")
        print('具体值', f"{avg_result1:.2f}")
        print('具体值', f"{avg_result2:.2f}")
        print('具体值', f"{avg_result3:.2f}")

        axs[i].plot(eps1, avg_value1, label='peak_0', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='peak_1', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='peak_2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Reward_G.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Loss
    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['peak_0', 'peak_1', 'peak_2']
    for i in range(3):
        avg_costs1 = np.load(os.path.join(envtype, type[i], step, 'cost1.npy'))
        avg_costs2 = np.load(os.path.join(envtype, type[i], step, 'cost2.npy'))
        avg_costs3 = np.load(os.path.join(envtype, type[i], step, 'cost3.npy'))
        avg_costs4 = np.load(os.path.join(envtype, type[i], step, 'cost4.npy'))
        eps1 = np.arange(1, len(avg_costs1) + 1)
        eps2 = np.arange(1, len(avg_costs2) + 1)
        eps3 = np.arange(1, len(avg_costs3) + 1)
        eps4 = np.arange(1, len(avg_costs4) + 1)
        axs[i].plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        # axs[i].plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_costs3, label='DQN2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_costs4, label='DQN3', color=(0.0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i] + ' ' + 'Loss')
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Loss.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 离线训练
    envtype = "./Distributed_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    avg_costs1 = np.load(os.path.join(envtype, 'peak_0', step, 'cost.npy'))
    avg_costs2 = np.load(os.path.join(envtype, 'peak_1', step, 'cost.npy'))
    avg_costs3 = np.load(os.path.join(envtype, 'peak_2', step, 'cost.npy'))
    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='peak_0', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='peak_1', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='peak_2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Offline_train Loss', fontsize=18)
    plt.xlabel('Episodes', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/Offline_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    envtype = "./Distributed_Model"
    step = "online_train"
    print(step+"\n")
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['Reward', 'G', 'Loss']
    filename = ['rewards.npy', 'G.npy', 'cost.npy']
    parametertype = ['peak_0', 'peak_1', 'peak_2']
    for i in range(3):
        avg_value1 = np.load(os.path.join(envtype, 'peak_0', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'peak_1', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'peak_2', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        avg_values = []
        avg_values.extend([avg_value1, avg_value2, avg_value3])
        for j in range(len(parametertype)):
            check_converged(avg_values[j], parametertype[j])

        result_episodes = 100
        avg_result1 = avg_value1[-result_episodes:].mean()
        avg_result2 = avg_value2[-result_episodes:].mean()
        avg_result3 = avg_value3[-result_episodes:].mean()
        print(type[i] + "\n")
        print('具体值', f"{avg_result1:.2f}")
        print('具体值', f"{avg_result2:.2f}")
        print('具体值', f"{avg_result3:.2f}")

        axs[i].plot(eps1, avg_value1, label='peak_0', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='peak_1', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='peak_2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Online_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Online_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_process_3D():
    envtype = "./Distributed_Model"
    Parameter = "process"
    # 预训练
    step = "pre_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 创建图形和子图
    fig = plt.figure(figsize=(10, 6))
    axs = [fig.add_subplot(2, 1, i + 1, projection='3d') for i in range(2)]

    type = ['Reward', 'G']
    filename = ['rewards.npy', 'G.npy']
    processes = ['process_12', 'process_18', 'process_24']

    for i in range(2):
        # 初始化z轴数据
        z_data = []
        for process in processes:
            # 加载数据
            avg_value = np.load(os.path.join(envtype, process, step, filename[i]))
            z_data.append(avg_value)

        # 初始化x轴数据（episodes）
        eps = np.arange(1, len(z_data[0]) + 1)

        # 初始化y轴数据（不同的进程）
        y_data = [np.full_like(eps, j) for j in range(len(processes))]

        # 绘制三维线图
        for j, avg_value in enumerate(z_data):
            axs[i].plot(eps, y_data[j], avg_value, label=processes[j],
                        color=plt.cm.viridis(j / len(processes)), linestyle='-', linewidth=2)

        axs[i].set_title('Pre_train ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Processes')
        axs[i].set_zlabel(type[i])
        axs[i].legend()
        axs[i].grid(True)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Reward_G.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Loss
    plt.figure(figsize=(100, 30))
    type = ['process_12', 'process_18', 'process_24']
    fig, axs = plt.subplots(len(type), 1)
    for i in range(len(type)):
        avg_costs1 = np.load(os.path.join(envtype, type[i], step, 'cost1.npy'))
        avg_costs2 = np.load(os.path.join(envtype, type[i], step, 'cost2.npy'))
        avg_costs3 = np.load(os.path.join(envtype, type[i], step, 'cost3.npy'))
        avg_costs4 = np.load(os.path.join(envtype, type[i], step, 'cost4.npy'))
        eps1 = np.arange(1, len(avg_costs1) + 1)
        eps2 = np.arange(1, len(avg_costs2) + 1)
        eps3 = np.arange(1, len(avg_costs3) + 1)
        eps4 = np.arange(1, len(avg_costs4) + 1)
        axs[i].plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        # axs[i].plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_costs3, label='DQN2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_costs4, label='DQN3', color=(0.0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i] + ' ' + 'Loss')
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Loss.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 离线训练
    envtype = "./Distributed_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    avg_costs1 = np.load(os.path.join(envtype, 'process_12', step, 'cost.npy'))
    avg_costs2 = np.load(os.path.join(envtype, 'process_18', step, 'cost.npy'))
    avg_costs3 = np.load(os.path.join(envtype, 'process_24', step, 'cost.npy'))
    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='process_12', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='process_18', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='process_24', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Offline_train Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.savefig(dir_path + '/Offline_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    envtype = "./Distributed_Model"
    step = "online_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['Reward', 'G', 'Loss']
    filename = ['rewards.npy', 'G.npy', 'cost.npy']
    for i in range(3):
        avg_value1 = np.load(os.path.join(envtype, 'process_12', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'process_18', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'process_24', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        axs[i].plot(eps1, avg_value1, label='process_12', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='process_18', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='process_24', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Online_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Online_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()
def plot_peak_3D():
    envtype = "./Distributed_Model"
    Parameter = "peak"
    # 预训练
    step = "pre_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 创建图形和子图
    fig = plt.figure(figsize=(10, 6))
    axs = [fig.add_subplot(2, 1, i + 1, projection='3d') for i in range(2)]

    type = ['Reward', 'G']
    filename = ['rewards.npy', 'G.npy']
    processes = ['process_12', 'process_18', 'process_24']

    for i in range(2):
        # 初始化z轴数据
        z_data = []
        for process in processes:
            # 加载数据
            avg_value = np.load(os.path.join(envtype, process, step, filename[i]))
            z_data.append(avg_value)

        # 初始化x轴数据（episodes）
        eps = np.arange(1, len(z_data[0]) + 1)

        # 初始化y轴数据（不同的进程）
        y_data = [np.full_like(eps, j) for j in range(len(processes))]

        # 绘制三维线图
        for j, avg_value in enumerate(z_data):
            axs[i].plot(eps, y_data[j], avg_value, label=processes[j],
                        color=plt.cm.viridis(j / len(processes)), linestyle='-', linewidth=2)

        axs[i].set_title('Pre_train ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Processes')
        axs[i].set_zlabel(type[i])
        axs[i].legend()
        axs[i].grid(True)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Reward_G.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Loss
    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(2, 1)
    type = ['peak_0', 'peak_1', 'peak_2']
    for i in range(2):
        avg_costs1 = np.load(os.path.join(envtype, type[i], step, 'cost1.npy'))
        avg_costs2 = np.load(os.path.join(envtype, type[i], step, 'cost2.npy'))
        avg_costs3 = np.load(os.path.join(envtype, type[i], step, 'cost3.npy'))
        avg_costs4 = np.load(os.path.join(envtype, type[i], step, 'cost4.npy'))
        eps1 = np.arange(1, len(avg_costs1) + 1)
        eps2 = np.arange(1, len(avg_costs2) + 1)
        eps3 = np.arange(1, len(avg_costs3) + 1)
        eps4 = np.arange(1, len(avg_costs4) + 1)
        axs[i].plot(eps1, avg_costs1, label='DQN1', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        # axs[i].plot(eps2, avg_costs2, label='DQN2', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_costs3, label='DQN2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].plot(eps4, avg_costs4, label='DQN3', color=(0.0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].set_title('Pre_train' + ' ' + type[i] + ' ' + 'Loss')
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Pre_train Loss.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 离线训练
    envtype = "./Distributed_Model"
    step = "offline_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    avg_costs1 = np.load(os.path.join(envtype, 'peak_0', step, 'cost.npy'))
    avg_costs2 = np.load(os.path.join(envtype, 'peak_1', step, 'cost.npy'))
    avg_costs3 = np.load(os.path.join(envtype, 'peak_2', step, 'cost.npy'))
    eps1 = np.arange(1, len(avg_costs1) + 1)
    eps2 = np.arange(1, len(avg_costs2) + 1)
    eps3 = np.arange(1, len(avg_costs3) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(eps1, avg_costs1, label='peak_0', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
    plt.plot(eps2, avg_costs2, label='peak_1', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
    plt.plot(eps3, avg_costs3, label='peak_2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Offline_train Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.savefig(dir_path + '/Offline_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 在线训练
    envtype = "./Distributed_Model"
    step = "online_train"
    dir_path = os.path.join(envtype, 'img', Parameter, step)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.figure(figsize=(100, 30))
    fig, axs = plt.subplots(3, 1)
    type = ['Reward', 'G', 'Loss']
    filename = ['rewards.npy', 'G.npy', 'cost.npy']
    for i in range(3):
        avg_value1 = np.load(os.path.join(envtype, 'peak_0', step, filename[i]))
        avg_value2 = np.load(os.path.join(envtype, 'peak_1', step, filename[i]))
        avg_value3 = np.load(os.path.join(envtype, 'peak_2', step, filename[i]))
        eps1 = np.arange(1, len(avg_value1) + 1)
        eps2 = np.arange(1, len(avg_value2) + 1)
        eps3 = np.arange(1, len(avg_value3) + 1)
        axs[i].plot(eps1, avg_value1, label='peak_0', color=(0.9, 0.7, 0.0), linestyle='-', linewidth=2)
        axs[i].plot(eps2, avg_value2, label='peak_1', color=(0, 0.4, 0.8), linestyle='-', linewidth=2)
        axs[i].plot(eps3, avg_value3, label='peak_2', color=(0.0, 0.7, 0.3), linestyle='-', linewidth=2)
        axs[i].set_title('Online_train' + ' ' + type[i])
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel(type[i])
        axs[i].grid(True, linestyle='--', alpha=0.7)

    # 调整子图之间的间距
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_path + '/Online_train Result.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def check_converged(avg_value, type):
    threshold = 0.003
    n = 20
    window_size = 5
    # 计算相邻数据点的变化率
    change_rates = [abs(avg_value[i + 1] - avg_value[i]) / avg_value[i] for i in range(len(avg_value) - 1)]
    converged = False
    print(change_rates)
    for i in range(len(change_rates) - n + 1):
        if np.all(np.abs(change_rates[i:i + n - 1]) < threshold):
            converged = True
            print("开始收敛值", i)
            print(change_rates[i:i + n])
            break
        else:
            converged = False
    # # 计算移动平均
    # moving_averages = [sum(avg_value[i:i + window_size]) / window_size for i in range(len(avg_value) - window_size+ 1)]
    # stable = abs(moving_averages[-1] - moving_averages[-2]) < threshold
    if converged:
        print(type+"数据收敛。")
    else:
        print(type+"数据未收敛。")

# plot_process_3D()
# plot_C_3D()
# plot_peak_3D()
plot_DQNs()
# plot_process()
# plot_C()
# plot_peak()
