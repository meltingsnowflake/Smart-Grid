import random
import numpy as np  # 导入 numpy
# 设置全局随机种子
seed_value = 42
random.seed(seed_value)   # 设置 Python 标准库的随机种子
np.random.seed(seed_value) # 设置 NumPy 的随机种子
# 控制参数
m = 24  # 进程数
n = 1  # 设备数
CI = 10500  # 储电设备基础储能上限
U = 220  # 电压初始化
WI = 3000  # 总需求基础值
alpha = 10  # 电网供电时间成本
beta = 0.1*np.array([random.uniform(0.2 * alpha, 0.4 * alpha) for _ in range(n)])
sigma = 0.1*np.array([random.uniform(1, 2) for _ in range(n)])
P_i = np.random.randint(600, 900, size=n)  # 生成 2 到 4（含）的随机整数矩阵

# k 晨昏划分点
base_value = int(m * 3 / 5)  # m 的 5 分之 3
fluctuation = int(base_value * 0.1)  # 浮动的上下限
k = random.randint(base_value - fluctuation, base_value + fluctuation)
store_enable = [1 if i < k else 0 for i in range(m)]

a_s = np.zeros((n, m), dtype=int)
a_p = np.zeros((n, m), dtype=int)

M = np.array(list(range(1, m + 1)))  # 进程列表

# 初始化 fi
fi = np.zeros(m, dtype=int)  # 创建大小为 m 的全 0 一维矩阵
num_ones = int(m * 0.1)  # 计算需要置为 1 的数量
start_idx = k  # 从 k 开始
# shift = random.randint(0, num_ones)  # 位移 0 到 m * 0.1 的随机值
shift = 0
new_start_idx = min(start_idx + shift, m - num_ones)  # 确保不会超出数组边界
fi[new_start_idx: new_start_idx + num_ones] = 1

# 初始化 储能上限C 和 剩余电量E
C = np.array([random.uniform(CI * 0.9, CI * 1.1) for _ in range(n)])  # 随机上下浮动 10%
E = np.zeros(n)  # 全为 0 的一维矩阵

# 需求分配矩阵
W_ij = np.random.uniform(WI * 0.9, WI * 1.1, size=(n, m))
for j in range(m):
    if fi[j] == 1:
        for i in range(n):
            W_ij[i, j] = random.uniform(1.1, 1.2) * WI

# 总需求数组
W = np.zeros(m, dtype=int)
for j in range(m):
    temp_w = 0
    for i in range(n):
        temp_w += W_ij[i, j]
    W[j] = temp_w

# 生成 I_b_ij 和 I_g_ij 矩阵
I_b_m = np.random.uniform(2, 4, size=n)
I_g_m = 15  # 电网供给最大电流
I_b_j = np.zeros(m)
I_g_j = np.zeros(m)
I_b_ij = np.zeros((n, m))  # 储电设备供给电量，全 0 矩阵
I_g_ij = np.zeros((n, m))  # 电网供给电流，全 0 矩阵

# 测试输出
print("m(任务数):", m)
print("n(设备数):", n)
print("k(晨昏分界线):", k)
print("M (任务数):\n", M)
print("fi (高峰期判断):\n", fi)
print("store_enable (储能判断):\n", store_enable)
print("C (储能上限):\n", C)
print("E (剩余电量):\n", E)
print("W (总需求):\n", W)
print("W_ij (需求分配矩阵):\n", W_ij)
print("I_b_ij (储电设备供给电量):\n", I_b_ij)
print("I_g_ij (电网供给电流):\n", I_g_ij)
print("alpha (电网供电时间成本):\n", alpha)
print("beta (储电设备供电时间成本):\n", beta)
print("I_b_m (最大储电设备供给电量):\n", I_b_m)
print("I_g_m (最大电网供给电流):\n", I_g_m)
print("P_i (储电设备产电效率):\n", P_i)
print("sigma (储电成本):\n", sigma)
