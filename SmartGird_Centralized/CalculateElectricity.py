import random

import VariableInitialization as v
import numpy as np  # 导入 numpy
# 设置全局随机种子
seed_value = 42
random.seed(seed_value)   # 设置 Python 标准库的随机种子
np.random.seed(seed_value) # 设置 NumPy 的随机种子

# 供电模型

# 总供电
E_p = np.zeros(v.m, dtype=int)  # 全 0 矩阵


def compute_e_p(j):
    temp_g = 0
    temp_b = 0
    for i in range(v.n):  # 遍历每个设备
        temp_g += v.I_g_ij[i, j]  # 正确的矩阵索引方式
        temp_b += v.I_b_ij[i, j]  # 正确的矩阵索引方式
    E_p[j] = (temp_g + temp_b) * v.U  # 计算总供电


# 总供电成本
G_p = np.zeros(v.m, dtype=int)  # 全 0 矩阵


def compute_g_p(j):
    temp_g = 0
    temp_b = 0
    for i in range(v.n):  # 遍历每个设备
        temp_g += v.W_ij[i, j]/v.U-v.I_b_ij[i, j] # 正确的矩阵索引方式
        temp_b += v.beta[i] * v.I_b_ij[i, j] * v.I_b_ij[i, j]  # 正确的矩阵索引方式
    G_p[j] = v.alpha * temp_g ** 2 + temp_b  # 计算总供电


# 计算储能设备电流
def compute_I_b(j):
    v.I_b_j[j] = 0
    for i in range(v.n):
        v.I_b_j[j] += v.I_b_ij[i, j]


# 计算电网供电电流
def compute_I_g(j):
    v.I_g_j[j] = 0
    for i in range(v.n):
        v.I_g_ij[i, j] = v.W_ij[i, j]/v.U-v.I_b_ij[i, j]
        v.I_g_j[j] += v.I_g_ij[i, j]


# 储电模型

# 存储电量
E_l = np.zeros((v.n, v.m))
E_l_temp = np.zeros((v.n, v.m))


def compute_e_l(j):
    temp1 = temp2 = 0
    # 按照任务 j 累计 temp1 和 temp2
    for k in range(j+1):
        temp1 += v.a_s[:, k] * v.store_enable[k] * v.P_i  # 按列取出 a_s 的第 j 列，计算 temp1
        temp2 += v.a_p[:, k] * v.U * v.I_b_ij[:, k]  # 按列计算 temp2 的第 j 列
        for i in range(v.n):  # 遍历每个设备
            E_l[i, j] = temp1[i] - temp2[i]  # 计算 E_l[i, j]
    temp1 = temp2 = 0
    # 按照任务 j 累计 temp1 和 temp2
    for k in range(j+1):
        if k == j:  # 判断是否为最后一次循环
            temp2 += np.zeros(v.n)  # 在最后一次循环时将 temp2 设为 0
        else:
            temp2 += v.a_p[:, k] * v.U * v.I_b_ij[:, k]  # 按列计算 temp2 的第 j 列

        temp1 += v.a_s[:, k] * v.store_enable[k] * v.P_i  # 按列取出 a_s 的第 j 列，计算 temp1
        for i in range(v.n):  # 遍历每个设备
            E_l_temp[i, j] = temp1[i] - temp2[i]  # 计算 E_l[i, j]
        # print("temp1 (temp1):\n", temp1)
        # print("temp2 (temp1):\n", temp2)


# 储电成本
G_s = np.zeros(v.m, dtype=int)  # 全 0 矩阵


def compute_g_s(j):
    compute_e_l(j)
    temp = 0
    for i in range(v.n):  # 遍历每个设备
        temp += v.sigma[i] * E_l[i, j]
    G_s[j] = temp


# 总成本
G = np.zeros(v.m, dtype=int)  # 全 0 矩阵


def compute_g(j):
    compute_I_g(j)
    compute_g_p(j)
    compute_g_s(j)
    G[j] = G_p[j] + G_s[j]


index = 1
# 调用函数计算
compute_e_p(index)
compute_g_p(index)
compute_I_g(index)
compute_e_l(index)
compute_g_s(index)
compute_g(index)

# 输出结果
print("I_b_ij (储电设备供给电量):\n", v.I_b_ij)
print("I_g_ij (电网供给电流):\n", v.I_g_ij)
print("W_ij(各户需求):\n", v.W_ij)
print("I_b_j(储电设备总供电电流):\n", v.I_b_j)
print("I_g_j(电网总供电电流):\n", v.I_g_j)
print("E_p (总供电):\n", E_p)
print("G_p (总供电):\n", G_p)
print("E_l (存储电量):\n", E_l)
print("G_s (储电成本):\n", G_s)
