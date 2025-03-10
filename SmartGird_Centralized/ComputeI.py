import math
import random

import VariableInitialization as v
import CalculateElectricity as c
import numpy as np

random.seed()  # 重新设置随机种子，使用系统时间或其他源生成种子

dif = 1000
flag_f = 1000
flag = flag_f
count = 0
v.a_s = np.random.randint(0, 2, size=(v.n, v.m))  # 随机生成 0 和 1 的矩阵
v.a_p = np.random.randint(0, 2, size=(v.n, v.m))  # 随机生成 0 和 1 的矩阵


def compute_i(j):
    if v.W[j] <= v.U * v.I_g_m:  # 非高峰期
        NormalHeuristics(j)
        if flag != flag_f:
            if flag == -1:
                return -1
            else:
                return flag
    else:
        SpecialHeuristics(j)
        if flag != flag_f:
            if flag == -1:
                return -1
            else:
                return flag


def Normalinit_I(j, valid_indices):
    # print("非高峰期")
    global count
    for i in range(v.n):
        v.I_b_ij[i, j] = 0
    for i in valid_indices:
        v.I_b_ij[i, j] = min(v.I_b_m[i] - count, (c.E_l_temp[i, j]) / v.U) / 2
    count = (v.W[j] / v.U) / dif


def NormalHeuristics(j):
    global flag
    temp_goal = 10000
    e = 0.1 ** 25
    alpha = 0.90
    T = 1
    markov = 1
    accept = 0
    rand_accept = 0
    refuse = 0
    jump1 = 0
    jump2 = 0
    I_list = np.arange(v.n)

    # 根据 v.a_p[i,j] 过滤掉那些为零的 i
    valid_indices = I_list[v.a_p[I_list, j] != 0]  # 只保留 a_p[i,j] != 0 的 i
    Normalinit_I(j, valid_indices)

    # print("v.I_b_m", v.I_b_m)
    valid_indices = np.append(valid_indices, v.n)  # 将 v.n 添加到 valid_indices
    # print("valid_indices", valid_indices)
    # 判断 valid_indices 是否为空或只剩一个有效设备
    if valid_indices.size == 1:
        c.compute_g(j)
        # print("无有效设备")
        return
    while T > e:
        for t in range(markov):
            c.compute_g(j)
            # print("I_b_ij1 ", v.I_b_ij[0, j])
            # print("I_b_ij2 ", v.I_b_ij[1, j])
            # print("I_b_ij3 ", v.I_b_ij[2, j])
            # print("count", count)
            # print("E_l1 ", c.E_l[0, j])
            # print("E_l2 ", c.E_l[1, j])
            # print("E_l3 ", c.E_l[2, j])
            # print("E_l_temp1 ", c.E_l_temp[0, j])
            # print("E_l_temp2 ", c.E_l_temp[1, j])
            # print("E_l_temp3 ", c.E_l_temp[2, j])
            # print("I_g_j ", v.I_g_j[j])
            # 临时列表，用来分别存储符合条件的设备索引
            valid_temp1 = []  # 存储 I_b_ij >= count 的设备
            valid_temp2 = []  # 存储 I_b_ij <= v.I_b_m - count 的设备

            # 循环检查每个设备是否符合条件
            for i in valid_indices[:-1]:
                if v.I_b_ij[i, j] >= count:
                    valid_temp1.append(i)  # 加入 valid_temp1
                if v.I_b_ij[i, j] <= v.I_b_m[i] - count and v.I_b_ij[i, j] < c.E_l_temp[i, j] / v.U - count:
                    valid_temp2.append(i)  # 加入 valid_temp2
            if v.I_g_j[j] >= count:
                valid_temp1.append(valid_indices[-1])  # 加入 valid_temp1
            if v.I_g_j[j] <= v.I_g_m - count:
                valid_temp2.append(valid_indices[-1])  # 加入 valid_temp2
            # print("valid_temp1 ", valid_temp1)
            # print("valid_temp2 ", valid_temp2)
            # 检查 valid_temp1 和 valid_temp2 是否满足条件，必须各自至少有一个设备
            if len(valid_temp1) == 0 or len(valid_temp2) == 0:
                jump1 += 1
                continue  # 如果任何一个列表为空，跳过当前迭代

            # 从 valid_temp1 中随机选择 temp1
            temp1 = np.random.choice(valid_temp1)

            # 从 valid_temp2 中剔除与 temp1 相同的值后随机选择 temp2
            valid_temp2 = [i for i in valid_temp2 if i != temp1]  # 剔除与 temp1 相同的值
            if len(valid_temp2) == 0:  # 如果 valid_temp2 为空，跳过当前迭代
                jump2 += 1
                continue
            temp2 = np.random.choice(valid_temp2)

            # 更新 I_b_ij 的值
            if temp1 != v.n:
                v.I_b_ij[temp1, j] -= count
            else:
                v.I_g_j[j] -= count

            if temp2 != v.n:
                v.I_b_ij[temp2, j] += count
            else:
                v.I_g_j[j] += count

            c.compute_g(j)
            goal = c.G[j]
            if goal < temp_goal:
                temp_goal = goal
                goal = 0
                accept += 1
            else:
                # print("temp_goal", temp_goal)
                # print("goal", goal)
                # print("T", T)
                # print("exp ", math.exp((temp_goal / (temp_goal + goal) - goal / (temp_goal + goal)) / T))
                if math.exp((temp_goal / (temp_goal + goal) - goal / (temp_goal + goal)) / T) >= random.random():
                    temp_goal = goal
                    goal = 0
                    rand_accept += 1
                else:
                    # 恢复操作
                    if temp1 != v.n:
                        v.I_b_ij[temp1, j] += count
                    else:
                        v.I_g_j[j] += count
                    if temp2 != v.n:
                        v.I_b_ij[temp2, j] -= count
                    else:
                        v.I_g_j[j] -= count
                    goal = 0
                    refuse += 1

        T *= alpha

    # 打印结果（可选）
    # print("accept %d" % accept)
    # print("rand_accept %d" % rand_accept)
    # print("refuse %d" % refuse)
    # print("jump1 %d" % jump1)
    # print("jump2 %d" % jump2)


def Specialinit_I(j, removed_indices, valid_indices):
    global count
    temp_I_g = v.I_g_m
    num = 0
    for i in removed_indices:
        num += 1
        temp_I_g -= v.W_ij[i, j] / v.U
    count = (v.W[j] / v.U - temp_I_g) / dif
    for i in valid_indices:
        v.I_b_ij[i, j] = min(v.I_b_m[i] - count, c.E_l_temp[i, j] / v.U)
        temp_I_g -= v.W_ij[i, j] / v.U - v.I_b_ij[i, j]
    while temp_I_g > count:
        for i in valid_indices:
            if v.I_b_ij[i, j] > count:
                v.I_b_ij[i, j] -= count
                temp_I_g -= count
    # print("电流初始化分配完毕",temp_I_g)


def SpecialHeuristics(j):
    # print("高峰期")
    global flag
    temp_goal = 100000
    e = 0.1 ** 25
    alpha = 0.95
    T = 1
    markov = 1
    accept = 0
    rand_accept = 0
    refuse = 0
    jump1 = 0
    jump2 = 0
    I_list = np.arange(v.n)

    # print("v.I_b_m", v.I_b_m)
    # 根据 v.a_p[i,j] 过滤掉那些为零的 i
    valid_indices = I_list[v.a_p[I_list, j] != 0]  # 只保留 a_p[i,j] != 0 的 i
    # print("valid_indices", valid_indices)
    # 找到 a_p 中为 0 的位置（即需要剔除的元素的索引）
    removed_indices = I_list[v.a_p[I_list, j] == 0]
    Specialinit_I(j, removed_indices, valid_indices)
    # 判断 valid_indices 是否为空或只剩一个元素
    if valid_indices.size == 1:
        # print("只剩下一个有效设备，直接选择该设备")
        for i in range(v.n):
            v.I_b_ij[i, j] = 0
        v.I_b_ij[valid_indices[0], j] = v.W[j] / v.U - v.I_g_m
        c.compute_g(j)
        flag = valid_indices[0]
        return
    # print("count %f" % count)
    while T > e:
        for t in range(markov):
            c.compute_g(j)
            # print("I_b_ij1 ", v.I_b_ij[0, j])
            # print("I_b_ij2 ", v.I_b_ij[1, j])
            # print("I_b_ij3 ", v.I_b_ij[2, j])
            # 临时列表，用来分别存储符合条件的设备索引
            valid_temp1 = []  # 存储 I_b_ij >= count 的设备
            valid_temp2 = []  # 存储 I_b_ij <= v.I_b_m - count 的设备

            # 循环检查每个设备是否符合条件
            for i in valid_indices:
                if v.I_b_ij[i, j] >= count:
                    valid_temp1.append(i)  # 加入 valid_temp1
                if v.I_b_ij[i, j] <= v.I_b_m[i] - count and v.I_b_ij[i, j] < c.E_l_temp[i, j] / v.U - count:
                    valid_temp2.append(i)  # 加入 valid_temp2
            # 检查 valid_temp1 和 valid_temp2 是否满足条件，必须各自至少有一个设备
            # print("valid_temp1 ", valid_temp1)
            # print("valid_temp2 ", valid_temp2)
            if len(valid_temp1) == 0 or len(valid_temp2) == 0:
                jump1 += 1
                continue  # 如果任何一个列表为空，跳过当前迭代

            # 从 valid_temp1 中随机选择 temp1
            temp1 = np.random.choice(valid_temp1)

            # 从 valid_temp2 中剔除与 temp1 相同的值后随机选择 temp2
            valid_temp2 = [i for i in valid_temp2 if i != temp1]  # 剔除与 temp1 相同的值
            if len(valid_temp2) == 0:  # 如果 valid_temp2 为空，跳过当前迭代
                jump2 += 1
                continue
            temp2 = np.random.choice(valid_temp2)

            # 更新 I_b_ij 的值
            v.I_b_ij[temp1, j] -= count
            v.I_b_ij[temp2, j] += count

            c.compute_g(j)
            goal = c.G[j]
            if goal < temp_goal:
                temp_goal = goal
                goal = 0
                accept += 1
            else:
                # print("temp_goal ", temp_goal)
                # print("goal ", goal)
                # print("exp ", math.exp((temp_goal / (temp_goal + goal) - goal / (temp_goal + goal)) / T))
                if math.exp((temp_goal / (temp_goal + goal) - goal / (temp_goal + goal)) / T) >= random.random():
                    temp_goal = goal
                    goal = 0
                    rand_accept += 1
                else:
                    # 恢复操作
                    v.I_b_ij[temp1, j] += count
                    v.I_b_ij[temp2, j] -= count
                    goal = 0
                    refuse += 1

        T *= alpha
    # 打印结果（可选）
    # print("accept %d" % accept)
    # print("rand_accept %d" % rand_accept)
    # print("refuse %d" % refuse)
    # print("jump1 %d" % jump1)
    # print("jump2 %d" % jump2)
