import numpy as np

import CalculateElectricity as c
import VariableInitialization as v
import ComputeI as C


# class Environment_Centralized:
#
#     def __init__(self):
#         self.name = '集中式环境'
#         self.feature = 7
#         self.E_l_need = 0
#         self.epsilon = 0.01
#         self.maxG = 15000
#         self.n = 1
#         self.state = np.zeros((self.n, v.m, self.feature), dtype=float)
#         self.reward = np.zeros((self.n, v.m), dtype=float)
#         self.done = np.zeros((self.n, v.m), dtype=int)
#
#     def reset(self):
#         self.state = np.zeros((self.n, v.m, self.feature), dtype=float)
#         self.reward = np.zeros((self.n, v.m), dtype=float)
#         self.done = np.zeros((self.n, v.m), dtype=int)
#         v.a_s = np.zeros((self.n, v.m), dtype=int)
#         v.a_p = np.zeros((self.n, v.m), dtype=int)
#         v.I_b_ij = np.zeros((self.n, v.m))
#         c.compute_I_b(0)
#         c.compute_I_g(0)
#         return self.state[:, 0, :], self.reward[:, 0], self.done[:, 0]
#
#     def step(self, j, action):
#         # 每步动作前的环境计算
#         self.action_decoding(action, j)
#         # c.compute_I_b(j)
#         # c.compute_I_g(j)
#         c.compute_e_l(j)
#         # 对当前环境的判断
#
#         # 启发式前的环境检查
#         if self.state_check(j):
#             for i in range(self.n):
#                 self.state_transfer(i, j)
#             print("未通过环境检查")
#             pass
#         else:
#             print("v.I_b_ij1:\n", v.I_b_ij[:, j])
#             print("E_l1:\n", c.E_l[:, j])
#             C.compute_i(j)
#             print("v.I_b_ij2:\n", v.I_b_ij[:, j])
#             print("E_l2:\n", c.E_l[:, j])
#
#             # 对所有设备进行状态转移
#             for i in range(self.n):
#                 # print("I_b_ij:\n", v.I_b_ij[i, j])
#                 # print("I_b_m:\n", v.I_b_m[i])
#                 # print("W_ij:\n", v.W_ij[i, j])
#                 # print("W_ij/U - I_g_ij:\n", v.W_ij[i, j] / v.U - v.I_g_ij[i, j])
#                 # print("E_l:\n", c.E_l[i, j])
#                 print("C_i:\n", v.C[i])
#                 print("G:", c.G[j])
#                 # print("G_m:\n", max(c.G))
#                 print("不够电:", v.I_b_ij[i, j] < (v.W_ij[i, j] * (1 - self.epsilon) / v.U) - v.I_g_ij[i, j])
#                 print("超额:", v.I_b_ij[i, j] > (v.W_ij[i, j] * (1 + self.epsilon)) / v.U)
#                 print("I_g_ij", v.I_g_ij[:, j])
#                 # print("超限:\n", v.I_b_ij[i, j] > v.I_b_m[i])
#                 # print("电流小于0", v.I_b_ij[i, j] < 0.0)
#                 # print("超容量:\n", c.E_l[i, j] > v.C[i])
#                 # print("没电:\n", c.E_l[i, j] < 0)
#
#                 # if (v.I_b_ij[i, j] > min(v.I_b_m[i], v.W_ij[i, j] / v.U) or v.I_b_ij[i, j] < 0.0
#                 #     or v.I_b_ij[i, j] < v.W_ij[i, j] / v.U - v.I_g_ij[i, j]
#                 #     or c.E_l[i, j] < 0 or c.E_l[i, j] > v.C[i]):
#                 #     self.done[:, j] = 1
#                 #     self.reward[:, j] = -2000
#                 #     break
#                 # else:
#                 #     self.reward[i, j] = (max(c.G) - c.G[j]) / 100
#                 # 动作a_s=0 a_p=0的状态转移
#                 if v.a_s[i, j] == 0 and v.a_p[i, j] == 0:
#                     self.reward[i, j] = (self.maxG - c.G[j]) / 100
#                 # 动作a_s=0 a_p=1的状态转移
#                 elif v.a_s[i, j] == 0 and v.a_p[i, j] == 1:
#                     if (v.I_b_ij[i, j] > v.I_b_m[i] * (1 + self.epsilon) or v.I_b_ij[i, j] > (v.W_ij[i, j] * (
#                             1 + self.epsilon) / v.U)
#                             or v.I_b_ij[i, j] * (1 - self.epsilon) < 0.0
#                             or v.I_b_ij[i, j] < (v.W_ij[i, j] * (1 - self.epsilon) / v.U) - v.I_g_ij[i, j]
#                             or c.E_l[i, j] * (1 - self.epsilon) < 0 or c.E_l[i, j] > v.C[i] * (1 + self.epsilon)):
#                         self.done[:, j] = 1
#                         self.reward[:, j] = -5000
#                         break
#                     else:
#                         self.reward[i, j] = (self.maxG - c.G[j]) / 100
#                 # 动作a_s=1 a_p=0的状态转移
#                 elif v.a_s[i, j] == 1 and v.a_p[i, j] == 0:
#                     if c.E_l[i, j] > v.C[i]:
#                         self.done[:, j] = 1
#                         self.reward[:, j] = -5000
#                         break
#                     else:
#                         self.reward[i, j] = (self.maxG - c.G[j]) / 100
#                 # 动作a_s=1 a_p=1的状态转移
#                 elif v.a_s[i, j] == 1 and v.a_p[i, j] == 1:
#                     if (v.I_b_ij[i, j] > v.I_b_m[i] * (1 + self.epsilon) or v.I_b_ij[i, j] > (v.W_ij[i, j] * (
#                             1 + self.epsilon) / v.U)
#                             or v.I_b_ij[i, j] < (v.W_ij[i, j] * (1 - self.epsilon) / v.U) - v.I_g_ij[i, j]
#                             or c.E_l[i, j] * (1 - self.epsilon) < 0 or c.E_l[i, j] > v.C[i] * (1 + self.epsilon)):
#                         self.done[:, j] = 1
#                         self.reward[:, j] = -5000
#                         break
#                     else:
#                         self.reward[i, j] = (self.maxG - c.G[j]) / 100
#             # 高峰期开始前
#             if j < v.k:
#                 # 检测电流储备量
#                 self.E_l_need = sum([v.W[m] - v.U * v.I_g_m for m in range(v.k, v.k + v.num_ones)])
#                 pre_process = int(self.E_l_need // sum(v.P_i)) + 1
#                 print(pre_process)
#                 for i in range(pre_process):
#                     if j == v.k - pre_process + i:
#                         E_l_owner = sum([c.E_l[m, j] for m in range(self.n)])
#                         print(j)
#                         print("E_l_owner:", E_l_owner)
#                         print("总发电量", sum(v.P_i))
#                         print("发电轮次", (pre_process - i))
#                         print("sum(v.P_i) * (pre_process - i):", sum(v.P_i) * (pre_process - i))
#                         print("E_l_need:", self.E_l_need)
#                         if E_l_owner + sum(v.P_i) * (pre_process - i - 1) < self.E_l_need:
#                             print("储备电量不足:")
#                             self.done[:, j] = 1
#                             self.reward[:, j] = -5000
#                             break
#             else:
#                 self.E_l_need = 0
#             # 进程的判断
#             self.reward[:, j] += (1 + j) * 500
#             # 1轮次结束
#             if j == v.k - 1 and self.done[0, j] != 1:
#                 self.reward[:, j] += 10000
#             # 2轮次结束
#             if j == v.k + v.num_ones - 1 and self.done[0, j] != 1:
#                 self.reward[:, j] += 20000
#             # 3轮次结束
#             if j == v.m - 1:
#                 self.reward[:, j] += 30000
#                 self.done[:, j] = 1
#             for i in range(self.n):
#                 self.state_transfer(i, j)
#         return self.state[:, j, :], self.reward[:, j], self.done[:, j]
#
#     def action_decoding(self, action, j):
#         for i in range(self.n):
#             if action[i] == 0:
#                 v.a_s[i, j] = 0
#                 v.a_p[i, j] = 0
#             elif action[i] == 1:
#                 v.a_s[i, j] = 0
#                 v.a_p[i, j] = 1
#             elif action[i] == 2:
#                 v.a_s[i, j] = 1
#                 v.a_p[i, j] = 0
#             elif action[i] == 3:
#                 v.a_s[i, j] = 1
#                 v.a_p[i, j] = 1
#
#     def state_check(self, j):
#         result = False
#         print("a_p:\n", v.a_p[:, j])
#         print("a_s:\n", v.a_s[:, j])
#
#         # 对I_b与E_l的判断
#         if (v.I_g_m + sum([v.a_p[i, j] * min(v.I_b_m[i], c.E_l[i, j] / v.U) for i in range(self.n)])) * v.U < v.W[
#             j] or c.E_l[:, j].any() < 0 \
#                 or (sum([v.a_p[i, j] for i in range(self.n)]) == 0 and v.W[j] > v.U * v.I_g_m):
#             print("c.E_l[i, j]/v.U:\n", c.E_l[:, j] / v.U)
#             print("I_b_m:\n", v.I_b_m)
#             print("W_have:\n",
#                   (v.I_g_m + sum([v.a_p[i, j] * min(v.I_b_m[i], c.E_l[i, j] / v.U) for i in range(self.n)])) * v.U)
#             print("W:\n", v.W[j])
#             print("够电:\n",
#                   (v.I_g_m + sum([v.a_p[i, j] * min(v.I_b_m[i], c.E_l[i, j] / v.U) for i in range(self.n)])) * v.U < v.W[
#                       j])
#             print("有电", c.E_l[:, j].any() < 0)
#             print("高峰期:\n", v.W[j] > v.U * v.I_g_m)
#             print("全0:\n", sum([v.a_p[i, j] for i in range(self.n)]) == 0)
#             print("高峰期全0:\n", (v.W[j] > v.U * v.I_g_m and sum([v.a_p[i, j] for i in range(self.n)]) == 0))
#             # c.E_l[:, j] = 0
#             self.done[:, j] = 1
#             if sum([v.a_p[i, j] for i in range(self.n)]) == 0 and v.W[j] > v.U * v.I_g_m:
#                 self.reward[:, j] = -10000
#             else:
#                 self.reward[:, j] = -5000
#             for i in range(self.n):
#                 self.state_transfer(i, j)
#             result = True
#
#         return result
#
#     def state_transfer(self, i, j):
#         E_pre = (self.E_l_need - sum(c.E_l[:, j])) / v.U if (self.E_l_need - sum(c.E_l[:, j])) / v.U > 0 else 0
#         self.state[i, j] = [v.I_b_ij[i, j], c.E_l[i, j] / v.U, v.W_ij[i, j] / v.U, v.fi[j] * 2, j, E_pre, v.C[i]]


class Environment_Distributed:
    def __init__(self):
        self.name = '分布式环境'
        self.feature = 7
        self.E_l_need = 0
        self.epsilon = 0.01
        self.maxG = 3000
        self.state = np.zeros((v.n, v.m, self.feature), dtype=float)
        self.reward = np.zeros((v.n, v.m), dtype=float)
        self.done = np.zeros((v.n, v.m), dtype=int)

    def reset(self):
        self.state = np.zeros((v.n, v.m, self.feature), dtype=float)
        self.reward = np.zeros((v.n, v.m), dtype=float)
        self.done = np.zeros((v.n, v.m), dtype=int)
        v.a_s = np.zeros((v.n, v.m), dtype=int)
        v.a_p = np.zeros((v.n, v.m), dtype=int)
        v.I_b_ij = np.zeros((v.n, v.m))
        c.compute_I_b(0)
        c.compute_I_g(0)
        return self.state[:, 0, :], self.reward[:, 0], self.done[:, 0]

    def step(self, j, action):
        global pre_process
        self.done[:, j] = 0
        # 每步动作前的环境计算
        self.action_decoding(action, j)
        # c.compute_I_b(j)
        # c.compute_I_g(j)
        c.compute_e_l(j)
        # 对当前环境的判断

        # 启发式前的环境检查
        if self.state_check(j):
            for i in range(v.n):
                self.state_transfer(i, j)
            print("未通过环境检查")
            pass
        else:
            # print("v.I_b_ij1:\n", v.I_b_ij[:, j])
            # print("E_l1:\n", c.E_l[:, j])
            C.compute_i(j)
            # print("v.I_b_ij2:\n", v.I_b_ij[:, j])
            # print("E_l2:\n", c.E_l[:, j])

            # 对所有设备进行状态转移
            for i in range(v.n):
                # print("I_b_ij:\n", v.I_b_ij[i, j])
                # print("I_b_m:\n", v.I_b_m[i])
                # print("W_ij:\n", v.W_ij[i, j])
                # print("W_ij/U - I_g_ij:\n", v.W_ij[i, j] / v.U - v.I_g_ij[i, j])
                # print("E_l:\n", c.E_l[i, j])
                # print("C_i:\n", v.C[i])
                # print("beta:\n", v.beta)
                # print("sigma:\n", v.sigma)
                # print("G:", c.G[j])
                # print("G_p:", c.G_p[j])
                # print("G_s:", c.G_s[j])
                # print("G_m:\n", max(c.G))
                # print("不够电:", v.I_b_ij[i, j] < (v.W_ij[i, j]*(1 - self.epsilon) / v.U) - v.I_g_ij[i, j])
                # print("超额:", v.I_b_ij[i, j] > (v.W_ij[i, j]*(1+self.epsilon))/v.U)
                # print("I_g_ij", v.I_g_ij[:,j])
                # print("超限:\n", v.I_b_ij[i, j] > v.I_b_m[i])
                # print("电流小于0", v.I_b_ij[i, j] < 0.0)
                # print("超容量:\n", c.E_l[i, j] > v.C[i])
                # print("没电:\n", c.E_l[i, j] < 0)

                # if (v.I_b_ij[i, j] > min(v.I_b_m[i], v.W_ij[i, j] / v.U) or v.I_b_ij[i, j] < 0.0
                #     or v.I_b_ij[i, j] < v.W_ij[i, j] / v.U - v.I_g_ij[i, j]
                #     or c.E_l[i, j] < 0 or c.E_l[i, j] > v.C[i]):
                #     self.done[:, j] = 1
                #     self.reward[:, j] = -2000
                #     break
                # else:
                #     self.reward[i, j] = (max(c.G) - c.G[j]) / 100
                # 动作a_s=0 a_p=0的状态转移
                if v.a_s[i, j] == 0 and v.a_p[i, j] == 0:
                    self.reward[i, j] = (self.maxG - c.G[j]) / 100
                # 动作a_s=0 a_p=1的状态转移
                elif v.a_s[i, j] == 0 and v.a_p[i, j] == 1:
                    if (v.I_b_ij[i, j] > v.I_b_m[i] * (1 + self.epsilon) or v.I_b_ij[i, j] > (v.W_ij[i, j] * (
                            1 + self.epsilon) / v.U)
                            or v.I_b_ij[i, j] * (1 - self.epsilon) < 0.0
                            or v.I_b_ij[i, j] < (v.W_ij[i, j] * (1 - self.epsilon) / v.U) - v.I_g_ij[i, j]
                            or c.E_l[i, j] * (1 - self.epsilon) < 0 or c.E_l[i, j] > v.C[i] * (1 + self.epsilon)):
                        self.done[:, j] = 1
                        self.reward[:, j] = -5
                        break
                    else:
                        self.reward[i, j] = (self.maxG - c.G[j]) / 100
                # 动作a_s=1 a_p=0的状态转移
                elif v.a_s[i, j] == 1 and v.a_p[i, j] == 0:
                    if c.E_l[i, j] > v.C[i]:
                        self.done[:, j] = 1
                        self.reward[:, j] = -5
                        break
                    else:
                        self.reward[i, j] = (self.maxG - c.G[j]) / 100
                # 动作a_s=1 a_p=1的状态转移
                elif v.a_s[i, j] == 1 and v.a_p[i, j] == 1:
                    if (v.I_b_ij[i, j] > v.I_b_m[i] * (1 + self.epsilon) or v.I_b_ij[i, j] > (v.W_ij[i, j] * (
                            1 + self.epsilon) / v.U)
                            or v.I_b_ij[i, j] < (v.W_ij[i, j] * (1 - self.epsilon) / v.U) - v.I_g_ij[i, j]
                            or c.E_l[i, j] * (1 - self.epsilon) < 0 or c.E_l[i, j] > v.C[i] * (1 + self.epsilon)):
                        self.done[:, j] = 1
                        self.reward[:, j] = -5
                        break
                    else:
                        self.reward[i, j] = (self.maxG - c.G[j]) / 100
            # 高峰期开始前
            if j < v.k:
                # 检测电流储备量
                self.E_l_need = sum([v.W[m] - v.U * v.I_g_m for m in range(v.k, v.k + v.num_ones)])
                pre_process = int(self.E_l_need // sum(v.P_i)) + 1
                # print("pre_process", pre_process)
                for i in range(pre_process):
                    if j == v.k - pre_process + i:
                        E_l_owner = sum([c.E_l[m, j] for m in range(v.n)])
                        # print(j)
                        # print("E_l_owner:", E_l_owner)
                        # print("总发电量", sum(v.P_i))
                        # print("发电轮次", (pre_process - i))
                        # print("sum(v.P_i) * (pre_process - i):", sum(v.P_i) * (pre_process - i))
                        # print("E_l_need:", self.E_l_need)
                        if E_l_owner + sum(v.P_i) * (pre_process - i - 1) < self.E_l_need:
                            # print("储备电量不足:")
                            self.done[:, j] = 1
                            self.reward[:, j] = -5
                            break
            else:
                self.E_l_need = 0
            # 进程的判断
            # self.reward[:, j] += (1 + j) * 5000
            # # 1轮次结束
            # if j == v.k - 1 and self.done[0, j] != 1:
            #     self.reward[:, j] += 40000
            # # 2轮次结束
            # if j == v.k + v.num_ones - 1 and self.done[0, j] != 1:
            #     self.reward[:, j] += 50000
            # 3轮次结束
            if j == v.m - 1:
                # self.reward[:, j] += 80000
                self.done[:, j] = 1
            for i in range(v.n):
                self.state_transfer(i, j)
        return self.state[:, j, :], self.reward[:, j], self.done[:, j]

    def action_decoding(self, action, j):
        for i in range(v.n):
            if action[i] == 0:
                v.a_s[i, j] = 0
                v.a_p[i, j] = 0
            elif action[i] == 1:
                v.a_s[i, j] = 0
                v.a_p[i, j] = 1
            elif action[i] == 2:
                v.a_s[i, j] = 1
                v.a_p[i, j] = 0
            elif action[i] == 3:
                v.a_s[i, j] = 1
                v.a_p[i, j] = 1

    def state_check(self, j):
        result = False
        # print("a_p:\n", v.a_p[:, j])
        # print("a_s:\n", v.a_s[:, j])

        # 对I_b与E_l的判断
        if (v.I_g_m + sum([v.a_p[i, j] * min(v.I_b_m[i], c.E_l[i, j] / v.U) for i in range(v.n)])) * v.U < v.W[
            j] * (1 - self.epsilon) or c.E_l[:, j].any() < 0 \
                or (sum([v.a_p[i, j] for i in range(v.n)]) == 0 and v.W[j] > v.U * v.I_g_m):
            # print("c.E_l[i, j]/v.U:\n", c.E_l[:, j]/v.U)
            # print("I_b_m:\n", v.I_b_m)
            # print("W_have:\n",
            #       (v.I_g_m + sum([v.a_p[i, j] * min(v.I_b_m[i], c.E_l[i, j] / v.U) for i in range(v.n)])) * v.U)
            # print("W:\n", v.W[j])
            # print("够电:\n",
            #       (v.I_g_m + sum([v.a_p[i, j] * min(v.I_b_m[i], c.E_l[i, j] / v.U) for i in range(v.n)])) * v.U < v.W[
            #           j]* (1 - self.epsilon))
            # print("有电", c.E_l[:, j].any() < 0)
            # print("高峰期:\n", v.W[j] > v.U * v.I_g_m)
            # print("全0:\n", sum([v.a_p[i, j] for i in range(v.n)]) == 0)
            # print("高峰期全0:\n", (v.W[j] > v.U * v.I_g_m and sum([v.a_p[i, j] for i in range(v.n)]) == 0))
            # c.E_l[:, j] = 0
            self.done[:, j] = 1
            self.reward[:, j] = -5
            for i in range(v.n):
                self.state_transfer(i, j)
            result = True

        return result

    def state_transfer(self, i, j):
        E_pre = (self.E_l_need - sum(c.E_l[:, j])) / v.U if (self.E_l_need - sum(c.E_l[:, j])) / v.U > 0 else 0
        self.state[i, j] = [v.I_b_ij[i, j], c.E_l[i, j] / v.U, v.W_ij[i, j] / v.U, v.fi[j] * 2, j, E_pre, v.C[i]]
