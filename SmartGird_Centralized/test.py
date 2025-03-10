import numpy as np
import torch
import random

# a = [[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]]
# b = [[[1,2],[4,5],[3,6]],[[1,2],[4,5],[3,6]]]
# a = torch.tensor(a)
# b = torch.tensor(b)
# print(a.shape)
# print(b.shape)
# c = torch.matmul(a,b)
# print(c)
# print(c.shape)
#
# a = [[0, 0, 0], [0, 0, 0]]
# b = np.zeros((2, 3))
# b[1, 0] = 1
# print(b[1, 0])
# print(sum([b[i, 0] for i in range(2)]) == 0)

# a = np.random.randn(3)
# print(a)
# random.seed(11)
# for i in range(5):
#     a = np.random.randn(3)
#     print(a)

# import torch
# flag = torch.cuda.is_available()
# if flag:
#     print("CUDA可使用")
# else:
#     print("CUDA不可用")
#
# ngpu= 1
# # Decide which device we want to run on
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# print("驱动为：",device)
# print("GPU型号： ",torch.cuda.get_device_name(0))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建三条二维折线的数据
x1 = np.linspace(0, 10, 100)
z1 = np.sin(x1)  # 第一条折线

x2 = np.linspace(0, 10, 100)
z2 = np.cos(x2)  # 第二条折线

x3 = np.linspace(0, 10, 100)
z3 = np.sin(x3) + np.cos(x3)  # 第三条折线

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制第一条折线（y=0）
ax.plot(x1, np.zeros_like(x1), z1, label='Line 1', color='r')

# 绘制第二条折线（y=1）
ax.plot(x2, np.ones_like(x2), z2, label='Line 2', color='g')

# 绘制第三条折线（y=2）
ax.plot(x3, 2 * np.ones_like(x3), z3, label='Line 3', color='b')

# 设置坐标轴标签
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 设置 y 轴刻度（可选）
ax.set_yticks([0, 1, 2])

# 添加图例
ax.legend()

# 显示图形
plt.show()