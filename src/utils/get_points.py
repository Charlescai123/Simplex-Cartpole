import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import copy
import math

n_points_per_dim = 5

# # ======================= 2D cases =================#
# P_matrix_2 = np.array([[2.86418998, 3.82536023],
#                        [3.82536023, 10.2753182]])
#
#
# dim = 2
#
# eigen_values, eigen_vectors = np.linalg.eig(P_matrix_2)  # get eigen value and eigen vector
# print("eigen_values", eigen_values)
# D = np.diag(eigen_values)
#
# Q = eigen_vectors
#
# s0_list = []
# s1_list = []
#
# for i in range(n_points_per_dim):
#     angle = i * (2 * math.pi) / n_points_per_dim
#     y0 = math.sqrt(1 / eigen_values[0]) * math.cos(angle)
#     y1 = math.sqrt(1 / eigen_values[1]) * math.sin(angle)
#
#     s = Q @ np.array([y0, y1]).transpose()
#     s0_list.append(s[0])
#     s1_list.append(s[1])
#     result = s.transpose() @ P_matrix_2 @ s
#     # print("result should be 1", result)
#
#
# fig1 = plt.figure()
# p3 = plt.scatter(s0_list, s1_list, c='blue', s=8)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel(r'$dim 1$', fontsize=20)
# plt.ylabel(r"${dim 2}$", fontsize=20)
# plt.show()
# fig1.savefig(f'2d_case.png', bbox_inches='tight')


# ======================= 3D cases =================#

P_matrix_3 = np.array([[5.28012284, 2.26058234, 5.55120678],
                       [2.26058234, 1.13503516, 2.82913169],
                       [5.55120678, 2.82913169, 7.3854151]]) * 3

# P_matrix_3 = np.array([[2.86418998, 1.13537668, 3.82536023],
#                        [1.13537668, 0.81986347, 2.62785224],
#                        [3.82536023, 2.62785224, 10.27531829]])

# P_matrix_3 = np.eye(3)
# print(P_matrix_3)
dim = 3

eigen_values, eigen_vectors = np.linalg.eig(P_matrix_3)  # get eigen value and eigen vector

D = np.diag(eigen_vectors)

Q = eigen_vectors

s0_list = []
s1_list = []
s2_list = []

for i in range(n_points_per_dim):
    angle_1 = i * (2 * math.pi) / n_points_per_dim
    y0 = math.sqrt(1 / eigen_values[0]) * math.cos(angle_1)
    vector_in_2d = math.sin(angle_1)

    for j in range(n_points_per_dim):
        angle_2 = j * (2 * math.pi) / n_points_per_dim
        y1 = vector_in_2d * math.sqrt(1 / eigen_values[1]) * math.cos(angle_2)
        y2 = vector_in_2d * math.sqrt(1 / eigen_values[2]) * math.sin(angle_2)

        s = Q @ np.array([y0, y1, y2]).transpose()
        s0_list.append(s[0])
        s1_list.append(s[1])
        s2_list.append(s[2])
        result = s.transpose() @ P_matrix_3 @ s
        # print("result should be 1", result)

fig = plt.figure(figsize=(4, 4))
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
ax = fig.add_subplot(projection='3d')
ax.scatter(s0_list, s1_list, s2_list, marker='.')
ax.set_xlabel('$S_1$', fontsize=15)
ax.set_ylabel('$S_2$', fontsize=15)
ax.set_zlabel('$S_3$', fontsize=15)
plt.show()
# fig.savefig(f'3d_case.png', bbox_inches='tight')

# # ======================= 4D cases =================#
# P_matrix_4 = np.array([[2.86418998, 1.13537668, 3.82536023, 0.81212514],
#                        [1.13537668, 0.81986347, 2.62785224, 0.62859456],
#                        [3.82536023, 2.62785224, 10.27531829, 2.22805237],
#                        [0.81212514, 0.62859456, 2.22805237, 0.67150232]])
#
# eigen_values, eigen_vectors = np.linalg.eig(P_matrix_4)  # get eigen value and eigen vector
#
# D = np.diag(eigen_vectors)
#
# Q = eigen_vectors
#
# s0_list = []
# s1_list = []
# s2_list = []
# s3_list = []
#
# for i in range(n_points_per_dim):
#     angle_1 = i * (2 * math.pi) / n_points_per_dim
#     y0 = math.sqrt(1 / eigen_values[0]) * math.cos(angle_1)
#     vector_in_3d = math.sin(angle_1)
#
#     for k in range(n_points_per_dim):
#         angle_2 = k * (2 * math.pi) / n_points_per_dim
#         y1 = vector_in_3d * math.sqrt(1 / eigen_values[1]) * math.cos(angle_2)
#         vector_in_2d = vector_in_3d * math.sin(angle_2)
#
#         for j in range(n_points_per_dim):
#             angle_3 = j * (2 * math.pi) / n_points_per_dim
#             y2 = vector_in_2d * math.sqrt(1 / eigen_values[2]) * math.cos(angle_3)
#             y3 = vector_in_2d * math.sqrt(1 / eigen_values[3]) * math.sin(angle_3)
#
#             s = Q @ np.array([y0, y1, y2, y3]).transpose()
#             s0_list.append(s[0])
#             s1_list.append(s[1])
#             s2_list.append(s[2])
#             s3_list.append(s[3])
#             result = s.transpose() @ P_matrix_4 @ s
#             print("result should be 1", result)
