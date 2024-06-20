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
from src import DDPGAgent, DDPGParams
from src.envs import CartpoleParams, Cartpole, states2observations

# todo x_dot [-3, 3]
# todo theta_dot [-4.5, 4.5]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x_t_list = np.linspace(-0.9, 0.9, 45)
theta_t_list = np.linspace(-0.8, 0.8, 40)
trajectory_length = 150
P_matrix = np.array([[4.6074554, 1.49740096, 5.80266046, 0.99189224],
                     [1.49740096, 0.81703147, 2.61779592, 0.51179642],
                     [5.80266046, 2.61779592, 11.29182733, 1.87117709],
                     [0.99189224, 0.51179642, 1.87117709, 0.37041435]])

cP = P_matrix

tP = np.zeros((2, 2))

# For velocity
# tP[0][0] = cP[1][1]
# tP[1][1] = cP[3][3]
# tP[0][1] = cP[1][3]
# tP[1][0] = cP[1][3]

# For position
tP[0][0] = cP[0][0]
tP[1][1] = cP[2][2]
tP[0][1] = cP[0][2]
tP[1][0] = cP[0][2]

wp, vp = LA.eig(tP)

theta = np.linspace(-np.pi, np.pi, 1000)

ty1 = (np.cos(theta)) / np.sqrt(wp[0])
ty2 = (np.sin(theta)) / np.sqrt(wp[1])

ty = np.stack((ty1, ty2))
tQ = inv(vp.transpose())
# tQ = vp.transpose()
tx = np.matmul(tQ, ty)

tx1 = np.array(tx[0]).flatten()
tx2 = np.array(tx[1]).flatten()


fig1 = plt.figure()
# plt.figure(num = 3,figsize = (10,5))
plt.plot(tx1, tx2, linewidth=2, color='black')
plt.vlines(x=-0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
plt.vlines(x=0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
plt.hlines(y=-0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
plt.hlines(y=0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
plt.show()

# plt.xlim(-1.2, 1.2)
# plt.ylim(-1, 1)
#
# # Loading model
# params = read_config("./config/icml/icml_test.json")
# model_path = "models/train_sparse_best"
#
#
# agent_our = DDPGAgent(params.agent_params,
#                       params.taylor_params,
#                       shape_observations=5,
#                       shape_action=1,
#                       model_path=model_path,
#                       mode="eval")
#
# env = Cartpole(params.cartpole_params)
# trajectory = []
# tx_list = []
#
# def interact_loop(x_t, theta_t, ai_agent):
#     init_states = [x_t, 0., theta_t, 0, False]
#     # init_states = [0, 0., 0, 0, False]k
#     env.reset(init_states)
#     # trajectory
#     trajectory = []
#     tx_list = []
#     position_list = []
#     angle_list = []
#
#     for step in range(trajectory_length):
#         current_states = copy.deepcopy(env.states)
#         tx = np.matmul(np.array(current_states)[:4], P_matrix) @ np.array(current_states)[:4].transpose()
#
#         position = current_states[0]
#         angle = current_states[2]
#
#         position_list.append(position)
#         angle_list.append(angle)
#
#         tx_list.append(tx)
#         trajectory.append(current_states)
#         observations, _ = states2observations(current_states)
#         action = ai_agent.get_action(observations, mode="eval")
#         env.step(action, action_mode='residual')
#
#     tx_array = np.array(tx_list)
#     position_array = np.abs(position_list)
#     angle_array = np.abs(angle_list)
#     return tx_array, position_array, angle_array
#
# for x_t in tqdm(x_t_list):
#     for theta_t in theta_t_list:
#
#         tx_array, position_array, angle_array = interact_loop(x_t, theta_t, ai_agent=agent_our)
#
#         if len(tx_array[tx_array > 1]) == 0:
#             p3 = plt.scatter(x_t, theta_t, c='blue', s=8)
#         elif len(position_array[position_array > 0.9]) == 0 \
#                 and len(angle_array[angle_array > 0.8]) == 0:
#             p4 = plt.scatter(x_t, theta_t, c='green', s=8)
#
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.xlabel(r'$x$', fontsize=20)
# plt.ylabel(r"${\Theta}$", fontsize=20)
# fig1.savefig(f'plot/iclr_our_75k_mf.pdf', format='pdf', bbox_inches='tight')
