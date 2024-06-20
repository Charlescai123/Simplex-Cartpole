import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA
from src.envs import MATRIX_P


def plot_safety_envelope(epsilon=0.6, p_mat=MATRIX_P):
    # p_mat *= 0.5
    # if p_mat is None:
    #     p_mat = self.p_mat
    cP = p_mat

    tP = np.zeros((2, 2))
    vP = np.zeros((2, 2))

    # For velocity
    vP[0][0] = cP[1][1]
    vP[1][1] = cP[3][3]
    vP[0][1] = cP[1][3]
    vP[1][0] = cP[1][3]

    # For position
    tP[0][0] = cP[0][0]
    tP[1][1] = cP[2][2]
    tP[0][1] = cP[0][2]
    tP[1][0] = cP[0][2]

    wp, vp = LA.eig(tP)
    wp_eps, vp_eps = LA.eig(tP / epsilon)
    # wp, vp = LA.eig(vP)

    theta = np.linspace(-np.pi, np.pi, 1000)

    ty1 = (np.cos(theta)) / np.sqrt(wp[0])
    ty2 = (np.sin(theta)) / np.sqrt(wp[1])

    ty1_eps = (np.cos(theta)) / np.sqrt(wp_eps[0])
    ty2_eps = (np.sin(theta)) / np.sqrt(wp_eps[1])

    ty = np.stack((ty1, ty2))
    tQ = inv(vp.transpose())
    # tQ = vp.transpose()
    tx = np.matmul(tQ, ty)

    ty_eps = np.stack((ty1_eps, ty2_eps))
    tQ_eps = inv(vp_eps.transpose())
    tx_eps = np.matmul(tQ_eps, ty_eps)

    tx1 = np.array(tx[0]).flatten()
    tx2 = np.array(tx[1]).flatten()

    tx_eps1 = np.array(tx_eps[0]).flatten()
    tx_eps2 = np.array(tx_eps[1]).flatten()

    # Safety envelope
    plt.plot(tx1, tx2, linewidth=8, color='grey', mew=2)
    # plt.plot(0, 0, 'k*', markersize=4, mew=0.6)  # global equilibrium (star)
    # plt.plot(0, 0, 'ko-', markersize=7, mew=1, markerfacecolor='none')  # global equilibrium (circle)

    # HAC switch envelope
    # if self.simplex_enable:
    #     plt.plot(tx_eps1, tx_eps2, 'k--', linewidth=0.8, label=r"$\partial\Omega_{HAC}$")

    # HPC switch envelope
    # plt.plot(tx_hpc1, tx_hpc2, 'b--', linewidth=0.8, label=r"$\partial\Omega_{HPC}$")


def plot_inference_phase(trajectories, action_modes, eq_point, plot_eq=False):
    print(f"len traj: {len(trajectories)}")
    print(f"len action: {len(action_modes)}")
    assert len(trajectories) == len(action_modes)

    # eq points
    if plot_eq and eq_point is not None:
        print(f"eq point: {eq_point}")
        plt.plot(eq_point[0], eq_point[2], '*', color=[0.4660, 0.6740, 0.1880], markersize=8)

    for i in range(len(trajectories) - 1):
        if action_modes[i] == "model" or action_modes[i] == "residual":
            plt.plot(trajectories[i][0], trajectories[i][1], '.', color=[0, 0.4470, 0.7410],
                     markersize=2)  # model trajectory
        elif action_modes[i] == "simplex":
            plt.plot(trajectories[i][0], trajectories[i][1], 'r.', markersize=2)  # simplex trajectory
        else:
            raise RuntimeError("Unrecognized action mode")

    # Add label
    # h1, = plt.plot(trajectories[-1][0], trajectories[-1][1], '.', color=[0, 0.4470, 0.7410], label="HPC",
    #                markersize=2)
    # if self.simplex_enable:
    #     h2, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'r.', label="HAC", markersize=2)

    # plt.plot(trajectories[0][:], trajectories[1][:], 'r.', markersize=2)  # trajectory
    h3, = plt.plot(trajectories[0][0], trajectories[0][1], 'ko', markersize=6, mew=1.2)  # initial state
    h4, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'kx', markersize=8, mew=1.2)  # end state


def plot_inference_phase_drl(trajectories, name, label, color=[0, 0.4470, 0.7410]):
    print(f"len traj: {len(trajectories)}")
    for i in range(len(trajectories) - 1):
        x = trajectories[i][0]
        y = trajectories[i][2]
        x_next = trajectories[i + 1][0]
        y_next = trajectories[i + 1][2]
        plt.plot([x, x_next], [y, y_next], linewidth=6, color=color)
    xx = trajectories[-2][0]
    yy = trajectories[-2][2]

    plt.plot(xx, yy, color=color, linestyle='solid', linewidth=2, label=label)

    # for i in range(len(trajectories) - 1):
    #     # if action_modes[i] == "model" or action_modes[i] == "residual":
    #     plt.plot(trajectories[i][0], trajectories[i][1], '.', color=[0, 0.4470, 0.7410],
    #              markersize=2)  # model trajectory
    # elif action_modes[i] == "simplex":
    #     plt.plot(trajectories[i][0], trajectories[i][1], 'r.', markersize=2)  # simplex trajectory
    # else:
    #     raise RuntimeError("Unrecognized action mode")

    # Add label
    # h1, = plt.plot(trajectories[-1][0], trajectories[-1][1], '.', color=[0, 0.4470, 0.7410], label="HPC",
    #                markersize=2)
    # if self.simplex_enable:
    #     h2, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'r.', label="HAC", markersize=2)

    # plt.plot(trajectories[0][:], trajectories[1][:], 'r.', markersize=2)  # trajectory
    h3, = plt.plot(trajectories[0][0], trajectories[0][2], 'ko', markersize=15, mew=1.2)  # initial state
    h4, = plt.plot(trajectories[-1][0], trajectories[-1][2], color='black', marker='*', markersize=24,
                   mew=2)  # end state

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('x', fontsize=35)
    plt.ylabel('$\\theta$', fontsize=35)
    plt.legend(loc="upper right", markerscale=4, handlelength=1.5, handletextpad=0.5, fontsize=22)


if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 10))
    plot_safety_envelope(p_mat=MATRIX_P / 0.6 * 0.4)
    # plot_safety_envelope()

    order = 1
    ep = "ep5"

    file_name11 = f'../../saved_trajectory_data/ep2/data/unsafe_trajectory1.txt'
    file_name12 = f'saved_trajectory_data/{ep}/data/unsafe_learn_trajectory1.txt'
    file_name13 = f'saved_trajectory_data/{ep}/data/safe_learn_trajectory1.txt'

    file_name21 = f'../../saved_trajectory_data/ep2/data/unsafe_trajectory2.txt'
    file_name22 = f'saved_trajectory_data/{ep}/data/unsafe_learn_trajectory2.txt'
    file_name23 = f'saved_trajectory_data/{ep}/data/safe_learn_trajectory2.txt'

    file_name31 = f'../../saved_trajectory_data/ep2/data/unsafe_trajectory3.txt'
    file_name32 = f'saved_trajectory_data/{ep}/data/unsafe_learn_trajectory3.txt'
    file_name33 = f'saved_trajectory_data/{ep}/data/safe_learn_trajectory3.txt'

    if order == 1:
        cnt11 = np.loadtxt(file_name11)
        plot_inference_phase_drl(cnt11, name=1, color='green', label='Pre-trained Policy')
        cnt12 = np.loadtxt(file_name12)
        plot_inference_phase_drl(cnt12, name=2, color='blue', label='Unsafe Continual Learning')
        cnt13 = np.loadtxt(file_name13)
        plot_inference_phase_drl(cnt13, name=3, color='red', label='SeC-Learning Machine')

    elif order == 2:
        cnt21 = np.loadtxt(file_name21)
        plot_inference_phase_drl(cnt21, name=1, color='green', label='Pre-trained Policy')
        cnt22 = np.loadtxt(file_name22)
        plot_inference_phase_drl(cnt22, name=2, color='blue', label='Unsafe Continual Learning')
        cnt23 = np.loadtxt(file_name23)
        plot_inference_phase_drl(cnt23, name=3, color='red', label='SeC-Learning Machine')

    elif order == 3:
        cnt31 = np.loadtxt(file_name31)
        plot_inference_phase_drl(cnt31, name=1, color='green', label='Pre-trained Policy')
        cnt32 = np.loadtxt(file_name32)
        plot_inference_phase_drl(cnt32, name=2, color='blue', label='Unsafe Continual Learning')
        cnt33 = np.loadtxt(file_name33)
        plot_inference_phase_drl(cnt33, name=3, color='red', label='SeC-Learning Machine')

    # fig.savefig(f"phase1.pdf")
    # fig.savefig(f"phase2.pdf")
    fig.savefig(f"saved_trajectory_data/{ep}/plot/phase{order}_{ep}.pdf")

    # for i in range(3):
    #     file_name = f'unsafe_trajectory{i + 1}.txt'
    #     cnt = np.loadtxt(file_name)
    #     plot_inference_phase_drl(cnt, name=i)
    #     print(cnt)
