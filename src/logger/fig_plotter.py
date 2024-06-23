import os
import re
import sys
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from numpy.linalg import inv
from numpy import linalg as LA
from src.utils.utils import check_dir, ActionMode, PlotMode
from matplotlib.collections import LineCollection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize="7.3", loc='best')


class FigPlotter:
    def __init__(self, plotter_cfg):
        self.params = plotter_cfg
        self.phase_dir = plotter_cfg.phase.save_dir
        self.trajectory_dir = plotter_cfg.trajectory.save_dir

        self.check_all_dir(plotter_cfg.phase.plot, plotter_cfg.trajectory.plot)

        # For live plot
        self.last_live_state = None
        self.last_live_action = 0
        self.last_live_action_mode = None
        self.last_live_safety_val = 0

        self.line_collections = []

    def reset_live_variables(self):
        self.last_live_state = []
        self.last_live_action = 0
        self.last_live_action_mode = None
        self.last_live_safety_val = 0

    def check_all_dir(self, phase_plot, trajectory_plot):
        if phase_plot:
            check_dir(self.phase_dir)

        if trajectory_plot:
            check_dir(self.trajectory_dir)

    def change_dir(self, old_dir: str, new_dir: str):
        try:
            pattern = rf'\b{old_dir}\b'
            self.phase_dir = re.sub(pattern, new_dir, self.phase_dir)
            self.trajectory_dir = re.sub(pattern, new_dir, self.trajectory_dir)
            self.check_all_dir(phase_plot=self.params.phase.plot,
                               trajectory_plot=self.params.trajectory.plot)
        except:
            raise RuntimeError("Failed to change the plotter existing directory")

    def phase_portrait(self, state_list, action_mode_list, x_set, theta_set, epsilon, p_mat, idx):
        # Figure name
        fig_name = f"{self.phase_dir}/phase{idx}.png"
        print(f"Plotting Phase to: {fig_name}...")

        # Phase
        self.plot_phase(state_list=state_list, action_mode_list=action_mode_list)

        # Safety envelope
        self.plot_envelope(p_mat=p_mat, epsilon=epsilon)

        # Safety set
        self.plot_safety_set(x_set=x_set, theta_set=theta_set)

        # plt.title(f"Inverted Pendulum Phase ($f = {freq} Hz$)", fontsize=14)
        plt.xlabel('x (m)', fontsize=18)
        plt.ylabel('$\\theta$ (rad)', fontsize=18)
        plt.legend(loc="lower left", markerscale=4, handlelength=1.2, handletextpad=0.5, bbox_to_anchor=(0.05, 0.05))
        plt.savefig(fig_name)

        print(f"Successfully plot phase: {fig_name}")
        plt.close()

    def plot_phase(self, state_list, action_mode_list, plot_mode=PlotMode.POSITION):
        assert len(state_list) == len(action_mode_list)

        states = np.array(state_list)
        x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

        if plot_mode == PlotMode.POSITION:
            trajectories = np.vstack((x, theta)).T
        elif plot_mode == PlotMode.VELOCITY:
            trajectories = np.vstack(([x_dot, theta_dot])).T
        else:
            raise NotImplementedError(f"Unrecognized plot mode: {plot_mode}")
        # eq points
        # if plot_eq and eq_point is not None:
        #     print(f"eq point: {eq_point}")
        #     plt.plot(eq_point[0], eq_point[2], '*', color=[0.4660, 0.6740, 0.1880], markersize=8)

        for i in range(len(trajectories) - 1):
            if action_mode_list[i] == ActionMode.STUDENT:
                plt.plot(trajectories[i][0], trajectories[i][1], '.', color=[0, 0.4470, 0.7410],
                         markersize=2)  # student trajectory
            elif action_mode_list[i] == ActionMode.TEACHER:
                plt.plot(trajectories[i][0], trajectories[i][1], 'r.', markersize=2)  # teacher trajectory
            else:
                raise RuntimeError(f"Unrecognized action mode: {action_modes[i]}")

        # Add label
        h1, = plt.plot(trajectories[-1][0], trajectories[-1][1], '.', color=[0, 0.4470, 0.7410], label="HP Student",
                       markersize=2)
        h2, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'r.', label="HA Teacher", markersize=2)
        h3, = plt.plot(trajectories[0][0], trajectories[0][1], 'ko', markersize=6, mew=1.2)  # initial state
        h4, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'kx', markersize=8, mew=1.2)  # end state

    def plot_trajectory(self, state_list, action_list, action_mode_list, safety_val_list, x_set, theta_set, action_set,
                        freq, fig_idx):
        # Figure name
        fig_name = f'{self.trajectory_dir}/trajectory{fig_idx}.png'
        print(f"Plotting Trajectory to: {fig_name}...")

        x_l, x_h = x_set
        th_l, th_h = theta_set
        f_l, f_h = action_set
        x_ticks = np.linspace(x_l, x_h, 5)
        th_ticks = np.linspace(th_l, th_h, 5)
        f_ticks = np.linspace(f_l, f_h, 5)

        # plt.close()
        # plt.clf()
        # print(f"current fig: {plt.gcf()}")

        n1 = len(state_list)
        n2 = len(action_list)
        n3 = len(action_mode_list)
        n4 = len(safety_val_list)
        assert n1 == n2
        assert n2 == n3
        assert n3 == n4

        trajectories = np.asarray(state_list)
        fig, axes = plt.subplots(3, 2, figsize=(12, 6))  # Create a 2x2 subplot grid
        fig.suptitle(f'Inverted Pendulum Trajectories ($f = {freq} Hz$)', fontsize=11, ha='center', y=0.97)

        for i in range(n1 - 1):
            self.line_segment(axes=axes,
                              state1=trajectories[i],
                              state2=trajectories[i + 1],
                              action1=action_list[i],
                              action2=action_list[i + 1],
                              safety_val1=safety_val_list[i],
                              safety_val2=safety_val_list[i + 1],
                              action_mode=action_mode_list[i],
                              i=i)

        # Add legend and label
        self.legend_and_label(axes, x_ticks, th_ticks, f_ticks)

        plt.tight_layout()  # Adjust spacing between subplots
        plt.savefig(fig_name, dpi=150)
        plt.close(fig)
        print(f"Successfully plot trajectory: {fig_name}")

    def live_plot_trajectory(self, axes, state, action, action_mode, safety_val, idx):
        if idx == 0:
            pass
        # if idx % 3 == 0:
        #     plt.clf()
        #     pass
        else:
            self.line_segment(axes=axes,
                              state1=self.last_live_state,
                              state2=state,
                              action1=self.last_live_action,
                              action2=action,
                              safety_val1=self.last_live_safety_val,
                              safety_val2=safety_val,
                              action_mode=action_mode,
                              i=idx - 1)
            plt.draw()
            # plt.show(block=False)
            plt.pause(0.00001)

        # Update the live variables
        self.last_live_state = state
        self.last_live_action = action
        self.last_live_action_mode = action_mode
        self.last_live_safety_val = safety_val

    def live_plot_trajectory2(self, axes, line_collections, state_list, action_list, action_mode_list, safety_val_list,
                              idx):
        if idx == 0:
            return
        # print(f"action_list: {action_list}")
        if len(state_list) > 30:
            state_list = state_list[-30:]
            action_list = action_list[-30:]
            action_mode_list = action_mode_list[-30:]
            safety_val_list = safety_val_list[-30:]

        trajectories = np.asarray(state_list)
        n = len(trajectories)
        # n = len(trajectories) if len(trajectories) < 100 else 100
        #
        colors = []
        segments = []
        for i in range(n - 1):
            if action_mode_list[i] == ActionMode.TEACHER:
                colors.append(0)
            else:
                colors.append(1)
            segment = np.column_stack([np.array([i, i + 1]), np.array([action_list[i], action_list[i + 1]])])
            print(f"segment: {segment}")
            segments.append(segment)
        for i in range(6):
            line_collections[i].set_segments(segments)
            line_collections[i].set_array(colors)

        # for i in range(n - 1):
        #     self.line_segment(axes=axes,
        #                       state1=trajectories[i],
        #                       state2=trajectories[i + 1],
        #                       action1=action_list[i],
        #                       action2=action_list[i + 1],
        #                       safety_val1=safety_val_list[i],
        #                       safety_val2=safety_val_list[i + 1],
        #                       action_mode=action_mode_list[i],
        #                       i=i)
        plt.draw()
        # plt.show(block=False)
        plt.pause(0.00001)

        # Update the live variables
        # self.last_live_state = state
        # self.last_live_action = action
        # self.last_live_action_mode = action_mode
        # self.last_live_safety_val = safety_val

    @staticmethod
    def legend_and_label(axes, x_ticks, th_ticks, f_ticks):
        # Add label and title (x)
        axes[0, 0].set_yticks(x_ticks)
        axes[0, 0].set_ylabel("x (m)")
        axes[0, 0].add_line(mlines.Line2D([], [], color=[0, 0.4470, 0.7410], linestyle='-', label='HPC'))
        axes[0, 0].add_line(mlines.Line2D([], [], color='red', linestyle='-', label='HAC'))
        legend_without_duplicate_labels(axes[0, 0])

        # Add label and title (x_dot)
        # axes[0, 1].set_yticks(np.linspace(-3, 3, 5))
        axes[0, 1].set_ylabel(r"$\dot{x}$ (m/s)")
        axes[0, 1].add_line(mlines.Line2D([], [], color=[0, 0.4470, 0.7410], linestyle='-', label='HPC'))
        axes[0, 1].add_line(mlines.Line2D([], [], color='red', linestyle='-', label='HAC'))
        legend_without_duplicate_labels(axes[0, 1])

        # Add label and title (theta)
        axes[1, 0].set_yticks(th_ticks)
        axes[1, 0].set_ylabel('$\\theta$ (rad)')
        axes[1, 0].add_line(mlines.Line2D([], [], color=[0, 0.4470, 0.7410], linestyle='-', label='HPC'))
        axes[1, 0].add_line(mlines.Line2D([], [], color='red', linestyle='-', label='HAC'))
        legend_without_duplicate_labels(axes[1, 0])

        # Add label and title (theta_dot)
        # axes[1, 1].set_yticks(np.linspace(-4.5, 4.5, 5))
        axes[1, 1].set_ylabel(r'$\dot{\theta}$ (rad/s)')
        axes[1, 1].add_line(mlines.Line2D([], [], color=[0, 0.4470, 0.7410], linestyle='-', label='HPC'))
        axes[1, 1].add_line(mlines.Line2D([], [], color='red', linestyle='-', label='HAC'))
        legend_without_duplicate_labels(axes[1, 1])

        # Add label and title (force)
        axes[2, 0].set_yticks(f_ticks)
        axes[2, 0].set_ylabel("force (N)")
        axes[2, 0].add_line(mlines.Line2D([], [], color=[0, 0.4470, 0.7410], linestyle='-', label='HPC'))
        axes[2, 0].add_line(mlines.Line2D([], [], color='red', linestyle='-', label='HAC'))
        legend_without_duplicate_labels(axes[2, 0])

        # Add label and title (safety values)
        # axes[2, 1].set_yticks(np.linspace(0, 3, 5))
        axes[2, 1].set_ylabel("safety value")
        axes[2, 1].add_line(mlines.Line2D([], [], color=[0, 0.4470, 0.7410], linestyle='-', label='HPC'))
        axes[2, 1].add_line(mlines.Line2D([], [], color='red', linestyle='-', label='HAC'))
        legend_without_duplicate_labels(axes[2, 1])

    @staticmethod
    def line_segment(axes, state1, state2, action1, action2, action_mode, safety_val1, safety_val2, i):
        if action_mode == ActionMode.STUDENT:
            # x
            axes[0, 0].plot([i, i + 1], [state1[0], state2[0]], '-', label='HPC', color=[0, 0.4470, 0.7410])

            # x_dot
            axes[0, 1].plot([i, i + 1], [state1[1], state2[1]], '-', label='HPC', color=[0, 0.4470, 0.7410])

            # theta
            axes[1, 0].plot([i, i + 1], [state1[2], state2[2]], '-', label='HPC', color=[0, 0.4470, 0.7410])

            # theta_dot
            axes[1, 1].plot([i, i + 1], [state1[3], state2[3]], '-', label='HPC', color=[0, 0.4470, 0.7410])

            # force/action
            axes[2, 0].plot([i, i + 1], [action1, action2], '-', label='HPC', color=[0, 0.4470, 0.7410])

            # safety values
            axes[2, 1].plot([i, i + 1], [safety_val1, safety_val2], '-', label='HPC', color=[0, 0.4470, 0.7410])

        elif action_mode == ActionMode.TEACHER:
            # x
            axes[0, 0].plot([i, i + 1], [state1[0], state2[0]], 'r-', label='HAC')

            # x_dot
            axes[0, 1].plot([i, i + 1], [state1[1], state2[1]], 'r-', label='HAC')

            # theta
            axes[1, 0].plot([i, i + 1], [state1[2], state2[2]], 'r-', label='HAC')

            # theta_dot
            axes[1, 1].plot([i, i + 1], [state1[3], state2[3]], 'r-', label='HAC')

            # force/action
            axes[2, 0].plot([i, i + 1], [action1, action2], 'r-', label='HAC')

            # safety values
            axes[2, 1].plot([i, i + 1], [safety_val1, safety_val2], 'r-', label='HAC')

        else:
            raise RuntimeError(f"Unrecognized action mode: {action_mode_list[i]}")

    @staticmethod
    def plot_safety_set(x_set=[-0.9, 0.9], theta_set=[-0.8, 0.8]):
        x_l, x_h = x_set
        th_l, th_h = theta_set

        # Safety Set
        plt.vlines(x=x_l, ymin=th_l, ymax=th_h, color='black', linewidth=2.5)
        plt.vlines(x=x_h, ymin=th_l, ymax=th_h, color='black', linewidth=2.5)
        plt.hlines(y=th_l, xmin=x_l, xmax=x_h, color='black', linewidth=2.5)
        plt.hlines(y=th_h, xmin=x_l, xmax=x_h, color='black', linewidth=2.5)

    @staticmethod
    def plot_envelope(p_mat, epsilon):
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
        plt.plot(tx1, tx2, linewidth=2, color='black')
        plt.plot(0, 0, 'k*', markersize=4, mew=0.6)  # global equilibrium (star)
        plt.plot(0, 0, 'ko-', markersize=7, mew=1, markerfacecolor='none')  # global equilibrium (circle)

        # HAC switch envelope
        # if self.simplex_enable:
        #     plt.plot(tx_eps1, tx_eps2, 'k--', linewidth=0.8, label=r"$\partial\Omega_{HAC}$")

        # HPC switch envelope
        # plt.plot(tx_hpc1, tx_hpc2, 'b--', linewidth=0.8, label=r"$\partial\Omega_{HPC}$")


if __name__ == '__main__':
    pass
    # plt.clf()
    # plt.close()
    # # Create a 3x2 subplot grid
    # live_fig, axes = plt.subplots(3, 2, figsize=(10, 6), num='Live Trajectory')
    # live_fig.suptitle(f'Inverted Pendulum Trajectory', fontsize=11, ha='center', y=0.97)
    # state1 = np.asarray([0, 1, 2, 3])
    # state2 = np.asarray([1, 2, 3, 4])
    # action1 = 1
    # action2 = 2
    # action_mode = ActionMode.TEACHER
    # safety_val1 = 0.5
    # safety_val2 = 2
    # i = 0
    #
    # # x
    # axes[0, 0].plot([i, i + 1], [state1[0], state2[0]], 'r-', label='HAC')
    #
    # # x_dot
    # axes[0, 1].plot([i, i + 1], [state1[1], state2[1]], 'r-', label='HAC')
    #
    # # theta
    # axes[1, 0].plot([i, i + 1], [state1[2], state2[2]], 'r-', label='HAC')
    #
    # # theta_dot
    # axes[1, 1].plot([i, i + 1], [state1[3], state2[3]], 'r-', label='HAC')
    #
    # # force/action
    # axes[2, 0].plot([i, i + 1], [action1, action2], 'r-', label='HAC')
    #
    # # safety values
    # axes[2, 1].plot([i, i + 1], [safety_val1, safety_val2], 'r-', label='HAC')
    # plt.show()
    # plt.pause(1023)
