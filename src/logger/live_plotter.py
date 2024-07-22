import time
import numpy as np
from PIL import Image
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from src.logger.fig_plotter import FigPlotter
from src.utils.utils import ActionMode

matplotlib.use('TkAgg')  # Use TkAgg as the matplotlib backend


class LivePlotter:
    def __init__(self, live_cfg):
        self.params = live_cfg
        self.window_size = live_cfg.live_trajectory.window_size

        # For live plot
        self.animation = None
        self.live_plot_counter = 0
        self.live_plot_last_counter = 0
        self.live_plot_flag = False
        self.live_fig = None
        self.live_axes = None
        self.live_line = None
        self.frames = []
        self.line_collections = []

        self.states = deque(maxlen=self.window_size)
        self.actions = deque(maxlen=self.window_size)
        self.action_modes = deque(maxlen=self.window_size)
        self.energies = deque(maxlen=self.window_size)

    def reset(self):
        self.animation = None
        self.live_plot_flag = False
        self.live_plot_counter = 0
        self.live_plot_last_counter = 0
        self.live_fig = None
        self.live_axes = None
        self.live_line = None
        self.frames = []
        self.line_collections = []

        self.states = deque(maxlen=self.window_size)
        self.actions = deque(maxlen=self.window_size)
        self.action_modes = deque(maxlen=self.window_size)
        self.energies = deque(maxlen=self.window_size)

    def update(self, state, action, action_mode, energy):
        self.states.append(state)
        self.actions.append(action)
        self.action_modes.append(action_mode)
        self.energies.append(energy)
        self.live_plot_counter += 1

    def animation_init(self):
        # x_l, x_h = x_set
        # th_l, th_h = theta_set
        # f_l, f_h = action_set
        # x_ticks = np.linspace(x_l, x_h, 5)
        # th_ticks = np.linspace(th_l, th_h, 5)
        # f_ticks = np.linspace(f_l, f_h, 5)

        """Initialize"""
        for i in range(6):
            self.line_collections[i].set_segments([])
        return self.line_collections[:]

    def animation_update(self, frame):
        # print(f"frame: {frame}")

        n = self.live_plot_counter
        if n == 0:
            return self.line_collections[:]
        data_size = len(self.states)

        # Sync animation update frequency with main thread
        if self.live_plot_last_counter == self.live_plot_counter:
            return self.line_collections[:]

        # Update sliding window
        if n > self.window_size:
            # Update x-axis range
            for axes in self.live_axes:
                axes.set_xlim(n - self.window_size, n - 1)

        colors = []
        segment_list = [[] for _ in range(6)]
        window_idx = n - data_size

        # Segment with colors
        for i in range(data_size - 1):
            if self.action_modes[i] == ActionMode.TEACHER:
                colors.append('red')
            else:
                colors.append([0, 0.4470, 0.7410])

            s1, s2 = self.states[i], self.states[i + 1]
            a1, a2 = self.actions[i], self.actions[i + 1]
            e1, e2 = self.energies[i], self.energies[i + 1]

            segment_state1 = np.column_stack([np.array([i + window_idx, i + window_idx + 1]), np.array([s1[0], s2[0]])])
            segment_state2 = np.column_stack([np.array([i + window_idx, i + window_idx + 1]), np.array([s1[1], s2[1]])])
            segment_state3 = np.column_stack([np.array([i + window_idx, i + window_idx + 1]), np.array([s1[2], s2[2]])])
            segment_state4 = np.column_stack([np.array([i + window_idx, i + window_idx + 1]), np.array([s1[3], s2[3]])])
            segment_action = np.column_stack([np.array([i + window_idx, i + window_idx + 1]), np.array([a1, a2])])
            segment_energy = np.column_stack([np.array([i + window_idx, i + window_idx + 1]), np.array([e1, e2])])

            segment_list[0].append(segment_state1)
            segment_list[1].append(segment_state2)
            segment_list[2].append(segment_state3)
            segment_list[3].append(segment_state4)
            segment_list[4].append(segment_action)
            segment_list[5].append(segment_energy)

        for i in range(6):
            self.line_collections[i].set_segments(segment_list[i])
            self.line_collections[i].set_colors(colors)

        # Update y-axis range
        states = np.asarray(self.states)
        dx_min, dx_max = min(states[:, 1]) - 1, max(states[:, 1]) + 1
        dth_min, dth_max = min(states[:, 3]) - 1, max(states[:, 3]) + 1
        e_min, e_max = min(self.energies) - 1, max(self.energies) + 1

        self.live_axes[1].set_ylim(dx_min, dx_max)
        self.live_axes[3].set_ylim(dth_min, dth_max)
        self.live_axes[5].set_ylim(e_min, e_max)
        plt.draw()

        # Append frames
        frame_img = np.array(self.live_fig.canvas.renderer.buffer_rgba())
        self.frames.append(Image.fromarray(frame_img))

        self.live_plot_last_counter = self.live_plot_counter

        return self.line_collections[:]

    def animation_run(self, x_set, theta_set, action_set, state, action, action_mode, energy):
        if self.live_plot_flag is False:
            self.live_plot_flag = True
            plt.clf()
            plt.close()
            print(f"Setting up figure canvas for live plot")

            x_l, x_h = x_set
            th_l, th_h = theta_set
            f_l, f_h = action_set
            x_ticks = np.linspace(x_l, x_h, 5)
            th_ticks = np.linspace(th_l, th_h, 5)
            f_ticks = np.linspace(f_l, f_h, 5)

            # Create a 3x2 subplot grid
            self.live_fig, self.live_axes = plt.subplots(3, 2, figsize=(10, 6), num='Live Trajectory')
            FigPlotter.legend_and_label(self.live_axes, x_ticks, th_ticks, f_ticks)
            plt.tight_layout()  # Adjust spacing between subplots

            self.live_axes = self.live_axes.flatten()
            self.line_collections = []
            for i in range(6):
                line_collection = LineCollection([], cmap='viridis', norm=plt.Normalize(0, 1))

                self.live_axes[i].set_xlim(0, self.window_size)
                self.live_axes[i].add_collection(line_collection)
                self.line_collections.append(line_collection)

            self.animation = animation.FuncAnimation(
                fig=self.live_fig,
                func=self.animation_update,
                interval=100,
                # frames=10,
                # init_func=lambda: self.animation_init(x_set, theta_set, action_set),
                init_func=self.animation_init,
                blit=False
            )
            plt.show(block=False)

        self.update(state=state, action=action, action_mode=action_mode, energy=energy)

    @staticmethod
    def line_segment(axes, action_mode, i):
        y1 = np.random.rand()
        y2 = np.random.rand()
        lines = []
        if action_mode == ActionMode.STUDENT:
            # x
            line, = axes[0, 0].plot([i, i + 1], [y1, y2], '-', label='HPC', color=[0, 0.4470, 0.7410])
            line.set_ydata([y1, y2])
            # axes[0, 0].set_xlim([0, 50])
            lines.append(line)

            # x_dot
            line, = axes[0, 1].plot([i, i + 1], [y1, y2], '-', label='HPC', color=[0, 0.4470, 0.7410])
            lines.append(line)

            # theta
            line, = axes[1, 0].plot([i, i + 1], [y1, y2], '-', label='HPC', color=[0, 0.4470, 0.7410])
            lines.append(line)

            # theta_dot
            line, = axes[1, 1].plot([i, i + 1], [y1, y2], '-', label='HPC', color=[0, 0.4470, 0.7410])
            lines.append(line)

            # force/action
            line, = axes[2, 0].plot([i, i + 1], [y1, y2], '-', label='HPC', color=[0, 0.4470, 0.7410])
            lines.append(line)

            # system energy
            line, = axes[2, 1].plot([i, i + 1], [y1, y2], '-', label='HPC', color=[0, 0.4470, 0.7410])
            lines.append(line)

            return lines

        elif action_mode == ActionMode.TEACHER:
            # x
            line, = axes[0, 0].plot([i, i + 1], [y1, y2], 'r-', label='HAC')
            lines.append(line)

            # x_dot
            line, = axes[0, 1].plot([i, i + 1], [y1, y2], 'r-', label='HAC')
            lines.append(line)

            # theta
            line, = axes[1, 0].plot([i, i + 1], [y1, y2], 'r-', label='HAC')
            lines.append(line)

            # theta_dot
            line, = axes[1, 1].plot([i, i + 1], [y1, y2], 'r-', label='HAC')
            lines.append(line)

            # force/action
            line, = axes[2, 0].plot([i, i + 1], [y1, y2], 'r-', label='HAC')
            lines.append(line)

            # system energy
            line, = axes[2, 1].plot([i, i + 1], [y1, y2], 'r-', label='HAC')
            lines.append(line)

            return lines
        else:
            raise RuntimeError(f"Unrecognized action mode: {action_mode}")
