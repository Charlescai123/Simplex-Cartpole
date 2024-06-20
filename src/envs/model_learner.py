# from lib.agent.ddpg import DDPGParams, DDPGAgent
# from lib.agent.network import TaylorParams
# from lib.env.cart_pole import CartpoleParams, Cartpole, states2observations, get_init_condition, MATRIX_P
# from lib.env.cart_pole import observations2states
# from lib.logger.logger import LoggerParams, Logger, plot_trajectory
# from lib.utils import ReplayMemory
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from numpy.linalg import pinv
import numpy as np
import copy


class OnlineModelLearner:

    def __init__(self, window_size, sample_freq):
        self._window_size = window_size
        self._T = 1 / sample_freq
        self.data_buffer = deque()
        self.eq_point = None

    def reset(self):
        self.data_buffer = deque()

    def sample(self, xi, ui):
        # Popleft when size exceeds
        if len(self.data_buffer) == self._window_size + 1:
            removed = self.data_buffer.popleft()
            # print(f"Data Buffer is oversize, removing data {removed}")

        self.data_buffer.append((xi, ui))
        print(f"Successfully cached data tuple {(xi, ui)}")

    def record_eq_point(self):
        eq = np.zeros((1, 4))
        for i in range(len(self.data_buffer) - 1):
            # print(f"buffer: {self.data_buffer[i][0]}")
            eq += self.data_buffer[i][0]
        # print(f"eq: {eq}")
        eq = eq / (len(self.data_buffer) - 1)
        self.eq_point = eq.squeeze()

    def system_identification(self, method="normal"):
        A_bar = np.zeros((4, 5))
        # if len(self.data_buffer) <= self._window_size:
        #     return None

        # print(f"self.X_kN1: {self.X_kN1}")
        # print(f"self.Y_kN: {self.Y_kN}")
        if method == "difference":
            Q = self.X_kN1 @ self.Y_kN.transpose()
            P = self.Y_kN @ self.Y_kN.transpose()

            # print(f"Q: {Q}")
            # print(f"P: {P}")
            # print(f"inv P: {np.linalg.inv(P)}")

            A_bar = Q @ np.linalg.pinv(P)

        elif method == "normal":

            N = len(self.data_buffer)

            X = np.array([]).reshape(4, 0)
            Y = np.array([]).reshape(5, 0)
            for i in range(N - 1):
                xi, ui = self.data_buffer[i][0].reshape(4, 1), self.data_buffer[i][1].reshape(1, 1)
                xi_next, ui_next = self.data_buffer[i + 1][0].reshape(4, 1), self.data_buffer[i + 1][1].reshape(1, 1)
                yi = np.vstack((xi, ui))

                Y = np.hstack((Y, yi))
                X = np.hstack((X, xi_next))

            A_bar = X @ np.linalg.pinv(Y)

        Ak = A_bar[:, :4]
        Bk = A_bar[:, 4].reshape(4, 1)

        Ac = (Ak - np.eye(4)) / self._T
        Bc = Bk / self._T

        return Ac, Bc

    def x_diff(self, idx1, idx2):
        length = len(self.data_buffer[idx1][0])
        diff = self.data_buffer[idx2][0] - self.data_buffer[idx1][0]
        return diff.reshape(length, 1)

    def y_diff(self, idx1, idx2):

        x1, u1 = self.data_buffer[idx1][0], self.data_buffer[idx1][1]
        x2, u2 = self.data_buffer[idx2][0], self.data_buffer[idx2][1]

        lx, lu = x1.size, u1.size

        y1 = np.hstack((x1, u1))
        y2 = np.hstack((x2, u2))

        diff = y2 - y1
        return diff.reshape(lx + lu, 1)

    @property
    def Y_kN(self):

        _Y_kN = np.array([]).reshape(5, 0)

        for i in range(self._window_size):
            for j in range(i + 1, self._window_size):
                _y_diff = self.y_diff(i, j)
                _Y_kN = np.hstack((_Y_kN, _y_diff))

        return _Y_kN

    @property
    def X_kN1(self):

        _X_kN1 = np.array([]).reshape(4, 0)
        # print(f"deque size: {len(self.data_buffer)}")
        for i in range(1, self._window_size + 1):
            for j in range(i + 1, self._window_size + 1):
                _x_diff = self.x_diff(i, j)
                # print(f"i, j: {i}, {j}")
                # print(_x_diff.shape)
                # print(f"x_diff: {_x_diff}")
                # print(f"_X_kN1: {_X_kN1}")
                _X_kN1 = np.hstack([_X_kN1, _x_diff])
        # print(f"shape: {_X_kN1.shape}")
        return _X_kN1

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, size):
        self._window_size = size

    @property
    def sample_period(self):
        return self._T

    @sample_period.setter
    def sample_period(self, tau):
        self._T = tau
