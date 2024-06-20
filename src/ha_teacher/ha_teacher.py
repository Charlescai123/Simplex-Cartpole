import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

from src.physical_design import MATRIX_P, F
from src.ha_teacher.mat_engine import MatEngine
from src.hp_student.agents.ddpg import DDPGAgent
from src.logger.logger import Logger, plot_trajectory
from src.utils.utils import safety_value, get_discrete_Ad_Bd, logger

np.set_printoptions(suppress=True)


class HATeacher:
    def __init__(self, teacher_cfg, cartpole_cfg):

        self.teacher_params = teacher_cfg
        self.cartpole_params = cartpole_cfg

        # Matlab Engine
        self.mat_engine = MatEngine(cfg=teacher_cfg.matlab_engine)

        # Configuration
        self.chi = teacher_cfg.chi
        self.epsilon = teacher_cfg.epsilon
        self.teacher_enable = teacher_cfg.teacher_enable

        # Real-time status
        self._plant_state = None
        self._patch_center = np.array([0, 0, 0, 0])
        self._center_update = True  # Patch center update flag
        self._patch_gain = F  # F_hat

    def update(self, states: np.ndarray):
        """
        Update real-time plant state and corresponding patch center if state is unsafe
        """

        self._plant_state = states
        safety_val = safety_value(states=states, p_mat=MATRIX_P)

        # Restore patch flag
        if safety_val < self.epsilon:
            self._center_update = True

        # States unsafe (outside safety envelope)
        else:
            # Update patch center with current plant state
            if self._center_update is True:
                self._patch_center = self._plant_state * self.chi
                self._center_update = False

    def get_action(self):
        """
        Get updated teacher action during real-time
        """

        # If teacher deactivated
        if self.teacher_enable is False:
            return None

        As, Bs = self.get_As_Bs_by_state(state=self._plant_state)
        Ak, Bk = get_discrete_Ad_Bd(Ac=As, Bc=Bs, T=1 / self.cartpole_params.frequency)

        # Call Matlab Engine for patch gain (F_hat)
        F_hat, t_min = self.mat_engine.system_patch(As=As, Bs=Bs, Ak=Ak, Bk=Bk)

        if t_min > 0:
            print(f"LMI has no solution, use last updated patch")
            # self._patch_gain = np.asarray(F_hat).squeeze()
        else:
            self._patch_gain = np.asarray(F_hat).squeeze()

        # State error form
        error_state = self._plant_state - self._patch_center
        redundancy_term = self._patch_center - Ak @ self._patch_center

        v1 = np.squeeze(redundancy_term[1] / Bk[1])
        v2 = np.squeeze(redundancy_term[3] / Bk[3])
        # v = np.linalg.pinv(self.Bk).squeeze() @ (np.eye(4) - self.Ak) @ sbar_star
        v = (v1 + v2) / 2
        teacher_action = self._patch_gain @ error_state + v

        logger.debug(f"v1: {v1}")
        logger.debug(f"v2: {v2}")
        logger.debug(f"v is: {v}")
        logger.debug(f"redundancy term: {redundancy_term}")
        logger.debug(f"patch gain: {self._patch_gain}")
        logger.debug(f"self._plant_state: {self._plant_state}")
        logger.debug(f"self._patch_center: {self._patch_center}")
        logger.debug(f"Generated teacher action: {teacher_action}")

        return teacher_action

    def get_As_Bs_by_state(self, state: np.ndarray):
        """
        Update the physical knowledge matrices A(s) and B(s) in real-time based on the current state
        """
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]

        As = np.zeros((4, 4))
        As[0][1] = 1
        As[2][3] = 1

        mc = self.cartpole_params.mass_cart
        mp = self.cartpole_params.mass_pole
        g = self.cartpole_params.gravity
        l = self.cartpole_params.length_pole / 2

        term = 4 / 3 * (mc + mp) - mp * np.cos(theta) * np.cos(theta)

        As[1][2] = -mp * g * np.sin(theta) * np.cos(theta) / (theta * term)
        As[1][3] = 4 / 3 * mp * l * np.sin(theta) * theta_dot / term
        As[3][2] = g * np.sin(theta) * (mc + mp) / (l * theta * term)
        As[3][3] = -mp * np.sin(theta) * np.cos(theta) * theta_dot / term

        Bs = np.zeros((4, 1))
        Bs[1] = 4 / 3 / term
        Bs[3] = -np.cos(theta) / (l * term)

        return As, Bs

    @property
    def plant_state(self):
        return self._plant_state

    @property
    def patch_center(self):
        return self._patch_center

    @property
    def patch_gain(self):
        return self._patch_gain
