import os
import time
import enum
import copy
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

from src.hp_student.agents.ddpg import DDPGAgent
from src.logger.logger import Logger, plot_trajectory
from src.utils.utils import ActionMode, safety_value, logger
from src.physical_design import MATRIX_P

np.set_printoptions(suppress=True)


class Coordinator:

    def __init__(self, config):
        self.params = config

        # Configurations
        self.teacher_learn = config.teacher_learn
        self.max_dwell_steps = config.max_dwell_steps

        # Real time status
        self._plant_state = None
        self._plant_action = 0
        self._dwell_step = 0
        self._action_mode = ActionMode.STUDENT
        self._last_action_mode = None

    def update(self, states: List[float]):
        self._plant_state = states

    def determine_action(self, hp_action, ha_action, epsilon=1):
        # print(f"last_action_mode: {self.last_action_mode}")
        # print(f"action_mode: {self._action_mode}")
        self._last_action_mode = self._action_mode

        # When Teacher deactivated
        if ha_action is None:
            self._action_mode = ActionMode.STUDENT
            self._plant_action = hp_action
            return hp_action, ActionMode.STUDENT

        safety_val = safety_value(np.array(self._plant_state), MATRIX_P)

        # Inside safety envelope (bounded by epsilon)
        if safety_val < epsilon:
            logger.debug(f"current safety status: {safety_val} < {epsilon}, system is safe")

            # Teacher already activated
            if self._last_action_mode == ActionMode.TEACHER:

                # Run for Max Dwell Steps
                if self._dwell_step <= self.max_dwell_steps:
                    logger.debug(
                        f"Teacher activated, run for max dwell time: {self._dwell_step}/{self.max_dwell_steps}")
                    self._action_mode = ActionMode.TEACHER
                    self._plant_action = ha_action
                    self._dwell_step += 1
                    return ha_action, ActionMode.TEACHER

                # Switch back to HPC
                else:
                    self._dwell_step = 0  # Reset the dwell steps
                    self._action_mode = ActionMode.STUDENT
                    self._plant_action = hp_action
                    logger.debug(f"Max dwell time achieved, switch back to HPC control")
                    return hp_action, ActionMode.STUDENT
            else:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = hp_action
                logger.debug(f"Continue HPC behavior")
                return hp_action, ActionMode.STUDENT

        # Outside safety envelope (bounded by epsilon)
        else:
            logger.debug(f"current safety status: {safety_val} >= {epsilon}, system is unsafe")
            self._action_mode = ActionMode.TEACHER
            self._plant_action = ha_action
            return ha_action, ActionMode.TEACHER

    @property
    def dwell_step(self):
        return self._dwell_step

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode

    @property
    def plant_state(self):
        return self._plant_state

    @plant_state.setter
    def plant_state(self, plant_state: np.array):
        self._plant_state = plant_state
