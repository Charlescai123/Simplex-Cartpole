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

    def __init__(self):

        # Real time status
        self._plant_action = 0
        self._action_mode = ActionMode.STUDENT
        self._last_action_mode = None

    def get_terminal_action(self, hp_action, ha_action, plant_state, epsilon=1, dwell_flag=False):
        self._last_action_mode = self._action_mode

        # When Teacher deactivated
        if ha_action is None:
            logger.debug("HA-Teacher deactivated, use HP-Student's action instead")
            self._action_mode = ActionMode.STUDENT
            self._plant_action = hp_action
            return hp_action, ActionMode.STUDENT

        safety_val = safety_value(plant_state, MATRIX_P)

        # Inside safety envelope (bounded by epsilon)
        if safety_val < epsilon:
            logger.debug(f"current safety status: {safety_val} < {epsilon}, system is safe")

            # Teacher already activated
            if self._last_action_mode == ActionMode.TEACHER:

                # Teacher Dwell time
                if dwell_flag is True:
                    if ha_action is None:
                        raise RuntimeError(f"Unrecognized HA-Teacher action {ha_action} for dwelling")
                    else:
                        logger.debug("HA-Teacher action continues in dwell time")
                        self._action_mode = ActionMode.TEACHER
                        self._plant_action = ha_action
                        return ha_action, ActionMode.TEACHER

                # Switch back to HPC
                else:
                    self._action_mode = ActionMode.STUDENT
                    self._plant_action = hp_action
                    logger.debug(f"Max dwell time achieved, switch back to HP-Student control")
                    return hp_action, ActionMode.STUDENT
            else:
                self._action_mode = ActionMode.STUDENT
                self._plant_action = hp_action
                logger.debug(f"Continue HP-Student action")
                return hp_action, ActionMode.STUDENT

        # Outside safety envelope (bounded by epsilon)
        else:
            logger.debug(f"current safety status: {safety_val} >= {epsilon}, system is unsafe")
            logger.debug(f"Use HA-Teacher action for safety concern")
            self._action_mode = ActionMode.TEACHER
            self._plant_action = ha_action
            return ha_action, ActionMode.TEACHER

    @property
    def plant_action(self):
        return self._plant_action

    @property
    def action_mode(self):
        return self._action_mode

    @property
    def last_action_mode(self):
        return self._last_action_mode
