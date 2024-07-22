import os
import gym
import math
import copy
import pyglet
import numpy as np
from numpy.linalg import inv
from gym.utils import seeding
from numpy import linalg as LA
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering

from src.physical_design import MATRIX_P, MATRIX_S
from src.utils.utils import energy_value, logger


class DrawText:
    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()


class Cartpole(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, config: DictConfig):
        self.params = config
        self.safety_set = dict(config.safety_set)

        # Random variable settings
        self._reset_rand = self.seed(seed=config.random_reset.seed)
        self._noise_rand = self.seed(seed=config.random_noise.seed)
        self._noise_apply = config.random_noise.actuator.apply
        self._noise_mean = config.random_noise.actuator.mean
        self._noise_stddev = config.random_noise.actuator.stddev
        self._high_performance_reward_factor = config.reward.high_performance_reward_factor

        # Cart-Pole settings
        self.gravity = config.gravity
        self.tau = 1 / config.frequency
        self.mass_cart = config.mass_cart
        self.mass_pole = config.mass_pole
        self.with_friction = config.with_friction
        self.total_mass = self.mass_cart + self.mass_pole
        self.half_length = config.length_pole * 0.5
        self.pole_mass_length_half = config.mass_pole * self.half_length
        self.friction_cart = config.friction_cart
        self.friction_pole = config.friction_pole
        self._f_min, self._f_max = config.force_bound

        # Runtime status
        self.state = None
        self.viewer = None
        self.state_dim = 4  # x, x_dot, theta, theta_dot
        self.state_observations_dim = 5  # x, x_dot, s_theta, c_theta, theta_dot
        self.action_dim = 1  # force input or voltage
        self.reward_list = []
        self.ut = 0

    @staticmethod
    def seed(seed=None):
        np_random, seed = seeding.np_random(seed)
        return np_random

    def step(self, action: float, action_mode=None):
        """
        action: the action injected to the plant
        return: a list of state
        """
        x, x_dot, theta, theta_dot, _ = self.state
        force = np.clip(action, a_min=self._f_min, a_max=self._f_max)

        # Actual force applied to plant after random noise
        if self._noise_apply:
            force += self._noise_rand.normal(loc=self._noise_mean,
                                             scale=self._noise_stddev)
        self.ut = force
        logger.debug(f"applied force is: {force}")

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        # kinematics of the inverted pendulum
        if self.with_friction:
            """ with friction"""
            temp \
                = (force + self.pole_mass_length_half * theta_dot ** 2 *
                   sin_th - self.friction_cart * x_dot) / self.total_mass

            th_acc = \
                (self.gravity * sin_th - cos_th * temp -
                 self.friction_pole * theta_dot / self.pole_mass_length_half) / \
                (self.half_length * (4.0 / 3.0 - self.mass_pole * cos_th ** 2 / self.total_mass))
            x_acc = temp - self.pole_mass_length_half * th_acc * cos_th / self.total_mass

        else:
            """without friction"""
            temp = (force + self.pole_mass_length_half * theta_dot ** 2 * sin_th) / self.total_mass
            th_acc = (self.gravity * sin_th - cos_th * temp) / \
                     (self.half_length * (4.0 / 3.0 - self.mass_pole * cos_th ** 2 / self.total_mass))
            x_acc = temp - self.pole_mass_length_half * th_acc * cos_th / self.total_mass

        if self.params.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc  # here we inject disturbances
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * th_acc  # here we inject disturbances
            failed = self.is_failed(x, theta)

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * th_acc
            theta = theta + self.tau * theta_dot
            failed = self.is_failed(x, theta)

        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))  # wrap to [-pi, pi]
        new_state = [x, x_dot, theta_rescale, theta_dot, failed]

        self.state = new_state  # to update animation
        return self.state

    def reset(self, reset_state=None):
        print(f"<====== Env Reset: Reset at predefined condition =====>")
        if reset_state is not None:
            self.state = reset_state
        else:
            self.state = self.params.initial_condition

    def random_reset(self, threshold, mode='train'):
        print("<====== Env Reset: Random ======>")
        x_l, x_h = self.safety_set['x']
        dx_l, dx_h = self.safety_set['x_dot']
        th_l, th_h = self.safety_set['theta']
        dth_l, dth_h = self.safety_set['theta_dot']

        flag = True
        while flag:
            rand_x = self._reset_rand.uniform(x_l, x_h)
            rand_dx = self._reset_rand.uniform(dx_l, dx_h)
            rand_th = self._reset_rand.uniform(th_l, th_h)
            rand_dth = self._reset_rand.uniform(dth_l, dth_h)

            energy = energy_value(
                state=np.array([rand_x, rand_dx, rand_th, rand_dth]), p_mat=MATRIX_P
            )
            if energy < threshold:
                flag = False

        self.state = [rand_x, rand_dx, rand_th, rand_dth, False]

    def render(self, mode='human', state=None, idx=0):
        screen_width = 600
        screen_height = 400
        world_width = self.safety_set['x'][1] * 2 + 1
        scale = screen_width / world_width
        cart_y = 120  # TOP OF CART
        pole_width = 10.0
        # pole_length = scale * self.params.length_pole
        pole_length = 137
        cart_width = 50.0
        cart_height = 30.0
        target_width = 45
        target_height = 45

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Target
            self.target_trans = rendering.Transform()
            # target = rendering.Image('./docs/target.svg', width=target_width, height=target_height)
            self.target = rendering.make_circle(12)
            self.target.set_color(.8, .8, .45)
            self.target.add_attr(self.target_trans)
            self.viewer.add_geom(self.target)

            # Cart
            l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
            self.cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            self.cart.add_attr(self.cart_trans)
            self.viewer.add_geom(self.cart)

            # Pole
            axle_offset = cart_height / 4.0
            l, r, t, b = -pole_width / 2, pole_width / 2, pole_length - pole_width / 2, -pole_width / 2
            self.pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.pole.set_color(.8, .6, .4)
            self.pole_trans = rendering.Transform(translation=(0, axle_offset))
            self.pole.add_attr(self.pole_trans)
            self.pole.add_attr(self.cart_trans)
            self.viewer.add_geom(self.pole)

            # Axle
            self.axle = rendering.make_circle(pole_width / 2)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            # Track line
            self.track = rendering.Line((0, cart_y), (screen_width, cart_y))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # Labels
            cnt_text = f'Step: {idx}'
            pos_text = f'x: {self.state[0]:.2f} m'
            ang_text = f'x: {self.state[2]:.2f} rad'
            self.cnt_label = pyglet.text.Label(cnt_text, font_size=26, font_name='Times New Roman',
                                               x=300, y=340, anchor_x='center', anchor_y='center',
                                               color=(0, 0, 0, 255))
            self.pos_label = pyglet.text.Label(pos_text, font_size=20, font_name='Times New Roman',
                                               x=170, y=40, anchor_x='center', anchor_y='bottom',
                                               color=(0, 0, 0, 255))
            self.ang_label = pyglet.text.Label(ang_text, font_size=20, font_name='Times New Roman',
                                               x=430, y=40, anchor_x='center', anchor_y='bottom',
                                               color=(0, 0, 0, 255))
            self.cnt_label.draw()
            self.pos_label.draw()
            self.ang_label.draw()
            self.viewer.add_geom(DrawText(self.cnt_label))
            self.viewer.add_geom(DrawText(self.pos_label))
            self.viewer.add_geom(DrawText(self.ang_label))

        if state is None:
            if self.state is None:
                return None
            else:
                s = self.state
        else:
            s = state

        # Change to red color to indicate system failure
        if s is not None:
            if self.is_trans_failed(s[0]):
                self.cart.set_color(1.0, 0, 0)
            if self.is_theta_failed(s[2]):
                self.pole.set_color(1.0, 0, 0)

        # Edit the pole polygon vertex
        l, r, t, b = -pole_width / 2, pole_width / 2, pole_length - pole_width / 2, -pole_width / 2
        self.pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cart_x = s[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        target_x = 0 * scale + screen_width / 2.0
        target_y = pole_length + cart_y

        # Update cart-pole translation and rotation
        self.cart_trans.set_translation(cart_x, cart_y)
        self.target_trans.set_translation(target_x, target_y)
        self.pole_trans.set_rotation(-s[2])

        # Update text on label
        self.cnt_label.text = f'Step: {idx}'
        self.pos_label.text = f"x: {self.state[0]:.2f} m"
        self.ang_label.text = f"theta: {self.state[2]:.2f} rad"

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def is_trans_failed(self, x):
        trans_failed = bool(x <= self.safety_set['x'][0]
                            or x >= self.safety_set['x'][1])
        return trans_failed

    def is_theta_failed(self, theta):
        theta_failed = bool(theta <= self.safety_set['theta'][0]
                            or theta >= self.safety_set['theta'][1])
        return theta_failed

    def is_failed(self, x, theta):
        return self.is_trans_failed(x) or self.is_theta_failed(theta)

    @staticmethod
    def get_tracking_error(p_matrix, state_real, state_reference):

        state = np.array(state_real[0:4])
        state = np.expand_dims(state, axis=0)
        state_ref = np.array(state_reference[0:4])
        state_ref = np.expand_dims(state_ref, axis=0)

        state_error = state - state_ref
        eLya1 = np.matmul(state_error, p_matrix)
        eLya = np.matmul(eLya1, np.transpose(state_error))

        error = -eLya

        return error

    def get_pP_and_vP(self):
        # P = MATRIX_P
        pP = np.zeros((2, 2))
        vP = np.zeros((2, 2))

        # For velocity
        vP[0][0] = P[1][1]
        vP[1][1] = P[3][3]
        vP[0][1] = P[1][3]
        vP[1][0] = P[1][3]

        # For position
        pP[0][0] = P[0][0]
        pP[1][1] = P[2][2]
        pP[0][1] = P[0][2]
        pP[1][0] = P[0][2]

        return pP, vP

    def reward_fcn(self, curr_state, action, state_next):

        observations, _ = state2observations(curr_state)
        set_point = self.params.set_point

        distance_score = self.get_distance_score(observations=observations, set_point=set_point)
        distance_reward = distance_score * self._high_performance_reward_factor

        lyapunov_reward_current = self.get_lyapunov_reward(MATRIX_P, curr_state)

        ##########
        tem_state_a = np.array(curr_state[:4])
        tem_state_b = np.expand_dims(tem_state_a, axis=0)
        tem_state_c = np.matmul(tem_state_b, np.transpose(MATRIX_S))
        tem_state_d = np.matmul(tem_state_c, MATRIX_P)
        lyapunov_reward_current_aux = np.matmul(tem_state_d, np.transpose(tem_state_c))
        ##########

        lyapunov_reward_next = self.get_lyapunov_reward(MATRIX_P, state_next)

        if self.params.reward.lyapunov_form == 'UCB':  # Use lyapunov form of UC Berkeley
            lyapunov_reward = lyapunov_reward_current - lyapunov_reward_next
        elif self.params.reward.lyapunov_form == 'Phy-DRL':  # Phy-DRL
            lyapunov_reward = lyapunov_reward_current_aux - lyapunov_reward_next
        else:
            raise RuntimeError(f"Unknown lyapunov reward form: {self.params.reward.lyapunov_form}")

        self.reward_list.append(np.squeeze(lyapunov_reward))

        lyapunov_reward *= self.params.reward.lyapunov_reward_factor
        action_penalty = -1 * self.params.reward.action_penalty * action * action
        r = distance_reward + lyapunov_reward + action_penalty

        return r, distance_score

    def get_distance_score(self, observations, set_point):
        distance_score_factor = 5  # to adjust the exponential gradients
        cart_position = observations[0]
        pendulum_angle_sin = observations[2]
        pendulum_angle_cos = observations[3]

        target_cart_position = set_point[0]
        target_pendulum_angle = set_point[2]

        pendulum_length = self.params.length_pole

        pendulum_tip_position = np.array(
            [cart_position + pendulum_length * pendulum_angle_sin, pendulum_length * pendulum_angle_cos])

        target_tip_position = np.array(
            [target_cart_position + pendulum_length * np.sin(target_pendulum_angle),
             pendulum_length * np.cos(target_pendulum_angle)])

        distance = np.linalg.norm(target_tip_position - pendulum_tip_position)

        distance_score = np.exp(-distance * distance_score_factor)
        return distance_score

    @staticmethod
    def get_lyapunov_reward(p_matrix, state_real):
        state = np.array(state_real[0:4])
        state = np.expand_dims(state, axis=0)
        Lya1 = np.matmul(state, p_matrix)
        Lya = np.matmul(Lya1, np.transpose(state))
        return Lya


def state2observations(state):
    x, x_dot, theta, theta_dot, failed = state
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def observations2state(observations, failed):
    x, x_dot, s_theta, c_theta, theta_dot = observations[:5]
    state = [x, x_dot, np.arctan2(s_theta, c_theta), theta_dot, failed]
    return state


def get_init_condition(n_points_per_dim=20):
    eigen_values, eigen_vectors = np.linalg.eig(MATRIX_P)  # get eigen value and eigen vector

    Q = eigen_vectors

    initial_condition_list = []

    for i in range(n_points_per_dim):
        angle_1 = i * math.pi / n_points_per_dim
        y0 = math.sqrt(1 / eigen_values[0]) * math.cos(angle_1)
        vector_in_3d = math.sin(angle_1)

        if vector_in_3d == 0:
            y1 = 0
            y2 = 0
            y3 = 0
            s = Q @ np.array([y0, y1, y2, y3]).transpose()
            # print(s.transpose() @ P_matrix_4 @ s)
            initial_condition_list.append([s[0], s[1], s[2], s[3], False])
            continue

        for k in range(n_points_per_dim):
            angle_2 = k * math.pi / n_points_per_dim
            y1 = vector_in_3d * math.sqrt(1 / eigen_values[1]) * math.cos(angle_2)
            vector_in_2d = vector_in_3d * math.sin(angle_2)

            if vector_in_2d == 0:
                y2 = 0
                y3 = 0
                s = Q @ np.array([y0, y1, y2, y3]).transpose()
                # print(s.transpose() @ P_matrix_4 @ s)
                initial_condition_list.append([s[0], s[1], s[2], s[3], False])
                continue

            for j in range(n_points_per_dim):
                angle_3 = j * math.pi / n_points_per_dim
                y2 = vector_in_2d * math.sqrt(1 / eigen_values[2]) * math.cos(angle_3)
                y3 = vector_in_2d * math.sqrt(1 / eigen_values[3]) * math.sin(angle_3)
                s = Q @ np.array([y0, y1, y2, y3]).transpose()
                # print(s.transpose() @ MATRIX_P @ s)
                initial_condition_list.append([s[0], s[1], s[2], s[3], False])

    print(f"Generating {len(initial_condition_list)} conditions for training ...")

    return initial_condition_list


if __name__ == "__main__":
    screen_width = 600
    screen_height = 400
    world_width = 0.9 * 2 + 1
    scale = screen_width / world_width
    cart_y = 100  # TOP OF CART
    pole_width = 10.0
    pole_length = scale * 0.64
    cart_width = 50.0
    cart_height = 30.0
    target_width = 25
    target_height = 25

    viewer = rendering.Viewer(screen_width, screen_height)
    target_trans = rendering.Transform()
    # target = rendering.Image('./docs/target.svg', width=target_width, height=target_height)
    target = rendering.make_circle(12)
    target.set_color(.8, .8, .45)

    target.add_attr(target_trans)
    viewer.add_geom(target)

    l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
    axle_offset = cart_height / 4.0
    cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

    is_normal_operation = False
    if not is_normal_operation:
        cart.set_color(1.0, 0, 0)
    cart_trans = rendering.Transform()
    cart.add_attr(cart_trans)
    viewer.add_geom(cart)

    l, r, t, b = -pole_width / 2, pole_width / 2, pole_length - pole_width / 2, -pole_width / 2
    pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    pole.set_color(.8, .6, .4)
    pole_trans = rendering.Transform(translation=(0, axle_offset))
    pole.add_attr(pole_trans)
    pole.add_attr(cart_trans)
    viewer.add_geom(pole)

    # import time
    # viewer.render()
    # time.sleep(123)

    axle = rendering.make_circle(pole_width / 2)
    axle.add_attr(pole_trans)
    axle.add_attr(cart_trans)
    axle.set_color(.5, .5, .8)
    viewer.add_geom(axle)
    track = rendering.Line((0, cart_y), (screen_width, cart_y))
    track.set_color(0, 0, 0)
    viewer.add_geom(track)
    _pole_geom = pole

    state = [-0., 0.6, 0.0, 0]
    x = state

    # Edit the pole polygon vertex
    pole = _pole_geom
    l, r, t, b = -pole_width / 2, pole_width / 2, pole_length - pole_width / 2, -pole_width / 2
    pole.v = [(l, b), (l, t), (r, t), (r, b)]

    cart_x = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    target_x = 0 * scale + screen_width / 2.0
    target_y = pole_length + cart_y

    cart_trans.set_translation(cart_x, cart_y)
    target_trans.set_translation(target_x, target_y)
    pole_trans.set_rotation(-x[2])
    mode = 'human'
    viewer.render(return_rgb_array=mode == 'rgb_array')
    import time

    time.sleep(20)
