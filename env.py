"""Wrapper of OpenAI Gym enviroment"""
from collections import deque
import cv2
import numpy as np
import gym


def preprocess_frame(screen, prev_screen, frame_size):
    screen = np.maximum(screen, prev_screen)
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(gray, (frame_size, frame_size))
    frame = frame.astype(np.float32, copy=False)
    frame /= 255.0
    return frame


class AtariEnv:
    def __init__(self,
                 name,
                 frame_skip,
                 num_frames,
                 frame_size,
                 one_life,
                 *,
                 no_op_start=30,
                 record=False,
                 output_dir=None):
        # env
        self.name = name
        self.env = gym.make(name)
        self.num_actions = self.env.action_space.n
        self.lives = self.env.env.ale.lives()

        if record:
            self.env = gym.wrappers.Monitor(self.env, output_dir, force=True)

        # frame
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.state_shape = (num_frames, frame_size, frame_size)
        self.frame_queue = deque(maxlen=num_frames)

        # game
        self.one_life = one_life
        self.no_op_start = no_op_start

        self.prev_screen = None
        self.end = True
        self.epsd_reward = 0.0
        self.recent_epsd_rewards = deque(maxlen=10)

    @property
    def average_rewards(self):
        """running average of episode rewards"""
        return np.mean(self.recent_epsd_rewards)

    def set_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()

    def reset(self):
        """reset env and frame queue, return initial state """
        self.recent_epsd_rewards.append(self.epsd_reward)
        self.end = False
        self.epsd_reward = 0.0

        for _ in range(self.num_frames-1):
            empty_frame = np.zeros(
                (self.frame_size, self.frame_size), dtype=np.float32)
            self.frame_queue.append(empty_frame)

        screen = self.env.reset()
        self.prev_screen = screen
        # no_op_start
        n = np.random.randint(0, self.no_op_start + 1)
        for i in range(n):
            screen, reward, *_ = self.env.step(0)
            self.epsd_reward += reward
            if i == n - 2: # next to the last
                self.prev_screen = screen

        frame = preprocess_frame(screen, self.prev_screen, self.frame_size)
        self.frame_queue.append(frame)
        return np.array(self.frame_queue)

    def step(self, action):
        """Perform action and return frame sequence and reward.

        return:
            state: [frames] of length num_frames, 0 if fewer is available
            reward: float
        """
        assert not self.end, 'Acting on an ended environment'

        screen = None
        reward = 0
        for i in range(self.frame_skip):
            if i > 0:
                self.prev_screen = screen

            screen, r, self.end, info = self.env.step(action)
            reward += r

            if self.one_life and info['ale.lives'] < self.lives:
                self.end = True

            if self.end:
                break

        frame = preprocess_frame(screen, self.prev_screen, self.frame_size)
        self.prev_screen = screen
        self.frame_queue.append(frame)
        state = np.array(self.frame_queue)
        self.epsd_reward += reward

        return state, np.sign(reward)

    def close(self):
        self.env.close()


if __name__ == '__main__':
    env = AtariEnv('SpaceInvadersNoFrameskip-v4', 4, 4, 84, False)
    """{
        0: 'Noop',
        1: 'Fire',
        2: 'Right',
        3: 'Left',
        4: 'RightFire',
        5: 'LeftFire'
    }"""
