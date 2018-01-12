"""Wrapper of OpenAI Gym enviroment"""
from collections import deque
import gym
import cv2
import numpy as np


def preprocess_frame(observ, output_size):
    gray = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
    output = cv2.resize(gray, (output_size, output_size))
    output = output.astype(np.float32, copy=False)
    return output


class AtariEnv:
    def __init__(self,
                 name,
                 num_frames,
                 frame_size,
                 record=False,
                 output_dir=None):

        self.name = name
        self.env = gym.make(name)
        self.num_actions = self.env.action_space.n

        if record:
            self.env = gym.wrappers.Monitor(self.env, output_dir, force=True)

        self.frame_size = frame_size
        self.num_frames = num_frames
        self.frame_queue = deque(maxlen=num_frames)

        self.end = True
        self.lives = None
        self.total_reward = 0.0

    def set_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()

    def reset(self):
        """reset env and frame queue, return initial state """
        self.end = False
        self.lives = None
        self.total_reward = 0.0

        for _ in range(self.num_frames-1):
            empty_frame = np.zeros((self.frame_size, self.frame_size))
            self.frame_queue.append(empty_frame)

        state0 = self.env.reset()
        state0 = preprocess_frame(state0, self.frame_size)
        self.frame_queue.append(state0)
        return np.array(self.frame_queue)

    def step(self, action):
        """Perform action and return frame sequence and reward.
        Return:
        state: [frames] of length num_frames, 0 if fewer is available
        reward: float
        """
        assert not self.end, 'Acting on an ended environment'

        observ, reward, self.end, info = self.env.step(action)
        observ = preprocess_frame(observ, self.frame_size)
        self.frame_queue.append(observ) # left is automatically popped
        self.total_reward += reward

        if self.lives is None:
            self.lives = info['ale.lives']
        if info['ale.lives'] < self.lives:
            self.end = True

        state = np.array(self.frame_queue)
        reward = np.sign(reward)
        return state, reward

    def close(self):
        self.env.close()


if __name__ == '__main__':

    env = AtariEnv('SpaceInvadersNoFrameskip-v4', 4, 84)
    """{
        0: 'Noop',
        1: 'Fire',
        2: 'Right',
        3: 'Left',
        4: 'RightFire',
        5: 'LeftFire'
    }"""
