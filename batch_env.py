import multiprocessing as mp
import ctypes
import numpy as np
import torch
import utils
from condv import MasterWorkersCV


def create_shared_nparray(shape, dtype):
    dtype2ctype = {
        np.float32: ctypes.c_float,
        np.int32: ctypes.c_int32,
    }
    if dtype not in dtype2ctype:
        assert False, 'dtype %s is not supported' % dtype

    ctype = dtype2ctype[dtype]
    size = int(np.prod(shape))
    raw_array = mp.RawArray(ctype, size)
    array = np.frombuffer(raw_array, dtype=dtype)
    array = array.reshape(shape)
    return array


class SharedBuffer:
    def __init__(self, num_envs, state_shape):
        self.next_states = create_shared_nparray(
            (num_envs,) + state_shape, np.float32)
        self.rewards = create_shared_nparray((num_envs,), np.float32)
        self.non_ends = create_shared_nparray((num_envs,), np.float32)
        self.actions = create_shared_nparray((num_envs,), np.int32)

    def to_cuda_tensors(self):
        next_states = torch.from_numpy(self.next_states).cuda()
        rewards = torch.from_numpy(self.rewards).cuda()
        non_ends = torch.from_numpy(self.non_ends).cuda()
        return next_states, rewards, non_ends

    def to_cpu_tensors(self):
        next_states = torch.from_numpy(self.next_states).clone()
        rewards = torch.from_numpy(self.rewards).clone()
        non_ends = torch.from_numpy(self.non_ends).clone()
        return next_states, rewards, non_ends


class BatchSyncEnv:
    def __init__(self, env_thunk, num_envs):
        self.env_thunk = env_thunk
        self.env = env_thunk()
        self.name = self.env.name
        self.num_envs = num_envs
        self.num_actions = self.env.num_actions
        self.state_shape = self.env.state_shape

        self.processes = []
        self.cv = MasterWorkersCV(self.num_envs)
        self.shared_buffer = SharedBuffer(self.num_envs, self.state_shape)

    def create_processes(self):
        for eid in range(self.num_envs):
            args = (self.env_thunk, self.cv, self.shared_buffer, eid)
            p = mp.Process(target=self._single_env_step, args=args)
            self.processes.append(p)

        for p in self.processes:
            p.start()

    @staticmethod
    def _single_env_step(env_thunk, cv, shared_buffer, eid):
        utils.set_all_seeds(eid)
        env = env_thunk()

        while True:
            cv.wait_for_work(eid)

            if env.end: # should only happen at the very beginning
                next_state = env.reset()
                shared_buffer.non_ends[eid] = 1.0
            else:
                next_state, reward = env.step(shared_buffer.actions[eid])
                shared_buffer.rewards[eid] = reward
                shared_buffer.non_ends[eid] = 1.0 - np.float32(env.end)
                if env.end:
                    next_state = env.reset()
            shared_buffer.next_states[eid][:] = next_state

            cv.work_done_maybe_notify_master(eid)

    def step(self, actions):
        """
        state -> action -> reward, non_end, next_state
        """
        # prepare actions
        if actions is not None:
            self.shared_buffer.actions[:] = actions

        self.cv.sync_round()
        states, rewards, non_ends = self.shared_buffer.to_cuda_tensors()
        return states, rewards, non_ends


def save_frames(name_tpl, states):
    import os
    import cv2

    dirname = os.path.dirname(name_tpl)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i, state in enumerate(states):
        state = states[i]
        name = name_tpl % (i)
        cv2.imwrite(name, cv2.resize(state[-1], (800, 800)))


if __name__ == '__main__':
    from env import AtariEnv
    utils.set_all_seeds(100009)

    num_envs = 8
    # env_thunk = lambda : AtariEnv('SpaceInvadersNoFrameskip-v4', 4, 4, 84)
    env_thunk = lambda : AtariEnv('PongNoFrameskip-v4', 4, 4, 84)
    benv = BatchSyncEnv(env_thunk, num_envs)
    benv.create_processes()
    actions = np.random.randint(0, benv.num_actions, (num_envs,))

    name_tpl = 'dev/batch_env_%s/' % benv.name
    for sidx in range(300):
        name_tpl_ = name_tpl + 'env%d_' + ('step%d.png' % sidx)
        states, rewards, non_ends = benv.step(actions)
        states = states.cpu().numpy() * 255.0
        save_frames(name_tpl_, states)
