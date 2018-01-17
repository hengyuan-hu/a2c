import multiprocessing as mp
import ctypes
import numpy as np
import torch
import utils
from env import AtariEnv
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
        self.states = create_shared_nparray((num_envs,) + state_shape, np.float32)
        self.rewards = create_shared_nparray((num_envs,), np.float32)
        self.non_ends = create_shared_nparray((num_envs,), np.float32)
        self.actions = create_shared_nparray((num_envs,), np.int32)

    def to_cuda_tensors(self):
        states = torch.from_numpy(self.states).cuda()
        rewards = torch.from_numpy(self.rewards).cuda()
        non_ends = torch.from_numpy(self.non_ends).cuda()
        return states, rewards, non_ends

    def to_cpu_tensors(self):
        states = torch.from_numpy(self.states).clone()
        rewards = torch.from_numpy(self.rewards).clone()
        non_ends = torch.from_numpy(self.non_ends).clone()
        return states, rewards, non_ends


class BatchSyncEnv:
    def __init__(self, env_thunk, num_envs):
        # self.envs = envs # TODO: use thunk
        self.env_thunk = env_thunk
        self.env = env_thunk()
        self.name = self.env.name
        self.num_envs = num_envs
        self.num_actions = self.env.num_actions
        self.state_shape = self.env.state_shape

        self.processes = []
        self.cv = MasterWorkersCV(self.num_envs)
        self.shared_buffer = SharedBuffer(self.num_envs, self.state_shape)

    def create_processes(self, total_frames, traj_len):
        for eid in range(self.num_envs):
            args = (self.env_thunk,
                    self.cv,
                    self.shared_buffer,
                    eid,
                    total_frames,
                    traj_len)

            p = mp.Process(target=self._single_env_process, args=args)
            self.processes.append(p)

        for p in self.processes:
            p.start()

    @staticmethod
    def _single_env_process(env_thunk, cv, shared_buffer, eid, total_frames, traj_len):
        utils.set_all_seeds(eid)
        env = env_thunk()

        for fid in range(-1, total_frames):
            cv.wait_for_work(eid)

            if env.end:
                new_traj = ((fid+1) % traj_len == 0)
                if new_traj:
                    # reset only if next state is the start of a new trajectory
                    state = env.reset()
                else:
                    # dummy state
                    state_shape = shared_buffer.states[eid].shape
                    state = np.zeros(state_shape, dtype=np.float32)

                shared_buffer.rewards[eid] = 0.0
                shared_buffer.non_ends[eid] = 0.0
            else:
                state, reward = env.step(shared_buffer.actions[eid])
                shared_buffer.rewards[eid] = reward
                shared_buffer.non_ends[eid] = 1.0 - np.float32(env.end)

            shared_buffer.states[eid][:]= state

            cv.work_done_maybe_notify_master(eid)

        print('env %d: %d frames exhausted' % (eid, total_frames))

    def step(self, actions):
        """actor should be a function that takes states and produce actions

        use the state -> reward -> action model
        """
        # prepare actions
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

    for i in range(len(states)):
        state = states[i]
        name = name_tpl % (i)
        cv2.imwrite(name, cv2.resize(state[-1], (800, 800)))


if __name__ == '__main__':
    from env import AtariEnv
    utils.set_all_seeds(100009)

    num_envs = 8
    env_thunk = lambda : AtariEnv('SpaceInvadersNoFrameskip-v4', 4, 4, 84)
    # envs = [AtariEnv('SpaceInvadersNoFrameskip-v4', 4, 4, 84)
    #         for _ in range(num_envs)]
    benv = BatchSyncEnv(env_thunk, num_envs)
    benv.create_processes(200000, 4)
    actions = np.random.randint(0, benv.num_actions, (num_envs,))
    # actions = np.array(list(range(benv.num_actions)))

    # name_tpl = 'dev/batch_env2_%s/' % benv.name
    # for sidx in range(300):
    #     name_tpl_ = name_tpl + 'env%d_' + ('step%d.png' % sidx)
    #     states, rewards, non_ends = benv.step(actions)
    #     states = states.cpu().numpy()
    #     save_frames(name_tpl_, states)
