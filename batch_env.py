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
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)
        self.num_actions = self.envs[0].num_actions

        self.processes = []
        self.cv = MasterWorkersCV(self.num_envs)
        self.shared_buffer = SharedBuffer(self.num_envs, envs[0].state_shape)

    @property
    def state_shape(self):
        batch = self.num_envs
        num_chan = self.envs[0].num_frames
        size = self.envs[0].frame_size
        return (batch, num_chan, size, size)

    def create_processes(self, total_frames, traj_len):
        for eid in range(self.num_envs):
            args = (self.envs[eid],
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
    def _single_env_process(env, cv, shared_buffer, eid, total_frames, traj_len):
        for fid in range(total_frames):
            cv.wait_for_work(eid)

            new_traj = (fid % traj_len == 0)
            if env.end:
                if new_traj:
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
                shared_buffer.non_ends[eid] = 1.0

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


if __name__ == '__main__':
    from env import AtariEnv

    num_envs = 128
    envs = [AtariEnv('SpaceInvadersNoFrameskip-v4', 4, 4, 84)
            for _ in range(num_envs)]
    benv = BatchSyncEnv(envs)
    benv.create_processes(200000, 4)
    actions = np.random.randint(0, envs[0].num_actions, (num_envs,))
