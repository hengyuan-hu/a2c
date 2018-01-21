import torch
import utils


class Experience:
    """One experience is a batch of trajectories."""

    def __init__(self, num_envs, traj_len, state_shape):
        self.num_envs = num_envs
        self.traj_len = traj_len
        self.state_shape = state_shape

        # buffers
        self.states = torch.cuda.FloatTensor(traj_len, num_envs, *state_shape)
        self.actions = torch.cuda.LongTensor(traj_len, num_envs)
        self.rewards = torch.cuda.FloatTensor(traj_len, num_envs)
        self.non_ends = torch.cuda.FloatTensor(traj_len, num_envs)
        self.returns = torch.cuda.FloatTensor(traj_len+1, num_envs)
        self.timestep_count = 0

    def add_timestep(self, states, actions, rewards, non_ends):
        """add a timestep obtained from batch_env.step()

        """
        self.states[self.timestep_count] = states
        self.actions[self.timestep_count] = actions
        self.rewards[self.timestep_count] = rewards
        self.non_ends[self.timestep_count] = non_ends

        self.timestep_count += 1

    def compute_returns(self, gamma, next_vals):
        """
        next_vals is used for bootstrapping returns
        """
        utils.assert_eq(self.timestep_count, self.traj_len)
        self.timestep_count = 0

        self.returns[-1] = next_vals
        for i in range(self.traj_len-1, -1, -1):
            self.returns[i] = (self.returns[i+1] * self.non_ends[i] * gamma
                               + self.rewards[i])

        batch = self.traj_len * self.num_envs
        states = self.states.view(batch, *self.state_shape).contiguous()
        actions = self.actions.view(batch, 1).contiguous()
        returns = self.returns[:-1].view(batch, 1).contiguous()
        return states, actions, returns


def test_experience1():
    exp = Experience(3, 3, (2,))
    actions = torch.cuda.LongTensor((1, 1, 0))

    states = torch.rand(3, 2).cuda()
    rewards = torch.rand(3, 1).cuda()
    non_ends = torch.ones(3, 1).cuda()
    print('adding t0')
    exp.add_timestep(states, actions, rewards, non_ends)

    states = torch.rand(3, 2).cuda()
    rewards = torch.rand(3, 1).cuda()
    non_ends[0] = 0
    print('adding t1')
    exp.add_timestep(states, actions, rewards, non_ends)

    states = torch.rand(3, 2).cuda()
    rewards = torch.rand(3, 1).cuda()
    non_ends[0] = 1
    non_ends[1] = 0
    print('adding t2')
    exp.add_timestep(states, actions, rewards, non_ends)

    _, _, returns = exp.compute_returns(1, torch.ones(3, 1).cuda())

    print('rewards')
    print(exp.rewards)

    print('non_ends')
    print(exp.non_ends)

    print('returns')
    print(returns.view(3, 3))
    print(returns.view(3, 3) - exp.rewards)


if __name__ == '__main__':
    test_experience1()
