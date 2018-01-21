from contextlib import contextmanager
import torch.nn as nn
from torch.autograd import Variable
import distribution as dist
import utils


class A2C:
    def __init__(self, net):
        self.net = net
        self.net.train(False)

    @contextmanager
    def train(self):
        self.net.train()
        yield
        self.net.train(False)

    def parameters(self):
        return self.net.parameters()

    def loss(self, states, actions, returns, ent_coef):
        assert self.net.training
        states = Variable(states)
        actions = Variable(actions)
        returns = Variable(returns)
        vals, pi_logits = self.net(states)

        advs = returns - vals
        vals_loss = 0.5 * advs.pow(2)
        actions_logp, entropys = dist.categorical_logp(pi_logits, actions, True)
        actions_loss = -(Variable(advs.data) * actions_logp)

        utils.assert_eq(vals_loss.size(), actions_logp.size())
        utils.assert_eq(vals_loss.size(), entropys.size())

        loss = actions_loss + vals_loss - entropys * ent_coef
        return loss, vals_loss, actions_loss, entropys

    def get_values(self, states):
        """
        params:
            states: Tensor, [batch, channel, size, size]

        return:
            vals: Tensor, [batch, 1]
        """
        # utils.assert_eq(type(states), torch.cuda.FloatTensor)
        assert not self.net.training

        states = Variable(states, volatile=True)
        vals, _ = self.net(states)
        return vals.data

    def get_actions(self, states, greedy, print_=False):
        """
        params:
            states: Tensor, [batch, channel, size, size]

        return:
            actions: Tensor, [batch, 1]
        """
        # utils.assert_eq(type(states), torch.cuda.FloatTensor)
        assert not self.net.training

        states = Variable(states, volatile=True)
        _, pi_logits = self.net(states)
        if print_:
            probs = nn.functional.softmax(pi_logits, 1)
            print(probs.data * 10000)

        actions = dist.categorical_sample(pi_logits, greedy)
        return actions.data


if __name__ == '__main__':
    from network import build_default_network

    num_frames = 2
    frame_size = 84
    num_actions = 5
    net = build_default_network(num_frames, frame_size, num_actions, None, None)
    a2c = A2C(net)

    batch = 2
    states = torch.rand(batch, num_frames, frame_size, frame_size)
    actions = torch.LongTensor([0] * batch).view(-1, 1)
    returns = torch.rand(batch, 1)
    with a2c.train():
        loss = a2c.loss(states, actions, returns, 0.01)
        print(loss)

    actions = a2c.get_actions(states, False)
