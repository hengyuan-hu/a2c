import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from noisy_net import NoisyLinear
import utils


class ActorCriticNetwork(nn.Module):
    def __init__(self, conv, fc, value, pi_logit):
        super(ActorCriticNetwork, self).__init__()
        self.conv = conv
        self.fc = fc
        self.value = value
        self.pi_logit = pi_logit

    def forward(self, x):
        assert x.data.max() <= 1.0

        batch = x.size(0)
        feat = self.conv(x)
        feat = feat.view(batch, -1)
        feat = self.fc(feat)
        val = self.value(feat)
        pi_logit = self.pi_logit(feat)
        return val, pi_logit


# ---------------------------------------
def _build_default_conv(in_channels, wn):
    if wn:
        wrapper = lambda x: weight_norm(x)
    else:
        wrapper = lambda x: x

    conv = nn.Sequential(
        wrapper(nn.Conv2d(in_channels, 32, 8, 4)),
        nn.ReLU(),
        wrapper(nn.Conv2d(32, 64, 4, 2)),
        nn.ReLU(),
        wrapper(nn.Conv2d(64, 64, 3, 1)),
        nn.ReLU()
    )
    return conv


def _build_fc(dims, num_actions, noise_std, wn):
    if noise_std is not None and noise_std > 0:
        assert not wn, 'weight norm is not supported for NoisyNet'
        layer_func = lambda indim, outdim: NoisyLinear(indim, outdim, noise_std)
    else:
        layer_func = lambda indim, outdim: weight_norm(
            nn.Linear(indim, outdim), dim=None)

    layers = []
    for i in range(0, len(dims) - 1):
        layers.append(layer_func(dims[i], dims[i+1]))
        layers.append(nn.ReLU())

    fc = nn.Sequential(*layers)
    value = layer_func(dims[-1], 1)
    pi_logit = layer_func(dims[-1], num_actions)
    return fc, value, pi_logit


def build_default_network(
        in_channels, in_size, num_actions, noise_std, wn, net_file):

    conv = _build_default_conv(in_channels, wn)

    in_shape = (1, in_channels, in_size, in_size)
    fc_in = utils.count_output_size(in_shape, conv)
    fc_hid = 512
    fc, value, pi_logit = _build_fc([fc_in, fc_hid], num_actions, noise_std, wn)

    net = ActorCriticNetwork(conv, fc, value, pi_logit)
    utils.init_net(net, net_file)
    return net


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    for noise_std in [0.0, 0.2]:
        x = torch.rand(1, 4, 84, 84)
        x = Variable(x)
        net = build_default_network(4, 84, 6, noise_std, None)
        print(net)
        print('Training?:', net.training)
        v, pi = net(x)
        print('value:', v)
        print('pi:', pi)
