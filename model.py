import torch.nn as nn
from noisy_net import NoisyLinear
import utils


class ActorCriticNetwork(nn.Module):
    def __init__(self, conv, fc, value, policy):
        super(ActorCriticNetwork, self).__init__()
        self.conv = conv
        self.fc = fc
        self.value = value
        self.policy = policy

    def forward(self, x):
        assert x.data.max() <= 1.0

        batch = x.size(0)
        feat = self.conv(x)
        feat = feat.view(batch, -1)
        feat = self.fc(feat)
        val = self.value(feat)
        pi = self.policy(feat)
        return val, pi


# ---------------------------------------
def _build_default_conv(in_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1),
        nn.ReLU()
    )
    return conv


def _build_fc(dims, num_actions, noise_std):
    if noise_std > 0:
        layer_func = lambda indim, outdim: NoisyLinear(indim, outdim, noise_std)
    else:
        layer_func = nn.Linear

    layers = []
    for i in range(0, len(dims) - 1):
        layers.append(layer_func(dims[i], dims[i+1]))
        layers.append(nn.ReLU())

    fc = nn.Sequential(*layers)
    value = layer_func(dims[-1], 1)
    policy = nn.Sequential(
        layer_func(dims[-1], num_actions), nn.Softmax(dim=1))
    return fc, value, policy


def build_default_network(in_channels, in_size, num_actions, noise_std, net_file):
    conv = _build_default_conv(in_channels)

    in_shape = (1, in_channels, in_size, in_size)
    fc_in = utils.count_output_size(in_shape, conv)
    fc_hid = 512
    fc, value, policy = _build_fc([fc_in, fc_hid], num_actions, noise_std)

    net = ActorCriticNetwork(conv, fc, value, policy)
    utils.init_net(net, net_file)
    return net


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    for noise_std in [0.0, 0.2]:
        x = torch.rand(1, 4, 84, 84)
        x = Variable(x)
        net = build_default_network(4, 84, 6, 0, None)
        print(net)
        print('Training?:', net.training)
        v, pi = net(x)
        print('value:', v)
        print('pi:', pi)
