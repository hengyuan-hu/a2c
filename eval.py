import os
import argparse
import torch
import numpy as np
import utils
from env import AtariEnv
import batch_env
import network
import a2c
import train


def parse_args():
    parser = argparse.ArgumentParser(description='eval a2c')
    parser.add_argument('--seed', type=int, default=1214201)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--greedy', action='store_true')

    args = parser.parse_args()
    return args


# TODO: unify this into train.evaluate
def evaluate(env, num_epsd, model, greedy):
    actions = np.zeros(env.num_actions)
    total_rewards = np.zeros(num_epsd)
    epsd_idx = 0
    epsd_iters = 0
    max_epsd_iters = 108000

    state = env.reset()
    while epsd_idx < num_epsd:
        state = torch.from_numpy(state).unsqueeze(0).cuda()
        action = model.get_actions(state, greedy)[0][0]
        actions[action] += 1
        state, _ = env.step(action)
        epsd_iters += 1

        if env.end or epsd_iters >= max_epsd_iters:
            total_rewards[epsd_idx] = env.epsd_reward
            print('>>>Eval: [%d/%d], rewards: %s' %
                  (epsd_idx+1, num_epsd, total_rewards[epsd_idx]))

            if epsd_idx < num_epsd - 1: # leave last reset to next run
                state = env.reset()

            epsd_idx += 1
            epsd_iters = 0


    avg_rewards = total_rewards.mean()
    print('>>>Eval: avg total rewards: %.2f' % avg_rewards)
    print('>>>Eval: actions dist:')
    probs = list(actions/actions.sum())
    for action, prob in enumerate(probs):
        print('\t action: %d, p: %.4f' % (action, prob))

    return avg_rewards


if __name__ == '__main__':
    args = parse_args()

    utils.set_all_seeds(args.seed)
    cfg = utils.Config.load(os.path.join(args.output, 'cfg.txt'))
    eval_env = AtariEnv(
        cfg.env_name, cfg.frame_skip, cfg.num_frames, cfg.frame_size, False)

    netfile = os.path.join(args.output, 'net.pth')
    num_actions = eval_env.num_actions
    net = network.build_default_network(
        cfg.num_frames, cfg.frame_size, num_actions, None, cfg.weight_norm, netfile)
    a2c_model = a2c.A2C(net.cuda())

    evaluate(eval_env, 5, a2c_model, args.greedy)
