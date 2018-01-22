import os
import argparse
import torch
import utils
import env
import batch_env
import network
import a2c
import train


def parse_args():
    parser = argparse.ArgumentParser(description='a2c')

    # optimization
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--weight_norm', action='store_true')

    # enviroment
    parser.add_argument('--env_name', default='PongNoFrameskip-v4')
    parser.add_argument('--num_envs', type=int, default=16)
    parser.add_argument('--traj_len', type=int, default=5,
                        help='number of forward steps in a2c')
    parser.add_argument('--num_frames', type=int, default=4,
                        help='number of stacked frames per state')
    parser.add_argument('--frame_size', type=int, default=84)
    parser.add_argument('--frame_skip', type=int, default=4)
    parser.add_argument('--total_frames', type=int, default=int(80e6),
                        help='total # of frame for envs (sum)')
    # misc
    parser.add_argument('--seed', type=int, default=100901)
    parser.add_argument('--log_per_steps', type=int, default=int(10e3))
    parser.add_argument('--output', default='dev/')
    parser.add_argument('--exp_name', type=str, default='')

    args = parser.parse_args()
    args.output = os.path.join(args.output, args.env_name)
    if args.exp_name:
        args.output = '%s_%s' % (args.output, args.exp_name)
    args.frames_per_env = args.total_frames // args.num_envs
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    utils.set_all_seeds(args.seed)
    cfg = utils.Config(vars(args))
    cfg.dump(os.path.join(cfg.output, 'cfg.txt'))

    # train env
    env_thunk = lambda : env.AtariEnv(
        cfg.env_name, cfg.frame_skip, cfg.num_frames, cfg.frame_size, True)
    train_env = batch_env.BatchSyncEnv(env_thunk, cfg.num_envs)
    train_env.create_processes()

    # eval env
    eval_env = env.AtariEnv(
        cfg.env_name, cfg.frame_skip, cfg.num_frames, cfg.frame_size, False)
    evaluator = lambda model, logger: train.evaluate(eval_env, 3, model, logger)

    num_actions = train_env.num_actions
    net = network.build_default_network(
        cfg.num_frames, cfg.frame_size, num_actions, None, args.weight_norm, None)
    print(net)

    a2c_model = a2c.A2C(net.cuda())
    train.train(a2c_model, train_env, cfg, evaluator)
