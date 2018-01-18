import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
from experience import Experience
import time


def update(model, optim, next_vals, exps, gamma, ent_coef, max_grad_norm, logger):
    states, actions, returns = exps.compute_returns(gamma, next_vals)
    # print('updating')
    # print(actions.view(5, 16))
    with model.train():
        loss, vals_loss, actions_loss, entropys = model.loss(
            states, actions, returns, ent_coef)

        loss = loss.mean()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
        # print('total norm', total_norm)
        optim.step()
        optim.zero_grad()
        logger.append('loss', loss.data[0])
        logger.append('val_loss', vals_loss.mean().data[0])
        logger.append('action_loss', actions_loss.mean().data[0])
        logger.append('entropy', entropys.mean().data[0])


def train(model, env, config, evaluator):
    logger = utils.Logger(os.path.join(config.output, 'train_log.txt'))
    optim = torch.optim.RMSprop(model.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)
    exp_buffer = Experience(env.num_envs, config.traj_len, env.state_shape)
    action_dist = np.zeros(env.num_actions)

    best_avg_rewards = -float('inf')
    # TODO: lrschedule='linear'

    # reset
    states, *_ = env.step(None)
    t = time.time()
    for fid in range(config.frames_per_env):
        actions = model.get_actions(states, False)
        # print(actions.view(-1))
        actions_np = actions.cpu().numpy().reshape(-1)
        next_states, rewards, non_ends = env.step(actions_np)
        exp_buffer.add_timestep(states, actions, rewards, non_ends)

        if (fid + 1) % config.traj_len == 0:
            next_vals = model.get_values(next_states)
            update(model,
                   optim,
                   next_vals,
                   exp_buffer,
                   config.gamma,
                   config.ent_coef,
                   config.max_grad_norm,
                   logger)

        states = next_states

        log_interval = 10000
        if (fid + 1) % log_interval == 0:
            num_updates = (fid + 1) // config.traj_len
            total_frames = (fid + 1) * env.num_envs
            frame_rate = int(log_interval * env.num_envs / (time.time() - t))
            logger.write(
                'Step %d, Total Frames: %d Frame rate: %d, Updates: %d' % (
                    fid+1, total_frames, frame_rate, num_updates))
            logger.write(logger.log())

            avg_rewards = evaluator(model, logger)
            if avg_rewards > best_avg_rewards:
                best_avg_rewards = avg_rewards
                path = os.path.join(config.output, 'net.pth')
                torch.save(model.net.state_dict(), path)

            t = time.time()


def evaluate(env, num_epsd, model, logger):
    actions = np.zeros(env.num_actions)
    total_rewards = np.zeros(num_epsd)
    epsd_idx = 0
    epsd_iters = 0
    max_epsd_iters = 108000

    state = env.reset()
    while epsd_idx < num_epsd:
        # # save state
        # import cv2, os
        # filename = 'dev/debug/epsd%d_f%d.png' % (epsd_idx, epsd_iters)
        # dirname = os.path.dirname(filename)
        # if not os.path.exists(dirname):
        #     os.makedirs(dirname)
        # frame = state[0] * 255.0
        # cv2.imwrite(filename, cv2.resize(frame, (800, 800)))

        state = torch.from_numpy(state).unsqueeze(0).cuda()
        # if epsd_iters % 30 == 0:
        #     action = model.get_actions(state, False, True)[0][0]
        # else:
        #     action = model.get_actions(state, False)[0][0]
        action = model.get_actions(state, False)[0][0]
        actions[action] += 1
        state, _ = env.step(action)
        epsd_iters += 1

        # if epsd_iters % 100 == 0:
        #     print('state max:', state.max())
        #     print('state min:', state.min())

        # if epsd_iters > 400:
        #     break

        if env.end or epsd_iters >= max_epsd_iters:
            total_rewards[epsd_idx] = env.epsd_reward
            logger.write('>>>Eval: [%d/%d], rewards: %s' %
                         (epsd_idx+1, num_epsd, total_rewards[epsd_idx]))

            if epsd_idx < num_epsd - 1: # leave last reset to next run
                state = env.reset()

            epsd_idx += 1
            epsd_iters = 0


    avg_rewards = total_rewards.mean()
    logger.write('>>>Eval: avg total rewards: %.2f' % avg_rewards)
    logger.write('>>>Eval: actions dist:')
    probs = list(actions/actions.sum())
    for action, prob in enumerate(probs):
        logger.write('\t action: %d, p: %.4f' % (action, prob))

    return avg_rewards


# def save_frames(name_tpl, states):
#     import os
#     import cv2

#     dirname = os.path.dirname(name_tpl)
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)

#     for i in range(len(states)):
#         state = states[i]
#         name = name_tpl % (i)
#         cv2.imwrite(name, cv2.resize(state[-1], (800, 800)))
