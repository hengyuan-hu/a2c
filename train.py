import os
import time
import numpy as np
import torch
import torch.nn as nn
import utils
from experience import Experience


def update(model, optim, next_vals, exps, gamma, ent_coef, max_grad_norm, logger):
    states, actions, returns = exps.compute_returns(gamma, next_vals)
    with model.train():
        loss, vals_loss, actions_loss, entropys = model.loss(
            states, actions, returns, ent_coef)

        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
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
    # action_dist = np.zeros(env.num_actions)

    best_avg_rewards = -float('inf')
    # TODO: lrschedule='linear'

    # reset
    states, *_ = env.step(None)
    t = time.time()
    for fid in range(config.frames_per_env):
        actions = model.get_actions(states, False)
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

        if (fid + 1) % config.log_per_steps == 0:
            num_updates = (fid + 1) // config.traj_len
            total_frames = (fid + 1) * env.num_envs
            frame_rate = int(config.log_per_steps * env.num_envs / (time.time() - t))
            logger.write('Step %s, Total Frames: %s, Updates: %s, Frame Rate: %d' % (
                utils.num2str(fid + 1),
                utils.num2str(total_frames),
                utils.num2str(num_updates),
                frame_rate))
            logger.log(delimiter=', ')

            avg_rewards = evaluator(model, logger)
            if avg_rewards > best_avg_rewards:
                best_avg_rewards = avg_rewards
                path = os.path.join(config.output, 'net.pth')
                torch.save(model.net.state_dict(), path)

            t = time.time()


def evaluate(env, num_epsd, model, logger):
    action_dist = np.zeros(env.num_actions)
    total_rewards = np.zeros(num_epsd)
    epsd_idx = 0
    epsd_iters = 0
    max_epsd_iters = 108000

    state = env.reset()
    while epsd_idx < num_epsd:
        state = torch.from_numpy(state).unsqueeze(0).cuda()
        action = model.get_actions(state, False)[0][0]
        action_dist[action] += 1
        state, _ = env.step(action)
        epsd_iters += 1

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
    logger.write('>>>Eval: action dist:')
    probs = list(action_dist/action_dist.sum())
    for action, prob in enumerate(probs):
        logger.write('\t action: %d, p: %.4f' % (action, prob))

    return avg_rewards
