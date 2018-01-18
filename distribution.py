import torch
import torch.nn as nn
import utils


def categorical_sample(logits, greedy):
    utils.assert_eq(logits.dim(), 2)

    if greedy:
        samples = logits.max(1, keepdim=True)[1]
    else:
        probs = nn.functional.softmax(logits, 1)
        samples = probs.multinomial(1)
    return samples


def categorical_logp(logits, actions, calc_entropy):
    utils.assert_eq(logits.dim(), 2)

    logp = nn.functional.log_softmax(logits, 1)
    actions_logp = logp.gather(1, actions)

    if not calc_entropy:
        return actions_logp

    p = nn.functional.softmax(logits, 1)
    entropys = -(p * logp).sum(1, keepdim=True)
    return actions_logp, entropys


if __name__ == '__main__':
    from torch.autograd import Variable
    #TODO: check shape of logp and entropy
    logits = Variable(torch.rand(4, 3)) # batch, num_actions
    actions = Variable(torch.LongTensor([0, 1, 2, 1]))
    actions = actions.view(-1, 1)

    actions_logp, entropys = categorical_logp(logits, actions, True)
    print(actions_logp)
    print(entropys)
