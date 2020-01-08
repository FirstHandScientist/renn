import numpy as np
import torch


def corr(t1, t2):
    if t1.device.type == 'cpu':
        return np.corrcoef(t1.data.numpy(), t2.data.numpy())[0][1]
    else:
        return np.corrcoef(t1.data.cpu().numpy(), t2.data.cpu().numpy())[0][1]

def l2(t1, t2):
  return ((t1 - t2)**2).mean().item()

def l1(t1, t2):
  return ((t1 - t2).abs()).mean().item()


def get_scores(true_ub, test_ub):
    """
    Compute the l1 and corr.
    """
    unary_marginals, binary_marginals = true_ub
    unary_marginals_mf, binary_marginals_mf = test_ub
    marginals = torch.cat([unary_marginals, binary_marginals[:, 1, 1]], 0)
    marginals_mf = torch.cat([unary_marginals_mf, binary_marginals_mf[:, 1, 1]], 0)

    corr_unary_mf = corr(unary_marginals, unary_marginals_mf)
    corr_binary_mf = corr(binary_marginals[:, 1, 1], binary_marginals_mf[:, 1, 1])
    corr_mf = corr(marginals, marginals_mf)

    l1_unary_mf = l1(unary_marginals, unary_marginals_mf)
    l1_binary_mf = l1(binary_marginals[:, 1, 1], binary_marginals_mf[:, 1, 1])
    l1_mf = l1(marginals, marginals_mf)

    return (l1_mf, corr_mf)
