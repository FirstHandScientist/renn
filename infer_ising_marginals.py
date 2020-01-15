import sys
import os
import argparse
from parse import parse
import json
import random
import shutil
import copy
import pickle
import torch
from torch import cuda
import numpy as np
import time
import logging

import pandas as pd
from torch.nn.init import xavier_uniform_
from functools import partial

from rens.models import ising as ising_models
from rens.utils.utils import corr, l2, l1, get_scores, binary2unary_marginals, get_freer_gpu
from rens.models.inference_ising import bp_infer, p2cbp_infer, mean_field_infer, bethe_net_infer, kikuchi_net_infer

# Model options
parser = argparse.ArgumentParser()
parser.add_argument('--n', default=5, type=int, help="ising grid size")
parser.add_argument('--exp_iters', default=5, type=int, help="how many times to run the experiment")
parser.add_argument('--msg_iters', default=200, type=int, help="max number of inference steps")
parser.add_argument('--enc_iters', default=200, type=int, help="max number of encoder grad steps")
parser.add_argument('--eps', default=1e-5, type=float, help="threshold for stopping inference/sgd")
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--state_dim', default=200, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--agreement_pen', default=10, type=float, help='')
parser.add_argument('--device', default='cpu', type=str, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--optmz_alpha', action='store_true', help='whether to optimize alphas in alpha bp')
parser.add_argument('--damp', default=0.5, type=float, help='')
parser.add_argument('--unary_std', default=1.0, type=float, help='')

parser.add_argument('--task', default="infer_result_n5_std1.0.txt", type=str, help='the task to carry on.')
parser.add_argument('--sleep', default=1, type=int, help='sleep a time a beginning.')


    

def run_marginal_exp(args, seed=3435, verbose=True):
    '''compare the marginals produced by mean field, loopy bp, and inference network'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ising = ising_models.Ising(args.n, args.unary_std)
    ising.push2device(args.device)

    log_Z = ising.log_partition_ve()
    unary_marginals, binary_marginals = ising.marginals()
    p_get_scores = partial(get_scores, true_ub=(unary_marginals, binary_marginals))
    ising.generate_region_graph()

    all_scores ={}
    if 'mf' in args.method:
        time_start = time.time()
        mrgnl_mf = mean_field_infer(ising, args)
        scores_mf = p_get_scores(test_ub=(mrgnl_mf[1], mrgnl_mf[2]))
        time_end = time.time()
        all_scores['mf'] = {'l1': scores_mf[0], 'corr': scores_mf[1],\
                            'time': time_end - time_start,\
                            'logz_err': torch.abs(log_Z - mrgnl_mf[0]).to('cpu').data.numpy()}
        print('Finish {} ...'.format('mf'))

    # loopy bp
    if 'bp' in args.method:
        time_start = time.time()
        mrgnl_bp = bp_infer(ising, args, 'lbp')
        scores_bp = p_get_scores(test_ub=(mrgnl_bp[1], mrgnl_bp[2]))
        time_end = time.time()
        all_scores['bp'] = {'l1': scores_bp[0], 'corr': scores_bp[1],\
                            'time': time_end - time_start,\
                            'logz_err': torch.abs(log_Z - mrgnl_bp[0]).to('cpu').data.numpy()}
        print('Finish {} ...'.format('bp'))

    # damped bp
    if 'dbp' in args.method:
        time_start = time.time()
        mrgnl_dbp = bp_infer(ising, args, 'dampbp')
        scores_dbp = p_get_scores(test_ub=(mrgnl_dbp[1], mrgnl_dbp[2]))
        time_end = time.time()
        all_scores['dbp'] = {'l1': scores_dbp[0], 'corr': scores_dbp[1], \
                             'time': time_end - time_start, \
                             'logz_err': torch.abs(log_Z - mrgnl_dbp[0]).to('cpu').data.numpy()}
        print('Finish {} ...'.format('dbp'))
        time_end = time.time()



    # alhpa bp
    if 'abp' in args.method:
        time_start = time.time()
        mrgnl_abp = bp_infer(ising, args, 'alphabp')
        scores_abp = p_get_scores(test_ub=(mrgnl_abp[1], mrgnl_abp[2]))
        time_end = time.time()
        all_scores['abp'] = {'l1': scores_abp[0], 'corr': scores_abp[1], 'time': time_end - time_start}
        print('Finish {} ...'.format('alpha bp'))
        

    # Generalized belief propagation
    if 'gbp' in args.method:
        time_start = time.time()
        mrgnl_gbp = p2cbp_infer(ising, args)
        scores_gbp = p_get_scores(test_ub=(mrgnl_gbp[1].to(unary_marginals), mrgnl_gbp[2].to(unary_marginals)))
        time_end = time.time()
        all_scores['gbp'] = {'l1': scores_gbp[0], 'corr': scores_gbp[1], \
                             'time': time_end - time_start, \
                             'logz_err': torch.abs(log_Z - mrgnl_gbp[0]).to('cpu').data.numpy()}
        print('Finish {} ...'.format('gbp'))

    # Bethe net
    if 'bethe' in args.method:
        time_start = time.time()
        bethe_net = bethe_net_infer(ising, args)
        mrgnl_bethe = bethe_net()
        scores_bethe = p_get_scores(test_ub=(mrgnl_bethe[1], mrgnl_bethe[2]))
        time_end = time.time()
        all_scores['bethe'] = {'l1': scores_bethe[0], 'corr': scores_bethe[1], \
                               'time': time_end - time_start, \
                               'logz_err': torch.abs(log_Z - mrgnl_bethe[0]).to('cpu').data.numpy()}
        print('Finish {} ...'.format('bethe'))




    # Generalized net
    if 'kikuchi' in args.method:
        time_start = time.time()
        kikuchi_net = kikuchi_net_infer(ising, args)
        mrgnl_kikuchi = kikuchi_net()
        scores_kikuchi = p_get_scores(test_ub=(mrgnl_kikuchi[1].to(unary_marginals), mrgnl_kikuchi[2].to(unary_marginals)))
        time_end = time.time()
        all_scores['kikuchi'] = {'l1': scores_kikuchi[0], 'corr': scores_kikuchi[1],\
                                 'time': time_end - time_start, \
                                 'logz_err': torch.abs(log_Z - mrgnl_kikuchi[0]).to('cpu').data.numpy()}
        print('Finish {} ...'.format('kikuchi'))
        

    if verbose:
        print("This round results:\n {}".format(pd.DataFrame(all_scores)))
    return all_scores

    
if __name__ == '__main__':
    args = parser.parse_args()
    # run multiple number of experiments, and collect the stats of performance.
    # args.method = ['mf', 'bp', 'gbp', 'bethe', 'kikuchi']
    # parsing the task first
    _, num_node, unary_std = parse("{}_n{}_std{}.txt", args.task)
    args.n = int(num_node)
    args.unary_std = float(unary_std)
    

    args.method = ['mf','bp', 'dbp','gbp','bethe', 'kikuchi']
    
    time.sleep(np.random.randint(args.sleep))
    if args.device != 'cpu':
        args.device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))
    
    results = {key: {'l1':[], 'corr':[], 'logz_err':[], 'time':[]} for key in args.method}

    for k in range(args.exp_iters):
        d = run_marginal_exp(args, k+10)
        for key, value in d.items():
            for crt, score in value.items():
                results[key][crt].append(score)

    for key, value in results.items():
        for crt, score in value.items():
            results[key][crt] = {'mu': np.array(score).mean().round(decimals=6), \
                                 'std': np.std(np.array(score)).round(decimals=6)}
    
    print('Average results: \n')
    with pd.option_context('display.max_rows', None, 'display.max_columns', 1000):
        print(pd.DataFrame.from_dict(results, orient='index'))

    pkl_dir = args.task.replace('txt', 'pkl')
    with open(pkl_dir, 'wb') as handle:
        pickle.dump(results, handle)

    sys.exit(0)

  
