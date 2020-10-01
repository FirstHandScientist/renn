import sys
import os
from parse import parse
import argparse
import json
import random
import time
import shutil
import copy
import pickle
import torch
from torch import cuda
from collections import deque
import numpy as np
import time
import logging
import pandas as pd
from torch.nn.init import xavier_uniform_
from functools import partial
from torch.utils.data import DataLoader
from rens.models import ising as ising_models
from rens.utils.utils import corr, l2, l1, get_scores, binary2unary_marginals
from rens.utils.utils import clip_optimizer_params, get_freer_gpu
from rens.models.inference_ising import bp_infer, p2cbp_infer, mean_field_infer, bethe_net_infer, kikuchi_net_infer
from rens.utils.utils import generate_dataset
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
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--agreement_pen', default=10, type=float, help='')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--optmz_alpha', action='store_true', help='whether to optimize alphas in alpha bp')
parser.add_argument('--damp', default=0.5, type=float, help='')
parser.add_argument('--unary_std', default=1.0, type=float, help='')

parser.add_argument('--structure', default='grid', type=str, help='the graph type of ising model: grid | full_connected')
parser.add_argument('--data_dir', default='data', type=str, help='dataset dir')
parser.add_argument('--data_regain', action='store_true', help='if to regenerate dataset')
parser.add_argument('--train_size', default=100, type=int, help='the size of training samples')
parser.add_argument('--valid_size', default=100, type=int, help='the size of valid samples')
parser.add_argument('--test_size', default=50, type=int, help='the size of testing samples')
parser.add_argument('--batch_size', default=100, type=int, help='the size of batch samples')
parser.add_argument('--train_iters', default=200, type=int, help='the number of iterations to train')
parser.add_argument('--infer', default='ve', type=str, help='the inference method to use')
parser.add_argument('--fifo_maxlen', default=4, type=int, help='')
parser.add_argument('--conv_tol', default=1e-3, type=float, help='')
parser.add_argument('--t2i_ratio', default=20, type=int, help='training to inference ratio')
parser.add_argument('--sleep', default=1, type=int, help='sleep a time a beginning.')
parser.add_argument('--device', default='cpu', type=str, help='which gpu to use')

parser.add_argument('--task', default="log_msg/train_grid_score_n5_std0.1_pen0_algo.ve.txt", type=str, help='the task to carry on.')

    

def main(args, seed=3435, verbose=True):
    '''compare the marginals produced by mean field, loopy bp, and inference network'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ising = ising_models.Ising(n=args.n, mrf_para=args.mrf_para, unary_std=args.unary_std, device=args.device, structure=args.structure)
    

    
    ising.push2device(args.device)
      
        
        # number of neighbors - 1?
        

    # exact computation on ising
    unary_marginals, binary_marginals = ising.marginals()
    
    log_Z = ising.log_partition_ve()
    
    
    # prepare dataset
    train_data_loader = DataLoader(dataset=args.dataset['train'],
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=False)
    test_data_loader = DataLoader(dataset=args.dataset['test'],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    
    
    

    # training with exact inference (variable elimination)
    if args.infer == 've':
        inference_method = ising.log_partition_ve
    elif args.infer == 'mf':
        inference_method = partial(mean_field_infer, ising=ising, args=args)
    elif args.infer == 'lbp':
        inference_method = partial(bp_infer, ising=ising, args=args, solver='lbp')
    elif args.infer == 'dbp':
        inference_method = partial(bp_infer, ising=ising, args=args, solver='dampbp')
    elif args.infer == 'gbp':
        inference_method = p2cbp_infer(ising=ising, args=args)
    elif args.infer == 'bethe':
        inference_method = bethe_net_infer(ising=ising, args=args)        
    elif args.infer == 'kikuchi':
        inference_method = kikuchi_net_infer(ising=ising, args=args)
        

    else:
        print("Your assigned inference method is not available.")
        os.exit(1)
    # define the optimizer
    if args.infer in ['ve', 'mf', 'lbp', 'dbp']:
        optimizer = torch.optim.Adam([ising.unary, ising.binary], lr=args.lr * 10)
    elif args.infer in ['gbp']:
        optimizer = torch.optim.Adam([ising.unary, ising.binary], lr=args.lr * 3)

    elif args.infer in ['bethe', "kikuchi"]:
        optimizer = torch.optim.Adam([{"params": ising.parameters() , "lr":args.lr * 3},
                                      {"params": inference_method.encoder.parameters() , \
                                       "lr": args.lr}])



    best_nll = float('inf')
    avg_time_per_iter = []
    fifo_loss = deque(maxlen=args.fifo_maxlen)
    for cur_iter in range(args.train_iters):
        time_begin = time.time()
            
        if args.infer in ['kikuchi', 'bethe']:
            # for renn and bethe cases
            _, _ , _ = inference_method()
            log_Z_computer = partial(inference_method.neg_free_energy)
            # maybe, use the inferred marginals to get a partition function, unary, binary ---> bethe energy
            train_avg_nll = ising.trainer(train_data_loader, log_Z_computer, args.t2i_ratio, optimizer, args.clip, args.agreement_pen, args.infer)
        elif args.infer in ['ve', 'mf', 'lbp', 'dbp']:
            # for mf, lbp, gbp cases
            train_avg_nll = ising.trainer(train_data_loader, inference_method, args.t2i_ratio, optimizer, args.clip, args.agreement_pen, args.infer)
        elif args.infer in ['gbp']:
            # ising._init_disfactor()
            # ising.attach_region_factors(ising.region_graph)
            _ = inference_method()
            log_Z_computer = partial(inference_method.neg_free_energy)
            train_avg_nll = ising.trainer(train_data_loader, log_Z_computer, args.t2i_ratio, optimizer, args.clip, args.agreement_pen, args.infer)

        
        time_end = time.time()
        test_avg_nll = ising.test_nll(test_data_loader, inference_method)
        if best_nll > test_avg_nll:
            best_nll = test_avg_nll
        print("Iter: {:5d} | train_avg_nll:{:8.5f} | test_avg_nll: {:8.5f} | best_nll: {:8.5f} | true_nll {:8.5f}, iter_time {:8.5f}".format(cur_iter, train_avg_nll, test_avg_nll, best_nll, args.true_nll['test'], time_end - time_begin))
        avg_time_per_iter.append(time_end - time_begin)
        fifo_loss.append(train_avg_nll.detach().data)
        
        # check if converge
        if len(fifo_loss) == args.fifo_maxlen:
            list_loss = list(fifo_loss)
            diff = torch.FloatTensor([torch.abs(i-j) for i,j in zip(list_loss[:-1], list_loss[1:])])
            # if the differences between epochs all are smaller than loss_tol
            # then do not training further
            if (diff < args.conv_tol).sum() >= diff.size(0):
                break

    avg_time_per_iter = np.array(avg_time_per_iter).mean()
    print("Average time per epoch: {:8.5f}".format(avg_time_per_iter/args.t2i_ratio))
    return best_nll

    
if __name__ == '__main__':
    args = parser.parse_args()
    # run multiple number of experiments, and collect the stats of performance.
    _, num_node, mrf_para, unary_std, penalty, algorithm = parse("{}score_n{}_{}_std{}_pen{}_algo.{}.txt", args.task)
    args.n = int(num_node)
    args.unary_std = float(unary_std)
    args.agreement_pen = float(penalty)
    args.infer = str(algorithm)
    args.mrf_para = mrf_para
    assert (args.mrf_para.startswith('G') or args.mrf_para.startswith('U')) \
        and float(args.mrf_para[1:])>0 , 'Un-supported distribution to sample for potential factors.'

    time.sleep(np.random.randint(args.sleep))
    if args.device != 'cpu':
        args.device = torch.device('cuda:{}'.format(int(get_freer_gpu()) ))


    args = generate_dataset(args)
    d = main(args, np.random.randint(1000))

  
