#!/usr/bin/env python3
import sys
import os

import argparse
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
import ising as ising_models
from torch.nn.init import xavier_uniform_

parser = argparse.ArgumentParser()

# Model options
parser.add_argument('--n', default=5, type=int, help="ising grid size")
parser.add_argument('--exp_iters', default=5, type=int, help="how many times to run the experiment")
parser.add_argument('--msg_iters', default=200, type=int, help="max number of inference steps")
parser.add_argument('--enc_iters', default=200, type=int, help="max number of encoder grad steps")
parser.add_argument('--eps', default=1e-5, type=float, help="threshold for stopping inference/sgd")
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--state_dim', default=200, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--agreement_pen', default=10, type=float, help='')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')

def corr(t1, t2):
  return np.corrcoef(t1.data.cpu().numpy(), t2.data.cpu().numpy())[0][1]

def l2(t1, t2):
  return ((t1 - t2)**2).mean().item()

def l1(t1, t2):
  return ((t1 - t2).abs()).mean().item()

def bp_infer(ising, msg_iters, solver, optmz_alpha=False):
  '''Do belief propagation with given solver'''
  messages = torch.zeros(ising.n**2, ising.n**2, 2).fill_(0.5).cuda()
  unary_marginals_lbp, binary_marginals_lbp = ising.lbp_marginals(messages)
  if optmz_alpha:
    optimizer = torch.optim.Adam([ising.alpha_wgt], lr=0.05)

  for i in range(msg_iters):
    if solver is 'lbp':
      messages = ising.lbp_update(1, messages).detach()
      unary_marginals_lbp_new, binary_marginals_lbp_new = ising.lbp_marginals(messages)
    elif solver is 'alphabp':
      new_messages = ising.alphabp_update(1, messages)
      unary_marginals_lbp_new, binary_marginals_lbp_new = ising.alphabp_marginals(new_messages)
      if optmz_alpha:
        optimizer.zero_grad()
        loss = ising.free_energy_mf(unary_marginals_lbp_new)
        loss.backward()
        optimizer.step()
        
      messages = new_messages.detach()
      


    delta_unary = l2(unary_marginals_lbp_new, unary_marginals_lbp) 
    delta_binary = l2(binary_marginals_lbp_new[:, 1, 1], binary_marginals_lbp[:, 1, 1])
    delta = delta_unary + delta_binary
    if delta < args.eps:
      break
    unary_marginals_lbp = unary_marginals_lbp_new.detach()
    binary_marginals_lbp = binary_marginals_lbp_new.detach()

  log_Z_lbp = -ising.bethe_energy(unary_marginals_lbp, binary_marginals_lbp)  
  return log_Z_lbp, unary_marginals_lbp, binary_marginals_lbp


def main(args):

  cuda.set_device(args.gpu)  

  def run_marginal_exp(n, seed=3435):
    '''compare the marginals produced by mean field, loopy bp, and inference network'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    ising = ising_models.Ising(n)
    encoder = ising_models.TransformerInferenceNetwork(n, args.state_dim, args.num_layers)

    if args.gpu >= 0:
      ising.cuda()
      
      ising.mask = ising.mask.cuda()
      # number of neighbors - 1?
      ising.degree = ising.degree.cuda()  
      encoder.cuda()

    log_Z = ising.log_partition_ve()
    unary_marginals, binary_marginals = ising.marginals()

    # mean field
    unary_marginals_mf = torch.zeros(ising.n**2).fill_(0.5).cuda()
    binary_marginals_mf = ising.mf_binary_marginals(unary_marginals_mf)
    
    for i in range(args.msg_iters):
      unary_marginals_mf_new = ising.mf_update(1, unary_marginals_mf)
      binary_marginals_mf_new = ising.mf_binary_marginals(unary_marginals_mf_new)
      delta_unary = l2(unary_marginals_mf_new, unary_marginals_mf) 
      delta_binary = l2(binary_marginals_mf_new[:, 1, 1], binary_marginals_mf[:, 1, 1])
      delta = delta_unary + delta_binary
      if delta < args.eps:
        break
      
      unary_marginals_mf = unary_marginals_mf_new.detach()
      binary_marginals_mf = binary_marginals_mf_new.detach()

    log_Z_mf = -ising.bethe_energy(unary_marginals_mf, binary_marginals_mf)
    log_Z_mf_energy = -ising.free_energy_mf(unary_marginals_mf)


    
    # loopy bp
    log_Z_lbp, unary_marginals_lbp, binary_marginals_lbp = bp_infer(ising, args.msg_iters, 'lbp')
    # alhpa bp
    log_Z_alphabp, unary_marginals_alphabp, binary_marginals_alphabp = bp_infer(ising, args.msg_iters, 'alphabp', True)

    # inference network
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    unary_marginals_enc = torch.zeros_like(unary_marginals).fill_(0.5)
    binary_marginals_enc = torch.zeros_like(binary_marginals).fill_(0.25)
    
    for i in range(args.enc_iters):
      optimizer.zero_grad()
      unary_marginals_enc_new, binary_marginals_enc_new = encoder(ising.binary_idx)
      bethe_enc = ising.bethe_energy(unary_marginals_enc_new, binary_marginals_enc_new)
      agreement_loss = encoder.agreement_penalty(ising.binary_idx, unary_marginals_enc_new,
                                                 binary_marginals_enc_new)
      (bethe_enc + args.agreement_pen*agreement_loss).backward()
      optimizer.step()
      delta_unary = l2(unary_marginals_enc_new, unary_marginals_enc) 
      delta_binary = l2(binary_marginals_enc_new[:, 1, 1], binary_marginals_enc[:, 1, 1])
      delta = delta_unary + delta_binary
      if delta < args.eps:
        break
      
      unary_marginals_enc = unary_marginals_enc_new.detach()
      binary_marginals_enc = binary_marginals_enc_new.detach()
      
    log_Z_enc = -ising.bethe_energy(unary_marginals_enc, binary_marginals_enc)  
      
    # compute the marginals
    marginals = torch.cat([unary_marginals, binary_marginals[:, 1, 1]], 0)
    marginals_mf = torch.cat([unary_marginals_mf, binary_marginals_mf[:, 1, 1]], 0)
    marginals_lbp = torch.cat([unary_marginals_lbp, binary_marginals_lbp[:, 1, 1]], 0)
    marginals_alphabp = torch.cat([unary_marginals_alphabp, binary_marginals_alphabp[:, 1, 1]], 0)
    marginals_enc = torch.cat([unary_marginals_enc, binary_marginals_enc[:, 1, 1]], 0)

    # compute the corrcoefs
    corr_unary_mf = corr(unary_marginals, unary_marginals_mf)
    corr_unary_lbp = corr(unary_marginals, unary_marginals_lbp)
    corr_unary_alphabp = corr(unary_marginals, unary_marginals_alphabp)
    corr_unary_enc = corr(unary_marginals, unary_marginals_enc)
    corr_binary_mf = corr(binary_marginals[:, 1, 1], binary_marginals_mf[:, 1, 1])
    corr_binary_lbp = corr(binary_marginals[:, 1, 1], binary_marginals_lbp[:, 1, 1])
    corr_binary_alphabp = corr(binary_marginals[:, 1, 1], binary_marginals_alphabp[:, 1, 1])
    corr_binary_enc = corr(binary_marginals[:, 1, 1], binary_marginals_enc[:, 1, 1])
    corr_mf = corr(marginals, marginals_mf)
    corr_lbp = corr(marginals, marginals_lbp)
    corr_alphabp = corr(marginals, marginals_alphabp)
    corr_enc = corr(marginals, marginals_enc)

    l1_unary_mf = l1(unary_marginals, unary_marginals_mf)
    l1_unary_lbp = l1(unary_marginals, unary_marginals_lbp)
    l1_unary_alphabp = l1(unary_marginals, unary_marginals_alphabp)
    l1_unary_enc = l1(unary_marginals, unary_marginals_enc)
    
    l1_binary_mf = l1(binary_marginals[:, 1, 1], binary_marginals_mf[:, 1, 1])
    l1_binary_lbp = l1(binary_marginals[:, 1, 1], binary_marginals_lbp[:, 1, 1])
    l1_binary_alphabp = l1(binary_marginals[:, 1, 1], binary_marginals_alphabp[:, 1, 1])
    l1_binary_enc = l1(binary_marginals[:, 1, 1], binary_marginals_enc[:, 1, 1])
    l1_mf = l1(marginals, marginals_mf)
    l1_lbp = l1(marginals, marginals_lbp)
    l1_alphabp = l1(marginals, marginals_alphabp)
    l1_enc = l1(marginals, marginals_enc)

    log_Z_diff_mf = abs(log_Z.item() - log_Z_mf.item())
    log_Z_diff_lbp = abs(log_Z.item() - log_Z_lbp.item())
    log_Z_diff_alphabp = abs(log_Z.item() - log_Z_alphabp.item())
    log_Z_diff_enc = abs(log_Z.item() - log_Z_enc.item())
    print('log_Z error:', log_Z_diff_mf, abs(log_Z_mf_energy.item() - log_Z_mf.item()), log_Z_diff_lbp, log_Z_diff_enc)
    
    return [corr_mf, corr_lbp, corr_alphabp, corr_enc, l1_mf, l1_lbp, l1_alphabp, l1_enc]

  data = []
  # run multiple number of experiments, and collect the stats of performance.
  for k in range(args.exp_iters):
    d = run_marginal_exp(args.n, k+1)
    data.append(d)
    print(k+1, d)

  data = np.array(data).mean(0)
  stats = "N: %d, Corr_mf: %.4f, Corr_lbp: %.4f, Corr_alphabp: %.4f,, Corr_inf: %.4f, " + \
    "L1_mf: %.6f, L1_lbp: %.6f, L1_alphabp: %.6f,, L1_inf: %.6f, "
  print(stats % (args.n, data[0], data[1], data[2], data[3], data[4], data[5], data[6],data[7]))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
