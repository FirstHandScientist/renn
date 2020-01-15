import numpy as np
import torch
import os

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


def binary2unary_marginals(binary_idx, binary_marginals, n):

    unary_marginals_all = [[] for _ in range(n**2)]
    for k, (i,j) in enumerate(binary_idx):            
        binary_marginal = binary_marginals[k]
        unary_marginals_all[i].append(binary_marginal.sum(1))
        unary_marginals_all[j].append(binary_marginal.sum(0))            
    unary_marginals = [torch.stack(unary, 0).mean(0)[1] for unary in unary_marginals_all]
    
    return torch.stack(unary_marginals, 0)

def get_binary_layer_of_region_graph(graph):
    """
    Input: the region graph, with multiple layers of regions

    Ouput: (idx, region layer)
    """
    assert hasattr(graph, "region_layers"), "Region does not has attr region_layers"
    
    for key, layer in graph.region_layers.items():
        if len(layer[0]) == 2 :
            return (key, layer)

def get_binary_marginals_of_region_graph(graph, binary_regions, key='belief'):
    """
    Compute the binary marginals of region graph.
    binary_regions: list of regions (with two nodes) for which to compute marginals.
    """
    _, binary_region_layer = get_binary_layer_of_region_graph(graph)
    
    binary_marginals = []
    
    for idx, pair in enumerate(binary_regions):
        if pair in binary_region_layer:
            # for already gathered belief in gbp
            binary_marginals.append(graph.nodes[pair][key].values)
        else:
            # for belief not gathered in gbp
            pair_belief = 0
            parents_of_pair = graph.get_supernode_of(pair)
            for p_node in parents_of_pair:
                to_marginal_idx = tuple(sorted(set(p_node) - set(pair)))
                p_belief = graph.nodes[p_node][key].copy()
                p_belief.marginalize([str(i) for i in to_marginal_idx], inplace=True)
                p_belief.normalize(inplace=True)
                pair_belief += p_belief.values

            
            binary_marginals.append( pair_belief / len(parents_of_pair))

    return torch.stack(binary_marginals, 0)
    
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)
