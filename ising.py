import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import math
from pgmpy.models import RegionGraph
from pgmpy.factors.discrete import DiscreteFactor


def logadd(x, y):
    d = torch.max(x,y)  
    return torch.log(torch.exp(x-d) + torch.exp(y-d)) + d    

def logsumexp(x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
        return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
        return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d    


class ResidualLayer(nn.Module):
  def __init__(self, in_dim = 100,
               out_dim = 100):
    super(ResidualLayer, self).__init__()
    self.lin1 = nn.Linear(in_dim, out_dim)
    self.lin2 = nn.Linear(out_dim, out_dim)

  def forward(self, x):
    return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class SelfAttention(nn.Module):
    def __init__(self, dim = 100):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.dense = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        # x : b x l x hidden
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attn_scores = torch.matmul(query, key.transpose(1,2)) # b x l x l
        attn_scores = attn_scores 
        if mask is not None:
            attn_scores = attn_scores + mask.to(attn_scores.device)
        attn_probs = F.softmax(attn_scores, dim= -1)
        context = torch.matmul(attn_probs, value) # b x l x h
        out = self.ln(self.dense(context) + x)
        return out

class TransformerInferenceNetwork(nn.Module):
    def __init__(self, n, state_dim = 100, num_layers = 1, mlp_out_dim=4):
        super(TransformerInferenceNetwork, self).__init__()
        self.n = n
        self.node_emb = nn.Parameter(torch.randn(1, n**2, state_dim))
        self.attn_layers = nn.ModuleList([SelfAttention(state_dim) for _ in range(num_layers)])
        local_times = int(math.log(mlp_out_dim,2))
        self.mlp = nn.Sequential(ResidualLayer(state_dim*local_times, state_dim*local_times), 
                                 ResidualLayer(state_dim*local_times, state_dim*local_times), 
                                 nn.Linear(state_dim*local_times, mlp_out_dim))
        self.num_layers = num_layers
        self.mask = torch.zeros(n**2, n**2).fill_(0)
        self.binary_mlp = nn.Sequential(ResidualLayer(state_dim, state_dim),
                                 nn.Linear(state_dim, 1))
        self.state_dim = state_dim
        for i in range(n**2):
            for j in range(i+1, n **2):                        
                if i + 1 == j and (i+1) % n != 0:
                    self.mask[i][j] = 0
                    self.mask[j][i] = 0
                if j - i == n and i < n**2 - 1:
                    self.mask[i][j] = 0
                    self.mask[j][i] = 0

    def forward(self, binary_idx, unary = None, binary = None):
        x = self.node_emb
        for l in range(self.num_layers):
            x = self.attn_layers[l](x, self.mask) # 1 x n**2 x state_dim

        # here output of attn_layers is embedding vector e
        binary_features = []
        for (i,j) in binary_idx:
            emb_ij = torch.cat([x[0][i], x[0][j]], 0) # state_dim*2            
            binary_features.append(emb_ij)
        binary_features = torch.stack(binary_features, 0) # |E| x state_dim*2        
        binary_logits = self.mlp(binary_features) 
        # binary_logits = [h_1, h_2, ... h_n2]
        binary_prob = F.softmax(binary_logits, dim = 1)
        binary_marginals = binary_prob.view(-1, 2, 2)
        unary_marginals_all = [[] for _ in range(self.n**2)]
        for k, (i,j) in enumerate(binary_idx):            
            binary_marginal = binary_marginals[k]
            unary_marginals_all[i].append(binary_marginal.sum(1))
            unary_marginals_all[j].append(binary_marginal.sum(0))            
        unary_marginals = [torch.stack(unary, 0).mean(0)[1] for unary in unary_marginals_all]
        return torch.stack(unary_marginals), binary_marginals

    def agreement_penalty(self, binary_idx, unary_marginals, binary_marginals):
        loss = 0

        for k, (i,j) in enumerate(binary_idx):            
            binary_marginal = binary_marginals[k]
            loss += (binary_marginal.sum(1)[1] - unary_marginals[i])**2
            loss += (binary_marginal.sum(0)[1] - unary_marginals[j])**2
        return loss

class GeneralizedInferenceNetwork(TransformerInferenceNetwork):

    def forward(self, graph=None):
        region_idx = graph.region_layers["R0"]
        x = self.node_emb
        for l in range(self.num_layers):
            x = self.attn_layers[l](x, self.mask) # 1 x n**2 x state_dim

        # here output of attn_layers is embedding vector e
        region_features = []
        for region in region_idx:
            emb_rg = torch.cat([x[0][i] for i in region], 0) # state_dim*2            
            region_features.append(emb_rg)
            
        region_features = torch.stack(region_features, 0) # |R0| x state_dim*2        
        region_logits = self.mlp(region_features) 
        # region_logits = [h_1, h_2, ... h_n2]
        region_prob = F.softmax(region_logits, dim = 1)
        r0_beliefs = region_prob.view(len(region_idx), 2, 2, 2, 2)
        # cast to R1 and R2
        r1_beliefs, consist_error1= self.marginal_down(r0_beliefs, graph)
        r2_beliefs, consist_error2= self.marginal_down(r1_beliefs, graph)
        infered_beliefs = {"R0": r0_beliefs, "R1": r1_beliefs, "R2": r2_beliefs}
        
        return infered_beliefs, consist_error1 + consist_error2

    
    def marginal_down(self, r_beliefs, graph):
        """
        Given the beliefs in a layer, R0 or R1, cast to the marginals in children layer.
        
        """
        if r_beliefs.dim() - 1 == 4:
            # input is in R0
            r1_beliefs, mismatch = self.belief_mismatch_regions_below(cast_job="R0R1", graph=graph, p_beliefs=r_beliefs)

            return r1_beliefs, mismatch

        
        elif r_beliefs.dim() - 1 == 2:
            # input is in R1
            r2_beliefs, mismatch = self.belief_mismatch_regions_below(cast_job="R1R2", graph=graph, p_beliefs=r_beliefs)

            return r2_beliefs, mismatch


    def belief_mismatch_regions_below(self, cast_job, graph, p_beliefs):
        
        parent_layer=graph.region_layers[cast_job[:2]]
        child_layer=graph.region_layers[cast_job[2:]]
        mismatch = 0
        r1_beliefs = []
        for child in child_layer:
            r0_parents = graph.get_parents(child)
            r0_parents_beliefs = p_beliefs[[parent_layer.index(i) for i in r0_parents]]
            all_mgnl2child = self.mgnl2child(r0_parents, child, r0_parents_beliefs)

            child_belief = all_mgnl2child.mean(dim=0)

            r1_beliefs.append(child_belief)

            mismatch += torch.sum((all_mgnl2child - child_belief ) ** 2)

        return torch.stack(r1_beliefs), mismatch

        

    def mgnl2child(self, parents, child, parent_beliefs):
        """
        Calculation of cast marginals of parents to child nodes.
        Parameters
        ----------
        parents and child nodes, parent_beliefs
        
        Examples
        ----------
        >>> parents = [(1,2,11,12), (2, 3, 12, 13)]
        >>> child = (2,3)
        >>> parent_beliefs = torch.randn(2,2,2,2,2)
        """
        child_beliefs = []
        for idx, p_node in enumerate(parents):
            set_diff = set(p_node) - set(child)
            reduce_idx = [list(p_node).index(i) for i in set_diff]
            child_beliefs.append(parent_beliefs[idx].sum(dim=reduce_idx))

        return torch.stack(child_beliefs)

    def read_marginals(self, binary_idx, infer_beliefs, graph):
        """
        Read out all the unary and binary marginals from optimized network.

        """
        # 1. read binary marginals from infer_beliefs first and put at corresponding position in
        # binary_marginals
        r1_nodes = graph.region_layers["R1"]
        idx_map_list = [binary_idx.index(i) for i in r1_nodes]
                
        binary_marginals = torch.zeros(len(binary_idx), 2, 2)
        # feed in the marginals from infer_beliefs
        binary_marginals[idx_map_list] = infer_beliefs["R1"]
        
        # find the rest binary marginals
        idx_diff = set(binary_idx).difference(set(r1_nodes))
        for idx in idx_diff:
            p_nodes = [i for i in graph.region_layers["R0"] if graph._is_subset(idx, i)]
            p_beliefs_idx = [graph.region_layers["R0"].index(i) for i in p_nodes]
            p_nodes_beliefs = infer_beliefs["R0"][p_beliefs_idx]
            binary_marginals[binary_idx.index(idx)] = self.mgnl2child(p_nodes, idx, p_nodes_beliefs)

        # compute the unary
        unary_marginals_all = [[] for _ in range(self.n**2)]
        for k, (i,j) in enumerate(binary_idx):            
            binary_marginal = binary_marginals[k]
            unary_marginals_all[i].append(binary_marginal.sum(1))
            unary_marginals_all[j].append(binary_marginal.sum(0))            
        unary_marginals = [torch.stack(unary, 0).mean(0)[1] for unary in unary_marginals_all]
        return torch.stack(unary_marginals), binary_marginals


    def kikuchi_energy(self, log_phis, infer_beliefs, counts):
        """Compute the Kikuchi free energy"""
        energy = 0
        for key, beliefs in infer_beliefs.items():
            num_regions = beliefs.size(0)
            beliefs = beliefs.reshape(num_regions, -1)
            lphi = torch.from_numpy(log_phis[key]).reshape(num_regions, -1).type(beliefs.dtype)
            regions_energy = torch.sum(beliefs * (beliefs.log() - lphi), dim=1)
            layer_counts = torch.from_numpy(counts[key]).type(beliefs.dtype)
            energy += (layer_counts * regions_energy).sum()

        return energy
        


class Ising(nn.Module):
    def __init__(self, n):
        super(Ising, self).__init__()
        self.n = n
        self.unary = nn.Parameter(torch.randn(n**2))
        self.binary = nn.Parameter(torch.randn(n**2, n**2))
        self.alpha_wgt = nn.Parameter(torch.randn(n**2, n**2) * 0.01 + 1)
        self.mask = self.binary_mask(n)
        self.binary_idx = []
        for i in range(n**2):
            for j in range(n**2):
                if self.mask[i][j].item() > 0:
                    self.binary_idx.append((i,j))
        self.neighbors = [self.get_neighbor(k) for k in range(self.n**2)]
        self.degree = torch.Tensor([len(v)-1 for v in self.neighbors]).float()
        self.EPS = 1e-6
        self.region_graph = None
        self._init_disfactor()

    def _init_disfactor(self):
        binary = self.binary*self.mask
        unary = self.unary
        unary1 = self.unary
        unary0 = -self.unary
        # bethe free energy computation of unary part
        self.log_unary_factors = {}
        for i in range(len(unary)):
            self.log_unary_factors[(i,)] = DiscreteFactor([str(i)], [2], torch.stack([unary0[i], unary1[i]], 0).data.numpy())
        
        
        
        self.log_binary_factors = {}
        for k, (i,j) in enumerate(self.binary_idx):
            binary_factor = binary[i][j] # J_ij
            binary_factor = torch.stack([binary_factor, -binary_factor], 0)
            binary_factor = torch.stack([binary_factor, -binary_factor], 1) # 2 x 2
            d_binary_factor = DiscreteFactor([str(i), str(j)], [2,2], binary_factor.data.numpy())
            self.log_binary_factors[(i,j)] = d_binary_factor
            self.log_binary_factors[(j,i)] = d_binary_factor

        return self

    def binary_mask(self, n):
        # binary vector of size n**2 x n**2
        # denote the connectivity in the grid of ising model
        mask = torch.zeros(n**2, n**2)
        for i in range(n**2):
            for j in range(i+1, n **2):                        
                if i + 1 == j and (i+1) % n != 0:
                    mask[i][j] = 1
                if j - i == n and i < n**2 - 1:
                    mask[i][j] = 1
        return mask


    def generate_region_graph(self, step=1):
        '''Convert the pairwise Ising model (MRF) into a RegionGraph'''
        graph = RegionGraph()
        clusters = self._get_R0(step)
        graph.add_nodes_from(clusters)
        graph.cluster_variation()
        
        # calculate the factor for each region and attach to graph
        self.attach_region_factors(graph)
        self.region_graph = graph
        return self

    def attach_region_factors(self, graph):
        """
        Input: Graph
        Output: the region graph factors
        """
        if not hasattr(self, "log_layer_phi"):
            self.log_phis = {}
            
        for ri, regions in graph.region_layers.items():
            self.log_phis[ri] = []
            for region_j in regions:
                the_phi = self._region_factor(region_j)
                graph.nodes[region_j]["log_phi"] = the_phi
                try:
                    self.log_phis[ri].append(the_phi.values)
                except:
                    print("error")

        for key, value in self.log_phis.items():
            self.log_phis[key] = np.array(value)
        
        return self

    def _region_factor(self, node):
        """
        Input: node
        Example: (0, 1, 10, 11), (34, 35), (74,)
        
        Output: DiscreteFactor 
        """
        region_sum = DiscreteFactor(tuple([str(i) for i in node]), \
                                    [2] * len(node), \
                                    np.zeros(2 ** len(node)))
        if len(node) ==1:
            region_sum.sum(self.log_unary_factors[node], inplace=True)
            return region_sum
        
        elif len(node) ==2:
            for i in node:
                region_sum.sum(self.log_unary_factors[(i,)], inplace=True)
                
            region_sum.sum(self.log_binary_factors[node], inplace=True)
            return region_sum
        
        elif len(node) == 4:
            for i in node:
                region_sum.sum(self.log_unary_factors[(i,)], inplace=True)

            for pair in itertools.combinations(node, 2):
                if pair in self.log_binary_factors:
                    region_sum.sum(self.log_binary_factors[pair], inplace=True)

            return region_sum
            
        

    def _get_R0(self, step=1):
        """
        Do the R0 clustering.
        Given the ising model, return a list of clusters (tuples),
        where each tuple represent a cluster/region.
        Working for step = 1 now... need revise for generalized calculation.
        """
        regions = []
        n = self.n
        i, j = 0,0
        while i>=0 and i< n - step:
            j = 0
            while j>=0 and j< n - step:
                cornor = i * n + j
                candidate = [cornor + col + row * n for row in range(step+1) for col in range(step+1)]
                regions.append(tuple(candidate))
                j += step
                
            i += step

        return regions
    
    
    
    def broadcast_sum(self, indices, reduce_idx, factors):
        union_idx = set()
        k = []
        views = []
        for factor_idx in indices:
            assert(reduce_idx in factor_idx)
            union_idx = union_idx.union(set(factor_idx))
            views.append([])
            k.append(0)
        union_idx = list(union_idx)
        union_idx.sort()
        tensors = []
        new_idx = []
        for i, idx in enumerate(union_idx):
            if idx == reduce_idx:
                reduce_i = i
            else:
                new_idx.append(idx)
            for j, factor_idx in enumerate(indices):
                if idx in factor_idx:
                    views[j].append(factors[j].size(k[j]))
                    k[j] += 1
                else:
                    views[j].append(1)
        for j in range(len(k)):
            assert k[j] == len(indices[j])
    
        tensors = [factors[j].view(views[j]).contiguous() for j in range(len(indices))]
        sum_tensor = sum(tensors)
        self.new_factors.append([union_idx, sum_tensor])
        new_factor = logsumexp(sum_tensor, reduce_i)
        return [new_idx, new_factor]

    def sum_factor(self, factors, reduce_idx):
        indices = []
        idx_factors = []
        new_factors = []
        while factors:
            factor = factors.pop()
            if reduce_idx in factor[0]:
                indices.append(factor[0])
                idx_factors.append(factor[1])
            else:
                new_factors.append(factor)    
        new_factors.append(self.broadcast_sum(indices, reduce_idx, idx_factors))
        # print('')
        # print("reduce_idx", reduce_idx)
        # print("indices", indices)
        # print("new_factor", new_factors[-1][0])
        # print('')
        return new_factors

    def log_energy(self, x):
        # x = b x n**2
        # unary =  n**2
        # binary =  n**2 x n**2
        binary = self.binary*self.mask
        unary = self.unary
        unary_x = x * unary.unsqueeze(0) # b x n**2
        binary_x = torch.matmul(x, binary)*x # b x n**2        
        return (unary_x + binary_x).sum(1)


    def log_partition_ve(self, order = None):
        # calculate log partition of an ising model via variable elimination
        # unary : n**2 of unary log potentials
        # binary: n**2 x n**2 edge log potentials
        if order is None:
            order = list(range(self.n**2))
        n = self.n
        binary = self.binary*self.mask
        unary = self.unary
        factors = []
        for i in range(n**2):
            unary_factor = torch.stack([-unary[i], unary[i]], 0)
            factors.append([[i], unary_factor])
        for i in range(n**2):
            for j in range(i+1, n **2):                        
                if (i + 1 == j and (i+1) % n != 0) or (j - i == n and i < n**2 - 1):
                    binary_factor = torch.stack([binary[i][j], -binary[i][j]], 0)
                    binary_factor = torch.stack([binary_factor, -binary_factor], 1)
                    factors.append([[i, j], binary_factor])
        assert(len(factors) == n**2 + 2*n*(n-1))
        self.new_factors = []
        for i in order:
            factors = self.sum_factor(factors, i)
        log_Z = factors[0][-1]
        return log_Z

    def marginals(self):
        log_Z = self.log_partition_ve()
        log_Z.backward()
        unary_marginals = self.unary.grad
        binary_marginals = self.binary.grad
        unary_marginals = (unary_marginals + 1)*0.5
        binary_marginals_list = []
        for (i,j) in self.binary_idx:
            p_i1 = unary_marginals[i]
            p_i0 = 1 - p_i1
            p_j1 = unary_marginals[j]
            p_j0 = 1-p_j1
            p_i1j1 = 0.25*(binary_marginals[i][j]-1+2*p_i1+2*p_j1)
            p_i1j0 = p_i1 - p_i1j1
            p_i0j1 = p_j1 - p_i1j1
            p_i0j0 = p_i0 - p_i0j1
            binary_marginal = torch.stack([torch.stack([p_i0j0, p_i0j1], 0),
                                           torch.stack([p_i1j0, p_i1j1], 0)], 0)            
            binary_marginals_list.append(binary_marginal)
        return unary_marginals, torch.stack(binary_marginals_list, 0)

    def sample(self, samples, log_Z = None, new_factors = None):
        # grid version of forward-filtering backward-sampling
        n = self.n
        if log_Z is None:
            log_Z = self.log_partition_ve()
            new_factors = self.new_factors
        x = torch.zeros(samples, n**2).long()
        log_px = torch.zeros(samples, n**2)
        for i, factor in zip(reversed(range(n**2)), reversed(new_factors)):
            # print(i, n**2)
            assert(i == factor[0][0])
            idx = factor[0]
            factor_size = [samples] + list(factor[1].size())            
            f_expand = factor[1].unsqueeze(0).expand(factor_size)
            sample_size = [samples] + [1]*(len(factor_size)-1)
            for j, k in enumerate(idx[1:]):
                sample_k = x[:, k] 
                sample_expand_size = list(f_expand.size())
                sample_expand_size[j+2] = 1
                samples_k = sample_k.view(sample_size).expand(sample_expand_size)
                f_expand = torch.gather(f_expand, j+2, samples_k)
            f = f_expand.view(samples, factor[1].size(0))
            p = F.softmax(f, dim=1)
            s = torch.multinomial(p, 1)
            log_ps = torch.gather(p.log(), 1, s)
            x[:, i].copy_(s.squeeze(1))
            log_px[:, i].copy_(log_ps.squeeze(1))            
        return 2*x.float()-1, log_px.sum(1)
    
    def get_neighbor(self, k):
        i = k // self.n
        j = k % self.n
        n_ij = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]
        n = []
        for (a,b) in n_ij:
            if not(a == -1 or a == self.n or b == -1 or b == self.n):
                n.append(self.n*a + b)
        n.sort()
        return n
            
    def mf_update(self, num_iters = 1, mean = None):
        # mean : n ** 2 of mean-field marginals
        binary = self.binary*self.mask
        unary = self.unary
        if mean is None:
            mean = self.unary.new(self.n**2).fill_(0.5)
        for _ in range(num_iters):
            for n in np.random.permutation(range(self.n**2)):
                message = 0
                for k in self.neighbors[n]:
                    if n < k:
                        binary_nk = binary[n][k]
                    else:
                        binary_nk = binary[k][n]
                    binary_nk = binary_nk
                    mean_k = mean[k]*2-1
                    message += mean_k*binary_nk
                message += unary[n]
                mean[n] = F.sigmoid(2*message)
        return mean
    
    def mf_binary_marginals(self, unary_marginals):
        binary_marginals = []
        for (i, j) in self.binary_idx:
            p_i0j0 = (1-unary_marginals[i])*(1-unary_marginals[j])
            p_i0j1 = (1-unary_marginals[i])*unary_marginals[j]
            p_i1j0 = unary_marginals[i]*(1-unary_marginals[j])
            p_i1j1 = unary_marginals[i]*unary_marginals[j]
            binary_marginal = torch.stack([torch.stack([p_i0j0, p_i0j1], 0),
                                           torch.stack([p_i1j0, p_i1j1], 0)], 0)            
            binary_marginals.append(binary_marginal)
        return torch.stack(binary_marginals, 0)
    
    def lbp_update_node(self, n, unary, binary, messages):
        '''Update the messages sent from node n to its neighbors'''
        updated_message = []
        for k in self.neighbors[n]:
            # compute the message from node n to its neighbor k
            unary_factor = torch.stack([-unary[n], unary[n]], 0) # 2
            # why this comparison ?
            if n < k:
                binary_factor = binary[n][k]
            else:
                binary_factor = binary[k][n]
            binary_factor = torch.stack([binary_factor, -binary_factor], 0)
            binary_factor = torch.stack([binary_factor, -binary_factor], 1) # 2 x 2
            messages_jn = []
            for j in self.neighbors[n]:
                if j != k:
                    messages_jn.append(messages[j][n].log()) # 2
            messages_jn = torch.stack(messages_jn, 0).sum(0)# 2
            message = messages_jn + unary_factor
            message = message.unsqueeze(1) + binary_factor 
            log_message = logsumexp(message, 0) # 2
            message = F.softmax(log_message, dim = 0)
            updated_message.append(message.detach())
        return torch.stack(updated_message)


    def alphabp_update_node(self, n, unary, binary, messages):
        '''Update the messages sent from node n to its neighbors'''
        updated_message = []
        for k in self.neighbors[n]:
            # compute the message from node n to its neighbor k
            unary_factor = torch.stack([-unary[n], unary[n]], 0) # 2
            # why this comparison ?
            if n < k:
                binary_factor = binary[n][k]
                alpha = self.alpha_wgt[n][k]
            else:
                binary_factor = binary[k][n]
                alpha = self.alpha_wgt[k][n]

            binary_factor = torch.stack([binary_factor, -binary_factor], 0)
            binary_factor = torch.stack([binary_factor, -binary_factor], 1) # 2 x 2
            messages_jn = []
            for j in self.neighbors[n]:
                if j != k:
                    messages_jn.append(messages[j][n].log()) # 2
            messages_jn = torch.stack(messages_jn, 0).sum(0)# 2
            message = messages_jn + unary_factor

            old_message_kn = messages[k][n] + self.EPS
            old_message_nk = messages[n][k] + self.EPS

            message = message.unsqueeze(1) \
                + binary_factor * alpha \
                + old_message_kn.log() * (1 - alpha)
            log_message = logsumexp(message, 0) # 2
            log_message = log_message + old_message_nk.log() * (1 - alpha)
            # normalize and convert to real domain 
            message = F.softmax(log_message, dim = 0)
            # assert torch.isnan(message).sum()==0, "Encounter nan in message update..."
            if torch.isnan(message).sum()>0:
                print("Encounter nan in message update...")
            updated_message.append(message)
        return torch.stack(updated_message)

    def lbp_update(self, num_iters = 1, messages = None):
        binary = self.binary*self.mask
        unary = self.unary
        if messages is None:
            messages = self.unary.new(self.n**2, self.n**2, 2).fill_(0.5)
        for _ in range(num_iters):
            for n in np.random.permutation(range(self.n**2)):
                # update message from node n to its neighbors
                messages[n][self.neighbors[n]] = self.lbp_update_node(n, unary, binary, messages)
                
        return messages

    def alphabp_update(self, num_iters = 1, messages = None, linear_response = False):
        # messages update of alpha BP
        # todo: linear response to adjust alpha
        
        binary = self.binary*self.mask
        unary = self.unary
        if messages is None:
            messages = self.unary.new(self.n**2, self.n**2, 2).fill_(0.5)
            
        new_messages = messages.detach()
        for _ in range(num_iters):
            for n in np.random.permutation(range(self.n**2)):
                # update message from node n to its neighbors
                new_messages[n][self.neighbors[n]] = self.alphabp_update_node(n, unary, binary, messages)
                
        return new_messages

    def alphabp_marginals(self, messages):
        '''Get the unary and binary marginals from alphabp messages '''
        binary = self.binary*self.mask
        unary = self.unary
        unary_marginals = []
        binary_marginals = []
        for n in range(self.n**2):
            unary_factor = torch.stack([-unary[n], unary[n]], 0) # 2 
            for k in self.neighbors[n]:
                unary_factor = unary_factor + messages[k][n].log()
            unary_prob = F.softmax(unary_factor, dim = 0)
            unary_marginals.append(unary_prob[1])        
        unary_marginals = torch.stack(unary_marginals, 0)
        binary_marginals = self.mf_binary_marginals(unary_marginals)

        return unary_marginals, binary_marginals


    def lbp_marginals(self, messages):
        binary = self.binary*self.mask
        unary = self.unary
        unary_marginals = []
        binary_marginals = []
        for n in range(self.n**2):
            unary_factor = torch.stack([-unary[n], unary[n]], 0) # 2 
            for k in self.neighbors[n]:
                unary_factor = unary_factor + messages[k][n].log()
            unary_prob = F.softmax(unary_factor, dim = 0)
            unary_marginals.append(unary_prob[1])        
        unary_marginals = torch.stack(unary_marginals, 0)
        for (i,j) in self.binary_idx:
            assert(i < j)
            binary_factor = binary[i][j]
            binary_factor = torch.stack([binary_factor, -binary_factor], 0)
            binary_factor = torch.stack([binary_factor, -binary_factor], 1) # 2 x 2
            unary_factor_i = torch.stack([-unary[i], unary[i]], 0) # 2 
            unary_factor_j = torch.stack([-unary[j], unary[j]], 0) # 2 
            for k in self.neighbors[i]:
                if k != j:
                    unary_factor_i += messages[k][i]
            for k in self.neighbors[j]:
                if k != i:
                    unary_factor_j += messages[k][j]
            binary_marginal = unary_factor_i.unsqueeze(1) + unary_factor_j.unsqueeze(0)
            binary_marginal = binary_marginal + binary_factor
            binary_marginal = F.softmax(binary_marginal.view(-1), dim = 0)
            binary_marginal = binary_marginal.view(2, 2)
            binary_marginals.append(binary_marginal)
        return unary_marginals, torch.stack(binary_marginals, 0)

    def free_energy_mf(self, unary_marginals):
        '''Compute the free energy for fully factorized estimations'''
        binary = self.binary*self.mask
        unary = self.unary
        unary1 = self.unary
        unary0 = -self.unary
        # bethe free energy computation of unary part
        unary_marginals1 = unary_marginals
        unary_marginals0 = 1 - unary_marginals
        unary_beliefs = torch.stack([unary_marginals0, unary_marginals1], 1) # 100 x 2
        # entropy and unary part of average energy
        bethe_unary = (unary_marginals0.log() - unary0)*unary_marginals0 + \
            (unary_marginals1.log() - unary1)*unary_marginals1
        free_energy = bethe_unary.sum()
        # include the binary part of free energy
        for k, (i,j) in enumerate(self.binary_idx):
            binary_factor = binary[i][j] # J_ij
            binary_factor = torch.stack([binary_factor, -binary_factor], 0)
            binary_factor = torch.stack([binary_factor, -binary_factor], 1) # 2 x 2
            
            marginal_ij = unary_beliefs[i].unsqueeze(1) * unary_beliefs[j].unsqueeze(0)
            free_energy += - (marginal_ij * binary_factor).sum()
        
        return free_energy
        
        
    def bethe_energy(self, unary_marginals, binary_marginals):
        binary = self.binary*self.mask
        unary = self.unary
        unary1 = self.unary
        unary0 = -self.unary
        # bethe free energy computation of unary part
        unary_marginals1 = unary_marginals
        unary_marginals0 = 1 - unary_marginals
        # why the minus unary here ???
        bethe_unary = (unary_marginals0.log() - unary0)*unary_marginals0 + (
            unary_marginals1.log() - unary1)*unary_marginals1
        bethe_unary = self.degree*bethe_unary
        bethe = -bethe_unary.sum()
        # bethe free energy computation of binary part
        for k, (i,j) in enumerate(self.binary_idx):
            binary_marginal = binary_marginals[k]
            binary_factor = binary[i][j] # J_ij
            binary_factor = torch.stack([binary_factor, -binary_factor], 0)
            binary_factor = torch.stack([binary_factor, -binary_factor], 1) # 2 x 2
            unary_factor_i = torch.stack([-unary[i], unary[i]], 0) # 2 
            unary_factor_j = torch.stack([-unary[j], unary[j]], 0) # 2 
            unary_factor_i = unary_factor_i.unsqueeze(1)
            unary_factor_j = unary_factor_j.unsqueeze(0)
            binary_factor_ij = binary_factor + unary_factor_i + unary_factor_j
            binary_factor_ij = binary_marginal*(binary_marginal.log() - binary_factor_ij)
            bethe += binary_factor_ij.sum()
        return bethe

if __name__ == '__main__':
    num_n = 15
    model = Ising(num_n)
    log_Z = model.log_partition_ve()
    unary_marginals, binary_marginals = model.marginals()
    model.generate_region_graph()
    
    encoder = GeneralizedInferenceNetwork(num_n, 200, 1, mlp_out_dim=2**4)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    unary_marginals_enc = torch.zeros_like(unary_marginals).fill_(0.5)
    binary_marginals_enc = torch.zeros_like(binary_marginals).fill_(0.25)
    from ising_marginals import corr, l1, l2
    for i in range(200):
        optimizer.zero_grad()
        infer_beliefs, consist_error = encoder(model.region_graph)
        kikuchi_energy = encoder.kikuchi_energy(log_phis=model.log_phis,\
                                                infer_beliefs=infer_beliefs, \
                                                counts=model.region_graph.collect_region_count())
        loss = kikuchi_energy + 10 * consist_error
        loss.backward()
        print(i,loss)
        unary_marginals_enc_new, binary_marginals_enc_new =\
            encoder.read_marginals(binary_idx=model.binary_idx,\
                                   infer_beliefs=infer_beliefs, \
                                   graph=model.region_graph)
    
        delta_unary = l2(unary_marginals_enc_new, unary_marginals_enc) 
        delta_binary = l2(binary_marginals_enc_new[:, 1, 1], binary_marginals_enc[:, 1, 1])
        delta = delta_unary + delta_binary
        if delta < 1e-5:
            break
      
        unary_marginals_enc = unary_marginals_enc_new.detach()
        binary_marginals_enc = binary_marginals_enc_new.detach()
    
        optimizer.step()

    
    
    marginals = torch.cat([unary_marginals, binary_marginals[:, 1, 1]], 0)
    marginals_enc = torch.cat([unary_marginals_enc, binary_marginals_enc[:, 1, 1]], 0)
    corr_unary_enc = corr(unary_marginals, unary_marginals_enc)
    corr_enc = corr(marginals, marginals_enc)
    
    l1_unary_enc = l1(unary_marginals, unary_marginals_enc)
    l1_binary_enc = l1(binary_marginals[:, 1, 1], binary_marginals_enc[:, 1, 1])
    l1_enc = l1(marginals, marginals_enc)
    print(corr_enc, l1_enc)
