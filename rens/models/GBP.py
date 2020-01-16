import torch
from pgmpy.factors.discrete import PTDiscreteFactor
import numpy
from random import shuffle

class parent2child_algo(object):
    """The generalized belief propagation algorithm."""
    def __init__(self, graph, n_iters=100, eps=1e-5):
        self._new_factor = PTDiscreteFactor
        self.EPS = eps
        self.n_iters = n_iters # default number iteration
        self.graph = graph # the region graph to do inference on
        
        self._init_msgs()
        self.graph_edges = [edge for edge in self.graph.edges()]
        
        pass


    def _init_msgs(self):
        """"
        Initialize the message at each directed edges of self.graph
        The message is itself is an object of PTDiscreteFactor.
        """
        for ie in self.graph.edges():
            child = ie[-1]
            crdt = self.graph.nodes[child]['log_phi'].cardinality
            device = self.graph.nodes[child]['log_phi'].values.device
            values = torch.randn(self.graph.nodes[child]['log_phi'].values.size()).abs().to(device)
            values = values / values.sum()
            self.graph.edges[ie]['log_msg'] = \
                self._new_factor([str(i) for i in child], \
                                 crdt, \
                                 values.detach().log())
            # 'old_msg' is used to monitor the convergence of algorithm
            self.graph.edges[ie]['old_msg'] = \
                self._new_factor([str(i) for i in child], \
                                 crdt, \
                                 values.detach().log())
        return self

    def check_converge(self):
        """
        Check if messages in graph have converged.
        """
        err = 0
        for idx, ie in enumerate(self.graph.edges()):
            diff = self.graph.edges[ie]['log_msg'].values.exp() - self.graph.edges[ie]['old_msg'].values.exp()
            err += torch.abs(diff).mean()
            self.graph.edges[ie]['old_msg'].values = self.graph.edges[ie]['log_msg'].values.clone()
        err = err / (idx+1)
        # print(err)
        if err < self.EPS:
            return True
        else:
            return False
        
        
    
    
    def inference(self):
        """
        Propagate beliefs in region graph among regions.
        """

        for i in range(self.n_iters):
            # propagate messages
            
            for ie in numpy.random.permutation(range(len(self.graph_edges))):
                
                self.update_msg(self.graph_edges[ie])
            
            
            # print(i)
            if self.check_converge():
                break

        
        return self

    def read_beliefs(self):
        """
        Read out the marginals/beliefs given the region graph.
        """
        # iter through each node in region graph to create the beliefs
        for node in self.graph.nodes:
            self.graph.nodes[node]['belief'] = self.gather_node_belief(node)

        return
    def kikuchi_energy(self):
        energy = 0
        graph = self.graph
        for node in graph.nodes():
            graph.nodes[node]['belief'].values = torch.clamp(graph.nodes[node]['belief'].values, min=1e-8)
            energy += torch.sum(graph.nodes[node]['belief'].values * \
                       (graph.nodes[node]['belief'].values.log() - \
                        graph.nodes[node]['log_phi'].values.detach())) * \
                        graph.nodes[node]['weight']

        return energy

    def gather_node_belief(self, node):
        """
        Gather belief for each node in graph from the computed messages.
        """
        # accumulate the f_a
        belief = self.graph.nodes[node]['log_phi'].copy()

        # accumulate msgs from parents
        for p_node in self.graph.get_parents(node):
            log_msg = self.graph.edges[(p_node, node)]['log_msg']
            belief.sum(log_msg, inplace=True)

        # accumulate msgs in belief_descendant_set
        for pair in self.belief_descendant_set(node):
            p_node, c_node = pair
            log_msg = self.graph.edges[(p_node, c_node)]['log_msg']
            belief.sum(log_msg, inplace=True)

        assert belief.variables == self.graph.nodes[node]['log_phi'].variables
        belief.to_real()
        belief.normalize(inplace=True)
        assert torch.isnan(belief.values).sum() == 0


        return belief

    def belief_descendant_set(self, node):
        """
        Get the pairs (sender, receiver) such that:
        receiver in descendants of node;
        sender is a parent of receiver, but not the node, either in descendants of the node.

        """
        descendants = self.graph.get_descendants_of([node], discard_self=True)
        descendants_and_self = self.graph.get_descendants_of([node], discard_self=False)

        candidate_set = set()
        for d_node in descendants:
            parents = self.graph.get_parents(d_node)
            for p_node in parents:
                if p_node not in descendants_and_self:
                    candidate_set.add((p_node, d_node))

        return list(candidate_set)
    

    def update_msg(self, edge):
        """
        Update the message along the input edge.
        
        """
        
        factor_PxC = self.graph.edges[edge]['PxC'].copy()
        factor_prod = factor_PxC #self.get_divided_factor(edge)
        set_N = self.get_set_N(edge)
        set_D = self.get_set_D(edge)

        
        accumulate_msg = self.graph.edges[edge]['log_msg'].copy()
        accumulate_msg.values.zero_()
        # accumulate the set of N
        for arc in set_N:
            # assert  accumulate_msg.var
            accumulate_msg.sum(self.graph.edges[arc]['log_msg'], inplace=True)

        # accumulate the factor A_P\A_R
        accumulate_msg.sum(factor_prod, inplace=True)

        # do marginals here
        # To real domain
        accumulate_msg.to_real()

        # marginalize set
        parent, child = edge
        to_be_marginalize = list(set(parent) - set(child))
        accumulate_msg.marginalize([str(i) for i in to_be_marginalize], inplace=True)
        
        # back To log domain
        accumulate_msg.to_log()

        # accumulate the set of D
        for arc in set_D:
            accumulate_msg.sum(self.graph.edges[arc]['log_msg'], inplace=True, minus=True)


        # normalization in log domain
        accumulate_msg.normalize(inplace=True, log_domain=True)

        # all msg should be positive
        # if torch.any(accumulate_msg.values <= 0) != False:
        #     print('err')

        
        assert accumulate_msg.variables == self.graph.edges[edge]["log_msg"].variables
        assert torch.isnan(accumulate_msg.values).sum() == 0
        
        # update msg
        accumulate_msg.to_real()
        self.graph.edges[edge]["log_msg"].to_real()
        self.graph.edges[edge]["log_msg"].sum(accumulate_msg, inplace=True, minus=False)
        
        self.graph.edges[edge]['log_msg'].normalize(inplace=True, log_domain=False)
        self.graph.edges[edge]['log_msg'].to_log()
        

        return self

    def _safe_log(self, value):
        value[torch.isclose(value, torch.zeros(1))] = self.EPS
        return value.log()

    def get_divided_factor(self, edge):
        parent, child = edge
        factor_parent = self.graph.nodes[parent]["log_phi"]
        factor_child = self.graph.nodes[child]["log_phi"]
        # factors are in log domain
        factor_div = factor_parent.sum(factor_child, inplace=False, minus=True)
        factor_div.to_real()
        factor_div.marginalize(factor_child.variables, inplace=True)
        
        factor_div.to_log()
        return factor_div

    def get_set_N(self, edge):

        parent, child = edge
        descendants_parent = self.graph.get_descendants_of([parent], discard_self=False)
        descendants_child = self.graph.get_descendants_of([child], discard_self=False)
        set_p_diff_c = set(descendants_parent) - set(descendants_child)
        
        set_n = set()
        for receiver in set_p_diff_c:
            parents_of_reveiver = self.graph.get_parents(receiver)
            for p_candidate in parents_of_reveiver:
                if p_candidate not in descendants_parent:
                    set_n.add((p_candidate, receiver))

        return list(set_n)

    def get_set_D(self, edge):
        parent, child = edge
        descendants_parent = self.graph.get_descendants_of([parent], discard_self=False)
        descendants_child = self.graph.get_descendants_of([child], discard_self=False)
        set_p_diff_c = set(descendants_parent) - set(descendants_child)
        
        set_d = set()
        for receiver in descendants_child:
            parents_of_reveiver = self.graph.get_parents(receiver)
            for p_candidate in parents_of_reveiver:
                if p_candidate in set_p_diff_c:
                    set_d.add((p_candidate, receiver))
        # this discard if not in original paper, typo of original paper?
        set_d.discard(edge)
        # for sender in set_p_diff_c:
        #     children_of_reveiver = self.graph.get_children(sender)
        #     for receiver_candidate in children_of_reveiver:
        #         if receiver_candidate not in descendants_child:
        #             set_d.add((sender, receiver_candidate))

        return list(set_d)

if __name__ == "__main__":
    from pgmpy.models import RegionGraph

    rg = RegionGraph()
    rg.add_nodes_from([('1','2','3', '5'),\
                       ('1','2','4', '6'),\
                       ('1','3','4', '7')])
    rg.add_nodes_from([('1','2'),\
                       ('1','4'),\
                       ('1','3')])
    rg.add_nodes_from([('1',)])
    rg.add_edges_from([(('1','2'), ('1',)), \
                       (('1','3'), ('1',)), \
                       (('1','4'), ('1',))])
    rg.add_edges_from([(('1','2','3', '5'), ('1','2')),\
                       (('1','2','3', '5'), ('1','3'))])
    rg.add_edges_from([(('1','2','4', '6'), ('1','2')),\
                       (('1','2','4', '6'), ('1','4'))])
    rg.add_edges_from([(('1','3','4', '7'), ('1','4')),\
                       (('1','3','4', '7'), ('1','3'))])
    
    gbp = parent2child_algo(rg)
    print('n', gbp.get_set_N((('1','2'), ('1',))))
    print('d', gbp.get_set_D((('1','2'), ('1',))))
    
    print('n', gbp.get_set_N((('1','2','3', '5'), ('1','2'))))
    print('d', gbp.get_set_D((('1','2','3', '5'), ('1','2'))))
    
    print('n', gbp.get_set_N((('1','2','4', '6'), ('1','2'))))
    print('d', gbp.get_set_D((('1','2','4', '6'), ('1','2'))))

    print('belief set:', gbp.belief_descendant_set(('1','2','3', '5')))
    print('belief set:', gbp.belief_descendant_set(('1','2','4', '6')))
    print('belief set:', gbp.belief_descendant_set(('1','3','4', '7')))
    print('belief set:', gbp.belief_descendant_set(('1','2')))
    print('belief set:', gbp.belief_descendant_set(('1','3')))
    print('belief set:', gbp.belief_descendant_set(('1',)))






    
