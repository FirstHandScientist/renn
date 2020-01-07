import torch
from pgmpy.factors.discrete import PTDiscreteFactor

class parent2child_algo(object):
    """The generalized belief propagation algorithm."""
    def __init__(self, graph, n_iters=200, eps=1e-5):
        self._new_factor = PTDiscreteFactor
        self.EPS = eps
        self.n_iters = n_iters # default number iteration
        self.graph = graph # the region graph to do inference on
        
        self._init_msgs()
        
        pass


    def _init_msgs(self):
        """"
        Initialize the message at each directed edges of self.graph
        The message is itself is an object of PTDiscreteFactor.
        """
        for ie in self.graph.edges():
            child = ie[-1]
            crdt = self.graph.nodes[child]['log_phi'].cardinality
            values = torch.ones_like(self.graph.nodes[child]['log_phi'].values)
            values = values / values.sum()
            self.graph.edges[ie]['log_msg'] = \
                self._new_factor([str(i) for i in child], \
                                 crdt, \
                                 values.log())
            # 'old_msg' is used to monitor the convergence of algorithm
            self.graph.edges[ie]['old_msg'] = \
                self._new_factor([str(i) for i in child], \
                                 crdt, \
                                 values.log())
        return self

    def check_converge(self):
        """
        Check if messages in graph have converged.
        """
        err = 0
        for idx, ie in enumerate(self.graph.edges()):
            diff = self.graph.edges[ie]['log_msg'].values.exp() - self.graph.edges[ie]['old_msg'].values.exp()
            err += torch.abs(diff).mean()
            self.graph.edges[ie]['old_msg'] = self.graph.edges[ie]['log_msg'] 
        err = err / (idx+1)
        print(err)
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
            for ie in self.graph.edges():
                self.update_msg(ie)
            print(i)
            if self.check_converge():
                break

        print(i)
        return self

    def read_beliefs(self, graph):
        """
        Read out the marginals/beliefs given the region graph.
        """
        pass

    def update_msg(self, edge):
        """
        Update the message along the input edge.
        
        """
        factor_prod = self.get_divided_factor(edge)
        set_N = self.get_set_N(edge)
        set_D = self.get_set_D(edge)

        
        accumulate_msg = self.graph.nodes[edge[1]]['log_phi'].copy()
        accumulate_msg.values.zero_()

        for arc in set_N:
            accumulate_msg.sum(self.graph.edges[arc]['log_msg'], inplace=True)

        accumulate_msg.sum(factor_prod, inplace=True)


        for arc in set_D:
            accumulate_msg.sum(self.graph.edges[arc]['log_msg'], inplace=True, minus=True)


        # To real domain
        accumulate_msg.values = accumulate_msg.values.exp()

        if torch.any(accumulate_msg.values <= 0).item() != False:
            accumulate_msg.values[accumulate_msg.values == 0] = self.EPS
            

        # marginalize set
        parent, child = edge
        to_be_marginalize = list(set(parent) - set(child))
        accumulate_msg.marginalize([str(i) for i in to_be_marginalize], inplace=True)


        if torch.any(accumulate_msg.values <= 0).item() != False:
            print('err')
        # normalization
        accumulate_msg.normalize(inplace=True, log_domain=False)

        # all msg should be positive
        # if torch.any(accumulate_msg.values <= 0) != False:
        #     print('err')

        # back to log domain
        accumulate_msg.values = self._safe_log(accumulate_msg.values)
        
        assert accumulate_msg.variables == self.graph.edges[edge]["log_msg"].variables
        assert torch.isnan(accumulate_msg.values).sum() == 0
        self.graph.edges[edge]["log_msg"] = accumulate_msg
        

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


    
