import torch
from pgmpy.factors.discrete import PTDiscreteFactor

class parent2child_algo(object):
    """The generalized belief propagation algorithm."""
    def __init__(self, graph, n_iters=200):
        self._new_factor = PTDiscreteFactor
        self.n_iters = n_iters # default number iteration
        self.graph = graph # the region graph to do inference on
        # self._init_msgs()
        
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
            self.graph.edges[ie]['lg_msg'] = \
                self._new_factor([str(i) for i in child], \
                                 crdt, \
                                 values.log())
        return self

        
    def inference(self, region_graph):
        """
        Propagate beliefs in region graph among regions.
        """
        self.upate_msg(ie)

        for _ in range(self.n_iters):
            # propagate messages        
            for idx, ie in region_graph.get_edges():
                self.upate_msg(ie)

            if self.check_converge(region_graph):
                break

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
        parent, child = edge
        
        set_N = self.get_set_N(parent, child)
        set_D = self.get_set_D(descendants_parent, descendants_child)

        # get N set
        # get D set
        
        pass

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


    
