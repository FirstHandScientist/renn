import torch
from pgmpy.factors.discrete import PTDiscreteFactor

class parent2child_algo(object):
    """The generalized belief propagation algorithm."""
    def __init__(self, graph, n_iters=200):
        self._new_factor = PTDiscreteFactor
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
        
        """
