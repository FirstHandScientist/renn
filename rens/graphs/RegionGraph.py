#!/usr/bin/env python3

from collections import defaultdict

import numpy as np
from itertools import combinations

from pgmpy.base import UndirectedGraph, DAG
from pgmpy.factors import factor_product
from pgmpy.extern.six.moves import filter, range, zip


class RegionGraph(DAG):
    r"""
    Base class for representing Region Graph.

    Cluster graph is an DAG graph which is associated with a subset of variables. The graph contains undirected
    edges that connects clusters whose scopes have a non-empty intersection.

    Formally, a cluster graph is  :math:`\mathcal{U}` for a set of factors :math:`\Phi` over :math:`\mathcal{X}` is an
    undirected graph, each of whose nodes :math:`i` is associated with a subset :math:`C_i \subseteq X`. A cluster
    graph must be family-preserving - each factor :math:`\phi \in \Phi` must be associated with a cluster C, denoted
    :math:`\alpha(\phi)`, such that :math:`Scope[\phi] \subseteq C_i`. Each edge between a pair of clusters :math:`C_i`
    and :math:`C_j` is associated with a sepset :math:`S_{i,j} \subseteq C_i \cap C_j`.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is created. The data is an edge list

    Examples
    --------
    Create an empty RegionGraph with no nodes and no edges

    >>> from pgmpy.models import RegionGraph
    >>> G = RegionGraph()

    G can be grown by adding clique nodes.

    **Nodes:**

    Add a tuple (or list or set) of nodes as single clique node.

    >>> G.add_node(('a', 'b', 'c'))
    >>> G.add_nodes_from([('a', 'b'), ('a', 'b', 'c')])

    **Edges:**

    G can also be grown by adding edges.

    >>> G.add_edge(('a', 'b', 'c'), ('a', 'b'))

    or a list of edges

    >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
    ...                   (('a', 'b', 'c'), ('a', 'c'))])
    """

    def __init__(self, ebunch=None):
        super(RegionGraph, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []
        self.region_layers = {}
        

    def add_node(self, node, **kwargs):
        """
        Add a single node to the cluster graph.

        Parameters
        ----------
        node: node
            A node should be a collection of nodes forming a clique. It can be
            a list, set or tuple of nodes

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> G = RegionGraph()
        >>> G.add_node(('a', 'b', 'c'))
        """
        if not isinstance(node, (list, set, tuple)):
            raise TypeError(
                "Node can only be a list, set or tuple of nodes forming a clique"
            )

        node = tuple(node)
        super(RegionGraph, self).add_node(node, **kwargs)

    def add_nodes_from(self, nodes, **kwargs):
        """
        Add multiple nodes to the cluster graph.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b'), ('a', 'b', 'c')])
        """
        for node in nodes:
            self.add_node(node, **kwargs)

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between two clique nodes.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any list or set or tuple of nodes forming a clique.

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        """
        set_u = set(u)
        set_v = set(v)
        if set_u.isdisjoint(set_v):
            raise ValueError("No sepset found between these two edges.")

        super(RegionGraph, self).add_edge(u, v)

    def add_factors(self, *factors):
        """
        Need to be rewritten... to accommodate the factors attachment of RegionGraph...


        Associate a factor to the graph.
        See factors class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.factors.factors object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = RegionGraph()
        >>> student.add_node(('Alice', 'Bob'))
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[3, 2],
        ...                 values=np.random.rand(6))
        >>> student.add_factors(factor)
        """
        for factor in factors:
            factor_scope = set(factor.scope())
            nodes = [set(node) for node in self.nodes()]
            if factor_scope not in nodes:
                raise ValueError(
                    "Factors defined on clusters of variable not" "present in model"
                )

            self.factors.append(factor)

    def _get_interset(self, u, v):
        """Given two nodes: u,v, return a node that is intersetion of u and v.
        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G._get_interset(('a', 'b', 'c'), ('a', 'b'))
        >>> ('a', 'b')
        """
        set_u = set(u)
        set_v = set(v)
        inter_set = set.intersection(set_u, set_v)
        return tuple(sorted(tuple(inter_set)))

    def get_next_region_layer(self, region_type=None):
        """ Give label R0, generate the children region layer R1.
        """
        label = "R{}".format(int(region_type[-1])+1)
        layer_nodes = self.region_layers[region_type]
        intersections = []
        for pair_nodes in combinations(layer_nodes,2):
            i_node, j_node = pair_nodes
            intersection_node = self._get_interset(i_node, j_node)
            # if empty set, go to next iteration
            if intersection_node is ():
                continue
            if intersection_node not in intersections:
                # add the new node and arcs to this nodes
                # self.add_node(intersection_node)
                # self.add_edges_from([(i_node, intersection_node), \
                #                      (j_node, intersection_node)])
                intersections.append(intersection_node)
        
        if intersections == []:
            # There is no next layer node generated
            return False
        else:
            # Do filtering, delete the node that is subset of the other nodes of the same layer
            filter_out_idx = []
            # test node i against from others
            for i in range(len(intersections)):
                for j in range(len(intersections)):
                    if i != j and self._is_subset(intersections[i], intersections[j]) and i not in filter_out_idx:
                        # self.remove_node(intersections[i])
                        filter_out_idx.append(i)

            # the final selected nodes for next layer
            selected = [intersections[idx] for idx in range(len(intersections)) if idx not in filter_out_idx]
            # add them into graph
            for c_node in selected:
                self.add_node(c_node)
                for p_node in layer_nodes:
                    if self._is_subset(c_node, p_node):
                        self.add_edges_from([(p_node, c_node)])
            
            self.region_layers[label] = selected
            return True

    def _is_subset(self, u, v):
        '''Check is u is subsect of v.'''
        set_u = set(u)
        set_v = set(v)
        if set_u.issubset(set_v):
            return True
        else:
            return False
        

    def cluster_variation(self, layer=5):
        """ Run the cluster variation to get R0, R1, ...
        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b'), ('b', 'c'), ('c','d')])
        >>> G.cluster_variation()
        >>> G.region_layers
        --------
        """
        assert list(self.edges) == [], "R0 layer should not contain edges"
        self.region_layers["R0"] = self.get_roots()
        for i in range(layer):
            target = "R{}".format(int(i))
            if self.get_next_region_layer(region_type=target):
                print("Children regions of " + target + " is successful.")
                
            else:
                print("No children regions of " + target + " any more.")
                break

        self.region_count()
        
        return self

    def region_count(self):
        """Given the generated region graph, calculate each region's count number from top to down."""
        for node in self.region_layers["R0"]:
            self.nodes[node]["weight"] = 1
            
        for i_layer in range(len(self.region_layers)-1):
            for i_node in self.region_layers["R{}".format(i_layer + 1)]:
                ancestors_counts = [self.nodes[the_node]["weight"] for the_node in self.get_ancestors_of([i_node])]
                self.nodes[i_node]["weight"] = 1 - sum(ancestors_counts)

        return self
       
    def collect_region_count(self):
        """
        Collect the region counts that already be generated.
        ----------
        Output:
        ----------
        Dict = {"R0": 1D array, "R1": 1d array, "R2": 1d array}

        """
        count_dict = {}
        for i_layer, node_list in self.region_layers.items():
            counts = [self.nodes[node]["weight"] for node in node_list]
            count_dict[i_layer] = np.array(counts)

        return count_dict



    def get_factors(self, node=None):
        """
        Return the factors that have been added till now to the graph.

        If node is not None, it would return the factor corresponding to the
        given node.

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = DiscreteFactor(['a', 'b', 'c'], [2, 2, 2], np.random.rand(8))
        >>> phi2 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi3 = DiscreteFactor(['a', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2, phi3)
        >>> G.get_factors()
        >>> G.get_factors(node=('a', 'b', 'c'))
        """
        if node is None:
            return self.factors
        else:
            nodes = [set(n) for n in self.nodes()]

            if set(node) not in nodes:
                raise ValueError("Node not present in Cluster Graph")

            factors = filter(lambda x: set(x.scope()) == set(node), self.factors)
            return next(factors)

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = RegionGraph()
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 value=np.random.rand(4))
        >>> student.add_factors(factor)
        >>> student.remove_factors(factor)
        """
        for factor in factors:
            self.factors.remove(factor)

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose cardinality we want. If node is not specified returns a
            dictionary with the given variable as keys and their respective cardinality
            as values.

        Returns
        -------
        int or dict : If node is specified returns the cardinality of the node.
                      If node is not specified returns a dictionary with the given
                      variable as keys and their respective cardinality as values.


        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = RegionGraph()
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 values=np.random.rand(4))
        >>> student.add_node(('Alice', 'Bob'))
        >>> student.add_factors(factor)
        >>> student.get_cardinality()
        defaultdict(<class 'int'>, {'Bob': 2, 'Alice': 2})

        >>> student.get_cardinality(node='Alice')
        2
        """
        if node:
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    if node == variable:
                        return cardinality

        else:
            cardinalities = defaultdict(int)
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    cardinalities[variable] = cardinality
            return cardinalities

    def get_partition_function(self):
        r"""
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

        Examples
        --------
        >>> from pgmpy.models import RegionGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = DiscreteFactor(['a', 'b', 'c'], [2, 2, 2], np.random.rand(8))
        >>> phi2 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi3 = DiscreteFactor(['a', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2, phi3)
        >>> G.get_partition_function()
        """
        if self.check_model():
            factor = self.factors[0]
            factor = factor_product(
                factor, *[self.factors[i] for i in range(1, len(self.factors))]
            )
            return np.sum(factor.values)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if factors are defined for all the cliques or not.
        * Check for running intersection property is not done explicitly over
          here as it done in the add_edges method.
        * Checks if cardinality information for all the variables is availble or not. If
          not it raises an error.
        * Check if cardinality of random variable remains same across all the
          factors.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for clique in self.nodes():
            factors = filter(lambda x: set(x.scope()) == set(clique), self.factors)
            if not any(factors):
                raise ValueError("Factors for all the cliques or clusters not defined.")

        cardinalities = self.get_cardinality()
        if len(set((x for clique in self.nodes() for x in clique))) != len(
            cardinalities
        ):
            raise ValueError("Factors for all the variables not defined.")

        for factor in self.factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                if cardinalities[variable] != cardinality:
                    raise ValueError(
                        "Cardinality of variable {var} not matching among factors".format(
                            var=variable
                        )
                    )

        return True

    def get_ancestors_of(self, obs_nodes_list, discard_self=True):
        """
        Returns a dictionary of all ancestors of all the observed nodes excluding the
        node itself.
        Parameters
        ----------
        obs_nodes_list: string, list-type
            name of all the observed nodes
        Examples
        --------
        """
        if not isinstance(obs_nodes_list, (list, tuple)):
            obs_nodes_list = [obs_nodes_list]

        for node in obs_nodes_list:
            if node not in self.nodes():
                raise ValueError("Node {s} not in not in graph".format(s=node))

        ancestors_list = set()
        nodes_list = set(obs_nodes_list)
        while nodes_list:
            node = nodes_list.pop()
            if node not in ancestors_list:
                nodes_list.update(self.predecessors(node))
            ancestors_list.add(node)

        if discard_self:
            for i in obs_nodes_list:
                ancestors_list.discard(i)
        return list(ancestors_list)

    def get_supernode_of(self, orphan, discard_self=True):
        """
        Go through the nodes in graph, find the node who is super-set of orphan.
        """
        candidate = set()
        for node in self.nodes():
            if self._is_subset(orphan, node):
                candidate.add(node)

        if discard_self:
            candidate.discard(orphan)

        return list(candidate)

    def get_descendants_of(self, obs_nodes_list, discard_self=True):
        """
        Returns a dictionary of all descendants of all the observed nodes excluding the
        node itself.
        Parameters
        ----------
        obs_nodes_list: string, list-type
            name of all the observed nodes
        --------
        """
        if not isinstance(obs_nodes_list, (list, tuple)):
            obs_nodes_list = [obs_nodes_list]

        for node in obs_nodes_list:
            if node not in self.nodes():
                raise ValueError("Node {s} not in not in graph".format(s=node))

        descendants_list = set()
        nodes_list = set(obs_nodes_list)
        while nodes_list:
            node = nodes_list.pop()
            if node not in descendants_list:
                nodes_list.update(self.successors(node))
            descendants_list.add(node)
        if discard_self:
            for i in obs_nodes_list:
                descendants_list.discard(i)
        return list(descendants_list)

    def copy(self):
        """
        Returns a copy of RegionGraph.

        Returns
        -------
        RegionGraph: copy of RegionGraph

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = RegionGraph()
        >>> G.add_nodes_from([('a', 'b'), ('b', 'c')])
        >>> G.add_edge(('a', 'b'), ('b', 'c'))
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> graph_copy = G.copy()
        >>> graph_copy.factors
        [<DiscreteFactor representing phi(a:2, b:2) at 0xb71b19cc>,
         <DiscreteFactor representing phi(b:2, c:2) at 0xb4eaf3ac>]
        >>> graph_copy.edges()
        [(('a', 'b'), ('b', 'c'))]
        >>> graph_copy.nodes()
        [('a', 'b'), ('b', 'c')]
        """
        copy = RegionGraph(self.edges())
        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            copy.add_factors(*factors_copy)
        return copy
if __name__ == "__main__":
    G = RegionGraph()
    G.add_nodes_from([(1, 2, 4, 5), (2, 3, 5, 6), (4, 5, 7, 8), (5,6,8,9)])
    G.cluster_variation()
    G.region_layers
    # from pgmpy.factors.discrete import DiscreteFactor
    # phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
    # phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
    # G.add_factors(phi1, phi2)
