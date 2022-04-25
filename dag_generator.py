import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class DAGGenerator:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph_path = None

    def generate_dag(self, nodes=5, edges=5):
        # get a random adjacency with no restrictions
        adj = np.random.randint(low=0, high=2, size=(nodes, nodes))
        # remove self-loops
        np.fill_diagonal(adj, 0)
        # orient undirected edges
        adj = self._orient_undirected_edges(adj)
        # remove cycles to obtain DAG
        adj = self._remove_cycles(adj)
        # remove edges randomly until there are only 'edge' edges remaining
        adj = self._remove_edges(adj, edges)
        # set graph
        self.graph = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph())
        print("Generated graph with {} nodes and {} edges".format(nodes, edges))
        return self

    def _remove_cycles(self, adj):
        """
        As long as there are cycles in the graph, call this function recursively and remove one cycle per call
        """
        G = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph())
        try:
            cycle = list(nx.find_cycle(G, orientation='original'))
            edge_index = np.random.randint(0, len(cycle))
            removed_edge = cycle[edge_index]
            a, b, _ = removed_edge
            G.remove_edge(a, b)
            return self._remove_cycles(nx.convert_matrix.to_numpy_array(G))
        except:
            return adj

    def _orient_undirected_edges(self, adj):
        """
        Orient all undirected edges in one or the other direction (each with probability 0.5)
        """
        # direct undirected edges randomly
        for i in range(0, len(adj)):
            row = adj[i]
            for j in range(0, len(row)):
                # if there is an undirected edge
                if adj[i][j] == 1 and adj[j][i] == 1:
                    i_to_j = 0
                    j_to_i = 1
                    # orient the edge randomly
                    direction = np.random.choice([i_to_j, j_to_i])
                    if direction == i_to_j:
                        adj[j][i] = 0
                    else:
                        adj[i][j] = 0
        return adj

    def _remove_edges(self, adj, wanted):
        """
        Remove edges from graph as long as there are more edges in the graph than configured in the generate_dag-call
        """
        edges = np.argwhere(adj == 1)
        to_remove = len(edges) - wanted
        if to_remove > 0:
            removal_indices = np.random.randint(0, len(edges), to_remove)
            removed_edges = edges[removal_indices]
            for i, j in removed_edges:
                adj[i][j] = 0
        return adj
