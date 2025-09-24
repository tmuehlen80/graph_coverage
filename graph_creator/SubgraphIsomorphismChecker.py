import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
from typing import List, Dict
import networkx as nx


class IsomorphicGrapCoverageCounter:
    def __init__(self, coverage_graphs: Dict[str, nx.DiGraph], graph_paths: List[str], node_match: List[str], edge_match: List[str]):
        """
        Initialize the IsomorphicGrapCoverageCounter.

        Args:
            coverage_graphs: Dictionary of coverage graphs.
            graph_paths: List of graph paths.
            node_match: List of node match attributes.
            edge_match: List of edge match attributes.
        """
        self.coverage_graphs = coverage_graphs
        self.graph_paths = graph_paths
        self.node_match = node_match
        self.edge_match = edge_match
        self.cov_data = {}
        for scen_name in self.coverage_graphs.keys():
            self.cov_data[scen_name] = []

        self.cov_data["degree"] = []
        self.cov_data["density"] = []
        self.cov_data["diameter"] = []


    def count_isomorphic_graphs(self):
        for graph_path in tqdm(self.graph_paths):
            with open(graph_path, "rb") as file:
                ag_nx = pickle.load(file)

            for key in self.coverage_graphs.keys():
                GM = nx.algorithms.isomorphism.DiGraphMatcher(
                    ag_nx, self.coverage_graphs[key],
                    node_match=nx.algorithms.isomorphism.categorical_node_match(self.node_match, [None] * len(self.node_match)),
                    edge_match=nx.algorithms.isomorphism.categorical_edge_match(self.edge_match, [None] * len(self.edge_match))
                )
                self.cov_data[key].append(GM.subgraph_is_isomorphic())
            
            self.cov_data["degree"].append(sum(dict(ag_nx.degree()).values()) / len(ag_nx.nodes()))
            self.cov_data["density"].append(nx.density(ag_nx))
            self.cov_data["diameter"].append(nx.diameter(ag_nx))

        self.cov_data_df = pd.DataFrame(self.cov_data)
