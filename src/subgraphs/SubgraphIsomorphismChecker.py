import pandas as pd
import networkx as nx
from tqdm import tqdm
import pickle
from typing import List, Dict, Optional
from collections import defaultdict


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
        self.cov_data["path"] = []


    def count_isomorphic_graphs(self):
        for graph_path in tqdm(self.graph_paths, desc="Checking isomorphic graphs"):
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
            self.cov_data["path"].append(graph_path)

        self.cov_data_df = pd.DataFrame(self.cov_data)

    def count_node_coverage(self, max_graphs: int = 10000) -> pd.DataFrame:
        """
        Count how many times each node in each graph is covered by each subgraph type.
        
        For each graph, finds all subgraph isomorphisms and tracks which nodes are covered.
        Each node gets a counter that increments by 1 for each subgraph that covers it.
        
        Args:
            max_graphs: Maximum number of graphs to process (default: 10000)
            
        Returns:
            DataFrame with columns:
                - graph_path: Path to the graph file
                - node_id: ID of the node in the graph
                - subgraph_name: Name of the subgraph type
                - coverage_count: Number of times this node is covered by this subgraph type
        """
        # Limit the number of graphs to process
        graph_paths_to_process = self.graph_paths[:max_graphs]
        
        # Store node coverage data: (graph_path, node_id, subgraph_name) -> count
        node_coverage_data = []
        
        for graph_path in tqdm(graph_paths_to_process, desc="Counting node coverage"):
            with open(graph_path, "rb") as file:
                ag_nx = pickle.load(file)
            
            # For each subgraph type, find all isomorphisms
            for subgraph_name, subgraph_template in self.coverage_graphs.items():
                GM = nx.algorithms.isomorphism.DiGraphMatcher(
                    ag_nx, subgraph_template,
                    node_match=nx.algorithms.isomorphism.categorical_node_match(
                        self.node_match, [None] * len(self.node_match)
                    ),
                    edge_match=nx.algorithms.isomorphism.categorical_edge_match(
                        self.edge_match, [None] * len(self.edge_match)
                    )
                )
                
                # Find all subgraph isomorphisms
                # Each mapping maps nodes from the host graph (ag_nx) to the pattern graph (subgraph_template)
                # Count how many times each node is covered by this subgraph type
                node_coverage_count = defaultdict(int)
                
                for mapping in GM.subgraph_isomorphisms_iter():
                    # mapping: {node_in_host_graph: node_in_pattern_graph}
                    # Increment counter for each node covered by this isomorphism
                    for node_id in mapping.keys():
                        node_coverage_count[node_id] += 1
                
                # Store the coverage data for this graph and subgraph
                # Include all nodes: those covered (with count > 0) and those not covered (count = 0)
                all_nodes = set(ag_nx.nodes())
                covered_nodes = set(node_coverage_count.keys())
                
                # Store nodes that are covered
                for node_id, count in node_coverage_count.items():
                    node_coverage_data.append({
                        'graph_path': graph_path,
                        'node_id': node_id,
                        'subgraph_name': subgraph_name,
                        'coverage_count': count
                    })
                
                # Store nodes that are not covered (coverage_count = 0)
                uncovered_nodes = all_nodes - covered_nodes
                for node_id in uncovered_nodes:
                    node_coverage_data.append({
                        'graph_path': graph_path,
                        'node_id': node_id,
                        'subgraph_name': subgraph_name,
                        'coverage_count': 0
                    })
        
        self.node_coverage_df = pd.DataFrame(node_coverage_data)
        return self.node_coverage_df
