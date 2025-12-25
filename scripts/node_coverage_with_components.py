#!/usr/bin/env python3
"""
Enhanced node coverage analysis that handles disconnected subgraphs.

This script:
1. Splits each graph into disconnected components
2. Analyzes component size (number of actors)
3. Applies simple patterns (2-actor) only to isolated components
4. Applies complex patterns (3+ actor) to all components
"""

import argparse
import glob
import os
import sys
import pickle
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from subgraphs.subgraph_types import get_all_subgraphs, get_simple_patterns, get_complex_patterns


def split_into_components(graph):
    """Split a directed graph into weakly connected components."""
    if graph.is_directed():
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    
    return [graph.subgraph(c).copy() for c in components]


def analyze_component_coverage(graph, node_match, edge_match):
    """Analyze coverage for a single graph by splitting into components."""
    components = split_into_components(graph)
    simple_patterns = get_simple_patterns()
    complex_patterns = get_complex_patterns()
    
    node_coverage = defaultdict(lambda: defaultdict(int))
    
    for component in components:
        num_nodes = component.number_of_nodes()
        
        # Apply simple patterns only to 2-node isolated components
        # Apply complex patterns to 3+ node components
        if num_nodes == 2:
            patterns_to_check = simple_patterns
        else:
            patterns_to_check = complex_patterns
        
        for pattern_name, pattern_graph in patterns_to_check.items():
            GM = nx.algorithms.isomorphism.DiGraphMatcher(
                component, pattern_graph,
                node_match=nx.algorithms.isomorphism.categorical_node_match(
                    node_match, [None] * len(node_match)
                ),
                edge_match=nx.algorithms.isomorphism.categorical_edge_match(
                    edge_match, [None] * len(edge_match)
                )
            )
            
            for mapping in GM.subgraph_isomorphisms_iter():
                for node_id in mapping.keys():
                    node_coverage[node_id][pattern_name] += 1
    
    return node_coverage


def main():
    parser = argparse.ArgumentParser(
        description="Analyze node coverage with disconnected component handling"
    )
    
    parser.add_argument(
        "--graph-pattern",
        type=str,
        required=True,
        help="Glob pattern to find graph pickle files"
    )
    
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=100,
        help="Maximum number of graphs to process"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--node-match",
        type=str,
        nargs="+",
        default=["actor_type", "lane_change", "is_on_intersection"],
        help="Node attributes to match"
    )
    
    parser.add_argument(
        "--edge-match",
        type=str,
        nargs="+",
        default=["edge_type"],
        help="Edge attributes to match"
    )
    
    args = parser.parse_args()
    
    if os.path.isabs(args.graph_pattern):
        graph_pattern = args.graph_pattern
    else:
        graph_pattern = os.path.join(PROJECT_ROOT, args.graph_pattern)
    
    graph_paths = sorted(glob.glob(graph_pattern))
    
    if not graph_paths:
        sys.exit(1)
    
    if len(graph_paths) > args.max_graphs:
        graph_paths = graph_paths[:args.max_graphs]
    
    all_patterns = get_all_subgraphs()
    coverage_data = []
    
    for graph_path in tqdm(graph_paths, desc="Analyzing graphs"):
        with open(graph_path, "rb") as f:
            actor_graph = pickle.load(f)
        
        for timestamp, G_nx in actor_graph.actor_graphs.items():
            graph_id = f"{os.path.basename(graph_path)}_{timestamp}"
            
            node_coverage = analyze_component_coverage(
                G_nx, args.node_match, args.edge_match
            )
            
            for node_id in G_nx.nodes():
                for pattern_name in all_patterns.keys():
                    count = node_coverage[node_id].get(pattern_name, 0)
                    coverage_data.append({
                        'graph_path': graph_id,
                        'node_id': node_id,
                        'subgraph_name': pattern_name,
                        'coverage_count': count
                    })
    
    df = pd.DataFrame(coverage_data)
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(PROJECT_ROOT, "node_coverage_with_components.csv")
    
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

