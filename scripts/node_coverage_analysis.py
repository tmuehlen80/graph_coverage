#!/usr/bin/env python3
"""
Script to analyze node-level coverage of subgraphs in actor graphs.

This script counts how many times each node in each graph is covered by each subgraph type.
It processes a subset of graphs (default: 10000) and saves the results to a CSV file.
"""

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from subgraphs.SubgraphIsomorphismChecker import IsomorphicGrapCoverageCounter
from subgraphs.subgraph_types import get_all_subgraphs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze node-level coverage of subgraphs in actor graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--graph-pattern",
        type=str,
        required=True,
        help="Glob pattern to find graph pickle files (e.g., 'actor_graphs/carla_w_intersection/*.pkl')"
    )
    
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=10000,
        help="Maximum number of graphs to process"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: node_coverage_<dataset_name>.csv in project root)"
    )
    
    parser.add_argument(
        "--node-match",
        type=str,
        nargs="+",
        default=["actor_type", "lane_change", "is_on_intersection"],
        help="Node attributes to match for isomorphism"
    )
    
    parser.add_argument(
        "--edge-match",
        type=str,
        nargs="+",
        default=["edge_type"],
        help="Edge attributes to match for isomorphism"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if os.path.isabs(args.graph_pattern):
        graph_pattern = args.graph_pattern
    else:
        graph_pattern = os.path.join(PROJECT_ROOT, args.graph_pattern)
    
    print(f"Looking for graphs matching pattern: {graph_pattern}")
    graph_paths = sorted(glob.glob(graph_pattern))
    
    if not graph_paths:
        print(f"Error: No graph files found matching pattern: {graph_pattern}")
        sys.exit(1)
    
    print(f"Found {len(graph_paths)} graph files")
    
    if len(graph_paths) > args.max_graphs:
        graph_paths = graph_paths[:args.max_graphs]
        print(f"Processing first {args.max_graphs} graphs")
    else:
        print(f"Processing all {len(graph_paths)} graphs")
    
    print("Loading subgraph types...")
    coverage_graphs = get_all_subgraphs()
    print(f"Loaded {len(coverage_graphs)} subgraph types: {list(coverage_graphs.keys())}")
    
    print("Initializing coverage counter...")
    isom_cov_counter = IsomorphicGrapCoverageCounter(
        coverage_graphs=coverage_graphs,
        graph_paths=graph_paths,
        node_match=args.node_match,
        edge_match=args.edge_match
    )
    
    print("Running node coverage analysis...")
    node_coverage_df = isom_cov_counter.count_node_coverage(max_graphs=args.max_graphs)
    
    print(f"Analysis complete! Processed {len(node_coverage_df['graph_path'].unique())} graphs")
    print(f"Total node coverage records: {len(node_coverage_df)}")
    
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(PROJECT_ROOT, output_path)
    else:
        dataset_name = os.path.basename(os.path.dirname(graph_pattern))
        if not dataset_name or dataset_name == ".":
            dataset_name = "graphs"
        output_path = os.path.join(PROJECT_ROOT, f"node_coverage_{dataset_name}.csv")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving results to: {output_path}")
    node_coverage_df.to_csv(output_path, index=False)
    print("Done!")
    
    print(f"Total graphs processed: {node_coverage_df['graph_path'].nunique()}")
    print(f"Total nodes analyzed: {node_coverage_df['node_id'].nunique()}")
    print(f"Total subgraph types: {node_coverage_df['subgraph_name'].nunique()}")
    print(f"\nCoverage by subgraph type:")
    coverage_summary = node_coverage_df.groupby('subgraph_name')['coverage_count'].agg(['sum', 'max'])
    print(coverage_summary[['sum', 'max']])
    print(f"\nNodes with coverage > 0: {(node_coverage_df['coverage_count'] > 0).sum()}")
    print(f"Nodes with coverage = 0: {(node_coverage_df['coverage_count'] == 0).sum()}")


if __name__ == "__main__":
    main()

