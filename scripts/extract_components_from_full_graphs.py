#!/usr/bin/env python3
"""
Extract connected components from full actor graphs and save them as separate files.

This script:
1. Reads full actor graphs (ActorGraph objects with multiple timestamps)
2. Extracts weakly connected components from each timestamp
3. Saves each component as a separate NetworkX graph file

Command-line Usage:

    # Process only first 100 graphs (for testing/development)
    python extract_components_from_full_graphs.py \\
        --input-dir actor_graphs/carla_actor_graph_setting_1_50_50_10_20_20_4_4_4/carla_actor_graph_setting_1_50_50_10_20_20_4_4_4_nx \\
        --output-dir actor_graphs/carla_actor_graph_setting_1_50_50_10_20_20_4_4_4/carla_actor_graph_setting_1_50_50_10_20_20_4_4_4_components_nx \\
        --max-graphs 100
    
    # Short form using -n flag
    python extract_components_from_full_graphs.py \\
        --input-dir actor_graphs/carla_nx \\
        --output-dir actor_graphs/carla_components_nx \\
        -n 50
    
    # Process all graphs
    python extract_components_from_full_graphs.py \\
        --input-dir actor_graphs/carla_nx \\
        --output-dir actor_graphs/carla_components_nx

Programmatic Usage (from notebook or other script):

    from scripts.extract_components_from_full_graphs import extract_components_from_directory
    
    # Extract components from only 100 graphs
    stats = extract_components_from_directory(
        input_dir='actor_graphs/carla_nx',
        output_dir='actor_graphs/carla_components_nx',
        max_graphs=100,
        min_nodes=2
    )
    print(f"Extracted {stats['total_components']} components from {stats['processed_files']} graphs")
"""

import argparse
import glob
import os
import sys
import pickle
from tqdm import tqdm
import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)


def extract_components_from_graph(graph_nx):
    """
    Extract weakly connected components from a directed graph.
    
    Args:
        graph_nx: NetworkX DiGraph
        
    Returns:
        List of component subgraphs
    """
    if graph_nx.is_directed():
        components = nx.weakly_connected_components(graph_nx)
    else:
        components = nx.connected_components(graph_nx)
    
    return [graph_nx.subgraph(c).copy() for c in components]


def process_actor_graph_file(input_path, output_dir, min_nodes=2):
    """
    Process a single ActorGraph file and extract components.
    
    Args:
        input_path: Path to input pickle file (ActorGraph object)
        output_dir: Directory to save component graphs
        min_nodes: Minimum number of nodes for a component to be saved
        
    Returns:
        Number of components extracted
    """
    try:
        # Skip map graph files (only process actor graph files)
        if '_map_graph.pkl' in input_path:
            return 0
        
        with open(input_path, "rb") as f:
            actor_graph = pickle.load(f)
        
        # Check if this is an ActorGraph object with actor_graphs attribute
        if not hasattr(actor_graph, 'actor_graphs'):
            # This might be a MapGraph or other type, skip it
            return 0
        
        # Get base filename without extension
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        
        component_count = 0
        
        # Process each timestamp in the ActorGraph
        for timestamp, graph_nx in actor_graph.actor_graphs.items():
            # Extract components
            components = extract_components_from_graph(graph_nx)
            
            # Save each component
            for comp_idx, component in enumerate(components):
                # Only save components with at least min_nodes nodes
                if component.number_of_nodes() >= min_nodes:
                    # Create output filename: base_timestamp_componentidx.pkl
                    output_filename = f"{base_filename}_{timestamp}_{comp_idx}.pkl"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save component as NetworkX graph
                    with open(output_path, "wb") as f:
                        pickle.dump(component, f)
                    
                    component_count += 1
        
        return component_count
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return 0


def extract_components_from_directory(input_dir, output_dir, max_graphs=None, 
                                      min_nodes=2, pattern="*_actor_graph.pkl", verbose=True):
    """
    Extract components from all graphs in a directory (programmatic interface).
    
    This method can be called from other scripts or notebooks to extract components
    from only N graphs without using command-line arguments.
    
    Note: The function automatically filters out MapGraph files (those ending with
    '_map_graph.pkl') and only processes ActorGraph objects.
    
    Args:
        input_dir: Directory containing full actor graph pickle files
        output_dir: Directory to save component graphs
        max_graphs: Maximum number of graphs to process (None for all)
        min_nodes: Minimum number of nodes for a component to be saved
        pattern: Glob pattern for input files (default: "*_actor_graph.pkl")
        verbose: Print progress messages
        
    Returns:
        dict: Statistics about the extraction
            - 'total_input_files': Total number of input files found
            - 'processed_files': Number of files processed
            - 'total_components': Total number of components extracted
            - 'output_dir': Path to output directory
    
    Example:
        >>> from scripts.extract_components_from_full_graphs import extract_components_from_directory
        >>> stats = extract_components_from_directory(
        ...     input_dir='actor_graphs/carla_nx',
        ...     output_dir='actor_graphs/carla_components_nx',
        ...     max_graphs=100,
        ...     min_nodes=2
        ... )
        >>> print(f"Extracted {stats['total_components']} components from {stats['processed_files']} graphs")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all input files
    input_pattern = os.path.join(input_dir, pattern)
    input_files = sorted(glob.glob(input_pattern))
    
    if not input_files:
        if verbose:
            print(f"No files found matching pattern: {input_pattern}")
        return {
            'total_input_files': 0,
            'processed_files': 0,
            'total_components': 0,
            'output_dir': output_dir
        }
    
    if verbose:
        print(f"Found {len(input_files)} input files")
    
    # Limit number of files if specified
    processed_files = input_files
    if max_graphs is not None and len(input_files) > max_graphs:
        processed_files = input_files[:max_graphs]
        if verbose:
            print(f"Processing first {max_graphs} files (out of {len(input_files)})")
    
    # Process files
    total_components = 0
    
    iterator = tqdm(processed_files, desc="Extracting components") if verbose else processed_files
    
    for input_path in iterator:
        num_components = process_actor_graph_file(
            input_path, 
            output_dir, 
            min_nodes=min_nodes
        )
        total_components += num_components
    
    if verbose:
        print(f"\nExtraction complete!")
        print(f"Total components extracted: {total_components}")
        print(f"Output directory: {output_dir}")
    
    return {
        'total_input_files': len(input_files),
        'processed_files': len(processed_files),
        'total_components': total_components,
        'output_dir': output_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract connected components from full actor graphs"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing full actor graph pickle files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save component graphs"
    )
    
    parser.add_argument(
        "--max-graphs",
        "-n",
        type=int,
        default=None,
        help="Maximum number of graphs to process (None for all). Use this to analyze only N graphs for testing/development."
    )
    
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=2,
        help="Minimum number of nodes for a component to be saved"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_actor_graph.pkl",
        help="Glob pattern for input files. Default: '*_actor_graph.pkl' to exclude map graphs. Use '*.pkl' to include all (with automatic filtering)."
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    if os.path.isabs(args.input_dir):
        input_dir = args.input_dir
    else:
        input_dir = os.path.join(PROJECT_ROOT, args.input_dir)
    
    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    
    # Use the programmatic interface
    stats = extract_components_from_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        max_graphs=args.max_graphs,
        min_nodes=args.min_nodes,
        pattern=args.pattern,
        verbose=True
    )
    
    # Exit with error if no files were found
    if stats['total_input_files'] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

