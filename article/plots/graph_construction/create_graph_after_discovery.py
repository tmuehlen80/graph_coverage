#!/usr/bin/env python3
"""
Script to create a graph showing all connections after relation discovery (before hierarchical selection).
This graph includes all discovered relations without applying redundancy prevention.
"""

import sys
from pathlib import Path
import pickle
import json
import networkx as nx

script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parents[2]  # article/plots/graph_construction -> article/plots -> article -> root
sys.path.insert(0, str(repo_root))

from graph_creator.MapGraph import MapGraph
from graph_creator.ActorGraph import ActorGraph
from graph_creator.create_graph import get_scenario_data


def create_graph_after_discovery(scenario_id, dataroot=None, timestamp=1.0):
    """
    Create a graph showing all connections after relation discovery (before hierarchical selection).
    
    Args:
        scenario_id: The scenario ID to process
        dataroot: Path to the data root directory
        timestamp: Timestamp to use for the graph
    """
    script_dir = Path(__file__).parent
    
    if dataroot is None:
        dataroot = repo_root / "argoverse_data" / "train"
    
    # Load scenario data
    scenario, map = get_scenario_data(dataroot, scenario_id)
    
    # Load graph settings
    settings_path = script_dir / "actor_graph_setting_1_50_50_10_20_20_4_4_4.json"
    if not settings_path.exists():
        settings_path = repo_root / "configs" / "graph_settings" / "actor_graph_setting_1_50_50_10_20_20_4_4_4.json"
    
    with open(settings_path, 'r') as f:
        actor_graph_setting = json.load(f)
    
    # Create map graph
    G_map = MapGraph.create_from_argoverse_map(map)
    
    # Create actor graph object manually to avoid calling create_actor_graphs
    actor_graph = ActorGraph()
    actor_graph.G_map = G_map
    actor_graph.timestamps = list((scenario.timestamps_ns - min(scenario.timestamps_ns)) * 10**-9)
    actor_graph.num_timesteps = len(scenario.timestamps_ns)
    
    # Get track data
    track_data = actor_graph._create_track_data_argoverse(scenario)
    actor_graph.track_lane_dict = track_data.track_lane_dict
    actor_graph.track_s_value_dict = track_data.track_s_value_dict
    actor_graph.track_xyz_pos_dict = track_data.track_xyz_pos_dict
    actor_graph.track_speed_lon_dict = track_data.track_speed_lon_dict
    actor_graph.track_actor_type_dict = track_data.track_actor_type_dict
    
    # Use the alternative method that creates graph from ALL relations
    actor_graph.create_actor_graphs_all_relations(
        G_map,
        actor_graph_setting['max_distance_lead_veh_m'],
        actor_graph_setting['max_distance_neighbor_forward_m'],
        actor_graph_setting['max_distance_neighbor_backward_m'],
        actor_graph_setting['max_distance_opposite_forward_m'],
        actor_graph_setting['max_distance_opposite_backward_m'],
        actor_graph_setting.get('max_node_distance_leading', 3),
        actor_graph_setting.get('max_node_distance_neighbor', 3),
        actor_graph_setting.get('max_node_distance_opposite', 3),
        actor_graph_setting.get('delta_timestep_s', 1.0)
    )
    
    # Create actor components (needed for compatibility)
    actor_graph.actor_components = {}
    for key, value in actor_graph.actor_graphs.items():
        components = list(nx.weakly_connected_components(value))
        subgraphs = [value.subgraph(c).copy() for c in components]
        actor_graph.actor_components[key] = subgraphs
    
    # Find the timestep index closest to the desired timestamp
    timestamps = list(actor_graph.actor_graphs.keys())
    if not timestamps:
        print("Error: No timestamps found in actor graph")
        return
    
    closest_timestamp = min(timestamps, key=lambda x: abs(x - timestamp))
    if closest_timestamp != timestamp:
        print(f"Warning: Timestamp {timestamp} not found. Using closest: {closest_timestamp}")
    timestamp = closest_timestamp
    
    G_after_discovery = actor_graph.actor_graphs[timestamp]
    
    # Save the graph
    output_path = script_dir / f"{scenario_id}_graph_after_discovery.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(G_after_discovery, f)
    
    # Save the timestamp used so we can use the same one when plotting
    timestamp_path = script_dir / f"{scenario_id}_timestamp.json"
    with open(timestamp_path, "w") as f:
        json.dump({"timestamp": timestamp}, f)
    
    print(f"Graph after discovery saved to: {output_path}")
    print(f"Graph has {G_after_discovery.number_of_nodes()} nodes and {G_after_discovery.number_of_edges()} edges")
    print(f"Using timestamp: {timestamp}")
    
    return G_after_discovery


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create graph after relation discovery (before hierarchical selection)")
    parser.add_argument("--scenario-id", type=str, default="0922f82c-0640-43a6-b5ce-c42bc729418e",
                       help="Scenario ID to process")
    parser.add_argument("--timestamp", type=float, default=1.0,
                       help="Timestamp to use for the graph")
    parser.add_argument("--dataroot", type=str, default=None,
                       help="Path to data root directory (default: repo_root/argoverse_data/train)")
    
    args = parser.parse_args()
    
    dataroot = Path(args.dataroot) if args.dataroot else repo_root / "argoverse_data" / "train"
    create_graph_after_discovery(args.scenario_id, dataroot=dataroot, timestamp=args.timestamp)
