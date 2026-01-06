#!/usr/bin/env python3
"""
Standalone script to generate graph construction visualization plots.
Shows graphs at two stages:
- After relation discovery (all connections, no hierarchical selection)
- Final graph (with hierarchical selection and redundancy prevention)
"""

import sys
from pathlib import Path
import pickle
import json

script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parents[2]  # article/plots/graph_construction -> article/plots -> article -> root
sys.path.insert(0, str(repo_root))

from graph_creator.MapGraph import MapGraph
from graph_creator.ActorGraph import ActorGraph
from graph_creator.create_graph import get_scenario_data, plot_scene_at_timestep


def create_graph_construction_plots(scenario_id, dataroot=None, timestamp=1.0):
    """
    Create visualization plots for graph construction at two stages.
    
    Args:
        scenario_id: The scenario ID to process
        dataroot: Path to the data root directory
        timestamp: Timestamp to use for the graph
    """
    script_dir = Path(__file__).parent
    
    # Load graph after discovery if it exists
    graph_after_discovery_path = script_dir / f"{scenario_id}_graph_after_discovery.pkl"
    if not graph_after_discovery_path.exists():
        print(f"Error: Graph after discovery not found at {graph_after_discovery_path}")
        print("Please run create_graph_after_discovery.py first")
        return
    
    with open(graph_after_discovery_path, "rb") as f:
        G_after_discovery = pickle.load(f)
    
    # Load the timestamp that was used when creating the graph
    timestamp_path = script_dir / f"{scenario_id}_timestamp.json"
    if timestamp_path.exists():
        with open(timestamp_path, "r") as f:
            timestamp_data = json.load(f)
            timestamp = timestamp_data["timestamp"]
            print(f"Using timestamp from saved graph: {timestamp}")
    else:
        print(f"Warning: Timestamp file not found, using provided timestamp: {timestamp}")
    
    # Load scenario data for plotting
    if dataroot is None:
        dataroot = repo_root / "argoverse_data" / "train"
    
    scenario, map = get_scenario_data(dataroot, scenario_id)
    
    # Load graph settings
    settings_path = script_dir / "actor_graph_setting_1_50_50_10_20_20_4_4_4.json"
    if not settings_path.exists():
        settings_path = repo_root / "configs" / "graph_settings" / "actor_graph_setting_1_50_50_10_20_20_4_4_4.json"
    
    with open(settings_path, 'r') as f:
        actor_graph_setting = json.load(f)
    
    # Create map graph and actor graph (with hierarchical selection)
    G_map = MapGraph.create_from_argoverse_map(map)
    actor_graph = ActorGraph.from_argoverse_scenario(scenario, G_map, **actor_graph_setting)
    
    # Use the original method that applies hierarchical selection
    actor_graph.create_actor_graphs(
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
    
    # Find the timestep - use the exact timestamp from the saved graph
    timestamps = list(actor_graph.actor_graphs.keys())
    if not timestamps:
        print("Error: No timestamps found in actor graph")
        return
    
    # Use the exact timestamp if it exists, otherwise find closest
    if timestamp in timestamps:
        print(f"Using exact timestamp: {timestamp}")
    else:
        closest_timestamp = min(timestamps, key=lambda x: abs(x - timestamp))
        print(f"Warning: Timestamp {timestamp} not found. Using closest: {closest_timestamp}")
        timestamp = closest_timestamp
    
    # Final graph (from actor_graph with hierarchical selection)
    G_final = actor_graph.actor_graphs[timestamp]
    
    # Create a temporary actor graph object for visualization
    class TempActorGraph:
        def __init__(self, graph):
            self.actor_graphs = {timestamp: graph}
    
    # Plot 1: After discovery (all connections)
    temp_actor_graph = TempActorGraph(G_after_discovery)
    plot_scene_at_timestep(
        scenario,
        map,
        timestep=timestamp,
        actor_graph=temp_actor_graph,
        save_path=script_dir / f"{scenario_id}_after_discovery.png",
        lane_label=False,
        actor_label_fontsize=12,
        actor_label_offset=2.0,
        legend_fontsize=12,
        title="After Relation Discovery"
    )
    
    # Plot 2: Final graph
    plot_scene_at_timestep(
        scenario,
        map,
        timestep=timestamp,
        actor_graph=actor_graph,
        save_path=script_dir / f"{scenario_id}_final.png",
        lane_label=False,
        actor_label_fontsize=12,
        actor_label_offset=2.0,
        legend_fontsize=12,
        title="Final Graph (Hierarchical Selection)"
    )
    
    print(f"Plots saved to {script_dir}")
    print(f"  - {scenario_id}_after_discovery.png")
    print(f"  - {scenario_id}_final.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create graph construction visualization plots")
    parser.add_argument("--scenario-id", type=str, default="0922f82c-0640-43a6-b5ce-c42bc729418e",
                       help="Scenario ID to process")
    parser.add_argument("--timestamp", type=float, default=1.0,
                       help="Timestamp to use for the graph")
    parser.add_argument("--dataroot", type=str, default=None,
                       help="Path to data root directory (default: repo_root/argoverse_data/train)")
    
    args = parser.parse_args()
    
    dataroot = Path(args.dataroot) if args.dataroot else repo_root / "argoverse_data" / "train"
    create_graph_construction_plots(args.scenario_id, dataroot=dataroot, timestamp=args.timestamp)
