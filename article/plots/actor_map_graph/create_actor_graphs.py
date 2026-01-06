#!/usr/bin/env python3
"""
Standalone script to generate three actor map graphs for a given scenario ID.
"""

import sys
from pathlib import Path
import pickle
import json

script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parents[2]
sys.path.insert(0, str(repo_root))

from graph_creator.MapGraph import MapGraph
from graph_creator.ActorGraph import ActorGraph
from graph_creator.create_graph import get_scenario_data, plot_scene_at_timestep

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    visualize_scenario
)
from av2.map.map_api import ArgoverseStaticMap


def create_actor_graphs(scenario_id, dataroot=None, timestamp=1.0):
    script_dir = Path(__file__).parent
    
    map_graph_path = script_dir / f"{scenario_id}_map_graph.pkl"
    actor_graph_path = script_dir / f"{scenario_id}_actor_graph.pkl"
    
    has_scenario_data = False
    scenario = None
    map = None
    
    if map_graph_path.exists() and actor_graph_path.exists():
        with open(map_graph_path, "rb") as f:
            G_map = pickle.load(f)
        with open(actor_graph_path, "rb") as f:
            actor_graph = pickle.load(f)
        
        if dataroot is not None:
            scenario, map = get_scenario_data(dataroot, scenario_id)
            has_scenario_data = True
    else:
        if dataroot is None:
            print(f"Error: Pickle files not found and dataroot not provided")
            return
        
        scenario, map = get_scenario_data(dataroot, scenario_id)
        has_scenario_data = True
        
        settings_path = script_dir / "actor_graph_setting_1_50_50_10_20_20_4_4_4.json"
        if not settings_path.exists():
            settings_path = repo_root / "configs" / "graph_settings" / "actor_graph_setting_1_50_50_10_20_20_4_4_4.json"
        with open(settings_path, 'r') as f:
            actor_graph_setting = json.load(f)
        
        G_map = MapGraph.create_from_argoverse_map(map)
        actor_graph = ActorGraph.from_argoverse_scenario(scenario, G_map, **actor_graph_setting)
        
        with open(script_dir / f"{scenario_id}_map_graph.pkl", "wb") as f:
            pickle.dump(G_map, f)
        with open(script_dir / f"{scenario_id}_actor_graph.pkl", "wb") as f:
            pickle.dump(actor_graph, f)
    
    timestamps = list(actor_graph.actor_graphs.keys())
    if not timestamps:
        print("Error: No timestamps found in actor graph")
        return
    
    closest_timestamp = min(timestamps, key=lambda x: abs(x - timestamp))
    if closest_timestamp != timestamp:
        print(f"Warning: Timestamp {timestamp} not found. Using closest: {closest_timestamp}")
    timestamp = closest_timestamp
    
    actor_graph.visualize_actor_graph(
        t_idx=timestamp,
        comp_idx=0,
        use_map_pos=True,
        save_path=script_dir / f"{scenario_id}_actor_graph.png",
        scenario_id=None
    )
    
    if has_scenario_data:
        plot_scene_at_timestep(
            scenario,
            map,
            timestep=timestamp,
            actor_graph=actor_graph,
            save_path=script_dir / f"{scenario_id}_scene_at_{timestamp}.png",
            lane_label=False,
            actor_label_fontsize=12,
            actor_label_offset=2.0,
            legend_fontsize=12,
            title=None
        )
    
    G_map.visualize_graph(save_path=script_dir / f"{scenario_id}_map_graph.png")


if __name__ == "__main__":
    chosen_scenario_id = "0922f82c-0640-43a6-b5ce-c42bc729418e"
    dataroot = repo_root / "argoverse_data" / "train"
    create_actor_graphs(chosen_scenario_id, dataroot=dataroot, timestamp=1.0)
