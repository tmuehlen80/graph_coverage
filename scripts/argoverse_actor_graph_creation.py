from pathlib import Path
from argparse import Namespace
import matplotlib.pyplot as plt
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon

from graph_creator.MapGraph import MapGraph
from graph_creator.ActorGraph import ActorGraph

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    visualize_scenario
)
from av2.map.map_api import ArgoverseStaticMap
import os
import pickle
from tqdm import tqdm
import networkx as nx
os.getcwd()
import json
import os
from graph_creator.create_graph import get_scenario_data, plot_scene_at_timestep


from pathlib import Path
# repo_root = Path("/Users/marius/code/graph_coverage")
repo_root = Path("/home/tmuehlen/repos/graph_coverage")
#dataroot = repo_root / "argoverse_data_tmp" / "train"
dataroot =  Path("/home/tmuehlen/argoverse_data_tmp") / "train"
print(repo_root)

scenario_folders = [f for f in dataroot.iterdir() if f.is_dir()]

# Print number of scenarios found
print(f"Found {len(scenario_folders)} scenarios")


# read the actor graph setting from the json file
settings_name = "actor_graph_setting_1_50_50_10_20_20_4_4_4"
with open(f"configs/graph_settings/{settings_name}.json", "rb") as f:
    actor_graph_setting = json.load(f)

name_aug = settings_name
os.makedirs(repo_root / "actor_graphs" / f"argoverse_{name_aug}", exist_ok=True)
os.makedirs(repo_root / "actor_graphs" / f"argoverse_{name_aug}_components_nx", exist_ok=True)
os.makedirs(repo_root / "actor_graphs" / f"argoverse_{name_aug}_nx", exist_ok=True)

for i, scenario in enumerate(scenario_folders[:17000]):
    log_id = scenario.name
    timestamp = 1.0
    print(i, "/", len(scenario_folders), log_id)
    if os.path.exists(repo_root / "actor_graphs" / f"argoverse_{name_aug}" / f"{log_id}_map_graph.pkl"):
        continue
    
    scenario, map = get_scenario_data(dataroot, log_id)
    G_map = MapGraph.create_from_argoverse_map(map)
    actor_graph = ActorGraph.from_argoverse_scenario(
                                    scenario, 
                                    G_map, 
                                    **actor_graph_setting,
    )
    with open(repo_root / "actor_graphs" / f"argoverse_{name_aug}" / f"{log_id}_map_graph.pkl", "wb") as f:
        pickle.dump(G_map, f)
    with open(repo_root / "actor_graphs" / f"argoverse_{name_aug}" / f"{log_id}_actor_graph.pkl", "wb") as f:
        pickle.dump(actor_graph, f)
    keys = list(actor_graph.actor_graphs.keys())
    for  key in keys:
        actor_graph.actor_components[key] = [actor_graph.actor_components[key][i] for i in range(len(actor_graph.actor_components[key])) if actor_graph.actor_components[key][i].size() > 1]
    timestamps = list(actor_graph.actor_graphs.keys())
    print(f"Creating {len(timestamps)} actor graphs")
    for timestamp in tqdm(timestamps):
        for component_idx in range(len(actor_graph.actor_components[timestamp])):
            save_path = repo_root / "actor_graphs" / f"argoverse_{name_aug}_components_nx" / f'graph_{log_id}_{str(timestamp).replace(".", "_")}_{component_idx}.pkl'
            with open(save_path, "wb") as file:
                pickle.dump(actor_graph.actor_components[timestamp][component_idx], file)
        save_path = repo_root / "actor_graphs" / f"argoverse_{name_aug}_nx" / f'graph_{log_id}_{str(timestamp).replace(".", "_")}.pkl'
        with open(save_path, "wb") as file:
            pickle.dump(actor_graph.actor_graphs[timestamp], file)

    