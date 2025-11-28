import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial import distance
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString
import pandas as pd
#from src.generate_traffic_data import clean_carla, spawn_scene, run_scene
import carla
import os
os.getcwd()
from datetime import datetime
import time
import random
from tqdm import tqdm
import networkx as nx
import numpy as np
import pickle
import json
#os.chdir('../..')
#os.getcwd()

from graph_creator.MapGraph import MapGraph
from graph_creator.ActorGraph import ActorGraph
from graph_creator.utilities import make_node_edge_df, check_offroad_actors
#from graph_creator.plot_graphs import plot_lane_map_advanced, add_actors_to_map, add_actor_edges_to_map


files = os.listdir("/home/tmuehlen/repos/graph_coverage/carla/data")
#scn_ids = [file.split("_")[1] for file in files if "tracks_with_min_distance_to_lane" in file]
scn_ids = [file.split("scene_")[1].split("_tracks")[0] for file in files if "tracks.parquet" in file]
scn_ids = sorted(scn_ids)

#scn_ids = [scn_id for scn_id in scn_ids if scn_id > "2025-09-05"]
print(len(scn_ids))
#scn_ids

# read the actor graph setting from the json file
from pathlib import Path
# repo_root = Path("/Users/marius/code/graph_coverage")
repo_root = Path("/home/tmuehlen/repos/graph_coverage")

settings_name = "actor_graph_setting_1_50_50_10_20_20_4_4_4"
with open(f"graph_settings/{settings_name}.json", "rb") as f:
    actor_graph_setting = json.load(f)

name_aug = settings_name
os.makedirs(repo_root / "actor_graphs" / f"carla_{name_aug}", exist_ok=True)
os.makedirs(repo_root / "actor_graphs" / f"carla_{name_aug}_components_nx", exist_ok=True)
os.makedirs(repo_root / "actor_graphs" / f"carla_{name_aug}_nx", exist_ok=True)


# run through all scens so far and create the graphs:
#os.makedirs("/home/tmuehlen/repos/graph_coverage/actor_graphs/carla_w_intersection", exist_ok=True)

count = 0

for i, scn_id in enumerate(scn_ids):
    print(i, "/", len(scn_ids), scn_id)
    if os.path.exists(repo_root / "actor_graphs" / f"carla_{name_aug}" / f"{scn_id}_map_graph.pkl"):
        continue
    
    g_map = MapGraph()
    g_map.read_graph_from_file(f'/home/tmuehlen/repos/graph_coverage/carla/data/scene_{scn_id}_map_graph.pickle')
    # tracks = pd.read_parquet(f'/home/tmuehlen/repos/graph_coverage/carla/data/scene_{scn_id}_tracks_with_min_distance_to_lane.parquet')
    tracks = pd.read_parquet(f'/home/tmuehlen/repos/graph_coverage/carla/data/scene_{scn_id}_tracks.parquet')
    tracks['road_lane_id'] = tracks.road_id.astype(str) + '_' + tracks.lane_id.astype(str)
    # hm, this does not work, because it results in different lengths per actor.
    # mask = tracks.distances_orig_lane < offroad_distance_m
    # tracks = tracks.loc[mask, :].reset_index(drop = True)
    timestamps = tracks.timestamp.unique().tolist()
    actors = tracks.actor_id.unique().tolist()
    ag = ActorGraph()
    ag_carla = ag.from_carla_scenario(tracks, 
        g_map, 
        **actor_graph_setting,
    )
    with open(repo_root / "actor_graphs" / f"carla_{name_aug}" / f"{scn_id}_map_graph.pkl", "wb") as f:
        pickle.dump(g_map, f)
    with open(repo_root / "actor_graphs" / f"carla_{name_aug}" / f"{scn_id}_actor_graph.pkl", "wb") as f:
        pickle.dump(ag_carla, f)
    offroad_df = check_offroad_actors(ag_carla, g_map)
    # only store the graphs, which have a max of 3 meters offroad distance
    if offroad_df.distance.max() < 3.0:
        count += 1
        # clean up the graphs and components, i.e. remove components with only one node:
        keys = list(ag_carla.actor_graphs.keys())
        for  key in keys:
            ag_carla.actor_components[key] = [ag_carla.actor_components[key][i] for i in range(len(ag_carla.actor_components[key])) if ag_carla.actor_components[key][i].size() > 1]
        ag_timestamps = list(ag_carla.actor_graphs.keys())
        timestamp_idx = 0
        component_idx = 0
        for timestamp_idx in range(len(ag_timestamps)):
            for component_idx in range(len(ag_carla.actor_components[ag_timestamps[timestamp_idx]])):
                save_path = repo_root / "actor_graphs" / f"carla_{name_aug}_components_nx" / f"graph_{scn_id}_{timestamp_idx}_{component_idx}.pkl"
                with open(save_path, "wb") as file:
                    pickle.dump(ag_carla.actor_components[ag_timestamps[timestamp_idx]][component_idx], file)
        save_path = repo_root / "actor_graphs" / f"carla_{name_aug}_nx" / f'graph_{scn_id}_{str(timestamp_idx).replace(".", "_")}.pkl'
        with open(save_path, "wb") as file:
            pickle.dump(ag_carla.actor_graphs[ag_timestamps[timestamp_idx]], file)
    else:
        print(f'{scn_id} has offroad actors')

