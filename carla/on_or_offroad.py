# This script is used to check if an actor is on a road or offroad and to store the result to the tracks parquet file
# precisely, the smallest distance to any lane in the map graph is calculated and stored in the tracks parquet file

import pandas as pd
import os
#os.chdir("carla")
from graph_creator.MapGraph import MapGraph
from shapely.geometry import Point
from tqdm import tqdm
import numpy as np

files = os.listdir("/home/tmuehlen/repos/graph_coverage/carla/data")
scn_ids = [file.split("_")[1] for file in files if "tracks.parquet" in file]
scn_ids = sorted(scn_ids)
scn_ids = [scn_id for scn_id in scn_ids if scn_id > "2025-09-05"]
print(len(scn_ids))

#scn_id = scn_ids[100]
for j, scn_id in enumerate(scn_ids):
    print(f'{j}/{len(scn_ids)}: {scn_id}')
    tracks = pd.read_parquet(f'/home/tmuehlen/repos/graph_coverage/carla/data/scene_{scn_id}_tracks.parquet')
    g_map = MapGraph()
    g_map.read_graph_from_file(f'/home/tmuehlen/repos/graph_coverage/carla/data/scene_{scn_id}_map_graph.pickle')
    distances_orig_lane = []
    distances_min_lane = []
    min_distances_lane_ids = []
    for i in tqdm(range(len(tracks))):
        road_lane_id = str(tracks.road_id.iloc[i]) + '_' + str(tracks.lane_id.iloc[i])
        distance_orig_lane = Point(tracks.actor_location_xyz.iloc[i]).distance(g_map.graph.nodes[road_lane_id]["node_info"].lane_polygon)
        if distance_orig_lane > 0.0:
            min_distance = np.inf
            for lane in g_map.graph.nodes(data=True):
                try:
                    distance = Point(tracks.actor_location_xyz.iloc[i]).distance(lane[1]["node_info"].lane_polygon)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_lane_id = lane[0]
                except:
                    print("error", i, road_lane_id)
            distances_orig_lane.append(distance_orig_lane)
            distances_min_lane.append(min_distance)
            min_distances_lane_ids.append(min_distance_lane_id)
            # print("min distance details: ", i,  distance_orig_lane, min_distance, road_lane_id, min_distance_lane_id)
        else:
            distances_orig_lane.append(distance_orig_lane)
            distances_min_lane.append(distance_orig_lane)
            min_distances_lane_ids.append(road_lane_id)
    tracks["distances_orig_lane"] = distances_orig_lane
    tracks["distances_min_lane"] = distances_min_lane
    tracks["min_distances_road_lane_id"] = min_distances_lane_ids
    tracks["road_lane_id"] = tracks.road_id.astype(str) + '_' + tracks.lane_id.astype(str)
    tracks.to_parquet(f'/home/tmuehlen/repos/graph_coverage/carla/data/scene_{scn_id}_tracks_with_min_distance_to_lane.parquet')

