from shapely.geometry import Point, Polygon
from pathlib import Path
from random import choices
from argparse import Namespace
import networkx as nx
import matplotlib.pyplot as plt
import av2.rendering.vector as vector_plotting_utils

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    visualize_scenario,
)
from av2.map.map_api import ArgoverseStaticMap



#TODO: Do this only once for efficienct
# Function to create a polygon from lane boundaries
def create_lane_polygon(left_boundary, right_boundary):
    left_points = [(point.x, point.y) for point in left_boundary.waypoints]
    right_points = [(point.x, point.y) for point in right_boundary.waypoints]
    # Combine left and right boundary points to form a polygon
    polygon_points = left_points + right_points[::-1]
    return Polygon(polygon_points)

# Function to find the lane ID for a given position
def find_lane_id(position, vector_lane_segments):
    point = Point(position[0], position[1])
    for lane_id, lane in vector_lane_segments.items():
        lane_polygon = create_lane_polygon(lane.left_lane_boundary, lane.right_lane_boundary)
        if lane_polygon.contains(point):
            return lane_id
    return None

# Iterate over each track and find the lane IDs for all timesteps
def find_lane_ids_for_track(track, map, num_timesteps):
    lane_ids = []
    for ii, object_state in enumerate(track.object_states):
        if ii == object_state.timestep:
            position = object_state.position
            lane_id = find_lane_id(position, map.vector_lane_segments)
            lane_ids.append(lane_id)
        else:
            lane_ids.append(None)
    
    if len(lane_ids) > num_timesteps:
        raise ValueError(f"There are too many lane IDs for track {track.track_id}")
    elif len(lane_ids) < num_timesteps:
        lane_ids.extend([None] * (num_timesteps - len(lane_ids)))
        
    return lane_ids

def create_track_lane_dict(scenario, map, num_timesteps):
    track_lane_ids = {}
    for track in scenario.tracks:
        track_lane_ids[track.track_id] = find_lane_ids_for_track(track, map, num_timesteps)
    return track_lane_ids

def get_scenario_data(dataroot,log_id):
    args = Namespace(**{"dataroot": Path(dataroot), "log_id": Path(log_id)})

    static_map_path = (
        args.dataroot / f"{log_id}" / f"log_map_archive_{log_id}.json"
    )
    map = ArgoverseStaticMap.from_json(static_map_path)

    static_scenario_path = (
        args.dataroot / f"{log_id}" / f"scenario_{log_id}.parquet"
    )

    scenario = scenario_serialization.load_argoverse_scenario_parquet(static_scenario_path)

    return scenario, map

def create_graph_from_lanes(map):
    G = nx.MultiDiGraph()

    # Add nodes with attributes
    for lane_id, lane in map.vector_lane_segments.items():
        lane_polygon = create_lane_polygon(lane.left_lane_boundary, lane.right_lane_boundary)
        G.add_node(lane_id, 
                   is_intersection=lane.is_intersection,
                   lane_type=lane.lane_type,
                   left_mark_type=lane.left_mark_type,
                   right_mark_type=lane.right_mark_type,
                   left_boundary=lane.left_lane_boundary,
                   right_boundary=lane.right_lane_boundary,
                   lane_polygon=lane_polygon)

    # Add edges for successors and predecessors (type 1)
    for lane_id, lane in map.vector_lane_segments.items():
        for successor_id in lane.successors:
            if successor_id in G:
                G.add_edge(lane_id, successor_id, edge_type='following')
        for predecessor_id in lane.predecessors:
            if predecessor_id in G:
                G.add_edge(predecessor_id, lane_id, edge_type='following')

    # Add edges for neighboring lanes (type 2)
    for lane_id, lane in map.vector_lane_segments.items():
        if lane.left_neighbor_id is not None:
            if lane.left_neighbor_id in G:
                G.add_edge(lane_id, lane.left_neighbor_id, edge_type='neighbor')
                G.add_edge(lane.left_neighbor_id, lane_id, edge_type='neighbor')
        if lane.right_neighbor_id is not None:
            if lane.right_neighbor_id in G:
                G.add_edge(lane_id, lane.right_neighbor_id, edge_type='neighbor') 
                G.add_edge(lane.right_neighbor_id, lane_id, edge_type='neighbor') #TODO: we can expand this by taking the line type into account 

    # Rename neighboring lanes from lanes in opposite direction by looking for loops.
    edges_opposite = []
    for node in G.nodes():
        for successor in G.successors(node):
            if G[node][successor][0]['edge_type'] == 'following':  # Why only 0? look at it tomorrow and see what's going on.
                for neighbor in G.successors(successor):
                    if G[successor][neighbor][0]['edge_type'] == 'neighbor':
                        for next_node in G.successors(neighbor):
                            if G[neighbor][next_node][0]['edge_type'] == 'following':
                                if G.has_edge(next_node, node) and G[next_node][node][0]['edge_type'] == 'neighbor':
                                    edges_opposite.append((successor, neighbor))
                                    edges_opposite.append((next_node, node))

    # Rename edges to 'opposite'
    for u, v in edges_opposite:
        G[u][v][0]['edge_type'] = 'opposite'

    return G


def visualize_map_graph(G, save_path=None):
    pos = {node: (data['lane_polygon'].centroid.x, data['lane_polygon'].centroid.y) for node, data in G.nodes(data=True)}
    labels = {node: node for node in G.nodes()}

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=50, font_size=8, font_color='black')

    # Draw edges with different styles based on edge type
    edge_type_fol = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'following']
    edge_type_nei = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'neighbor']
    edge_type_opp = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'opposite']

    nx.draw_networkx_edges(G, pos, edgelist=edge_type_fol, width=2, edge_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=edge_type_nei, width=1, edge_color='green')
    nx.draw_networkx_edges(G, pos, edgelist=edge_type_opp, width=1, edge_color='red')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def create_track_relationship_graph(G_map, track_lane_ids, scenario, follow_vehicle_steps):
    timestep_graphs = []

    for t in range(len(next(iter(track_lane_ids.values())))):
        G_t = nx.MultiDiGraph()

        # Add nodes with attributes
        for track_id, lane_ids in track_lane_ids.items():
            track = next(track for track in scenario.tracks if track.track_id == track_id)
            G_t.add_node(track_id, object_type=track.object_type, category=track.category)

        # Add edges based on the conditions
        for track_id_A, lane_ids_A in track_lane_ids.items():
            if lane_ids_A[t] is None:
                continue
            for track_id_B, lane_ids_B in track_lane_ids.items():
                if track_id_A == track_id_B or lane_ids_B[t] is None:
                    continue

                # Check for "following_lead" and "leading_vehicle"
                if nx.has_path(G_map, lane_ids_A[t], lane_ids_B[t]):
                    path = nx.shortest_path(G_map, lane_ids_A[t], lane_ids_B[t], weight=None)
                    if len(path) - 1 <= follow_vehicle_steps and all(G_map[u][v][0]['edge_type'] == 'following' for u, v in zip(path[:-1], path[1:])):
                        G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead')
                        G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle')

                # Check for "direct_neighbor_vehicle"
                if G_map.has_edge(lane_ids_A[t], lane_ids_B[t]) and G_map[lane_ids_A[t]][lane_ids_B[t]][0]['edge_type'] == 'neighbor':
                    G_t.add_edge(track_id_A, track_id_B, edge_type='direct_neighbor_vehicle')

                # Check for "neighbor_vehicle"
                if nx.has_path(G_map, lane_ids_A[t], lane_ids_B[t]):
                    path = nx.shortest_path(G_map, lane_ids_A[t], lane_ids_B[t], weight=None)
                    if len(path) - 1 <= follow_vehicle_steps and any(G_map[u][v][0]['edge_type'] == 'neighbor' for u, v in zip(path[:-1], path[1:])) and not G_t.has_edge(track_id_A, track_id_B):
                        G_t.add_edge(track_id_A, track_id_B, edge_type='neighbor_vehicle')
                        G_t.add_edge(track_id_B, track_id_A, edge_type='neighbor_vehicle')

        timestep_graphs.append(G_t)

        #TODO: opposite direction is missing

    return timestep_graphs

def visualize_actor_graph(G, save_path=None):
    pos = nx.spring_layout(G, scale=1.0, k =0.1)
    labels = {node: node for node in G.nodes() if G.degree(node) > 0}
    nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, nodelist=nodes_with_edges, labels=labels, with_labels=True, node_size=200, font_size=10, font_color='black')

    # Draw edges with different styles based on edge type
    edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'following_lead']
    edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'leading_vehicle']
    edge_type_direct_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'direct_neighbor_vehicle']
    edge_type_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'neighbor_vehicle']

    nx.draw_networkx_edges(G, pos, edgelist=edge_type_following_lead, width=2, edge_color='blue', label='following_lead')
    #nx.draw_networkx_edges(G, pos, edgelist=edge_type_leading_vehicle, width=2, edge_color='cyan', label='leading_vehicle')
    nx.draw_networkx_edges(G, pos, edgelist=edge_type_direct_neighbor_vehicle, width=2, edge_color='green', label='direct_neighbor_vehicle')
    nx.draw_networkx_edges(G, pos, edgelist=edge_type_neighbor_vehicle, width=2, edge_color='red', label='neighbor_vehicle')

    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_map(map, save_path = None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for _, ls in map.vector_lane_segments.items():
        vector_plotting_utils.draw_polygon_mpl(
            ax, ls.polygon_boundary, color="g", linewidth=0.5
        )
        vector_plotting_utils.plot_polygon_patch_mpl(
            ls.polygon_boundary, ax, color="g", alpha=0.2
        )

        # plot all pedestrian crossings
        for _, pc in map.vector_pedestrian_crossings.items():
            vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="r", linewidth=0.5)
            vector_plotting_utils.plot_polygon_patch_mpl(
                pc.polygon, ax, color="r", alpha=0.2
            )

    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()


if __name__=="__main__":
    # path to where the logs live
    repo_root = Path(__file__).parents[1]  
    dataroot = repo_root / "argoverse_data"/ "motion-forecasting" / "test" 

    

    # unique log identifier
    log_id = "ffed77d2-3c42-436c-93e5-6cc696b6e6bb"

    scenario, map = get_scenario_data(dataroot, log_id)
    G_map = create_graph_from_lanes(map)
    visualize_map_graph(G_map, save_path= (repo_root / "map_graph.png" ))
    visualize_scenario(scenario, map, save_path=(repo_root / "scenario_plot.mp4") )
    num_timesteps = len(scenario.timestamps_ns)
    track_lane_ids = create_track_lane_dict(scenario, map, num_timesteps)

    follow_vehicle_steps = 5  # Example value, adjust as needed
    timestep_track_graphs = create_track_relationship_graph(G_map, track_lane_ids, scenario, follow_vehicle_steps)
    visualize_actor_graph(timestep_track_graphs[0], save_path=(repo_root / "vehicle_graph.png"))

    plot_map(map, save_path=(repo_root / "map.png"))

    #for track_id, lane_ids in track_lane_ids.items():
    #    print(f"Track ID: {track_id}, Visited Lane IDs: {lane_ids}")
