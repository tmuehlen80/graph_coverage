from pathlib import Path
from argparse import Namespace
import matplotlib.pyplot as plt
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon
import numpy as np

from graph_creator.MapGraph import MapGraph
from graph_creator.ActorGraph import ActorGraph

from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    visualize_scenario,
)
from av2.map.map_api import ArgoverseStaticMap


def get_scenario_data(dataroot, log_id):
    args = Namespace(**{"dataroot": Path(dataroot), "log_id": Path(log_id)})

    static_map_path = args.dataroot / f"{log_id}" / f"log_map_archive_{log_id}.json"
    map = ArgoverseStaticMap.from_json(static_map_path)

    static_scenario_path = args.dataroot / f"{log_id}" / f"scenario_{log_id}.parquet"

    scenario = scenario_serialization.load_argoverse_scenario_parquet(static_scenario_path)

    return scenario, map


def plot_argoverse_map(map, save_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for _, ls in map.vector_lane_segments.items():
        vector_plotting_utils.draw_polygon_mpl(ax, ls.polygon_boundary, color="g", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(ls.polygon_boundary, ax, color="g", alpha=0.2)

        # plot all pedestrian crossings
        for _, pc in map.vector_pedestrian_crossings.items():
            vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="r", linewidth=0.5)
            vector_plotting_utils.plot_polygon_patch_mpl(pc.polygon, ax, color="r", alpha=0.2)

    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()


def plot_scene_at_timestep(scenario, map, timestep, actor_graph=None, save_path=None, lane_label=False):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot()

    for _, ls in map.vector_lane_segments.items():
        vector_plotting_utils.draw_polygon_mpl(ax, ls.polygon_boundary, color="grey", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(ls.polygon_boundary, ax, color="grey", alpha=0.2)

        if lane_label:
            # Write the label ID in the middle of each polygon
            polygon = Polygon(ls.polygon_boundary)
            centroid = polygon.centroid
            ax.text(
                centroid.x,
                centroid.y,
                str(ls.id),
                fontsize=5,
                ha="center",
                va="center",
                color="black",
            )

    # Plot all pedestrian crossings
    for _, pc in map.vector_pedestrian_crossings.items():
        vector_plotting_utils.draw_polygon_mpl(ax, pc.polygon, color="brown", linewidth=0.5)
        vector_plotting_utils.plot_polygon_patch_mpl(pc.polygon, ax, color="brown", alpha=0.2)

    # Plot the position of each actor at the given timestep
    # If actor_graph is provided, only plot actors that are in the graph (after filtering)
    if actor_graph is not None:
        # Find the closest timestep in the actor graph
        closest_timestep = min(actor_graph.actor_graphs.keys(), key=lambda x: abs(x - timestep))
        G = actor_graph.actor_graphs[closest_timestep]
        
        # Only plot actors that are in the actor graph (non-parked vehicles)
        # Use positions directly from the actor graph to ensure consistency with arrows
        for track_id in G.nodes():
            node_data = G.nodes[track_id]
            xyz_point = node_data['xyz']
            ax.plot(xyz_point.x, xyz_point.y, "bo", markersize=5)
            ax.text(
                xyz_point.x,
                xyz_point.y - 0.5,  # Reduced from 2.5 to 0.5 for closer positioning
                track_id,
                fontsize=8,
                ha="center",
                va="center",
                color="red",
            )
    else:
        # If no actor_graph provided, plot all actors (original behavior)
        for track in scenario.tracks:
            timestep_list = [step.timestep for step in track.object_states]
            if timestep in timestep_list:
                object_state = track.object_states[timestep_list.index(timestep)]
                position = object_state.position
                ax.plot(position[0], position[1], "bo", markersize=5)
                ax.text(
                    position[0],
                    position[1] - 0.5,  # Reduced from 2.5 to 0.5 for closer positioning
                    track.track_id,
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="red",
                )

    # If actor_graph is provided, draw the edges
    if actor_graph is not None:
        # Use the same G that was already found above

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "following_lead"]
        edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "leading_vehicle"]
        edge_type_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "neighbor_vehicle"]
        edge_type_opposite_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "opposite_vehicle"]

       # Collect all edges and their positions for overlap detection
        all_edge_positions = []
        
        # Draw edges with arrows and offset handling
        for edge_type_name, edges, color in [
            ("following_lead", edge_type_following_lead, 'blue'),
            ("leading_vehicle", edge_type_leading_vehicle, 'red'),
            ("neighbor_vehicle", edge_type_neighbor_vehicle, 'forestgreen'),
            ("opposite_vehicle", edge_type_opposite_vehicle, 'orange')
        ]:
            print(f"\n{edge_type_name}: {len(edges)} total edges")
            
            for u, v in edges:
                pos_u = (G.nodes[u]['xyz'].x, G.nodes[u]['xyz'].y)
                pos_v = (G.nodes[v]['xyz'].x, G.nodes[v]['xyz'].y)
                
                # Check if this edge overlaps with any existing edges
                overlap_count = 0
                for existing_pos in all_edge_positions:
                    existing_start, existing_end = existing_pos
                    # Check if edges share start or end points
                    if (pos_u == existing_start or pos_u == existing_end or 
                        pos_v == existing_start or pos_v == existing_end):
                        overlap_count += 1
                
                # Draw the arrow with offset using FancyArrowPatch for better control
                from matplotlib.patches import FancyArrowPatch
                
                if overlap_count > 0:
                    # Create offset arrow path
                    arrow = FancyArrowPatch(
                        pos_u, pos_v,  # Keep original start/end at nodes
                        connectionstyle=f"arc3,rad={0.1 * overlap_count}",  # Curved path for offset
                        arrowstyle="->",
                        color=color,
                        linewidth=1.5,
                        alpha=0.7,
                        mutation_scale=15
                    )
                    ax.add_patch(arrow)
                else:
                    # No overlap, draw straight arrow
                    ax.annotate("", xy=pos_v, xytext=pos_u,
                               arrowprops=dict(arrowstyle="->", color=color, linewidth=1.5, alpha=0.7))
                
                # Store this edge position for future overlap detection
                all_edge_positions.append((pos_u, pos_v))

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='following_lead'),
            plt.Line2D([0], [0], color='red', label='leading_vehicle'),
            plt.Line2D([0], [0], color='forestgreen', label='neighbor_vehicle'),
            plt.Line2D([0], [0], color='orange', label='opposite_vehicle')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()


if __name__ == "__main__":
    # path to where the logs live
    repo_root = Path(__file__).parents[1]
    dataroot = repo_root / "argoverse_data" / "train"

    # unique log identifier
    log_id = "000ace8b-a3d2-4228-bd87-91b66a9c5127"

    scenario, map = get_scenario_data(dataroot, log_id)
    G_map = MapGraph.create_from_argoverse_map(map)
    G_map.visualize_graph(save_path=(repo_root / "map_graph.png"))
    visualize_scenario(scenario, map, save_path=(repo_root / "scenario_plot.mp4") )
    # plot_argoverse_map(map, save_path=(repo_root / "map.png"))

    actor_graph = ActorGraph.from_argoverse_scenario(
                                    scenario, 
                                    G_map, 
                                    delta_timestep_s=1.0,
                                    max_distance_lead_veh_m=50,
                                    max_distance_opposite_forward_m=100,
                                    max_distance_opposite_backward_m=10,
                                    max_distance_neighbor_forward_m=50,
                                    max_distance_neighbor_backward_m=20,
                                    max_node_distance_leading=10,
                                    max_node_distance_neighbor=6,
                                    max_node_distance_opposite=6,
    )
    show_timestep = 1.0
    plot_scene_at_timestep(scenario, map, timestep=show_timestep, save_path=(repo_root / "map.png"))
    actor_graph.visualize_actor_graph(t_idx=show_timestep, comp_idx=0,  save_path=(repo_root / "actor_graph.png"))
