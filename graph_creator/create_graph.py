from pathlib import Path
from argparse import Namespace
import matplotlib.pyplot as plt
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon
import numpy as np

from MapGraph import MapGraph
from ActorGraph import ActorGraph

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
                color="blue",
            )

    # If actor_graph is provided, draw the edges
    if actor_graph is not None:
        # Find the closest timestep in the actor graph
        closest_timestep = min(actor_graph.actor_graphs.keys(), key=lambda x: abs(x - timestep))
        G = actor_graph.actor_graphs[closest_timestep]

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "following_lead"]
        edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "leading_vehicle"]
        edge_type_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "neighbor_vehicle"]
        edge_type_opposite_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "opposite_vehicle"]

        # Function to calculate offset positions for multiple arrows between same nodes
        def calculate_arrow_offset(pos_u, pos_v, edge_count, edge_index, offset_distance=0.5):
            """
            Calculate offset positions for arrows to avoid overlap.
            
            Args:
                pos_u: Start position (x, y)
                pos_v: End position (x, y)
                edge_count: Total number of edges between these nodes
                edge_index: Index of this edge (0, 1, 2, ...)
                offset_distance: Distance to offset perpendicular to arrow direction
            
            Returns:
                Tuple of (offset_start, offset_end) positions
            """
            if edge_count <= 1:
                return pos_u, pos_v
            
            # Calculate arrow direction vector
            dx = pos_v[0] - pos_u[0]
            dy = pos_v[1] - pos_u[1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            
            if arrow_length == 0:
                return pos_u, pos_v
            
            # Normalize direction vector
            dx_norm = dx / arrow_length
            dy_norm = dy / arrow_length
            
            # Calculate perpendicular vector (90 degree rotation)
            perp_dx = -dy_norm
            perp_dy = dx_norm
            
            # Calculate offset based on edge index
            # Center the offsets around the original line
            offset_multiplier = (edge_index - (edge_count - 1) / 2) * offset_distance
            
            # Apply offset to create a parallel line, but keep endpoints at nodes
            # We'll offset the control points for the arrow, not the start/end
            offset_start = (
                pos_u[0] + perp_dx * offset_multiplier,
                pos_u[1] + perp_dy * offset_multiplier
            )
            offset_end = (
                pos_v[0] + perp_dx * offset_multiplier,
                pos_v[1] + perp_dy * offset_multiplier
            )
            
            # Debug: print offset information
            if edge_count > 1:
                print(f"Edge {edge_index + 1}/{edge_count} between nodes: offset_multiplier={offset_multiplier:.2f}, "
                      f"offset_distance={offset_distance}, perp_vector=({perp_dx:.3f}, {perp_dy:.3f})")
                print(f"  Original: ({pos_u[0]:.1f}, {pos_u[1]:.1f}) -> ({pos_v[0]:.1f}, {pos_v[1]:.1f})")
                print(f"  Offset:   ({offset_start[0]:.1f}, {offset_start[1]:.1f}) -> ({offset_end[0]:.1f}, {offset_end[1]:.1f})")
            
            return offset_start, offset_end

        # Group edges by node pairs to count multiple edges
        def group_edges_by_nodes(edges):
            edge_groups = {}
            for u, v in edges:
                pair = tuple(sorted([u, v]))
                if pair not in edge_groups:
                    edge_groups[pair] = []
                edge_groups[pair].append((u, v))
            return edge_groups

        # Group ALL edges by node pairs to handle overlapping edges of different types
        def group_all_edges_by_nodes():
            all_edges = []
            all_edges.extend([(u, v, 'following_lead') for u, v in edge_type_following_lead])
            all_edges.extend([(u, v, 'leading_vehicle') for u, v in edge_type_leading_vehicle])
            all_edges.extend([(u, v, 'neighbor_vehicle') for u, v in edge_type_neighbor_vehicle])
            all_edges.extend([(u, v, 'opposite_vehicle') for u, v in edge_type_opposite_vehicle])
            
            edge_groups = {}
            for u, v, edge_type in all_edges:
                pair = tuple(sorted([u, v]))
                if pair not in edge_groups:
                    edge_groups[pair] = []
                edge_groups[pair].append((u, v, edge_type))
            return edge_groups

        # Group edges by shared nodes to handle overlapping arrows
        def group_edges_by_shared_nodes():
            all_edges = []
            all_edges.extend([(u, v, 'following_lead') for u, v in edge_type_following_lead])
            all_edges.extend([(u, v, 'leading_vehicle') for u, v in edge_type_leading_vehicle])
            all_edges.extend([(u, v, 'neighbor_vehicle') for u, v in edge_type_neighbor_vehicle])
            all_edges.extend([(u, v, 'opposite_vehicle') for u, v in edge_type_opposite_vehicle])
            
            # Group edges by shared nodes (either start or end)
            node_groups = {}
            for u, v, edge_type in all_edges:
                # Add to groups for both start and end nodes
                if u not in node_groups:
                    node_groups[u] = []
                if v not in node_groups:
                    node_groups[v] = []
                
                node_groups[u].append((u, v, edge_type))
                node_groups[v].append((u, v, edge_type))
            
            return node_groups

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
    log_id = "d25d1aaa-8bb2-4c6b-98a5-fa5aa2ddd2eb"

    scenario, map = get_scenario_data(dataroot, log_id)
    G_map = MapGraph.create_from_argoverse_map(map)
    G_map.visualize_graph(save_path=(repo_root / "map_graph.png"))
    # visualize_scenario(scenario, map, save_path=(repo_root / "scenario_plot.mp4") )
    # plot_argoverse_map(map, save_path=(repo_root / "map.png"))

    actor_graph = ActorGraph.from_argoverse_scenario(
        scenario, 
        G_map,
        max_number_lead_vehicle=1,
        max_number_neighbor=1,
        max_number_opposite=1,
        max_node_distance=3
    )
    show_timestep = 1.0
    plot_scene_at_timestep(scenario, map, timestep=show_timestep, save_path=(repo_root / "map.png"))
    actor_graph.visualize_actor_graph(t_idx=show_timestep, comp_idx=0,  save_path=(repo_root / "actor_graph.png"))
