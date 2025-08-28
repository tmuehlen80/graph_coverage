from pathlib import Path
from argparse import Namespace
import matplotlib.pyplot as plt
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon

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
                position[1] - 2.5,
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

        # Draw edges with arrows
        for u, v in edge_type_following_lead:
            pos_u = (G.nodes[u]['xyz'].x, G.nodes[u]['xyz'].y)
            pos_v = (G.nodes[v]['xyz'].x, G.nodes[v]['xyz'].y)
            ax.annotate("", xy=pos_v, xytext=pos_u,
                       arrowprops=dict(arrowstyle="->", color='blue', linewidth=1.5, alpha=0.7))

        for u, v in edge_type_leading_vehicle:
            pos_u = (G.nodes[u]['xyz'].x, G.nodes[u]['xyz'].y)
            pos_v = (G.nodes[v]['xyz'].x, G.nodes[v]['xyz'].y)
            ax.annotate("", xy=pos_v, xytext=pos_u,
                       arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5, alpha=0.7))

        for u, v in edge_type_neighbor_vehicle:
            pos_u = (G.nodes[u]['xyz'].x, G.nodes[u]['xyz'].y)
            pos_v = (G.nodes[v]['xyz'].x, G.nodes[v]['xyz'].y)
            ax.annotate("", xy=pos_v, xytext=pos_u,
                       arrowprops=dict(arrowstyle="->", color='forestgreen', linewidth=1.5, alpha=0.7))

        for u, v in edge_type_opposite_vehicle:
            pos_u = (G.nodes[u]['xyz'].x, G.nodes[u]['xyz'].y)
            pos_v = (G.nodes[v]['xyz'].x, G.nodes[v]['xyz'].y)
            ax.annotate("", xy=pos_v, xytext=pos_u,
                       arrowprops=dict(arrowstyle="->", color='orange', linewidth=1.5, alpha=0.7))

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
