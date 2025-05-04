from shapely.geometry import Point, LineString
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy as np
from graph_creator.models import TrackData, ActorType


class ActorGraph:
    def __init__(self):
        self.G_map = None
        self.timestamps_graphs = []
        self.num_timesteps = None
        self.follow_vehicle_steps = None
        self.track_lane_dict = None
        self.actor_graphs = {}

    def find_lane_id_from_pos(self, position):
        point = Point(position[0], position[1])
        for lane_id, data in self.G_map.graph.nodes(data=True):
            lane_polygon = data["node_info"].lane_polygon
            if lane_polygon.contains(point):
                return lane_id
        return None

    def find_lane_ids_for_track(self, track):
        """
        create a list of length of number of timesteps, where each element holds the lane_id of an object at that timestep.

        Missing information is represented by None.
        """
        lane_ids = []
        timestep_list = [step.timestep for step in track.object_states]
        for ii in range(self.num_timesteps):
            if ii in timestep_list:
                position = track.object_states[timestep_list.index(ii)].position
                lane_id = self.find_lane_id_from_pos(position)
                lane_ids.append(str(lane_id) if lane_id is not None else None)
            else:
                lane_ids.append(None)
        if not len(lane_ids) == self.num_timesteps:
            raise ValueError(f"There are too many lane IDs for track {track.track_id}")

        return lane_ids

    def _calculate_s_t_coordinates(self, position: Point, lane_id_str: str) -> Tuple[float, float]:
        """
        Calculate s and t coordinates for a point on a lane.
        s: distance along the lane from start
        t: lateral distance from center line (positive to the left)
        """
        # Get lane boundaries from graph
        left_boundary = self.G_map.graph.nodes[lane_id_str]["node_info"].left_boundary
        right_boundary = self.G_map.graph.nodes[lane_id_str]["node_info"].right_boundary

        # Get first and last points of each boundary using .coords
        left_coords = list(left_boundary.coords)
        right_coords = list(right_boundary.coords)
        left_start_coords = left_coords[0]
        left_end_coords = left_coords[-1]
        right_start_coords = right_coords[0]
        right_end_coords = right_coords[-1]


        # Calculate center line start and end points
        center_start = Point((left_start_coords[0] + right_start_coords[0]) / 2, (left_start_coords[1] + right_start_coords[1]) / 2)
        center_end = Point((left_end_coords[0] + right_end_coords[0]) / 2, (left_end_coords[1] + right_end_coords[1]) / 2)

        # Create center line from start to end point
        center_line = LineString([center_start, center_end]) # Use Point objects directly

        # Project actor position onto center line
        actor_pos_2d = Point(position.x, position.y)
        projected_point = center_line.interpolate(center_line.project(actor_pos_2d))

        # Calculate s-coordinate (distance from start of center line to projected point)
        s_coord = center_line.project(actor_pos_2d)

        # Calculate t-coordinate (perpendicular distance from actor to center line)
        # Use cross product to determine if point is left (negative) or right (positive) of center line
        center_vector = np.array([center_end.x - center_start.x, center_end.y - center_start.y])
        point_vector = np.array([position.x - center_start.x, position.y - center_start.y])
        cross_product = np.cross(center_vector, point_vector)
        t_coord = np.sign(cross_product) * actor_pos_2d.distance(projected_point)

        return s_coord, t_coord

    def _create_track_data_argoverse(self, scenario):
        track_lane_dict = {}
        track_s_value_dict = {}
        track_xyz_pos_dict = {}
        track_speed_lon_dict = {}
        track_actor_type_dict = {}

        for track in scenario.tracks:
            track_id = str(track.track_id)  # Convert to string
            lane_ids = self.find_lane_ids_for_track(track)
            track_lane_dict[track_id] = lane_ids
            # Initialize other dictionaries with None values for missing timesteps
            s_values = []
            xyz_positions = []
            speeds = []

            timestep_list = [step.timestep for step in track.object_states]
            for ii in range(self.num_timesteps):
                if ii in timestep_list:
                    state = track.object_states[timestep_list.index(ii)]
                    position = Point(state.position[0], state.position[1], 0.0)
                    xyz_positions.append(position)
                    if (
                        track_lane_dict[track_id][ii] is not None
                    ):  # there are cases where the lane is None, i.e. the actor is not on a lane.
                        s_coord, t_coord = self._calculate_s_t_coordinates(position, track_lane_dict[track_id][ii])
                        s_values.append(s_coord)
                    else:
                        s_values.append(np.nan)
                    # Calculate longitudinal speed from velocity
                    speed = np.sqrt(state.velocity[0] ** 2 + state.velocity[1] ** 2)
                    speeds.append(speed)
                else:
                    xyz_positions.append(Point(np.nan, np.nan, np.nan))  # Placeholder point
                    speeds.append(np.nan)
                    s_values.append(np.nan)

            track_s_value_dict[track_id] = s_values
            track_xyz_pos_dict[track_id] = xyz_positions
            track_speed_lon_dict[track_id] = speeds
            # Add actor type information using ActorType enum and mapping
            actor_type_str = track.object_type.value
            if actor_type_str == "PEDESTRIAN":
                track_actor_type_dict[track_id] = ActorType.PEDESTRIAN
            elif actor_type_str == "CYCLIST": # If you want separate CYCLIST type later
                 track_actor_type_dict[track_id] = ActorType.CYCLIST
            else: # Default to VEHICLE for all other types
                track_actor_type_dict[track_id] = ActorType.VEHICLE
        # Create and return the Pydantic model
        return TrackData(
            track_lane_dict=track_lane_dict,
            track_s_value_dict=track_s_value_dict,
            track_xyz_pos_dict=track_xyz_pos_dict,
            track_speed_lon_dict=track_speed_lon_dict,
            track_actor_type_dict=track_actor_type_dict,
        )

    @classmethod
    def from_argoverse_scenario(
        cls,
        scenario,
        G_Map,
        max_distance_lead_veh_m=100,
        max_distance_opposite_veh_m=100,
        max_distance_neighbor_forward_m=50,
        max_distance_neighbor_backward_m=50,
        delta_timestep_s=1.0,
    ):
        """
        Create an ActorGraph instance from an Argoverse scenario.

        Args:
            scenario: An Argoverse scenario object
            G_Map: A GraphMap object
            max_distance_lead_veh_m: Maximum distance in meters for leading vehicle relationships
            max_distance_opposite_veh_m: Maximum distance in meters for opposite vehicle relationships
            max_distance_neighbor_forward_m: Maximum distance in meters for forward neighbor vehicle relationships
            max_distance_neighbor_backward_m: Maximum distance in meters for backward neighbor vehicle relationships
        """
        instance = cls()
        instance.G_map = G_Map
        instance.timestamps = list((scenario.timestamps_ns  - min(scenario.timestamps_ns) ) * 10**-9)
        instance.num_timesteps = len(scenario.timestamps_ns)
        instance.max_distance_lead_veh_m = max_distance_lead_veh_m
        instance.max_distance_opposite_veh_m = max_distance_opposite_veh_m
        instance.max_distance_neighbor_forward_m = max_distance_neighbor_forward_m
        instance.max_distance_neighbor_backward_m = max_distance_neighbor_backward_m

        # Get track data as Pydantic model
        track_data = instance._create_track_data_argoverse(scenario)

        # Store the data from the Pydantic model
        instance.track_lane_dict = track_data.track_lane_dict
        instance.track_s_value_dict = track_data.track_s_value_dict
        instance.track_xyz_pos_dict = track_data.track_xyz_pos_dict
        instance.track_speed_lon_dict = track_data.track_speed_lon_dict
        instance.track_actor_type_dict = track_data.track_actor_type_dict

        instance.actor_graphs = instance.create_actor_graphs(
            G_Map,
            max_distance_lead_veh_m=max_distance_lead_veh_m,
            max_distance_neighbor_forward_m=max_distance_neighbor_forward_m,
            max_distance_neighbor_backward_m=max_distance_neighbor_backward_m,
            max_distance_opposite_m=max_distance_opposite_veh_m,
            delta_timestep_s=delta_timestep_s,
        )

        return instance

    @classmethod
    def from_carla_scenario(
        cls,
        scenario,
        G_Map,
        max_distance_lead_veh_m=100,
        max_distance_opposite_veh_m=100,
        max_distance_neighbor_forward_m=50,
        max_distance_neighbor_backward_m=50,
    ):
        """
        Create an ActorGraph instance from a CARLA scenario.

        Args:
            scenario: A pd dataframe with the following columns: 'track_id', 'timestep', 'x', 'y'
            G_Map: A GraphMap object
            max_distance_lead_veh_m: Maximum distance in meters for leading vehicle relationships
            max_distance_opposite_veh_m: Maximum distance in meters for opposite vehicle relationships
            max_distance_neighbor_forward_m: Maximum distance in meters for forward neighbor vehicle relationships
            max_distance_neighbor_backward_m: Maximum distance in meters for backward neighbor vehicle relationships
        """
        instance = cls()
        instance.G_map = G_Map
        instance.num_timesteps = scenario.timestamp.nunique()
        instance.timestamps = scenario.timestamp.unique().tolist()
        instance.max_distance_lead_veh_m = max_distance_lead_veh_m
        instance.max_distance_opposite_veh_m = max_distance_opposite_veh_m
        instance.max_distance_neighbor_forward_m = max_distance_neighbor_forward_m
        instance.max_distance_neighbor_backward_m = max_distance_neighbor_backward_m

        # Get track data as Pydantic model
        track_data = instance._create_track_data_carla(scenario)

        # Store the data from the Pydantic model
        instance.track_lane_dict = track_data.track_lane_dict
        instance.track_s_value_dict = track_data.track_s_value_dict
        instance.track_xyz_pos_dict = track_data.track_xyz_pos_dict
        instance.track_speed_lon_dict = track_data.track_speed_lon_dict
        instance.track_actor_type_dict = track_data.track_actor_type_dict

        instance.actor_graphs = instance.create_actor_graphs(
            G_Map,
            max_distance_lead_veh_m=max_distance_lead_veh_m,
            max_distance_neighbor_forward_m=max_distance_neighbor_forward_m,
            max_distance_neighbor_backward_m=max_distance_neighbor_backward_m,
            max_distance_opposite_m=max_distance_opposite_veh_m,
        )

        instance.actor_components = {}
        for key, value in instance.actor_graphs.items():
            components = list(nx.weakly_connected_components(value))
            subgraphs = [value.subgraph(c).copy() for c in components]
            instance.actor_components[key] = subgraphs

        return instance

    def _create_track_data_carla(self, scenario):
        """For carla, scenario is a pd df containing the time indexed actor data."""
        track_lane_dict = {}
        track_s_value_dict = {}
        track_xyz_pos_dict = {}
        track_speed_lon_dict = {}
        track_actor_type_dict = {}
        actors = scenario.actor_id.unique().tolist()

        # First pass: collect all data
        for actor in actors:
            actor_id = str(actor)  # Convert to string
            mask = scenario.actor_id == actor
            # Convert lane IDs to integers
            track_lane_dict[actor_id] = [lane_id for lane_id in scenario[mask].road_lane_id.tolist()]
            track_s_value_dict[actor_id] = scenario[mask].distance_from_lane_start.tolist()
            # Convert xyz coordinates to Shapely Points
            xyz_coords = scenario[mask].actor_location_xyz.tolist()
            track_xyz_pos_dict[actor_id] = [Point(x, y, z) for x, y, z in xyz_coords]
            track_speed_lon_dict[actor_id] = scenario[mask].actor_speed_lon.tolist()
            # Take the first entry from the actor_type list and map to ActorType
            actor_types_str_list = scenario[mask].actor_type.tolist()
            if actor_types_str_list:
                actor_type_str = actor_types_str_list[0].upper() # Convert to upper for consistency
                if actor_type_str.startswith("WALKER"): # Assuming Carla uses "PEDESTRIAN" string
                    track_actor_type_dict[actor_id] = ActorType.PEDESTRIAN
                elif actor_type_str.startswith("VEHICLE"): # 
                    track_actor_type_dict[actor_id] = ActorType.VEHICLE
            else:
                 # Handle cases where actor type might be missing, default to VEHICLE or raise error
                 track_actor_type_dict[actor_id] = ActorType.VEHICLE
        # Create and return the Pydantic model
        return TrackData(
            track_lane_dict=track_lane_dict,
            track_s_value_dict=track_s_value_dict,
            track_xyz_pos_dict=track_xyz_pos_dict,
            track_speed_lon_dict=track_speed_lon_dict,
            track_actor_type_dict=track_actor_type_dict,
        )

    def create_actor_graphs(
        self,
        G_map,
        max_distance_lead_veh_m,
        max_distance_neighbor_forward_m,
        max_distance_neighbor_backward_m,
        max_distance_opposite_m,
        delta_timestep_s=1.0,
    ):
        graph_timesteps = []
        graph_timesteps_idx = []
        current_timestep = 0.0
        while True:
            # Find closest timestep in self.timesteps
            closest_idx = min(range(len(self.timestamps)), 
                            key=lambda i: abs(self.timestamps[i] - current_timestep))
            closest_timestep = self.timestamps[closest_idx]
            
            # Stop if we would add same timestep again
            if graph_timesteps and closest_timestep == graph_timesteps[-1]:
                break
                
            # Add timestep and index to lists
            graph_timesteps.append(closest_timestep)
            graph_timesteps_idx.append(closest_idx)
            
            # Increment timestep
            current_timestep += delta_timestep_s

        timestep_graphs = {}
        for t in graph_timesteps_idx:
            G_t = nx.MultiDiGraph()

            # Add nodes with attributes
            for track_id, lane_ids in self.track_lane_dict.items():
                if lane_ids[t] is not None:
                    G_t.add_node(
                        track_id,
                        lane_id=lane_ids[t],
                        s=self.track_s_value_dict[track_id][t],
                        xyz=self.track_xyz_pos_dict[track_id][t],
                        lon_speed=self.track_speed_lon_dict[track_id][t],
                        actor_type=self.track_actor_type_dict[track_id],
                    )

            # Add edges based on the conditions
            keys = list(self.track_lane_dict.keys())
            for i in range(len(keys) - 1):
                track_id_A = keys[i]
                lane_ids_A = self.track_lane_dict[keys[i]]

                if lane_ids_A[t] is None:
                    continue

                for j in range(i + 1, len(keys)):
                    track_id_B = keys[j]
                    lane_ids_B = self.track_lane_dict[keys[j]]

                    if lane_ids_B[t] is None:
                        continue

                    # Check for "following_lead" and "leading_vehicle" in same lane
                    if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):

                        path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)

                        if len(path) == 1:  # i.e. both on same lane
                            if (self.track_s_value_dict[track_id_B][t] > self.track_s_value_dict[track_id_A][t]) and (
                                (self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])
                                < max_distance_lead_veh_m
                            ):
                                G_t.add_edge(
                                    track_id_B,
                                    track_id_A,
                                    edge_type="leading_vehicle",
                                    path_length=self.track_s_value_dict[track_id_B][t]
                                    - self.track_s_value_dict[track_id_A][t],
                                )
                                G_t.add_edge(
                                    track_id_A,
                                    track_id_B,
                                    edge_type="following_lead",
                                    path_length=self.track_s_value_dict[track_id_B][t]
                                    - self.track_s_value_dict[track_id_A][t],
                                )

                        # second case: both in different, but following lanes:
                        if len(path) > 1 and all(
                            G_map.graph[u][v][0]["edge_type"] == "following" for u, v in zip(path[:-1], path[1:])
                        ):
                            path_length = (
                                sum([G_map.graph.nodes[node]["node_info"].length for node in path[:-1]])
                                + self.track_s_value_dict[track_id_B][t]
                                - self.track_s_value_dict[track_id_A][t]
                            )
                            if path_length < max_distance_lead_veh_m:
                                G_t.add_edge(
                                    track_id_B,
                                    track_id_A,
                                    edge_type="leading_vehicle",
                                    path_length=path_length,
                                )
                                G_t.add_edge(
                                    track_id_A,
                                    track_id_B,
                                    edge_type="following_lead",
                                    path_length=path_length,
                                )

                        # TODO: PROBLEM: we are not checking i in relationt j,but not j to 1. We might miss a case here

                        if track_id_A == "188747" and track_id_B == "188849":
                            print(f"path: {path}")
                        # third case: on neighboring lanes, forward
                        if (
                            sum([G_map.graph[u][v][0]["edge_type"] == "neighbor" for u, v in zip(path[:-1], path[1:])])
                            == 1
                        ) and (
                            sum([G_map.graph[u][v][0]["edge_type"] == "following" for u, v in zip(path[:-1], path[1:])])
                            == len(path) - 2
                        ):
                            # remove the neighbor node, otherwise that stretch is counted twice.
                            path_length = (
                                sum(
                                    [
                                        G_map.graph.nodes[path[i]]["node_info"].length
                                        for i in range(len(path) - 1)
                                        if G_map.graph[path[i]][path[i + 1]][0]["edge_type"] != "neighbor"
                                    ]
                                )
                                + self.track_s_value_dict[track_id_B][t]
                                - self.track_s_value_dict[track_id_A][t]
                            )
                            if path_length < max_distance_neighbor_forward_m:
                                G_t.add_edge(
                                    track_id_B,
                                    track_id_A,
                                    edge_type="neighbor_vehicle",
                                    path_length=path_length,
                                )
                                G_t.add_edge(
                                    track_id_A,
                                    track_id_B,
                                    edge_type="neighbor_vehicle",
                                    path_length=path_length,
                                )

                        # fourth case: on opposite, directly next lane:
                        # TODO: Marius has to double check this logic.
                        if (
                            sum([G_map.graph[u][v][0]["edge_type"] == "opposite" for u, v in zip(path[:-1], path[1:])])
                            == 1
                        ) and (
                            sum([G_map.graph[u][v][0]["edge_type"] == "following" for u, v in zip(path[:-1], path[1:])])
                            == len(path) - 2
                        ):
                            # remove the opposite node, otherwise that stretch is counted twice. if there is something to check, than if this logic is correct.
                            # Taking -s from the other actor on the opposite direciton, as the s value should be in opposite direction as welll, hopefully..
                            path_length = (
                                sum(
                                    [
                                        G_map.graph.nodes[path[i]]["node_info"].length
                                        for i in range(len(path) - 1)
                                        if G_map.graph[path[i]][path[i + 1]][0]["edge_type"] != "opposite"
                                    ]
                                )
                                + G_map.graph.nodes[path[-1]]["node_info"].length
                                - self.track_s_value_dict[track_id_B][t]
                                - self.track_s_value_dict[track_id_A][t]
                            )
                            if path_length < max_distance_opposite_m:
                                G_t.add_edge(
                                    track_id_B, track_id_A, edge_type="opposite_vehicle", path_length=path_length
                                )
                                G_t.add_edge(
                                    track_id_A, track_id_B, edge_type="opposite_vehicle", path_length=path_length
                                )

            timestep_graphs[self.timestamps[t]] = G_t

        self.actor_graphs = timestep_graphs

        return self.actor_graphs

    def visualize_actor_graph(
        self, t_idx, comp_idx, use_map_pos=True, node_size=1600, save_path=None, 
        graph_or_component="graph", scenario_id=None
    ):
        if graph_or_component == "graph":
            G = self.actor_graphs[t_idx]
        elif graph_or_component == "component":
            G = self.actor_components[t_idx][comp_idx]

        # Calculate number of actors and scale figure size accordingly
        num_actors = len(G.nodes())
        base_size = 6  # Base size for small graphs
        scale_factor = max(1, num_actors / 5)  # Scale up for more than 5 actors
        fig_size = base_size * scale_factor
        
        if use_map_pos:
            pos = {node: (G.nodes[node]['xyz'].x, G.nodes[node]['xyz'].y) for node in G.nodes}
        else:
            pos = nx.spring_layout(G, scale=1.0, k=0.1)
        
        # Scale node size with figure size
        scaled_node_size = node_size * scale_factor
        
        labels = {node: node for node in G.nodes() if G.degree(node) > 0}
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]

        plt.figure(figsize=(fig_size, fig_size))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_edges, node_size=scaled_node_size)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10 * scale_factor, font_color="black")

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "following_lead"]
        edge_type_direct_neighbor_vehicle = [
            (u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "direct_neighbor_vehicle"
        ]
        edge_type_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "neighbor_vehicle"]
        edge_type_opposite_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "opposite_vehicle"]

        # Scale edge width with figure size
        edge_width = 2 * scale_factor
        
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_following_lead,
            width=edge_width,
            edge_color="blue",
            arrows=True,
            node_size=scaled_node_size,
            label="following_lead",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_direct_neighbor_vehicle,
            width=edge_width,
            edge_color="red",
            arrows=True,
            node_size=scaled_node_size,
            label="leading_vehicle",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_neighbor_vehicle,
            width=edge_width,
            edge_color="forestgreen",
            arrows=True,
            node_size=scaled_node_size,
            label="neighbor_vehicle",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_opposite_vehicle,
            width=edge_width,
            edge_color="orange",
            arrows=True,
            node_size=scaled_node_size,
            label="opposite_vehicle",
        )
        # Add legend with custom colors and labels
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='following_lead'), 
            plt.Line2D([0], [0], color='red', label='leading_vehicle'),
            plt.Line2D([0], [0], color='forestgreen', label='neighbor_vehicle'), 
            plt.Line2D([0], [0], color='orange', label='opposite_vehicle')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10 * scale_factor)

        # Add title with scenario_id and timestamp information
        title = f"Timestamp: {t_idx:.2f}s"
        if scenario_id is not None:
            title = f"Scenario: {scenario_id}, {title}"
        plt.title(title, fontsize=12 * scale_factor)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def visualize_s_t_calculation(self, position: Point, lane_id: str, save_path: str = None):
        """
        Visualize the s-t coordinate calculation for a given position and lane.

        Args:
            position: The actor's position as a Point
            lane_id: The lane ID as a string
            save_path: Optional path to save the plot
        """
        # Get lane boundaries from graph
        left_boundary = self.G_map.graph.nodes[lane_id]["node_info"].left_boundary
        right_boundary = self.G_map.graph.nodes[lane_id]["node_info"].right_boundary

        # Get first and last points of each boundary using .coords
        left_coords = list(left_boundary.coords)
        right_coords = list(right_boundary.coords)
        left_start_coords = left_coords[0]
        left_end_coords = left_coords[-1]
        right_start_coords = right_coords[0]
        right_end_coords = right_coords[-1]

        # Calculate center line start and end points
        center_start = Point((left_start_coords[0] + right_start_coords[0]) / 2, (left_start_coords[1] + right_start_coords[1]) / 2)
        center_end = Point((left_end_coords[0] + right_end_coords[0]) / 2, (left_end_coords[1] + right_end_coords[1]) / 2)

        # Create center line from start to end point
        center_line = LineString([center_start, center_end]) # Use Point objects directly

        # Project actor position onto center line
        actor_pos_2d = Point(position.x, position.y)
        projected_point = center_line.interpolate(center_line.project(actor_pos_2d))

        # Create the plot
        plt.figure(figsize=(10, 10))

        # Plot left boundary with arrows using .coords
        left_x = [coord[0] for coord in left_coords]
        left_y = [coord[1] for coord in left_coords]
        plt.plot(left_x, left_y, "b-", label="Left Boundary")
        # Add arrows to left boundary
        for i in range(len(left_x) - 1):
            plt.arrow(
                left_x[i],
                left_y[i],
                left_x[i + 1] - left_x[i],
                left_y[i + 1] - left_y[i],
                head_width=0.5,
                head_length=0.8,
                fc="blue",
                ec="blue",
                alpha=0.5,
            )

        # Plot right boundary with arrows using .coords
        right_x = [coord[0] for coord in right_coords]
        right_y = [coord[1] for coord in right_coords]
        plt.plot(right_x, right_y, "r-", label="Right Boundary")
        # Add arrows to right boundary
        for i in range(len(right_x) - 1):
            plt.arrow(
                right_x[i],
                right_y[i],
                right_x[i + 1] - right_x[i],
                right_y[i + 1] - right_y[i],
                head_width=0.5,
                head_length=0.8,
                fc="red",
                ec="red",
                alpha=0.5,
            )

        # Plot center line with arrow
        center_x = [center_start.x, center_end.x]
        center_y = [center_start.y, center_end.y]
        plt.plot(center_x, center_y, "g--", label="Center Line")
        # Add arrow to center line
        plt.arrow(
            center_start.x,
            center_start.y,
            center_end.x - center_start.x,
            center_end.y - center_start.y,
            head_width=0.5,
            head_length=0.8,
            fc="green",
            ec="green",
            alpha=0.5,
        )

        # Plot original point
        plt.plot(position.x, position.y, "ko", label="Original Point")

        # Plot projected point
        plt.plot(projected_point.x, projected_point.y, "ro", label="Projected Point")

        # Draw line from original to projected point
        plt.plot([position.x, projected_point.x], [position.y, projected_point.y], "k--", label="Projection Line")

        # Calculate and display s and t coordinates
        s_coord, t_coord = self._calculate_s_t_coordinates(position, lane_id)
        plt.title(f"s = {s_coord:.2f}m, t = {t_coord:.2f}m")

        # Add legend and make plot square
        plt.legend()
        plt.axis("equal")

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
