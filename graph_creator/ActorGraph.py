from shapely.geometry import Point, LineString
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class TrackData(BaseModel):
    track_lane_dict: Dict[str, List[Optional[str]]] = Field(
        description="Dictionary mapping track IDs to lists of lane IDs (can be None)"
    )
    track_s_value_dict: Dict[str, List[float]] = Field(description="Dictionary mapping track IDs to lists of s-values")
    track_xyz_pos_dict: Dict[str, List[Point]] = Field(description="Dictionary mapping track IDs to lists of 3D points")
    track_speed_lon_dict: Dict[str, List[float]] = Field(
        description="Dictionary mapping track IDs to lists of longitudinal speeds"
    )
    track_actor_type_dict: Dict[str, List[str]] = Field(description="Dictionary mapping track IDs to actor types.")

    @field_validator(
        "track_lane_dict",
        "track_s_value_dict",
        "track_xyz_pos_dict",
        "track_speed_lon_dict",
        "track_actor_type_dict",
    )

    @classmethod
    def validate_list_lengths(cls, v: Dict[str, List]) -> Dict[str, List]:
        if not v:  # Skip validation if dictionary is empty
            return v

        # Get all actors from the dictionary
        actors = list(v.keys())

        # Get the length of the first list for the first actor
        first_length = len(v[actors[0]])

        # Check that all lists for all actors have the same length
        for actor in actors:
            if len(v[actor]) != first_length:
                raise ValueError(
                    f"List length mismatch for actor {actor}. Expected {first_length}, got {len(v[actor])}"
                )

        return v

    @model_validator(mode="after")
    def validate_dict_consistency(self) -> "TrackData":
        # Get all dictionaries
        lane_dict = self.track_lane_dict
        s_value_dict = self.track_s_value_dict
        xyz_dict = self.track_xyz_pos_dict
        speed_dict = self.track_speed_lon_dict
        actor_dict = self.track_actor_type_dict
        # Get all unique actors across all dictionaries
        all_actors = set(lane_dict.keys()) | set(s_value_dict.keys()) | set(xyz_dict.keys()) | set(speed_dict.keys() | set(actor_dict.keys()))

        # Check that all dictionaries have the same actors
        for actor in all_actors:
            if not all(actor in d for d in [lane_dict, s_value_dict, xyz_dict, speed_dict, actor_dict]):
                raise ValueError(f"Actor {actor} is missing in one or more dictionaries")

        # Check that all lists have the same length for each actor
        for actor in all_actors:
            lengths = [
                len(lane_dict[actor]),
                len(s_value_dict[actor]),
                len(xyz_dict[actor]),
                len(speed_dict[actor]),
                len(actor_dict[actor]),
                # As actor_dict isnot time indexed, it does not need to be checked here.,
            ]
            if not all(l == lengths[0] for l in lengths):
                raise ValueError(f"Inconsistent list lengths for actor {actor}. Expected {lengths[0]}, got {lengths}")

        return self

    class Config:
        arbitrary_types_allowed = True


class ActorGraph:
    # add opposite direction
    # use segmeent length for cut off
    # go over alle scenarios
    # add relation for actors on same ID
    def __init__(self):
        self.G_map = None
        self.num_timesteps = None
        self.follow_vehicle_steps = None
        self.track_lane_dict = None
        self.actor_graphs = []

    def find_lane_id_from_pos(self, position):
        point = Point(position[0], position[1])
        for lane_id, data in self.G_map.graph.nodes(data=True):
            lane_polygon = data.get("lane_polygon")
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
                lane_ids.append(int(lane_id) if lane_id is not None else None)
            else:
                lane_ids.append(None)
        if not len(lane_ids) == self.num_timesteps:
            raise ValueError(f"There are too many lane IDs for track {track.track_id}")

        return lane_ids

    def _calculate_s_t_coordinates(self, position: Point, lane_id: int) -> Tuple[float, float]:
        """
        Calculate s and t coordinates for a point on a lane.
        s: distance along the lane from start
        t: lateral distance from center line (positive to the left)
        """
        # Get lane boundaries from graph
        left_boundary = self.G_map.graph.nodes[lane_id]["left_boundary"]
        right_boundary = self.G_map.graph.nodes[lane_id]["right_boundary"]

        # Get first and last points of each boundary
        left_start = left_boundary.waypoints[0]
        left_end = left_boundary.waypoints[-1]
        right_start = right_boundary.waypoints[0]
        right_end = right_boundary.waypoints[-1]

        # Calculate center line start and end points
        center_start = Point((left_start.x + right_start.x) / 2, (left_start.y + right_start.y) / 2)
        center_end = Point((left_end.x + right_end.x) / 2, (left_end.y + right_end.y) / 2)

        # Create center line from start to end point
        center_line = LineString([(center_start.x, center_start.y), (center_end.x, center_end.y)])

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
        # Create and return the Pydantic model
        return TrackData(
            track_lane_dict=track_lane_dict,
            track_s_value_dict=track_s_value_dict,
            track_xyz_pos_dict=track_xyz_pos_dict,
            track_speed_lon_dict=track_speed_lon_dict,
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
        instance.num_timesteps = len(scenario.timestamps_ns)
        instance.timestamps = scenario.timestamps_ns
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
            track_actor_type_dict[actor_id] = scenario[mask].actor_type.tolist()
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
    ):
        number_graphs = 10  # TODO: replace by delta_time .. Make sure time def in carla and argo is the same!
        timestep_delta = int(len(self.track_lane_dict) / number_graphs)

        timestep_graphs = {}
        print(len(self.track_lane_dict))
        for t in tqdm(range(0, len(self.track_lane_dict), timestep_delta)):
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
                #for track_id_A, lane_ids_A in self.track_lane_dict.items():
            
                if lane_ids_A[t] is None:
                    continue

                #for track_id_B, lane_ids_B in self.track_lane_dict.items():
                for j in range(i + 1, len(keys)):
                    track_id_B = keys[j]
                    lane_ids_B = self.track_lane_dict[keys[j]]

                    if lane_ids_B[t] is None:
                        continue
                    # Skip if we've already processed this pair.
                    # Should not happen with the above nested loops, but keep it to be on the safe side.
                    # if str(track_id_A) == str(track_id_B):
                    #    continue
                    #if (track_id_B, track_id_A) in G_t.edges():
                    #    continue

                    # Check for "following_lead" and "leading_vehicle"
                    if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):

                        path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)

                        if len(path) == 1:  # i.e. both on same lane
                            if (self.track_s_value_dict[track_id_B][t] > self.track_s_value_dict[track_id_A][t]) and (
                                (self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])
                                < max_distance_lead_veh_m
                            ):
                                # isn't this the wrong way around?
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

                        # second case: both on different, but following lanes:
                        if len(path) > 1 and all(
                            G_map.graph[u][v][0]["edge_type"] == "following" for u, v in zip(path[:-1], path[1:])
                        ):
                            path_length = (
                                sum([G_map.graph.nodes[node]["length"] for node in path[:-1]])
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
                                        G_map.graph.nodes[path[i]]["length"]
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
                        if (sum([G_map.graph[u][v][0]['edge_type']  == 'opposite' for u, v in zip(path[:-1], path[1:])]) == 1) and (sum([G_map.graph[u][v][0]['edge_type']  == 'following' for u, v in zip(path[:-1], path[1:])] )  == len(path) - 2):
                            # remove the opposite node, otherwise that stretch is counted twice. if there is something to check, than if this logic is correct.
                            # Taking -s from the other actor on the opposite direciton, as the s value should be in opposite direction as welll, hopefully..
                            path_length = sum([G_map.graph.nodes[path[i]]['length'] for i in range(len(path) - 1) if G_map.graph[path[i]][path[i + 1]][0]['edge_type'] != 'opposite'])  + G_map.graph.nodes[path[-1]]['length'] - self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
                            if path_length <  max_distance_opposite_m:
                                G_t.add_edge(track_id_B, track_id_A, edge_type='opposite_vehicle', path_length = path_length)
                                G_t.add_edge(track_id_A, track_id_B, edge_type='opposite_vehicle', path_length = path_length)
 

            timestep_graphs[t] = G_t

        return timestep_graphs

    def visualize_actor_graph(self, t_idx, comp_idx, use_map_pos = True, node_size = 1600, save_path=None, graph_or_component = 'graph'):

        if graph_or_component == 'graph':
            G = self.actor_graphs[t_idx]
        elif graph_or_component == 'component':
            G = self.actor_components[t_idx][comp_idx]

        #G = self.actor_graphs[timestep]
        # not sure, if G is anyhow needed? Why not always use self.actor_graphs[timestep]?
        if use_map_pos:
            pos = {node: (G.nodes[node]["xyz"].x, G.nodes[node]["xyz"].y) for node in G.nodes}
        else:
            pos = nx.spring_layout(G, scale=1.0, k=0.1)
        # node_size = 1600
        # Why remove lonely actors?
        # -> There are many of them on argopverse data.
        labels = {node: node for node in G.nodes() if G.degree(node) > 0}
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]

        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_edges, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black")

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "following_lead"]
        # edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'leading_vehicle']
        # I think with the distance based approach, we don't need to distinguish between direct and general neighbor?
        # TODO; yes
        edge_type_direct_neighbor_vehicle = [
            (u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "direct_neighbor_vehicle"
        ]
        edge_type_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "neighbor_vehicle"]
        edge_type_opposite_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "opposite_vehicle"]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_following_lead,
            width=2,
            edge_color="blue",
            arrows=True,
            node_size=node_size,
            label="following_lead",
        )
        # nx.draw_networkx_edges(G, pos, edgelist=edge_type_leading_vehicle, width=2, edge_color='cyan', label='leading_vehicle')
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_direct_neighbor_vehicle,
            width=2,
            edge_color="springgreen",
            arrows=True,
            node_size=node_size,
            label="direct_neighbor_vehicle",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_neighbor_vehicle,
            width=2,
            edge_color="forestgreen",
            arrows=True,
            node_size=node_size,
            label="neighbor_vehicle",
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_type_opposite_vehicle,
            width=2,
            edge_color="orange",
            arrows=True,
            node_size=node_size,
            label="opposite_vehicle",
        )

        plt.legend()
        if save_path:
            plt.savefig(save_path)
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
        left_boundary = self.G_map.graph.nodes[int(lane_id)]["left_boundary"]
        right_boundary = self.G_map.graph.nodes[int(lane_id)]["right_boundary"]

        # Get first and last points of each boundary
        left_start = left_boundary.waypoints[0]
        left_end = left_boundary.waypoints[-1]
        right_start = right_boundary.waypoints[0]
        right_end = right_boundary.waypoints[-1]

        # Calculate center line start and end points
        center_start = Point((left_start.x + right_start.x) / 2, (left_start.y + right_start.y) / 2)
        center_end = Point((left_end.x + right_end.x) / 2, (left_end.y + right_end.y) / 2)

        # Create center line from start to end point
        center_line = LineString([(center_start.x, center_start.y), (center_end.x, center_end.y)])

        # Project actor position onto center line
        actor_pos_2d = Point(position.x, position.y)
        projected_point = center_line.interpolate(center_line.project(actor_pos_2d))

        # Create the plot
        plt.figure(figsize=(10, 10))

        # Plot left boundary with arrows
        left_x = [p.x for p in left_boundary.waypoints]
        left_y = [p.y for p in left_boundary.waypoints]
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

        # Plot right boundary with arrows
        right_x = [p.x for p in right_boundary.waypoints]
        right_y = [p.y for p in right_boundary.waypoints]
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
