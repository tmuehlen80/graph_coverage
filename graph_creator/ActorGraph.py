from shapely.geometry import Point, LineString
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class TrackData(BaseModel):
    track_lane_dict: Dict[str, List[str]] = Field(description="Dictionary mapping track IDs to lists of lane IDs")
    track_s_value_dict: Dict[str, List[float]] = Field(description="Dictionary mapping track IDs to lists of s-values")
    track_xyz_pos_dict: Dict[str, List[Point]] = Field(description="Dictionary mapping track IDs to lists of 3D points")
    track_speed_lon_dict: Dict[str, List[float]] = Field(
        description="Dictionary mapping track IDs to lists of longitudinal speeds"
    )

    @field_validator(
        "track_lane_dict",
        "track_s_value_dict",
        "track_xyz_pos_dict",
        "track_speed_lon_dict",
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

        # Get all unique actors across all dictionaries
        all_actors = set(lane_dict.keys()) | set(s_value_dict.keys()) | set(xyz_dict.keys()) | set(speed_dict.keys())

        # Check that all dictionaries have the same actors
        for actor in all_actors:
            if not all(actor in d for d in [lane_dict, s_value_dict, xyz_dict, speed_dict]):
                raise ValueError(f"Actor {actor} is missing in one or more dictionaries")

        # Check that all lists have the same length for each actor
        for actor in all_actors:
            lengths = [
                len(lane_dict[actor]),
                len(s_value_dict[actor]),
                len(xyz_dict[actor]),
                len(speed_dict[actor]),
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
                lane_ids.append(lane_id)
            else:
                lane_ids.append(None)
        if not len(lane_ids) == self.num_timesteps:
            raise ValueError(f"There are too many lane IDs for track {track.track_id}")

        return lane_ids

    def _calculate_s_t_coordinates(self, position: Point, lane_id: str) -> Tuple[float, float]:
        """
        Calculate s and t coordinates for a given position and lane.
        
        Args:
            position: The actor's position as a Point
            lane_id: The lane ID as a string
            
        Returns:
            Tuple[float, float]: The s and t coordinates
        """
        # Get lane boundaries from graph
        left_boundary = self.G_map.graph.nodes[int(lane_id)]['left_boundary']
        right_boundary = self.G_map.graph.nodes[int(lane_id)]['right_boundary']
        
        # Calculate center line by averaging left and right boundaries
        # Ignore z-coordinate by using only x,y coordinates
        center_line = LineString([
            ((l.x + r.x)/2, (l.y + r.y)/2) 
            for l, r in zip(left_boundary.waypoints, right_boundary.waypoints)
        ])
        
        # Project actor position onto center line
        # Convert actor position to 2D point (ignore z)
        actor_pos_2d = Point(position.x, position.y)
        
        # Get the projected point on the center line
        projected_point = center_line.interpolate(center_line.project(actor_pos_2d))
        
        # Calculate s-coordinate (distance from start of center line to projected point)
        s_coord = center_line.project(actor_pos_2d)
        
        # Calculate t-coordinate (perpendicular distance from actor to center line)
        t_coord = actor_pos_2d.distance(projected_point)
        
        return s_coord, t_coord

    def _create_track_data_argoverse(self, scenario):
        track_lane_dict = {}
        track_s_value_dict = {}
        track_xyz_pos_dict = {}
        track_speed_lon_dict = {}
        
        for track in scenario.tracks:
            track_id = str(track.track_id)  # Convert to string
            lane_ids = self.find_lane_ids_for_track(track)
            # Convert None values to string 'None' and ensure all lane IDs are strings
            track_lane_dict[track_id] = [str(lane_id) if lane_id is not None else 'None' for lane_id in lane_ids]
            
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
                    if track_lane_dict[track_id][ii] != 'None': # there are cases where the lane is None, i.e. the actor is not on a lane.
                        s_coord, _ = self._calculate_s_t_coordinates(position, track_lane_dict[track_id][ii])
                        s_values.append(s_coord)
                    else:
                        s_values.append(np.nan)
                    # Calculate longitudinal speed from velocity
                    speed = np.sqrt(state.velocity[0]**2 + state.velocity[1]**2)
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
            track_speed_lon_dict=track_speed_lon_dict
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

        instance.actor_graphs = instance.create_actor_graphs(
            G_Map,
            max_distance_lead_veh_m=max_distance_lead_veh_m,
            max_distance_neighbor_forward_m=max_distance_neighbor_forward_m,
            max_distance_neighbor_backward_m=max_distance_neighbor_backward_m,
            max_distance_opposite_m=max_distance_opposite_veh_m,
        )

        return instance

    def _create_track_data_carla(self, scenario):
        """For carla, scenario is a pd df containing the time indexed actor data."""
        track_lane_dict = {}
        track_s_value_dict = {}
        track_xyz_pos_dict = {}
        track_speed_lon_dict = {}
        actors = scenario.actor_id.unique().tolist()

        # First pass: collect all data
        for actor in actors:
            actor_id = str(actor)  # Convert to string
            mask = scenario.actor_id == actor
            # Convert lane IDs to strings
            track_lane_dict[actor_id] = [str(lane_id) for lane_id in scenario[mask].road_lane_id.tolist()]
            track_s_value_dict[actor_id] = scenario[mask].distance_from_lane_start.tolist()
            # Convert xyz coordinates to Shapely Points
            xyz_coords = scenario[mask].actor_location_xyz.tolist()
            track_xyz_pos_dict[actor_id] = [Point(x, y, z) for x, y, z in xyz_coords]
            track_speed_lon_dict[actor_id] = scenario[mask].actor_speed_lon.tolist()

        # Create and return the Pydantic model
        return TrackData(
            track_lane_dict=track_lane_dict,
            track_s_value_dict=track_s_value_dict,
            track_xyz_pos_dict=track_xyz_pos_dict,
            track_speed_lon_dict=track_speed_lon_dict,
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

        timestep_graphs = []

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
                    )
                    # Do we need to add for information about the track here?

            # Add edges based on the conditions
            for track_id_A, lane_ids_A in self.track_lane_dict.items():
                if lane_ids_A[t] is None:
                    continue
                for track_id_B, lane_ids_B in self.track_lane_dict.items():
                    if lane_ids_B[t] is None:
                        continue
                    # Skip if we've already processed this pair
                    if str(track_id_A) == str(track_id_B):
                        continue
                    if (track_id_B, track_id_A) in G_t.edges():
                        continue

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

                        # still need to add opposite direction.

            timestep_graphs.append(G_t)

        return timestep_graphs

    def visualize_actor_graph(self, timestep, use_map_pos=True, node_size=1600, save_path=None):
        G = self.actor_graphs[timestep]
        # not sure, if G is anyhow needed? Why not always use self.actor_graphs[timestep]?
        if use_map_pos:
            pos = {node: self.actor_graphs[timestep].nodes[node]["xyz"][:2] for node in self.actor_graphs[0].nodes}
        else:
            pos = nx.spring_layout(G, scale=1.0, k=0.1)
        # node_size = 1600
        # Why remove lonely actors?
        labels = {node: node for node in self.actor_graphs[timestep].nodes() if G.degree(node) > 0}
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]

        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_edges, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black")

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "following_lead"]
        # edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'leading_vehicle']
        # I think with the distance based approach, we don't need to distinguish between direct and general neighbor?
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
