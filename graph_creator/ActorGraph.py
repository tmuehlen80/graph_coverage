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

    def find_lane_ids_from_pos(self, position):
        """
        Find all lane IDs that contain the given position.
        An actor can be on multiple map elements, so we return a list of lane IDs.
        Most of the time, there will be just a single element.
        """
        point = Point(position[0], position[1])
        lane_ids = []
        for lane_id, data in self.G_map.graph.nodes(data=True):
            lane_polygon = data["node_info"].lane_polygon
            if lane_polygon.contains(point):
                lane_ids.append(lane_id)
        return lane_ids if lane_ids else [None]

    def find_lane_ids_for_track(self, track):
        """
        create a list of length of number of timesteps, where each element holds a list of lane_ids of an object at that timestep.
        An actor can be on multiple map elements, so each element is a list of lane IDs.
        Most of the time, there will be just a single element in each list.

        Missing information is represented by [None].
        """
        lane_ids = []
        timestep_list = [step.timestep for step in track.object_states]
        for ii in range(self.num_timesteps):
            if ii in timestep_list:
                position = track.object_states[timestep_list.index(ii)].position
                lane_id_list = self.find_lane_ids_from_pos(position)
                # Convert lane IDs to strings, handling the case where lane_id_list might contain None
                lane_ids.append([str(lane_id) if lane_id is not None else None for lane_id in lane_id_list])
            else:
                lane_ids.append([None])
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
                    lane_id_list = track_lane_dict[track_id][ii]
                    if (
                        lane_id_list is not None and lane_id_list != [None]
                    ):  # there are cases where the lane is None, i.e. the actor is not on a lane.
                        # For now, use the first lane ID in the list. In the future, we might want to handle multiple lanes differently.
                        primary_lane_id = lane_id_list[0] if lane_id_list[0] is not None else None
                        if primary_lane_id is not None:
                            s_coord, t_coord = self._calculate_s_t_coordinates(position, primary_lane_id)
                            s_values.append(s_coord)
                        else:
                            s_values.append(np.nan)
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
        max_distance_opposite_forward_m=100,
        max_distance_opposite_backward_m=100,
        max_distance_neighbor_forward_m=50,
        max_distance_neighbor_backward_m=50,
        max_node_distance_leading=3,
        max_node_distance_neighbor=3,
        max_node_distance_opposite=3,
        delta_timestep_s=1.0,
    ):
        """
        Create an ActorGraph instance from an Argoverse scenario.

        Args:
            scenario: An Argoverse scenario object
            G_Map: A GraphMap object
            max_distance_lead_veh_m: Maximum distance in meters for leading vehicle relationships
            max_distance_opposite_forward_m: Maximum distance in meters for forward opposite vehicle relationships
            max_distance_opposite_backward_m: Maximum distance in meters for backward opposite vehicle relationships
            max_distance_neighbor_forward_m: Maximum distance in meters for forward neighbor vehicle relationships
            max_distance_neighbor_backward_m: Maximum distance in meters for backward neighbor vehicle relationships
            max_node_distance_leading: Maximum number of nodes for leading/following path checking
            max_node_distance_neighbor: Maximum number of nodes for neighbor path checking
            max_node_distance_opposite: Maximum number of nodes for opposite path checking
            delta_timestep_s: Time step increment in seconds
        """
        instance = cls()
        instance.G_map = G_Map
        instance.timestamps = list((scenario.timestamps_ns  - min(scenario.timestamps_ns) ) * 10**-9)
        instance.num_timesteps = len(scenario.timestamps_ns)
        instance.max_distance_lead_veh_m = max_distance_lead_veh_m
        instance.max_distance_opposite_forward_m = max_distance_opposite_forward_m
        instance.max_distance_opposite_backward_m = max_distance_opposite_backward_m
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
            max_distance_opposite_forward_m=max_distance_opposite_forward_m,
            max_distance_opposite_backward_m=max_distance_opposite_backward_m,
            max_node_distance_leading=max_node_distance_leading,
            max_node_distance_neighbor=max_node_distance_neighbor,
            max_node_distance_opposite=max_node_distance_opposite,
            delta_timestep_s=delta_timestep_s,
        )

        return instance

    @classmethod
    def from_carla_scenario(
        cls,
        scenario,
        G_Map,
        max_distance_lead_veh_m=100,
        max_distance_opposite_forward_m=100,
        max_distance_opposite_backward_m=100,
        max_distance_neighbor_forward_m=50,
        max_distance_neighbor_backward_m=50,
        max_node_distance_leading=3,
        max_node_distance_neighbor=3,
        max_node_distance_opposite=3,
        delta_timestep_s=1.0,
    ):
        """
        Create an ActorGraph instance from a CARLA scenario.

        Args:
            scenario: A pd dataframe with the following columns: 'track_id', 'timestep', 'x', 'y'
            G_Map: A GraphMap object
            max_distance_lead_veh_m: Maximum distance in meters for leading vehicle relationships
            max_distance_opposite_forward_m: Maximum distance in meters for forward opposite vehicle relationships
            max_distance_opposite_backward_m: Maximum distance in meters for backward opposite vehicle relationships
            max_distance_neighbor_forward_m: Maximum distance in meters for forward neighbor vehicle relationships
            max_distance_neighbor_backward_m: Maximum distance in meters for backward neighbor vehicle relationships
            max_node_distance_leading: Maximum number of nodes for leading/following path checking
            max_node_distance_neighbor: Maximum number of nodes for neighbor path checking
            max_node_distance_opposite: Maximum number of nodes for opposite path checking
        """
        instance = cls()
        instance.G_map = G_Map
        instance.num_timesteps = scenario.timestamp.nunique()
        instance.timestamps = scenario.timestamp.unique().tolist()
        instance.max_distance_lead_veh_m = max_distance_lead_veh_m
        instance.max_distance_opposite_forward_m = max_distance_opposite_forward_m
        instance.max_distance_opposite_backward_m = max_distance_opposite_backward_m
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
            max_distance_opposite_forward_m=max_distance_opposite_forward_m,
            max_distance_opposite_backward_m=max_distance_opposite_backward_m,
            max_node_distance_leading=max_node_distance_leading,
            max_node_distance_neighbor=max_node_distance_neighbor,
            max_node_distance_opposite=max_node_distance_opposite,
            delta_timestep_s=delta_timestep_s,
        )
        instance.actor_components = {}
        # print("instance.actor_graphs.keys(): ", instance.actor_graphs.keys())
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
            # Convert lane IDs to list of lists format, because we can be placed on two lanes at the same time (@ Thomas Stimmt das für Carla?)
            # Each lane_id becomes a single-element list [lane_id] to match the new format
            lane_ids = scenario[mask].road_lane_id.tolist()
            track_lane_dict[actor_id] = [[lane_id] if lane_id is not None else [None] for lane_id in lane_ids]
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

    def _has_path_within_distance(self, G, source, target, max_distance):
        """
        Check if there's a path from source to target with at most max_distance nodes.
        
        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            max_distance: Maximum number of nodes in the path
            
        Returns:
            True if there's a path with at most max_distance nodes, False otherwise
        """
        if source == target:
            return True
        
        if max_distance <= 1:
            return G.has_edge(source, target)
        
        # Use BFS to find shortest path
        try:
            path = nx.shortest_path(G, source, target)
            return len(path) <= max_distance
        except nx.NetworkXNoPath:
            return False



    def _find_relation_between_actors(self, track_id_A, track_id_B, t, G_map, max_distance_lead_veh_m, 
                                     max_distance_neighbor_forward_m, max_distance_opposite_forward_m, max_distance_opposite_backward_m):
        """
        Find the best relation from actor A to actor B.
        
        Args:
            track_id_A: ID of actor A
            track_id_B: ID of actor B  
            t: timestep index
            G_map: Map graph
            max_distance_lead_veh_m: Maximum distance for leading vehicle relations
            max_distance_neighbor_forward_m: Maximum distance for forward neighbor relations
            max_distance_opposite_forward_m: Maximum distance for forward opposite vehicle relations
            max_distance_opposite_backward_m: Maximum distance for backward opposite vehicle relations
        
        Returns:
            Tuple of (relation_type, path_length) or None if no relation found
        """
        primary_lane_A = self.track_lane_dict[track_id_A][t][0] if self.track_lane_dict[track_id_A][t] else None
        primary_lane_B = self.track_lane_dict[track_id_B][t][0] if self.track_lane_dict[track_id_B][t] else None
        
        if primary_lane_A is None or primary_lane_B is None:
            return None

        # Note: We cannot check for existing paths here because the graph is being built incrementally


        # Check for "following_lead" and "leading_vehicle" in same lane
        if nx.has_path(G_map.graph, primary_lane_A, primary_lane_B):
            path = nx.shortest_path(G_map.graph, primary_lane_A, primary_lane_B, weight=None)

            if len(path) == 1:  # i.e. both on same lane
                if (self.track_s_value_dict[track_id_B][t] > self.track_s_value_dict[track_id_A][t]) and (
                    (self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])
                    < max_distance_lead_veh_m
                ):
                    path_length = self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
                    return ("following_lead", path_length)

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
                    return ("following_lead", path_length)

            # third case: on neighboring lanes, forward
            if (
                sum([G_map.graph[u][v][0]["edge_type"] == "neighbor" for u, v in zip(path[:-1], path[1:])])
                == 1
            ) and (
                sum([G_map.graph[u][v][0]["edge_type"] == "following" for u, v in zip(path[:-1], path[1:])])
                == len(path) - 2
            ):
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
                    return ("neighbor_vehicle", path_length)

            # fourth case: on opposite, directly next lane:
            if (
                sum([G_map.graph[u][v][0]["edge_type"] == "opposite" for u, v in zip(path[:-1], path[1:])])
                == 1
            ) and (
                sum([G_map.graph[u][v][0]["edge_type"] == "following" for u, v in zip(path[:-1], path[1:])])
                == len(path) - 2
            ):
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
                # Determine which distance limit to use based on path length direction
                if path_length >= 0:  # Forward direction
                    max_distance = max_distance_opposite_forward_m
                else:  # Backward direction (negative path_length)
                    max_distance = max_distance_opposite_backward_m
                
                if abs(path_length) < max_distance:
                    return ("opposite_vehicle", path_length)

        return None

    def _choose_better_relation(self, relation_A_to_B, relation_B_to_A):
        """
        Choose the better relation between two actors based on path length and hierarchy.
        
        Hierarchy: following > neighbor > opposite
        """
        if relation_A_to_B is None and relation_B_to_A is None:
            return None, None
        
        if relation_A_to_B is None:
            return relation_B_to_A, "B_to_A"
        if relation_B_to_A is None:
            return relation_A_to_B, "A_to_B"
        
        # Get hierarchy scores
        hierarchy_A = self._get_relation_hierarchy_score(relation_A_to_B[0])
        hierarchy_B = self._get_relation_hierarchy_score(relation_B_to_A[0])
        
        # If same hierarchy, choose shorter path
        if hierarchy_A == hierarchy_B:
            if relation_A_to_B[1] <= relation_B_to_A[1]:
                return relation_A_to_B, "A_to_B"
            else:
                return relation_B_to_A, "B_to_A"
        
        # If different hierarchy, choose higher hierarchy
        if hierarchy_A > hierarchy_B:
            return relation_A_to_B, "A_to_B"
        else:
            return relation_B_to_A, "B_to_A"

    def _get_relation_hierarchy_score(self, relation_type):
        """
        Get hierarchy score for relation types.
        Higher score = higher priority.
        """
        hierarchy = {
            "leading_vehicle": 3,      # Highest priority
            "following_lead": 3,       # Same as leading_vehicle
            "neighbor_vehicle": 2,     # Medium priority
            "opposite_vehicle": 1      # Lowest priority
        }
        return hierarchy.get(relation_type, 0)

    def _explore_relations(self, t, G_map, max_distance_lead_veh_m, max_distance_neighbor_forward_m, 
                          max_distance_opposite_forward_m, max_distance_opposite_backward_m):
        """
        Exploration phase: find all potential relations between actors at timestep t.
        
        Args:
            t: timestep index
            G_map: Map graph
            max_distance_lead_veh_m: Maximum distance for leading vehicle relations
            max_distance_neighbor_forward_m: Maximum distance for forward neighbor relations
            max_distance_opposite_forward_m: Maximum distance for forward opposite vehicle relations
            max_distance_opposite_backward_m: Maximum distance for backward opposite vehicle relations
            
        Returns:
            Dictionary mapping actor_id to relation types and target actors with path lengths
        """
        relations_dict = {}
        
        keys = list(self.track_lane_dict.keys())
        for i in range(len(keys) - 1):
            track_id_A = keys[i]
            lane_ids_A = self.track_lane_dict[keys[i]]
            lane_id_list_A = lane_ids_A[t]

            if lane_id_list_A is None or lane_id_list_A == [None]:
                continue

            for j in range(i + 1, len(keys)):
                track_id_B = keys[j]
                lane_ids_B = self.track_lane_dict[keys[j]]
                lane_id_list_B = lane_ids_B[t]

                if lane_id_list_B is None or lane_id_list_B == [None]:
                    continue

                # Find relations in both directions
                relation_A_to_B = self._find_relation_between_actors(
                    track_id_A, track_id_B, t, G_map, 
                    max_distance_lead_veh_m, max_distance_neighbor_forward_m, max_distance_opposite_forward_m, max_distance_opposite_backward_m
                )
                
                relation_B_to_A = self._find_relation_between_actors(
                    track_id_B, track_id_A, t, G_map, 
                    max_distance_lead_veh_m, max_distance_neighbor_forward_m, max_distance_opposite_forward_m, max_distance_opposite_backward_m
                )

                # Choose the better relation based on path length and hierarchy
                best_relation, direction = self._choose_better_relation(relation_A_to_B, relation_B_to_A)
                
                if best_relation is not None:
                    relation_type, path_length = best_relation
                    
                    # Add the relation to the appropriate actor based on direction
                    if direction == "A_to_B":
                        # A has the relation to B
                        if track_id_A not in relations_dict:
                            relations_dict[track_id_A] = {}
                        if relation_type not in relations_dict[track_id_A]:
                            relations_dict[track_id_A][relation_type] = []
                        relations_dict[track_id_A][relation_type].append((track_id_B, path_length))
                    else:  # direction == "B_to_A"
                        # B has the relation to A
                        if track_id_B not in relations_dict:
                            relations_dict[track_id_B] = {}
                        if relation_type not in relations_dict[track_id_B]:
                            relations_dict[track_id_B][relation_type] = []
                        relations_dict[track_id_B][relation_type].append((track_id_A, path_length))
        
        return relations_dict

    def _add_leading_following_edges(self, G_t, relations_dict, max_node_distance):
        """
        Add leading/following edges to the graph in hierarchical order.
        
        Args:
            G_t: The graph at timestep t
            relations_dict: Dictionary of discovered relations
            max_node_distance: Maximum node distance for path checking
        """
        # Collect leading/following relations
        leading_following_relations = []
        for actor_id, relation_types in relations_dict.items():
            for relation_type in ["leading_vehicle", "following_lead"]:
                if relation_type in relation_types:
                    for target_actor_id, path_length in relation_types[relation_type]:
                        leading_following_relations.append((actor_id, target_actor_id, relation_type, path_length))
        
        # Sort by path_length (shortest first) - leading/following distances are always positive (map is one-directional)
        leading_following_relations.sort(key=lambda x: x[3])
        
        # Add edges bidirectionally - even though exploration only found one relation, 
        # graph construction needs both directions for complete path finding
        for actor_id, target_actor_id, relation_type, path_length in leading_following_relations:
            if not self._has_path_within_distance(G_t, actor_id, target_actor_id, max_node_distance):
                G_t.add_edge(
                    actor_id,
                    target_actor_id,
                    edge_type=relation_type,
                    path_length=path_length,
                )
            
            if not self._has_path_within_distance(G_t, target_actor_id, actor_id, max_node_distance):
                G_t.add_edge(
                    target_actor_id,
                    actor_id,
                    edge_type=relation_type,
                    path_length=path_length,
                )

    def _add_neighbor_edges(self, G_t, relations_dict, max_node_distance):
        """
        Add neighbor edges to the graph in hierarchical order.
        
        Args:
            G_t: The graph at timestep t
            relations_dict: Dictionary of discovered relations
            max_node_distance: Maximum node distance for path checking
        """
        # Collect neighbor relations
        neighbor_relations = []
        for actor_id, relation_types in relations_dict.items():
            if "neighbor_vehicle" in relation_types:
                for target_actor_id, path_length in relation_types["neighbor_vehicle"]:
                    neighbor_relations.append((actor_id, target_actor_id, path_length))
        
        # Sort by absolute path_length (shortest first)
        neighbor_relations.sort(key=lambda x: abs(x[2]))
        
        # Add edges bidirectionally
        for actor_id, target_actor_id, path_length in neighbor_relations:
            if not self._has_path_within_distance(G_t, actor_id, target_actor_id, max_node_distance):
                G_t.add_edge(
                    actor_id,
                    target_actor_id,
                    edge_type="neighbor_vehicle",
                    path_length=path_length,
                )
            
            if not self._has_path_within_distance(G_t, target_actor_id, actor_id, max_node_distance):
                G_t.add_edge(
                    target_actor_id,
                    actor_id,
                    edge_type="neighbor_vehicle",
                    path_length=path_length,
                )

    def _add_opposite_edges(self, G_t, relations_dict, max_node_distance, max_distance_opposite_forward_m, max_distance_opposite_backward_m):
        """
        Add opposite edges to the graph in hierarchical order.
        
        Args:
            G_t: The graph at timestep t
            relations_dict: Dictionary of discovered relations
            max_node_distance: Maximum node distance for path checking
            max_distance_opposite_forward_m: Maximum distance for forward opposite vehicle relations
            max_distance_opposite_backward_m: Maximum distance for backward opposite vehicle relations
        """
        # Collect opposite relations
        opposite_relations = []
        for actor_id, relation_types in relations_dict.items():
            if "opposite_vehicle" in relation_types:
                for target_actor_id, path_length in relation_types["opposite_vehicle"]:
                    opposite_relations.append((actor_id, target_actor_id, path_length))
        
        # Sort by absolute path_length (shortest first)
        opposite_relations.sort(key=lambda x: abs(x[2]))
        
        # Add edges bidirectionally
        for actor_id, target_actor_id, path_length in opposite_relations:
            # Check distance limits based on direction
            if path_length >= 0:
                max_distance = max_distance_opposite_forward_m
            else:
                max_distance = max_distance_opposite_backward_m
            
            # Check if there's already a path in the current graph that's shorter than max_node_distance
            has_short_path_A_to_B = self._has_path_within_distance(G_t, actor_id, target_actor_id, max_node_distance)
            has_short_path_B_to_A = self._has_path_within_distance(G_t, target_actor_id, actor_id, max_node_distance)
            
            # For opposite relations, check both directions together
            # If either direction has a path, don't add either direction
            should_add_A_to_B = (not has_short_path_A_to_B and abs(path_length) <= max_distance)
            should_add_B_to_A = (not has_short_path_B_to_A and abs(path_length) <= max_distance)
            
            if should_add_A_to_B:
                G_t.add_edge(
                    actor_id,
                    target_actor_id,
                    edge_type="opposite_vehicle",
                    path_length=path_length,
                )
            
            if should_add_B_to_A:
                G_t.add_edge(
                    target_actor_id,
                    actor_id,
                    edge_type="opposite_vehicle",
                    path_length=path_length,
                )

    def _construct_graph(self, G_t, relations_dict, max_node_distance_leading, max_node_distance_neighbor, max_node_distance_opposite, max_distance_opposite_forward_m, max_distance_opposite_backward_m):
        """
        Graph construction phase: add edges to the graph based on discovered relations.
        
        Args:
            G_t: The graph at timestep t
            relations_dict: Dictionary of discovered relations
            max_node_distance_leading: Maximum node distance for leading/following relations
            max_node_distance_neighbor: Maximum node distance for neighbor relations
            max_node_distance_opposite: Maximum node distance for opposite relations
            max_distance_opposite_forward_m: Maximum distance for forward opposite vehicle relations
            max_distance_opposite_backward_m: Maximum distance for backward opposite vehicle relations
        """
        # Hierarchical graph construction: add edges step by step
        # Step 1: Add leading/following relations (shortest first)
        self._add_leading_following_edges(G_t, relations_dict, max_node_distance_leading)
        
        # Step 2: Add neighbor relations (shortest first)
        self._add_neighbor_edges(G_t, relations_dict, max_node_distance_neighbor)
        
        # Step 3: Add opposite relations (shortest first)
        self._add_opposite_edges(G_t, relations_dict, max_node_distance_opposite, max_distance_opposite_forward_m, max_distance_opposite_backward_m)

    def create_actor_graphs(
        self,
        G_map,
        max_distance_lead_veh_m,
        max_distance_neighbor_forward_m,
        max_distance_neighbor_backward_m,
        max_distance_opposite_forward_m,
        max_distance_opposite_backward_m,
        max_node_distance_leading=3,
        max_node_distance_neighbor=3,
        max_node_distance_opposite=3,
        delta_timestep_s=1.0,
    ):
        graph_timesteps = []
        graph_timesteps_idx = []
        assert all(a <= b for a, b in zip(self.timestamps, self.timestamps[1:])), "graph timestamps are not sorted"

        current_timestep = self.timestamps[0]
        # hmm, why is the following necessary?
        while True:
            # Find closest timestep in self.timestamps
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
                lane_id_list = lane_ids[t]
                if lane_id_list is not None and lane_id_list != [None]:
                    #TODO: For now, use the first lane ID in the list as the primary lane
                    primary_lane_id = lane_id_list[0] if lane_id_list[0] is not None else None
                    if primary_lane_id is not None:
                        G_t.add_node(
                            track_id,
                            lane_id=primary_lane_id,
                            lane_ids=lane_id_list,  # Store all lane IDs for future use
                            s=self.track_s_value_dict[track_id][t],
                            xyz=self.track_xyz_pos_dict[track_id][t],
                            lon_speed=self.track_speed_lon_dict[track_id][t],
                            actor_type=self.track_actor_type_dict[track_id],
                        )

            # Exploration phase: discover all potential relations
            relations_dict = self._explore_relations(
                t, G_map, max_distance_lead_veh_m, max_distance_neighbor_forward_m, 
                max_distance_opposite_forward_m, max_distance_opposite_backward_m
            )

            # Graph construction phase: add edges based on discovered relations
            self._construct_graph(
                G_t, relations_dict, max_node_distance_leading, max_node_distance_neighbor, max_node_distance_opposite,
                max_distance_opposite_forward_m, max_distance_opposite_backward_m
            )

            timestep_graphs[self.timestamps[t]] = G_t

        self.actor_graphs = timestep_graphs

        # Add lane change attribute to nodes
        ag_timestamps = list(self.actor_graphs.keys())
        ag_timestamps = sorted(ag_timestamps)

        for i in range(1, len(ag_timestamps)):
            all_nodes = list(self.actor_graphs[ag_timestamps[i]].nodes)
            # node = all_nodes
            for node in all_nodes:
                lane_id =  self.actor_graphs[ag_timestamps[i]].nodes(data=True)[node]["lane_id"]
                start_points = [u for u, v, d in G_map.graph.in_edges(lane_id, data=True) if d.get('edge_type') == 'following']
                start_points.append(lane_id)
                # ToDo: Check if node exists in previous timestep
                if self.actor_graphs[ag_timestamps[i - 1]].has_node(node):
                    previous_lane_id =  self.actor_graphs[ag_timestamps[i - 1]].nodes(data=True)[node]["lane_id"]
                    if previous_lane_id in start_points:
                        lane_change = False
                    else:
                        lane_change = True
                    #if lane_change:
                    #     # print(node, previous_lane_id, lane_id, start_points, lane_change)
                # In principle, here there could also be added a check for lane merge, i.e. counting if the start points had more then 1 element (before adding the current lane_id)
                else:
                    lane_change = False
                self.actor_graphs[ag_timestamps[i]].nodes(data=True)[node]["lane_change"] = lane_change

        return self.actor_graphs



    def visualize_actor_graph(
        self, 
        t_idx, 
        comp_idx, 
        use_map_pos=True, 
        node_size=1600, 
        save_path=None, 
        graph_or_component="graph", 
        scenario_id=None,
        scale_plot=True
    ):
        if graph_or_component == "graph":
            G = self.actor_graphs[t_idx]
        elif graph_or_component == "component":
            G = self.actor_components[t_idx][comp_idx]

        # Calculate number of actors and scale figure size accordingly
        num_actors = len(G.nodes())
        base_size = 10  # Base size for small graphs
        if scale_plot:
            scale_factor = max(1, num_actors / 5)  # Scale up for more than 5 actors
        else:
            scale_factor = 1
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
