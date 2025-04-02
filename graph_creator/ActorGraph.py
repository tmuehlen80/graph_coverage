from shapely.geometry import Point
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

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
            lane_polygon = data.get('lane_polygon')
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
    
    
    def _create_track_lane_dict_argoverse(self, scenario):
        track_lane_dict = {}
        for track in scenario.tracks:
            track_lane_dict[track.track_id] = self.find_lane_ids_for_track(track)
    
        return track_lane_dict
    
    
    @classmethod
    def from_argoverse_scenario(cls, scenario, G_Map, follow_vehicle_steps=3):
        instance = cls()
        instance.G_map = G_Map
        instance.num_timesteps = len(scenario.timestamps_ns)
        instance.follow_vehicle_steps = follow_vehicle_steps
        instance.track_lane_dict = instance._create_track_lane_dict_argoverse(scenario)
        instance.actor_graphs = instance.create_actor_graphs(G_Map)

        return instance

    def _create_track_lane_dict_carla(self, scenario):
        track_lane_dict = {}
        actors = scenario.actor_id.unique().tolist()
        for actor in actors:
            mask = scenario.actor_id == actor
            track_lane_dict[actor] = scenario[mask].road_lane_id.tolist()

        return track_lane_dict


    @classmethod
    def from_carla_scenario(cls, scenario, G_Map, follow_vehicle_steps=3):
        """
        Args:
            scenario: A pd dataframe with the following columns: 'track_id', 'timestep', 'x', 'y'
            G_Map: A GraphMap object
            follow_vehicle_steps: The maximum number of timesteps a vehicle can follow another vehicle
        """
        instance = cls()
        instance.G_map = G_Map
        instance.num_timesteps = scenario.timestamp.nunique()
        instance.follow_vehicle_steps = follow_vehicle_steps
        instance.track_lane_dict = instance._create_track_lane_dict_carla(scenario)
        instance.actor_graphs = instance.create_actor_graphs(G_Map)

        return instance

    def create_actor_graphs(self, G_map):
        timestep_graphs = []

        # das ist aber irgendwie eine ziemlich interessante Art die Anzahl an timestamps herauszufinden...
        #for t in tqdm(range(len(next(iter(self.track_lane_dict.values()))))):
        for t in tqdm(range(10)):
            G_t = nx.MultiDiGraph()

            # Add nodes with attributes
            for track_id, lane_ids in self.track_lane_dict.items():
                if lane_ids[t] is not None:
                    G_t.add_node(track_id, lane_id=lane_ids[t])
                    # Do we need to add for information about the track here? 

            # Add edges based on the conditions
            for track_id_A, lane_ids_A in self.track_lane_dict.items():
                if lane_ids_A[t] is None:
                    continue
                for track_id_B, lane_ids_B in self.track_lane_dict.items():
                    if track_id_A == track_id_B or lane_ids_B[t] is None:
                        continue

                    # Check for "following_lead" and "leading_vehicle"
                    if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):
                        path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)
                        if len(path) - 1 <= self.follow_vehicle_steps and all(G_map.graph[u][v][0]['edge_type'] == 'following' for u, v in zip(path[:-1], path[1:])):
                            G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle')
                            G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead')
                            #if (t==20 and track_id_A == '73020' and track_id_B == 'AV'):
                            #    print("wrong")

                    # Check for "direct_neighbor_vehicle"
                    if G_map.graph.has_edge(lane_ids_A[t], lane_ids_B[t]) and G_map.graph[lane_ids_A[t]][lane_ids_B[t]][0]['edge_type'] == 'neighbor':
                        G_t.add_edge(track_id_A, track_id_B, edge_type='direct_neighbor_vehicle')

                    # Check for "neighbor_vehicle"
                    if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):
                        path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)
                        if len(path) - 1 <= self.follow_vehicle_steps and any(G_map.graph[u][v][0]['edge_type'] == 'neighbor' for u, v in zip(path[:-1], path[1:])) and not G_t.has_edge(track_id_A, track_id_B):
                            G_t.add_edge(track_id_A, track_id_B, edge_type='neighbor_vehicle')
                            G_t.add_edge(track_id_B, track_id_A, edge_type='neighbor_vehicle') # works in both direction - maybe another type here? 

                    # Check for "opposite_vehicle"
                    for u, v, data in G_map.graph.edges(data=True):
                        if data['edge_type'] == 'opposite':
                            if nx.has_path(G_map.graph, lane_ids_A[t], u) and nx.has_path(G_map.graph, lane_ids_B[t], v):
                                path_A = nx.shortest_path(G_map.graph, lane_ids_A[t], u, weight=None)
                                path_B = nx.shortest_path(G_map.graph, lane_ids_B[t], v, weight=None)
                                # This ensures that the both paths combined are limited in length. 
                                # Next, we are only looking for vehicle on a direct opposite lane, so no neighbors are allowed. If we allow neighbors, we have to careful to ignore double neighbors, that lead to the same lane again. 
                                if (len(path_A) - 1 + len(path_B) - 1 <= self.follow_vehicle_steps and
                                    all(G_map.graph[u][v][0]['edge_type'] != 'neighbor' for u, v in zip(path_A[:-1], path_A[1:])) and
                                    all(G_map.graph[u][v][0]['edge_type'] != 'neighbor' for u, v in zip(path_B[:-1], path_B[1:])) and
                                    sum(1 for u, v in zip(path_A[:-1], path_A[1:]) if G_map.graph[u][v][0]['edge_type'] == 'opposite') == 1 and
                                    sum(1 for u, v in zip(path_B[:-1], path_B[1:]) if G_map.graph[u][v][0]['edge_type'] == 'opposite') == 1):
                                    G_t.add_edge(track_id_A, track_id_B, edge_type='opposite_vehicle')
                                    G_t.add_edge(track_id_B, track_id_A, edge_type='opposite_vehicle')

            timestep_graphs.append(G_t)

        return timestep_graphs


    def _create_track_lane_dict_carla_w_details(self, scenario):
        """For carla, scenario is a pd df containing the time indexed actor data."""
        track_lane_dict = {}
        track_s_value_dict = {}
        track_xyz_pos_dict = {}
        track_speed_lon_dict = {}
        actors = scenario.actor_id.unique().tolist()
        for actor in actors:
            mask = scenario.actor_id == actor
            track_lane_dict[actor] = scenario[mask].road_lane_id.tolist() # potentially do a .reset_index(drop=True)?
            track_s_value_dict[actor] = scenario[mask].distance_from_lane_start.tolist() # potentially do a .reset_index(drop=True)?
            track_xyz_pos_dict[actor] = scenario[mask].actor_location_xyz.tolist()
            track_speed_lon_dict[actor] = scenario[mask].actor_speed_lon.tolist()

        return track_lane_dict, track_s_value_dict, track_xyz_pos_dict, track_speed_lon_dict


    @classmethod
    def from_carla_scenario_w_details(cls, scenario, G_Map, max_distance_m = 100):
        """
        Args:
            scenario: A pd dataframe with the following columns: 'track_id', 'timestep', 'x', 'y'
            G_Map: A GraphMap object
            follow_vehicle_steps: The maximum number of timesteps a vehicle can follow another vehicle
        """
        instance = cls()
        instance.G_map = G_Map
        instance.num_timesteps = scenario.timestamp.nunique()
        instance.timestamps = scenario.timestamp.unique().tolist()
        instance.max_distance_lead_veh_m = max_distance_m
        instance.max_distance_opposite_veh_m = max_distance_m
        instance.max_distance_neighbor_forward_m = max_distance_m / 2
        instance.max_distance_neighbor_backward_m = max_distance_m / 2

        instance.track_lane_dict, instance.track_s_value_dict, instance.track_xyz_pos_dict, instance.track_speed_lon_dict = instance._create_track_lane_dict_carla_w_details(scenario)

        instance.actor_graphs = instance.create_actor_graphs_w_details_2(G_Map, max_distance_lead_veh_m = 100, max_distance_neighbor_forward_m = 20, max_distance_neighbor_backward_m = 20, max_distance_opposite_m = 100)
        instance.actor_components = {}

        for key, value in instance.actor_graphs.items():
            components = list(nx.weakly_connected_components(value))
            subgraphs = [g.subgraph(c).copy() for c in components]
            instance.actor_components[key] = subgraphs

        return instance


    # def create_actor_graphs_w_details(self, G_map, max_distance_lead_veh_m, max_distance_neighbor_forward_m, max_distance_neighbor_backward_m, max_distance_opposite_m):
    #     timestep_graphs = []

    #     # das ist aber irgendwie eine ziemlich interessante Art die Anzahl an timestamps herauszufinden...
        
    #     #for t in tqdm(range(len(self.timestamps))):
    #     for t in tqdm([0, 100, 200, 300, 400]):
    #         G_t = nx.MultiDiGraph()

    #         # Add nodes with attributes
    #         for track_id, lane_ids in self.track_lane_dict.items():
    #             if lane_ids[t] is not None:
    #                 G_t.add_node(track_id, lane_id=lane_ids[t], 
    #                              s = self.track_s_value_dict[track_id][t], 
    #                              xyz = self.track_xyz_pos_dict[track_id][t], 
    #                              lon_speed = self.track_speed_lon_dict[track_id][t])
    #                 # Do we need to add for information about the track here? 


    #         for track_id_A, lane_ids_A in self.track_lane_dict.items():

    #             if lane_ids_A[t] is None:
    #                 continue

    #             for track_id_B, lane_ids_B in self.track_lane_dict.items():
    #                 if track_id_A == track_id_B or lane_ids_B[t] is None:
    #                     continue

    #                 # Check for "following_lead" and "leading_vehicle"
    #                 if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):

    #                     path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)

    #                     if len(path) == 1: # i.e. both on same lane
    #                         if (self.track_s_value_dict[track_id_B][t] > self.track_s_value_dict[track_id_A][t]) and ((self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]) < max_distance_lead_veh_m):
    #                             # isn't this the wrong way around?
    #                             G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle', path_length = self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])
    #                             G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead', path_length = self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])

    #                     # second case: both on different, but following lanes:
    #                     if len(path) > 1 and all(G_map.graph[u][v][0]['edge_type'] == 'following' for u, v in zip(path[:-1], path[1:])):
    #                         path_length = sum([G_map.graph.nodes[node]['length'] for node in path[:-1]]) + self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
    #                         #print(path_length)
    #                         if (path_length < max_distance_lead_veh_m) :
    #                             G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle', path_length = path_length)
    #                             G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead', path_length = path_length)

    #                     # third case: on neighboring lanes, forward
    #                     if (sum([G_map.graph[u][v][0]['edge_type']  == 'neighbor' for u, v in zip(path[:-1], path[1:])]) == 1) and (sum([G_map.graph[u][v][0]['edge_type']  == 'following' for u, v in zip(path[:-1], path[1:])] )  == len(path) - 2):
    #                         # remove the neighbor node, otherwise that stretch is counted twice.
    #                         path_length = sum([G_map.graph.nodes[path[i]]['length'] for i in range(len(path) - 1) if G_map.graph[path[i]][path[i + 1]][0]['edge_type'] != 'neighbor']) + self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
    #                         if path_length <  max_distance_neighbor_forward_m:
    #                             G_t.add_edge(track_id_B, track_id_A, edge_type='neighbor_vehicle', path_length = path_length)
    #                             G_t.add_edge(track_id_A, track_id_B, edge_type='neighbor_vehicle', path_length = path_length)

    #                     # fourth case: on opposite, directly next lane:
    #                     # if (sum([G_map.graph[u][v][0]['edge_type']  == 'neighbor' for u, v in zip(path[:-1], path[1:])]) == 1) and (sum([G_map.graph[u][v][0]['edge_type']  == 'following' for u, v in zip(path[:-1], path[1:])] )  == len(path) - 2):
    #                     #     # remove the neighbor node, otherwise that stretch is counted twice.
    #                     #     path_length = sum([G_map.graph.nodes[path[i]]['length'] for i in range(len(path) - 1) if G_map.graph[path[i]][path[i + 1]][0]['edge_type'] != 'neighbor']) + self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
    #                     #     if path_length <  max_distance_neighbor_forward_m:
    #                     #         G_t.add_edge(track_id_B, track_id_A, edge_type='neighbor_vehicle', path_length = path_length)
    #                     #         G_t.add_edge(track_id_A, track_id_B, edge_type='neighbor_vehicle', path_length = path_length)

    #                     # still need to add opposite direction.

    #         timestep_graphs.append(G_t)

    #     return timestep_graphs



    def create_actor_graphs_w_details(self, G_map, max_distance_lead_veh_m, max_distance_neighbor_forward_m, max_distance_neighbor_backward_m, max_distance_opposite_m):
        timestep_graphs = {}

        # das ist aber irgendwie eine ziemlich interessante Art die Anzahl an timestamps herauszufinden...
        
        #for t in tqdm(range(len(self.timestamps))):
        for t in tqdm([0, 100, 200, 300, 400]):
            G_t = nx.MultiDiGraph()

            # Add nodes with attributes
            for track_id, lane_ids in self.track_lane_dict.items():
                if lane_ids[t] is not None:
                    G_t.add_node(track_id, lane_id=lane_ids[t], 
                                 s = self.track_s_value_dict[track_id][t], 
                                 xyz = self.track_xyz_pos_dict[track_id][t], 
                                 lon_speed = self.track_speed_lon_dict[track_id][t])
                    # Do we need to add for information about the track here? 


            #for track_id_A, lane_ids_A in self.track_lane_dict.items():
            keys = list(self.track_lane_dict.keys())
            for i in range(len(keys) - 1):
                track_id_A = keys[i]
                lane_ids_A = self.track_lane_dict[keys[i]]

                if lane_ids_A[t] is None:
                    continue

                #for track_id_B, lane_ids_B in self.track_lane_dict.items():
                for j in range(i + 1, len(keys)):
                    track_id_B = keys[j]
                    lane_ids_B = self.track_lane_dict[keys[j]]
                    if track_id_A == track_id_B or lane_ids_B[t] is None:
                        continue

                    # Check for "following_lead" and "leading_vehicle"
                    if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):

                        path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)

                        if len(path) == 1: # i.e. both on same lane
                            if (self.track_s_value_dict[track_id_B][t] > self.track_s_value_dict[track_id_A][t]) and ((self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]) < max_distance_lead_veh_m):
                                # isn't this the wrong way around?
                                G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle', path_length = self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])
                                G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead', path_length = self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t])

                        # second case: both on different, but following lanes:
                        if len(path) > 1 and all(G_map.graph[u][v][0]['edge_type'] == 'following' for u, v in zip(path[:-1], path[1:])):
                            path_length = sum([G_map.graph.nodes[node]['length'] for node in path[:-1]]) + self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
                            #print(path_length)
                            if (path_length < max_distance_lead_veh_m) :
                                G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle', path_length = path_length)
                                G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead', path_length = path_length)

                        # third case: on neighboring lanes, forward
                        if (sum([G_map.graph[u][v][0]['edge_type']  == 'neighbor' for u, v in zip(path[:-1], path[1:])]) == 1) and (sum([G_map.graph[u][v][0]['edge_type']  == 'following' for u, v in zip(path[:-1], path[1:])] )  == len(path) - 2):
                            # remove the neighbor node, otherwise that stretch is counted twice.
                            path_length = sum([G_map.graph.nodes[path[i]]['length'] for i in range(len(path) - 1) if G_map.graph[path[i]][path[i + 1]][0]['edge_type'] != 'neighbor']) + self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
                            if path_length <  max_distance_neighbor_forward_m:
                                G_t.add_edge(track_id_B, track_id_A, edge_type='neighbor_vehicle', path_length = path_length)
                                G_t.add_edge(track_id_A, track_id_B, edge_type='neighbor_vehicle', path_length = path_length)

                        # fourth case: on opposite, directly next lane:
                        if (sum([G_map.graph[u][v][0]['edge_type']  == 'opposite' for u, v in zip(path[:-1], path[1:])]) == 1) and (sum([G_map.graph[u][v][0]['edge_type']  == 'following' for u, v in zip(path[:-1], path[1:])] )  == len(path) - 2):
                            # remove the opposite node, otherwise that stretch is counted twice. if there is something to check, than if this logic is correct.
                            # Taking -s from the other actor on the opposite direciton, as the s value should be in opposite direction as welll, hopefully..
                            path_length = sum([G_map.graph.nodes[path[i]]['length'] for i in range(len(path) - 1) if G_map.graph[path[i]][path[i + 1]][0]['edge_type'] != 'opposite'])  + G_map.graph.nodes[path[-1]]['length'] - self.track_s_value_dict[track_id_B][t] - self.track_s_value_dict[track_id_A][t]
                            if path_length <  max_distance_opposite_m:
                                G_t.add_edge(track_id_B, track_id_A, edge_type='opposite_vehicle', path_length = path_length)
                                G_t.add_edge(track_id_A, track_id_B, edge_type='opposite_vehicle', path_length = path_length)

                        # still need to add opposite direction.

            timestep_graphs[t] = G_t

        return timestep_graphs


    def visualize_actor_graph(self, t_idx, comp_idx, use_map_pos = True, node_size = 1600, save_path=None, graph_or_componentes = 'graph'):

        if graph_or_componentes == 'graph':
            G = self.actor_graphs[t_idx]
        elif graph_or_componentes == 'component':
            G = self.actor_components[t_idx][comp_idx]

        # not sure, if G is anyhow needed? Why not always use self.actor_graphs[timestep]?
        if use_map_pos:
            pos = {node: G.nodes[node]['xyz'][:2] for node in G.nodes}
        else:
            pos = nx.spring_layout(G, scale=1.0, k=0.1)
        #node_size = 1600
        # Why remove lonely actors?
        labels = {node: node for node in G.nodes() if G.degree(node) > 0}
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]

        plt.figure(figsize=(6, 6))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_edges, node_size=node_size)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'following_lead']
        #edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'leading_vehicle']
        # I think with the distance based approach, we don't need to distinguish between direct and general neighbor?
        edge_type_direct_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'direct_neighbor_vehicle']
        edge_type_neighbor_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'neighbor_vehicle']
        edge_type_opposite_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'opposite_vehicle']

        nx.draw_networkx_edges(G, pos, edgelist=edge_type_following_lead, width=2, edge_color='blue', arrows=True, node_size=node_size, label='following_lead')
        #nx.draw_networkx_edges(G, pos, edgelist=edge_type_leading_vehicle, width=2, edge_color='cyan', label='leading_vehicle')
        nx.draw_networkx_edges(G, pos, edgelist=edge_type_direct_neighbor_vehicle, width=2, edge_color='springgreen', arrows=True,  node_size=node_size,label='direct_neighbor_vehicle')
        nx.draw_networkx_edges(G, pos, edgelist=edge_type_neighbor_vehicle, width=2, edge_color='forestgreen', arrows=True, node_size=node_size, label='neighbor_vehicle')
        nx.draw_networkx_edges(G, pos, edgelist=edge_type_opposite_vehicle, width=2, edge_color='orange', arrows=True, node_size=node_size, label='opposite_vehicle')

        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
