from shapely.geometry import Point
import networkx as nx
import matplotlib.pyplot as plt

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
        lane_ids = []
        for ii, object_state in enumerate(track.object_states):
            if ii == object_state.timestep:
                position = object_state.position
                lane_id = self.find_lane_id_from_pos(position)
                lane_ids.append(lane_id)
            else:
                lane_ids.append(None)
        
        if len(lane_ids) > self.num_timesteps:
            raise ValueError(f"There are too many lane IDs for track {track.track_id}")
        elif len(lane_ids) < self.num_timesteps:
            lane_ids.extend([None] * (self.num_timesteps - len(lane_ids)))
            
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

    def create_actor_graphs(self, G_map):
        timestep_graphs = []

        for t in range(len(next(iter(self.track_lane_dict.values())))):
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
                            G_t.add_edge(track_id_A, track_id_B, edge_type='following_lead')
                            G_t.add_edge(track_id_B, track_id_A, edge_type='leading_vehicle')

                    # Check for "direct_neighbor_vehicle"
                    if G_map.graph.has_edge(lane_ids_A[t], lane_ids_B[t]) and G_map.graph[lane_ids_A[t]][lane_ids_B[t]][0]['edge_type'] == 'neighbor':
                        G_t.add_edge(track_id_A, track_id_B, edge_type='direct_neighbor_vehicle')

                    # Check for "neighbor_vehicle"
                    if nx.has_path(G_map.graph, lane_ids_A[t], lane_ids_B[t]):
                        path = nx.shortest_path(G_map.graph, lane_ids_A[t], lane_ids_B[t], weight=None)
                        if len(path) - 1 <= self.follow_vehicle_steps and any(G_map.graph[u][v][0]['edge_type'] == 'neighbor' for u, v in zip(path[:-1], path[1:])) and not G_t.has_edge(track_id_A, track_id_B):
                            G_t.add_edge(track_id_A, track_id_B, edge_type='neighbor_vehicle')
                            G_t.add_edge(track_id_B, track_id_A, edge_type='neighbor_vehicle') # works in both direction - maybe another type here? 

            timestep_graphs.append(G_t)

        return timestep_graphs

    def visualize_actor_graph(self,timestep, save_path=None):
        G = self.actor_graphs[timestep]
        pos = nx.spring_layout(G, scale=1.0, k=0.1)
        labels = {node: node for node in self.actor_graphs[timestep].nodes() if G.degree(node) > 0}
        nodes_with_edges = [node for node in G.nodes() if G.degree(node) > 0]

        plt.figure(figsize=(12, 12))
        nx.draw(G, pos, nodelist=nodes_with_edges, labels=labels, with_labels=True, node_size=200, font_size=10, font_color='black')

        # Draw edges with different styles based on edge type
        edge_type_following_lead = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'following_lead']
        #edge_type_leading_vehicle = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'leading_vehicle']
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
