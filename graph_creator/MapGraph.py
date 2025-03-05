import networkx as nx
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

class MapGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    # Function to create a polygon from lane boundaries
    def _create_lane_polygon(self, left_boundary, right_boundary):
        left_points = [(point.x, point.y) for point in left_boundary.waypoints]
        right_points = [(point.x, point.y) for point in right_boundary.waypoints]
        polygon_points = left_points + right_points[::-1]
        return Polygon(polygon_points)


    @classmethod
    def create_from_argoverse_map(cls, map):
        instance = cls()
        G = instance.graph

        # Add nodes with attributes
        for lane_id, lane in map.vector_lane_segments.items():
            lane_polygon = instance._create_lane_polygon(lane.left_lane_boundary, lane.right_lane_boundary)
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
                    G.add_edge(lane.right_neighbor_id, lane_id, edge_type='neighbor')

        # Rename neighboring lanes from lanes in opposite direction by looking for loops.
        edges_opposite = []
        for node in G.nodes():
            for successor in G.successors(node):
                if G[node][successor][0]['edge_type'] == 'following':
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

        return instance
    

    def visualize_graph(self, save_path=None):
        pos = {node: (data['lane_polygon'].centroid.x, data['lane_polygon'].centroid.y) for node, data in self.graph.nodes(data=True)}
        labels = {node: node for node in self.graph.nodes()}

        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, labels=labels, with_labels=True, node_size=50, font_size=8, font_color='black') # TODO: update for directed graph

        # Draw edges with different styles based on edge type
        edge_type_fol = [(u, v) for u, v, d in self.graph.edges(data=True) if d['edge_type'] == 'following']
        edge_type_nei = [(u, v) for u, v, d in self.graph.edges(data=True) if d['edge_type'] == 'neighbor']
        edge_type_opp = [(u, v) for u, v, d in self.graph.edges(data=True) if d['edge_type'] == 'opposite']

        nx.draw_networkx_edges(self.graph, pos, edgelist=edge_type_fol, width=2, edge_color='blue')
        nx.draw_networkx_edges(self.graph, pos, edgelist=edge_type_nei, width=1, edge_color='green')
        nx.draw_networkx_edges(self.graph, pos, edgelist=edge_type_opp, width=1, edge_color='red')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    

