import networkx as nx
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import carla
import pickle


class MapGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    # Function to create a polygon from lane boundaries
    def _create_lane_polygon(self, left_boundary, right_boundary):
        left_points = [(point.x, point.y) for point in left_boundary.waypoints]
        right_points = [(point.x, point.y) for point in right_boundary.waypoints]
        polygon_points = left_points + right_points[::-1]
        return Polygon(polygon_points)

    def _calculate_boundary_length(self, boundary):
        """Calculate the length of a boundary by summing the distances between consecutive waypoints."""
        length = 0.0
        for i in range(len(boundary.waypoints) - 1):
            p1 = boundary.waypoints[i]
            p2 = boundary.waypoints[i + 1]
            length += np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        return length

    @classmethod
    def create_from_argoverse_map(cls, map):
        instance = cls()
        G = instance.graph

        # Add nodes with attributes
        for lane_id, lane in map.vector_lane_segments.items():
            lane_polygon = instance._create_lane_polygon(lane.left_lane_boundary, lane.right_lane_boundary)
            left_length = instance._calculate_boundary_length(lane.left_lane_boundary)
            right_length = instance._calculate_boundary_length(lane.right_lane_boundary)
            avg_length = (left_length + right_length) / 2.0
            
            G.add_node(
                lane_id,
                is_intersection=lane.is_intersection,
                lane_type=lane.lane_type,
                left_mark_type=lane.left_mark_type,
                right_mark_type=lane.right_mark_type,
                left_boundary=lane.left_lane_boundary,
                right_boundary=lane.right_lane_boundary,
                lane_polygon=lane_polygon,
                length=avg_length,
            )

        # Add edges for successors and predecessors (type 1)
        for lane_id, lane in map.vector_lane_segments.items():
            for successor_id in lane.successors:
                if successor_id in G:
                    G.add_edge(lane_id, successor_id, edge_type="following")
            for predecessor_id in lane.predecessors:
                if predecessor_id in G:
                    G.add_edge(predecessor_id, lane_id, edge_type="following")

        # Add edges for neighboring lanes (type 2)
        for lane_id, lane in map.vector_lane_segments.items():
            if lane.left_neighbor_id is not None:
                if lane.left_neighbor_id in G:
                    G.add_edge(lane_id, lane.left_neighbor_id, edge_type="neighbor")
                    G.add_edge(lane.left_neighbor_id, lane_id, edge_type="neighbor")
            if lane.right_neighbor_id is not None:
                if lane.right_neighbor_id in G:
                    G.add_edge(lane_id, lane.right_neighbor_id, edge_type="neighbor")
                    G.add_edge(lane.right_neighbor_id, lane_id, edge_type="neighbor")

        # Rename neighboring lanes from lanes in opposite direction by looking for loops.
        edges_opposite = []
        for node in G.nodes():
            for successor in G.successors(node):
                if G[node][successor][0]["edge_type"] == "following":
                    for neighbor in G.successors(successor):
                        if G[successor][neighbor][0]["edge_type"] == "neighbor":
                            for next_node in G.successors(neighbor):
                                if G[neighbor][next_node][0]["edge_type"] == "following":
                                    if G.has_edge(next_node, node) and G[next_node][node][0]["edge_type"] == "neighbor":
                                        edges_opposite.append((successor, neighbor))
                                        edges_opposite.append((next_node, node))

        # Rename edges to 'opposite'
        for u, v in edges_opposite:
            G[u][v][0]["edge_type"] = "opposite"

        return instance

    def _to_2d(self, location):
        return (location.x, location.y)

    @classmethod
    def create_from_carla_map(cls, map):
        """map has to be something like
        world = client.get_world()
        map = world.get_map()

        using a similar strategy as the global route planer:
        https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/global_route_planner.py#L118

        """
        instance = cls()
        G = instance.graph
        topo = map.get_topology()

        for item in topo:
            G.add_edge(
                f"{item[0].road_id}_{item[0].lane_id}",
                f"{item[1].road_id}_{item[1].lane_id}",
                edge_type="following",
            )
            if item[0].get_right_lane():
                if item[0].get_right_lane().lane_type == carla.libcarla.LaneType.Driving:
                    G.add_edge(
                        f"{item[0].road_id}_{item[0].lane_id}",
                        f"{item[0].get_right_lane().road_id}_{item[0].get_right_lane().lane_id}",
                        edge_type="neighbor",
                    )
            if item[0].get_left_lane():
                if item[0].get_left_lane().lane_type == carla.libcarla.LaneType.Driving:
                    if np.sign(item[0].lane_id) == np.sign(item[0].get_left_lane().lane_id):
                        G.add_edge(
                            f"{item[0].road_id}_{item[0].lane_id}",
                            f"{item[0].get_left_lane().road_id}_{item[0].get_left_lane().lane_id}",
                            edge_type="neighbor",
                        )
                    else:
                        G.add_edge(
                            f"{item[0].road_id}_{item[0].lane_id}",
                            f"{item[0].get_left_lane().road_id}_{item[0].get_left_lane().lane_id}",
                            edge_type="opposite",
                        )
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["lane_type"] = str(item[0].lane_type)
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["is_intersection"] = item[0].is_intersection
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["left_mark_type"] = str(item[0].left_lane_marking.type)
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["right_mark_type"] = str(item[0].right_lane_marking.type)
            wps = item[0].next_until_lane_end(0.25)
            lane_length = sum(
                [wps[i].transform.location.distance(wps[i + 1].transform.location) for i in range(len(wps) - 1)]
            )
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["length"] = lane_length
            # Create lane polygon
            forward_vector = item[0].transform.get_forward_vector()
            right_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)  # Perpendicular to forward
            # Compute lane width
            lane_width = item[0].lane_width
            # Compute boundary points
            left_boundary_start = item[0].transform.location + right_vector * (lane_width / 2.0)
            right_boundary_start = item[0].transform.location - right_vector * (lane_width / 2.0)
            left_boundary_end = item[1].transform.location + right_vector * (lane_width / 2.0)
            right_boundary_end = item[1].transform.location - right_vector * (lane_width / 2.0)
            lane_polygon = Polygon(
                [
                    instance._to_2d(left_boundary_start),
                    instance._to_2d(left_boundary_end),
                    instance._to_2d(right_boundary_end),
                    instance._to_2d(right_boundary_start),
                ]
            )
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["lane_polygon"] = lane_polygon

        return instance

    def visualize_graph(self, save_path=None):
        pos = {
            node: (data["lane_polygon"].centroid.x, data["lane_polygon"].centroid.y)
            for node, data in self.graph.nodes(data=True)
        }
        labels = {node: node for node in self.graph.nodes()}

        plt.figure(figsize=(12, 12))
        node_size = 50
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size)
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=labels,
            font_size=8,
            font_color="black",
            verticalalignment="bottom",
        )

        # move label away from nodes..
        # Draw edges with different styles based on edge type
        edge_type_fol = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "following"]
        edge_type_nei = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "neighbor"]
        edge_type_opp = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "opposite"]

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edge_type_fol,
            width=2,
            edge_color="blue",
            node_size=node_size,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edge_type_nei,
            width=1,
            edge_color="green",
            node_size=node_size,
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edge_type_opp,
            width=1,
            edge_color="red",
            node_size=node_size,
        )

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def store_graph_to_file(self, save_path=str):
        # nx.write_graphml(self.graph, save_path)
        # nx.write_gml(self.graph, save_path)
        with open(save_path, "wb") as file:
            pickle.dump(self.graph, file)

    def read_graph_from_file(self, load_path=str):
        with open(load_path, "rb") as file:
            self.graph = pickle.load(file)
