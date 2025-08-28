import networkx as nx
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import numpy as np

import pickle
from graph_creator.models import NodeInfo


class MapGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    @classmethod
    def create_from_argoverse_map(cls, map):
        instance = cls()
        G = instance.graph

        # Add nodes with attributes
        for lane_id, lane in map.vector_lane_segments.items():
            # Create NodeInfo instance from the lane
            node_info = NodeInfo.from_argoverse_lane(lane)
            
            # Add node with NodeInfo as attribute, ensuring lane_id is string
            G.add_node(str(lane_id), node_info=node_info)

        # Add edges for successors and predecessors (type 1)
        for lane_id, lane in map.vector_lane_segments.items():
            for successor_id in lane.successors:
                if str(successor_id) in G:
                    G.add_edge(str(lane_id), str(successor_id), edge_type="following")
            for predecessor_id in lane.predecessors:
                if str(predecessor_id) in G:
                    G.add_edge(str(predecessor_id), str(lane_id), edge_type="following")

        # Add edges for neighboring lanes (type 2)
        for lane_id, lane in map.vector_lane_segments.items():
            if lane.left_neighbor_id is not None:
                if str(lane.left_neighbor_id) in G:
                    G.add_edge(str(lane_id), str(lane.left_neighbor_id), edge_type="neighbor")
                    G.add_edge(str(lane.left_neighbor_id), str(lane_id), edge_type="neighbor")
            if lane.right_neighbor_id is not None:
                if str(lane.right_neighbor_id) in G:
                    G.add_edge(str(lane_id), str(lane.right_neighbor_id), edge_type="neighbor")
                    G.add_edge(str(lane.right_neighbor_id), str(lane_id), edge_type="neighbor")

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
        import carla
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

            node_info = NodeInfo.from_carla_lane(item)
            G.nodes[f"{item[0].road_id}_{item[0].lane_id}"]["node_info"] = node_info

        return instance

    @classmethod
    def create_from_carla_map_wo_nodeinfo(cls, map):
        """map has to be something like
        world = client.get_world()
        map = world.get_map()

        using a similar strategy as the global route planer:
        https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/global_route_planner.py#L118

        """
        import carla
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
            node: (data["node_info"].lane_polygon.centroid.x, data["node_info"].lane_polygon.centroid.y)
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

        # Helper function to offset edges orthogonally to their direction
        def offset_edge_positions(edges, offset_distance=0.5):
            """Offset edge positions orthogonally to make overlapping edges visible"""
            offset_pos = pos.copy()
            
            for u, v in edges:
                if u in pos and v in pos:
                    # Calculate edge direction vector
                    dx = pos[v][0] - pos[u][0]
                    dy = pos[v][1] - pos[u][1]
                    edge_length = np.sqrt(dx**2 + dy**2)
                    
                    if edge_length > 0:
                        # Normalize and rotate 90 degrees to get orthogonal direction
                        orthogonal_dx = -dy / edge_length
                        orthogonal_dy = dx / edge_length
                        
                        # Apply offset to both nodes
                        offset_pos[u] = (pos[u][0] + orthogonal_dx * offset_distance, 
                                       pos[u][1] + orthogonal_dy * offset_distance)
                        offset_pos[v] = (pos[v][0] + orthogonal_dx * offset_distance, 
                                       pos[v][1] + orthogonal_dy * offset_distance)
            
            return offset_pos

        # Separate edges by type
        edge_type_fol = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "following"]
        edge_type_nei = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "neighbor"]
        edge_type_opp = [(u, v) for u, v, d in self.graph.edges(data=True) if d["edge_type"] == "opposite"]

        # Draw edges with offsets to prevent overlap
        fol_pos = offset_edge_positions(edge_type_fol, offset_distance=0.3)
        nei_pos = offset_edge_positions(edge_type_nei, offset_distance=-0.3)
        opp_pos = offset_edge_positions(edge_type_opp, offset_distance=0.6)

        # Draw following edges (blue)
        nx.draw_networkx_edges(
            self.graph,
            fol_pos,
            edgelist=edge_type_fol,
            width=2,
            edge_color="blue",
            node_size=node_size,
            label="Following",
        )
        
        # Draw neighbor edges (green)
        nx.draw_networkx_edges(
            self.graph,
            nei_pos,
            edgelist=edge_type_nei,
            width=1,
            edge_color="green",
            node_size=node_size,
            label="Neighbor",
        )
        
        # Draw opposite edges (red)
        nx.draw_networkx_edges(
            self.graph,
            opp_pos,
            edgelist=edge_type_opp,
            width=1,
            edge_color="red",
            node_size=node_size,
            label="Opposite",
        )

        # Add legend
        plt.legend(loc='upper right', fontsize=10)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()

    def store_graph_to_file(self, save_path=str):
        with open(save_path, "wb") as file:
            pickle.dump(self.graph, file)

    def read_graph_from_file(self, load_path=str):
        with open(load_path, "rb") as file:
            self.graph = pickle.load(file)
