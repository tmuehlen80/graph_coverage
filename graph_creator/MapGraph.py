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

        # Update is_intersection status from argoverse to include all overlapping lanes. 
        intersection_status = instance.analyze_intersection_status(map)

        for lane_id, lane in map.vector_lane_segments.items():
            node_info = NodeInfo.from_argoverse_lane(lane, is_intersection=intersection_status[str(lane_id)])
            
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
                    # Only add if the edge doesn't already exist
                    if not G.has_edge(str(lane_id), str(lane.left_neighbor_id)):
                        G.add_edge(str(lane_id), str(lane.left_neighbor_id), edge_type="neighbor")
                    if not G.has_edge(str(lane.left_neighbor_id), str(lane_id)):
                        G.add_edge(str(lane.left_neighbor_id), str(lane_id), edge_type="neighbor")
            if lane.right_neighbor_id is not None:
                if str(lane.right_neighbor_id) in G:
                    # Only add if the edge doesn't already exist
                    if not G.has_edge(str(lane_id), str(lane.right_neighbor_id)):
                        G.add_edge(str(lane_id), str(lane.right_neighbor_id), edge_type="neighbor")
                    if not G.has_edge(str(lane.right_neighbor_id), str(lane_id)):
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

        # Convert neighbor edges to opposite edges
        for u, v in edges_opposite:
            # Just rename the edge type from 'neighbor' to 'opposite'
            G[u][v][0]["edge_type"] = "opposite"
            G[v][u][0]["edge_type"] = "opposite"
        
        return instance

    def _to_2d(self, location):
        return (location.x, location.y)

    def argoverse_check_intersection(self, lane1, lane2, lane1_info, lane2_info):
        """
        Check if two lanes intersect based on the specified criteria.
        
        Args:
            lane1: First Argoverse lane object
            lane2: Second Argoverse lane object  
            lane1_info: NodeInfo object for first lane
            lane2_info: NodeInfo object for second lane
            
        Returns:
            bool: True if lanes should be considered intersecting
        """
        # Step 1: If either lane already has is_intersection=True from Argoverse, set both to True
        if lane1.is_intersection or lane2.is_intersection:
            return True
            
        # Step 2: Use shapely to check if polygons intersect
        polygon1 = lane1_info.lane_polygon
        polygon2 = lane2_info.lane_polygon
        
        if not polygon1.intersects(polygon2):
            return False
            
        # Step 3: Check intersection area - if more than 10% of total area, consider intersecting
        intersection_area = polygon1.intersection(polygon2).area
        total_area = polygon1.area + polygon2.area
        
        if total_area > 0:
            intersection_ratio = intersection_area / total_area
            return intersection_ratio > 0.1  # 10% threshold
            
        return False

    def analyze_intersection_status(self, map):
        """
        Analyze Argoverse map to determine intersection status for each lane.
        
        Args:
            map: Argoverse map object
            
        Returns:
            dict: Dictionary mapping lane IDs (as strings) to intersection status (bool)
        """
        # First, create temporary NodeInfo instances for polygon analysis
        temp_lane_infos = {}
        for lane_id, lane in map.vector_lane_segments.items():
            temp_lane_infos[str(lane_id)] = NodeInfo.from_argoverse_lane(lane)
        
        intersection_status = {}
        lane_ids = list(temp_lane_infos.keys())
        
        # Initialize with original Argoverse intersection status
        for lane_id, lane in map.vector_lane_segments.items():
            intersection_status[str(lane_id)] = lane.is_intersection
        
        # Check for polygon-based intersections only between non-intersection lanes
        for i in range(len(lane_ids)):
            for j in range(i + 1, len(lane_ids)):
                lane1_id = lane_ids[i]
                lane2_id = lane_ids[j]
                lane1 = map.vector_lane_segments[int(lane1_id)]
                lane2 = map.vector_lane_segments[int(lane2_id)]
                
                # Only check polygon intersection if neither lane is already marked as intersection
                if not lane1.is_intersection and not lane2.is_intersection:
                    polygon1 = temp_lane_infos[lane1_id].lane_polygon
                    polygon2 = temp_lane_infos[lane2_id].lane_polygon
                    
                    if polygon1.intersects(polygon2):
                        intersection_area = polygon1.intersection(polygon2).area
                        total_area = polygon1.area + polygon2.area
                        
                        if total_area > 0:
                            intersection_ratio = intersection_area / total_area
                            if intersection_ratio > 0.1:  # 10% threshold
                                intersection_status[lane1_id] = True
                                intersection_status[lane2_id] = True
        
        return intersection_status

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
        
        # Create truncated labels by removing common leading characters
        labels = self._create_truncated_labels(list(self.graph.nodes()))
        
        plt.figure(figsize=(12, 12))
        node_size = 50
        
        # Calculate label positions below nodes with collision detection
        label_pos = self._calculate_label_positions(pos, labels, node_size)
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=node_size)
        
        # Draw labels at calculated positions (not on nodes)
        nx.draw_networkx_labels(
            self.graph,
            label_pos,
            labels=labels,
            font_size=6,  # Smaller font size
            font_color="darkred",  # Changed to darkred for better visibility against blue arrows
            font_weight="bold",  # Make text more readable
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

    def _create_truncated_labels(self, node_ids):
        """
        Create truncated labels by removing common leading characters.
        
        Args:
            node_ids: List of node IDs (strings)
            
        Returns:
            Dictionary mapping original node IDs to truncated labels
        """
        if not node_ids:
            return {}
        
        # Convert to strings and find common leading characters
        str_ids = [str(node_id) for node_id in node_ids]
        
        # Find the longest common prefix
        if len(str_ids) == 1:
            common_prefix = ""
        else:
            # Find common prefix by comparing characters
            common_prefix = ""
            min_length = min(len(s) for s in str_ids)
            
            for i in range(min_length):
                char = str_ids[0][i]
                if all(s[i] == char for s in str_ids):
                    common_prefix += char
                else:
                    break
        
        # Create truncated labels
        labels = {}
        for node_id in node_ids:
            str_id = str(node_id)
            if str_id.startswith(common_prefix):
                # Remove common prefix and keep the rest
                truncated = str_id[len(common_prefix):]
                # If truncated is empty, keep at least one character
                if not truncated:
                    truncated = str_id[-1] if len(str_id) > 0 else str_id
                labels[node_id] = truncated
            else:
                # Fallback: keep original if no common prefix
                labels[node_id] = str_id
        
        return labels

    def _calculate_label_positions(self, pos, labels, node_size, label_offset=0.8):
        """
        Calculate label positions below nodes with collision detection to prevent overlapping.
        
        Args:
            pos: Dictionary of node positions
            labels: Dictionary of node labels
            node_size: Size of nodes (used for spacing calculations)
            label_offset: Vertical offset below nodes
            
        Returns:
            Dictionary of label positions
        """
        label_pos = {}
        used_positions = []
        
        # Convert node_size to coordinate units (approximate)
        # node_size is in points, we need to estimate coordinate units
        # This is a rough approximation - you may need to adjust based on your data
        spacing = node_size * 0.01  # Adjust this multiplier based on your coordinate system
        
        for node, label in labels.items():
            if node not in pos:
                continue
                
            x, y = pos[node]
            # Start with position below the node
            label_x, label_y = x, y - label_offset
            
            # Check for collisions and adjust position
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts:
                collision = False
                
                # Check collision with existing labels
                for used_x, used_y in used_positions:
                    if abs(label_x - used_x) < spacing and abs(label_y - used_y) < spacing:
                        collision = True
                        break
                
                if not collision:
                    break
                
                # Try different positions: left, right, above, diagonal
                if attempts % 4 == 0:
                    label_x, label_y = x - label_offset, y - label_offset  # Left below
                elif attempts % 4 == 1:
                    label_x, label_y = x + label_offset, y - label_offset  # Right below
                elif attempts % 4 == 2:
                    label_x, label_y = x, y + label_offset  # Above
                else:
                    # Diagonal positions
                    angle = (attempts // 4) * np.pi / 4
                    label_x = x + label_offset * np.cos(angle)
                    label_y = y + label_offset * np.sin(angle)
                
                attempts += 1
            
            # If we still have collision, just place it with some random offset
            if attempts >= max_attempts:
                import random
                label_x = x + random.uniform(-label_offset, label_offset)
                label_y = y + random.uniform(-label_offset, label_offset)
            
            label_pos[node] = (label_x, label_y)
            used_positions.append((label_x, label_y))
        
        return label_pos

    def store_graph_to_file(self, save_path=str):
        with open(save_path, "wb") as file:
            pickle.dump(self.graph, file)

    def read_graph_from_file(self, load_path=str):
        with open(load_path, "rb") as file:
            self.graph = pickle.load(file)
