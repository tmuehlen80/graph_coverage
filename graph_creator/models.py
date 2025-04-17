from typing import List
from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import Polygon, LineString
import numpy as np

class NodeInfo(BaseModel):
    """Pydantic model for node information in the graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    lane_id: int = Field(..., description="Unique identifier for the lane")
    is_intersection: bool = Field(..., description="Whether the node represents an intersection")
    length: float = Field(..., description="Length of the lane in meters")
    lane_polygon: Polygon = Field(..., description="Polygon representing the lane area")
    left_boundary: LineString = Field(..., description="Left boundary of the lane as a polyline")
    right_boundary: LineString = Field(..., description="Right boundary of the lane as a polyline")

    @staticmethod
    def _create_lane_polygon(left_boundary, right_boundary):
        """Create a polygon from lane boundaries."""
        left_points = [(point.x, point.y) for point in left_boundary.waypoints]
        right_points = [(point.x, point.y) for point in right_boundary.waypoints]
        polygon_points = left_points + right_points[::-1]
        return Polygon(polygon_points)

    @staticmethod
    def _calculate_boundary_length(boundary):
        """Calculate the length of a boundary by summing the distances between consecutive waypoints."""
        length = 0.0
        for i in range(len(boundary.waypoints) - 1):
            p1 = boundary.waypoints[i]
            p2 = boundary.waypoints[i + 1]
            length += np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        return length

    @classmethod
    def from_argoverse_lane(cls, lane) -> 'NodeInfo':
        """Create a NodeInfo instance from an Argoverse lane object.
        
        Args:
            lane: An Argoverse lane object containing lane information
            
        Returns:
            NodeInfo instance with the lane information
        """
        # Convert left boundary waypoints to LineString
        left_points = [(pt.x, pt.y) for pt in lane.left_lane_boundary.waypoints]
        left_boundary = LineString(left_points)
        
        # Convert right boundary waypoints to LineString
        right_points = [(pt.x, pt.y) for pt in lane.right_lane_boundary.waypoints]
        right_boundary = LineString(right_points)
        
        # Create lane polygon from boundaries
        lane_polygon = cls._create_lane_polygon(lane.left_lane_boundary, lane.right_lane_boundary)
        
        # Calculate lane length
        left_length = cls._calculate_boundary_length(lane.left_lane_boundary)
        right_length = cls._calculate_boundary_length(lane.right_lane_boundary)
        avg_length = (left_length + right_length) / 2.0
        
        return cls(
            lane_id=lane.id,
            is_intersection=lane.is_intersection,
            length=avg_length,
            lane_polygon=lane_polygon,
            left_boundary=left_boundary,
            right_boundary=right_boundary
        )

    @classmethod
    def from_carla_lane(cls, lane) -> 'NodeInfo':
        """Create a NodeInfo instance from a Carla lane object.
        
        Args:
            lane: A Carla lane object containing lane information
            
        Returns:
            NodeInfo instance with the lane information
        """
        # TODO @Thomas
        return None 