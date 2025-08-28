from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from shapely.geometry import Polygon, LineString, Point
import numpy as np
from enum import Enum


def _to_2d(location):
    return (location.x, location.y)

class NodeInfo(BaseModel):
    """Pydantic model for node information in the graph."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    lane_id: str = Field(..., description="Unique identifier for the lane")
    is_intersection: bool = Field(..., description="Whether the node represents an intersection")
    length: float = Field(..., description="Length of the lane in meters")
    lane_polygon: Polygon = Field(..., description="Polygon representing the lane area")
    left_boundary: LineString = Field(None, description="Left boundary of the lane as a polyline")
    right_boundary: LineString = Field(None, description="Right boundary of the lane as a polyline")

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
            lane_id=str(lane.id),
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
        import carla
        # TODO @Thomas
        wps = lane[0].next_until_lane_end(0.25)
        lane_length = sum(
            [wps[i].transform.location.distance(wps[i + 1].transform.location) for i in range(len(wps) - 1)]
        )
        # Create lane polygon
        forward_vector = lane[0].transform.get_forward_vector()
        right_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)  # Perpendicular to forward
        # Compute lane width
        lane_width = lane[0].lane_width
        # Compute boundary points
        left_boundary_start = lane[0].transform.location + right_vector * (lane_width / 2.0)
        right_boundary_start = lane[0].transform.location - right_vector * (lane_width / 2.0)
        left_boundary_end = lane[1].transform.location + right_vector * (lane_width / 2.0)
        right_boundary_end = lane[1].transform.location - right_vector * (lane_width / 2.0)
        lane_polygon = Polygon(
            [
                _to_2d(left_boundary_start),
                _to_2d(left_boundary_end),
                _to_2d(right_boundary_end),
                _to_2d(right_boundary_start),
            ]
        )

        return cls(lane_id = F"{lane[0].road_id}_{lane[0].lane_id}",
                   is_intersection = lane[0].is_intersection,
                   length = lane_length,
                   lane_polygon = lane_polygon
                   )

class ActorType(Enum):
    VEHICLE = "VEHICLE"  # default
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"  # maybe delete this

class TrackData(BaseModel):
    track_lane_dict: Dict[str, List[List[Optional[str]]]] = Field(
        description="Dictionary mapping track IDs to lists of lists of lane IDs. Each inner list represents lane IDs for a timestep (can contain None)"
    )
    track_s_value_dict: Dict[str, List[float]] = Field(description="Dictionary mapping track IDs to lists of s-values")
    track_xyz_pos_dict: Dict[str, List[Point]] = Field(description="Dictionary mapping track IDs to lists of 3D points")
    track_speed_lon_dict: Dict[str, List[float]] = Field(
        description="Dictionary mapping track IDs to lists of longitudinal speeds"
    )
    track_actor_type_dict: Dict[str, ActorType] = Field(description="Dictionary mapping track IDs to actor types.")

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
        actor_dict = self.track_actor_type_dict
        # Get all unique actors across all dictionaries
        all_actors = (
            set(lane_dict.keys())
            | set(s_value_dict.keys())
            | set(xyz_dict.keys())
            | set(speed_dict.keys())
            | set(actor_dict.keys())
        )

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
            ]
            if not all(l == lengths[0] for l in lengths):
                raise ValueError(f"Inconsistent list lengths for actor {actor}. Expected {lengths[0]}, got {lengths}")

        return self

    class Config:
        arbitrary_types_allowed = True