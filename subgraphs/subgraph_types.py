"""
Manually defined subgraph types for coverage analysis.

This module contains predefined subgraph patterns that represent
common traffic scenarios for isomorphism-based coverage analysis.
"""
import networkx as nx
from graph_creator.models import ActorType


def create_lead_vehicle_with_neighbor_vehicle():
    """Lead vehicle in front with neighbor vehicle."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="following_lead")
    G.add_edge("a", "c", edge_type="neighbor_vehicle")
    G.add_edge("c", "a", edge_type="neighbor_vehicle")
    return G


def create_lead_vehicle_with_neighbor_vehicle_intersection():
    """Lead vehicle in front with neighbor vehicle at intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="following_lead")
    G.add_edge("a", "c", edge_type="neighbor_vehicle")
    G.add_edge("c", "a", edge_type="neighbor_vehicle")
    return G


def create_lead_vehicle_following_vehicle_in_back():
    """Lead vehicle in front with following vehicle in the back."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="following_lead")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    return G


def create_cut_in():
    """Cut-in scenario."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=True, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="following_lead")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    return G


def create_cut_in_intersection():
    """Cut-in scenario at intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=True, is_on_intersection=True)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="following_lead")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    return G


def create_cut_out():
    """Cut-out scenario."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=True, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("d", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="neighbor_vehicle")
    G.add_edge("b", "a", edge_type="neighbor_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="following_lead")
    return G


def create_cut_out_intersection():
    """Cut-out scenario at intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=True, is_on_intersection=True)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("d", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="neighbor_vehicle")
    G.add_edge("b", "a", edge_type="neighbor_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="following_lead")
    return G


def create_lead_neighbor_opposite_vehicle():
    """Lead, neighbor, and opposite vehicle scenario."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("d", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("e", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="neighbor_vehicle")
    G.add_edge("b", "a", edge_type="neighbor_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="following_lead")
    G.add_edge("a", "e", edge_type="opposite_vehicle")
    G.add_edge("e", "a", edge_type="opposite_vehicle")
    return G


def create_lead_neighbor_opposite_vehicle_intersection():
    """Lead, neighbor, and opposite vehicle scenario at intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("d", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("e", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="neighbor_vehicle")
    G.add_edge("b", "a", edge_type="neighbor_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="following_lead")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="following_lead")
    G.add_edge("a", "e", edge_type="opposite_vehicle")
    G.add_edge("e", "a", edge_type="opposite_vehicle")
    return G


def get_all_subgraphs():
    """
    Get all predefined subgraph types as a dictionary.
    
    Returns:
        dict: Dictionary mapping scenario names to NetworkX MultiDiGraph objects.
              Keys are descriptive scenario names, values are the graph objects.
    
    Example:
        >>> coverage_graphs = get_all_subgraphs()
        >>> coverage_graphs["cut_in"]  # Access a specific scenario
    """
    return {
        "lead_vehicle_in_front_with_neighbor_vehicle": create_lead_vehicle_with_neighbor_vehicle(),
        "lead_vehicle_in_front_with_neighbor_vehicle_intersection": create_lead_vehicle_with_neighbor_vehicle_intersection(),
        "lead_vehicle_in_front_following_vehicle_in_the_back": create_lead_vehicle_following_vehicle_in_back(),
        "cut_in": create_cut_in(),
        "cut_in_intersection": create_cut_in_intersection(),
        "cut_out": create_cut_out(),
        "cut_out_intersection": create_cut_out_intersection(),
        "lead_neighbor_opposite_vehicle": create_lead_neighbor_opposite_vehicle(),
        "lead_neighbor_opposite_vehicle_intersection": create_lead_neighbor_opposite_vehicle_intersection(),
    }

