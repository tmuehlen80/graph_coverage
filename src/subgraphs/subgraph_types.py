"""
Manually defined subgraph types for coverage analysis.

This module contains predefined subgraph patterns that represent
common traffic scenarios for isomorphism-based coverage analysis.

Patterns are organized into two groups:
1. SIMPLE_PATTERNS: 2-actor interactions (valid only if isolated/disconnected)
2. COMPLEX_PATTERNS: 3+ actors with special cases or 4+ actors

IMPORTANT - Edge Direction Semantics:
- "following_lead": edge from follower TO lead vehicle (A follows B: A→B)
- "leading_vehicle": edge from lead TO follower vehicle (B leads A: B→A)
- "neighbor_vehicle": bidirectional, same edge type both ways (symmetric relation)
- "opposite_vehicle": bidirectional, same edge type both ways (symmetric relation)
"""
import networkx as nx
from graph_creator.models import ActorType


# ==============================================================================
# SIMPLE 2-ACTOR PATTERNS (valid only if disconnected from other actors)
# ==============================================================================

def create_simple_following():
    """Two vehicles following each other (isolated car-following scenario)."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    return G


def create_simple_opposite():
    """Two vehicles in opposite directions (isolated oncoming traffic)."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="opposite_vehicle")
    G.add_edge("b", "a", edge_type="opposite_vehicle")
    return G


def create_simple_neighbor():
    """Two vehicles side by side in adjacent lanes (isolated)."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="neighbor_vehicle")
    G.add_edge("b", "a", edge_type="neighbor_vehicle")
    return G


# ==============================================================================
# COMPLEX PATTERNS (3+ actors with special cases OR 4+ actors)
# ==============================================================================


def create_lead_vehicle_with_neighbor_vehicle():
    """Lead vehicle in front with neighbor vehicle."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
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
    G.add_edge("b", "a", edge_type="leading_vehicle")
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
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="leading_vehicle")
    return G


def create_cut_in():
    """Cut-in scenario."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=True, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="leading_vehicle")
    return G


def create_cut_in_intersection():
    """Cut-in scenario at intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=True, is_on_intersection=True)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="leading_vehicle")
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
    G.add_edge("c", "a", edge_type="leading_vehicle")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="leading_vehicle")
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
    G.add_edge("c", "a", edge_type="leading_vehicle")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="leading_vehicle")
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
    G.add_edge("c", "a", edge_type="leading_vehicle")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="leading_vehicle")
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
    G.add_edge("c", "a", edge_type="leading_vehicle")
    G.add_edge("a", "d", edge_type="following_lead")
    G.add_edge("d", "a", edge_type="leading_vehicle")
    G.add_edge("a", "e", edge_type="opposite_vehicle")
    G.add_edge("e", "a", edge_type="opposite_vehicle")
    return G


def create_platoon_with_intersection():
    """Three-vehicle platoon with one vehicle on intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("b", "c", edge_type="following_lead")
    G.add_edge("c", "b", edge_type="leading_vehicle")
    return G


def create_opposite_traffic_at_intersection():
    """Three vehicles with opposite traffic and one on intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("a", "c", edge_type="opposite_vehicle")
    G.add_edge("c", "a", edge_type="opposite_vehicle")
    return G


def create_lead_with_neighbor_at_intersection():
    """Lead vehicle with neighbor vehicle, one approaching intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("a", "c", edge_type="neighbor_vehicle")
    G.add_edge("c", "a", edge_type="neighbor_vehicle")
    return G


def create_triple_opposite_traffic_intersection():
    """Three vehicles in opposite traffic configuration with intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="opposite_vehicle")
    G.add_edge("b", "a", edge_type="opposite_vehicle")
    G.add_edge("b", "c", edge_type="opposite_vehicle")
    G.add_edge("c", "b", edge_type="opposite_vehicle")
    return G


def create_four_vehicle_intersection_platoon():
    """Four-vehicle platoon with one vehicle on intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("d", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("b", "c", edge_type="following_lead")
    G.add_edge("c", "b", edge_type="leading_vehicle")
    G.add_edge("c", "d", edge_type="following_lead")
    G.add_edge("d", "c", edge_type="leading_vehicle")
    return G


def create_four_vehicle_opposite_intersection():
    """Four vehicles with lead-following and opposite traffic at intersection."""
    G = nx.MultiDiGraph()
    G.add_node("a", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("b", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_node("c", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=True)
    G.add_node("d", actor_type=ActorType.VEHICLE, lane_change=False, is_on_intersection=False)
    G.add_edge("a", "b", edge_type="following_lead")
    G.add_edge("b", "a", edge_type="leading_vehicle")
    G.add_edge("a", "c", edge_type="following_lead")
    G.add_edge("c", "a", edge_type="leading_vehicle")
    G.add_edge("a", "d", edge_type="opposite_vehicle")
    G.add_edge("d", "a", edge_type="opposite_vehicle")
    return G


def get_all_subgraphs():
    """
    Get all predefined subgraph types as a dictionary.
    
    Patterns are organized into simple (2-actor) and complex (3+ actor) patterns.
    
    Returns:
        dict: Dictionary mapping scenario names to NetworkX MultiDiGraph objects.
              Keys are descriptive scenario names, values are the graph objects.
    
    Example:
        >>> coverage_graphs = get_all_subgraphs()
        >>> coverage_graphs["simple_following"]  # Access a 2-actor pattern
    """
    return {
        # ===== SIMPLE 2-ACTOR PATTERNS (valid only if isolated) =====
        "simple_following": create_simple_following(),
        "simple_opposite": create_simple_opposite(),
        "simple_neighbor": create_simple_neighbor(),
        
        # ===== COMPLEX PATTERNS: 3-actor + special case =====
        # Original 3-vehicle scenarios with intersection/lane-change
        "lead_vehicle_in_front_with_neighbor_vehicle_intersection": create_lead_vehicle_with_neighbor_vehicle_intersection(),
        
        # Lane change scenarios (3+ actors + lane_change)
        "cut_in": create_cut_in(),
        "cut_in_intersection": create_cut_in_intersection(),
        
        # New 3-vehicle + intersection patterns
        "platoon_with_intersection": create_platoon_with_intersection(),
        "opposite_traffic_at_intersection": create_opposite_traffic_at_intersection(),
        "lead_with_neighbor_at_intersection": create_lead_with_neighbor_at_intersection(),
        "triple_opposite_traffic_intersection": create_triple_opposite_traffic_intersection(),
        
        # ===== COMPLEX PATTERNS: 4+ actors =====
        # Original 3-vehicle scenarios (now understood as needing 4+ or special case)
        "lead_vehicle_in_front_with_neighbor_vehicle": create_lead_vehicle_with_neighbor_vehicle(),
        "lead_vehicle_in_front_following_vehicle_in_the_back": create_lead_vehicle_following_vehicle_in_back(),
        
        # 4-vehicle scenarios
        "cut_out": create_cut_out(),
        "cut_out_intersection": create_cut_out_intersection(),
        "four_vehicle_intersection_platoon": create_four_vehicle_intersection_platoon(),
        "four_vehicle_opposite_intersection": create_four_vehicle_opposite_intersection(),
        
        # 5-vehicle scenarios
        "lead_neighbor_opposite_vehicle": create_lead_neighbor_opposite_vehicle(),
        "lead_neighbor_opposite_vehicle_intersection": create_lead_neighbor_opposite_vehicle_intersection(),
    }


def get_simple_patterns():
    """
    Get only the simple 2-actor patterns.
    These are valid only for isolated/disconnected subgraphs.
    
    Returns:
        dict: Dictionary of 2-actor patterns
    """
    return {
        "simple_following": create_simple_following(),
        "simple_opposite": create_simple_opposite(),
        "simple_neighbor": create_simple_neighbor(),
    }


def get_complex_patterns():
    """
    Get only the complex patterns (3+ actors).
    These patterns require either special cases (intersection/lane_change) for 3 actors,
    or 4+ actors.
    
    Returns:
        dict: Dictionary of complex patterns
    """
    all_patterns = get_all_subgraphs()
    simple_keys = set(get_simple_patterns().keys())
    return {k: v for k, v in all_patterns.items() if k not in simple_keys}

