# ActorGraph Quick Reference

## Common Operations

### Creating ActorGraph Instances

#### From Argoverse Data
```python
actor_graph = ActorGraph.from_argoverse_scenario(
    scenario, 
    G_map,
    max_node_distance_leading=6,      # Allow longer following chains
    max_node_distance_neighbor=3,     # Standard neighbor connectivity
    max_node_distance_opposite=2      # Strict opposite relations
)
```

#### From CARLA Data
```python
actor_graph = ActorGraph.from_carla_scenario(
    scenario, 
    G_map,
    max_node_distance_leading=4,      # Medium following chains
    max_node_distance_neighbor=3,     # Standard neighbor connectivity
    max_node_distance_opposite=2      # Strict opposite relations
)
```

### Custom Graph Creation
```python
actor_graph.create_actor_graphs(
    G_map,
    max_distance_lead_veh_m=100,           # Leading vehicle distance limit
    max_distance_neighbor_forward_m=50,    # Forward neighbor distance limit
    max_distance_neighbor_backward_m=50,   # Backward neighbor distance limit
    max_distance_opposite_forward_m=100,   # Forward opposite distance limit
    max_distance_opposite_backward_m=100,  # Backward opposite distance limit
    max_node_distance_leading=6,           # Leading/following path limit
    max_node_distance_neighbor=3,          # Neighbor path limit
    max_node_distance_opposite=2,          # Opposite path limit
    delta_timestep_s=1.0                  # Time step increment
)
```

## Parameter Tuning Guide

### For Dense Traffic (Many Vehicles)
```python
# Allow longer paths to capture complex relationships
max_node_distance_leading=8      # Long following chains
max_node_distance_neighbor=5     # Extended neighbor paths
max_node_distance_opposite=3     # Medium opposite paths
```

### For Sparse Traffic (Few Vehicles)
```python
# Keep paths short for clarity
max_node_distance_leading=3      # Short following chains
max_node_distance_neighbor=2     # Direct neighbors only
max_node_distance_opposite=1     # Direct opposites only
```

### For Highway Scenarios
```python
# Long following chains, few neighbors/opposites
max_node_distance_leading=10     # Very long following chains
max_node_distance_neighbor=2     # Direct neighbors only
max_distance_neighbor_forward_m=100   # Wide neighbor detection
max_distance_neighbor_backward_m=20   # Limited backward neighbors
```

### For Urban Scenarios
```python
# Balanced approach for complex intersections
max_node_distance_leading=5      # Medium following chains
max_node_distance_neighbor=4     # Extended neighbor paths
max_node_distance_opposite=3     # Medium opposite paths
max_distance_opposite_forward_m=80   # Moderate opposite detection
```

## Troubleshooting Guide

### Problem: Too Many Edges
**Symptoms**: Graph is overly dense, hard to interpret
**Solutions**:
1. Reduce `max_node_distance_*` values
2. Check if path checking is working correctly
3. Verify graph updates happen after each edge addition

### Problem: Missing Relations
**Symptoms**: Expected edges don't exist
**Solutions**:
1. Increase distance limits in exploration phase
2. Check if actors are within detection range
3. Verify map graph connectivity

### Problem: Redundant Opposite Edges
**Symptoms**: Multiple opposite edges between same pair
**Solutions**:
1. Ensure `_add_opposite_edges()` checks both directions together
2. Verify path checking uses updated graph state
3. Check if `should_add_A_to_B` logic is correct

### Problem: Performance Issues
**Symptoms**: Graph creation takes too long
**Solutions**:
1. Reduce `max_node_distance_*` values
2. Limit exploration distance limits
3. Check for unnecessary path finding calls

## Common Code Patterns

### Adding New Relation Types
```python
# 1. Add detection in exploration phase
def _find_relation_between_actors(self, ...):
    # ... existing logic ...
    if new_condition:
        return ("new_relation_type", path_length)
    
# 2. Create edge addition method
def _add_new_relation_edges(self, G_t, relations_dict, max_node_distance):
    # ... implementation ...
    
# 3. Integrate in construction phase
def _construct_graph(self, ...):
    # ... existing calls ...
    self._add_new_relation_edges(G_t, relations_dict, max_node_distance)
```

### Modifying Distance Calculations
```python
# In _find_relation_between_actors()
if relation_type == "neighbor_vehicle":
    # Custom distance calculation
    path_length = custom_distance_function(...)
    return (relation_type, path_length)
```

### Custom Path Checking
```python
# Override path checking method
def _has_path_within_distance(self, G, source, target, max_distance):
    # Custom path finding logic
    return custom_path_check(G, source, target, max_distance)
```

## Performance Tips

### For Large Datasets
1. **Start Conservative**: Use low `max_node_distance` values initially
2. **Profile Path Finding**: Monitor NetworkX shortest_path calls
3. **Batch Processing**: Consider processing timesteps in batches
4. **Memory Management**: Monitor memory usage with large graphs

### For Real-time Applications
1. **Limit Exploration**: Use smaller distance limits
2. **Cache Results**: Store computed relations when possible
3. **Incremental Updates**: Update graphs incrementally rather than recreating
4. **Parallel Processing**: Consider parallel exploration for independent timesteps

## Debugging Commands

### Check Graph Structure
```python
# Verify graph properties
G = actor_graph.actor_graphs[timestep]
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Check edge types
edge_types = [d['edge_type'] for u, v, d in G.edges(data=True)]
print(f"Edge types: {set(edge_types)}")
```

### Verify Relations Dictionary
```python
# Check exploration phase output
relations = actor_graph._explore_relations(t, G_map, ...)
for actor_id, relation_types in relations.items():
    print(f"Actor {actor_id}: {relation_types}")
```

### Test Path Checking
```python
# Verify path checking works
has_path = actor_graph._has_path_within_distance(G, source, target, max_distance)
print(f"Path exists: {has_path}")
```
