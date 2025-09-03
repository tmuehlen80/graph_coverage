# ActorGraph Technical Architecture

## System Overview

The ActorGraph system is designed to create temporal graphs representing vehicle relationships in traffic scenarios. It follows a two-phase architecture that separates concerns between relation discovery and graph construction.

## Core Design Principles

### 1. Separation of Concerns
- **Exploration Phase**: Pure data collection, no graph manipulation
- **Construction Phase**: Pure graph building, no relation discovery
- **Clear Interfaces**: Well-defined data structures passed between phases

### 2. Hierarchical Processing
- **Leading/Following**: Highest priority, processed first
- **Neighbor**: Medium priority, processed second  
- **Opposite**: Lowest priority, processed last
- **Shortest-First**: Within each type, process shortest paths first

### 3. Incremental Graph Updates
- **Immediate Updates**: Graph state updated after every edge addition
- **Path Consistency**: All path checks use current graph state
- **No Race Conditions**: Sequential processing prevents timing issues

## Data Flow Architecture

```
Input Data → Exploration Phase → Relations Dictionary → Construction Phase → Final Graph
     ↓              ↓                    ↓                    ↓              ↓
Track Data → Find Relations → Store in relations_dict → Add Edges → NetworkX Graph
```

### Data Structures

#### `relations_dict` Structure
```python
{
    actor_id_A: {
        "leading_vehicle": [(actor_id_B, path_length), ...],
        "neighbor_vehicle": [(actor_id_C, path_length), ...],
        "opposite_vehicle": [(actor_id_D, path_length), ...]
    },
    actor_id_B: {
        # Similar structure for other actors
    }
}
```

#### Graph Node Attributes
```python
{
    "lane_id": "primary_lane_id",
    "lane_ids": ["lane1", "lane2"],  # All lanes actor is on
    "s": s_value,                    # Longitudinal position
    "xyz": Point(x, y, z),           # 3D position
    "lon_speed": speed,              # Longitudinal speed
    "actor_type": ActorType.VEHICLE, # Actor classification
    "lane_change": boolean           # Whether lane changed from previous timestep
}
```

#### Graph Edge Attributes
```python
{
    "edge_type": "leading_vehicle" | "following_lead" | "neighbor_vehicle" | "opposite_vehicle",
    "path_length": float             # Distance in meters (can be negative)
}
```

## Method Architecture

### Core Methods

#### `create_actor_graphs()`
- **Entry Point**: Main orchestrator method
- **Responsibilities**: 
  - Timestep management
  - Node creation
  - Phase coordination
  - Lane change detection
- **Returns**: Dictionary of timestep → NetworkX graph

#### `_explore_relations()`
- **Input**: Timestep, map graph, distance limits
- **Process**: 
  - Iterate through all actor pairs
  - Find relations in both directions
  - Choose optimal relation using hierarchy
  - Store in relations_dict
- **Output**: Populated relations_dict

#### `_construct_graph()`
- **Input**: Empty graph, relations_dict, max_node_distance values
- **Process**: 
  - Call specialized edge addition methods in order
  - Ensure proper hierarchical processing
- **Output**: Populated graph with all edges

### Specialized Edge Addition Methods

#### `_add_leading_following_edges()`
- **Processing**: Leading → Following (bidirectional)
- **Sorting**: By path_length (shortest first)
- **Path Checking**: Uses max_node_distance_leading

#### `_add_neighbor_edges()`
- **Processing**: Neighbor relations (bidirectional)
- **Sorting**: By abs(path_length) (handles negatives)
- **Path Checking**: Uses max_node_distance_neighbor

#### `_add_opposite_edges()`
- **Processing**: Opposite relations (bidirectional)
- **Sorting**: By abs(path_length) (handles negatives)
- **Path Checking**: Uses max_node_distance_opposite
- **Distance Limits**: Separate forward/backward limits

## Algorithm Complexity

### Time Complexity
- **Exploration Phase**: O(n² × m) where n = actors, m = map complexity
- **Construction Phase**: O(e × p) where e = edges, p = path checking complexity
- **Overall**: O(n² × m + e × p)

### Space Complexity
- **Relations Dictionary**: O(n × r) where r = relations per actor
- **Graph Storage**: O(n + e) where e = total edges
- **Overall**: O(n × r + e)

## Performance Considerations

### Optimization Strategies
1. **Incremental Updates**: Graph updated immediately, not in batches
2. **Path Caching**: NetworkX shortest_path used efficiently
3. **Early Termination**: Path checking stops at max_node_distance
4. **Sorted Processing**: Shortest paths processed first

### Bottleneck Areas
1. **Path Finding**: NetworkX shortest_path calls during construction
2. **Actor Pair Iteration**: O(n²) complexity in exploration
3. **Map Graph Traversal**: Complex map structures can slow relation discovery

## Error Handling and Validation

### Input Validation
- **Timestamps**: Asserted to be sorted
- **Distance Limits**: Must be positive values
- **Map Graph**: Must be valid NetworkX graph
- **Actor Data**: Must have consistent structure

### Runtime Checks
- **Path Existence**: Handles NetworkXNoPath exceptions
- **Graph State**: Ensures graph is updated after each edge
- **Data Consistency**: Validates actor existence across timesteps

## Extension Points

### Adding New Relation Types
1. **Exploration**: Add detection logic in `_find_relation_between_actors()`
2. **Construction**: Create new `_add_*_edges()` method
3. **Integration**: Add to `_construct_graph()` hierarchy
4. **Parameters**: Add corresponding max_node_distance parameter

### Custom Distance Metrics
1. **Path Length Calculation**: Modify in relation detection
2. **Sorting Logic**: Update sorting keys in edge addition methods
3. **Validation**: Add parameter validation in factory methods

### Alternative Graph Types
1. **Weighted Edges**: Modify edge attributes and path finding
2. **Directed vs Undirected**: Change NetworkX graph type
3. **Multi-Graph Support**: Already implemented with MultiDiGraph

## Testing Strategy

### Unit Testing
- **Individual Methods**: Test each phase independently
- **Edge Cases**: Test with empty data, single actors, etc.
- **Parameter Validation**: Test boundary conditions

### Integration Testing
- **End-to-End**: Test complete graph creation pipeline
- **Data Consistency**: Verify graph structure matches input data
- **Performance**: Test with realistic dataset sizes

### Regression Testing
- **Graph Properties**: Verify consistent edge/node counts
- **Path Properties**: Ensure shortest paths are maintained
- **Parameter Sensitivity**: Test different distance limit combinations
