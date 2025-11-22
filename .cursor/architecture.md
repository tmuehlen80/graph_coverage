# ActorGraph Technical Architecture

## System Overview

The ActorGraph system is designed to create temporal graphs representing vehicle relationships in traffic scenarios. It follows a two-phase architecture that separates concerns between relation discovery and graph construction.

## Core Design Principles

### 1. Separation of Concerns: Two-Phase Architecture
- **Exploration Phase (Discovery)**: 
  - Finds ALL potential relations between actors based on distance limits
  - Uses map graph to determine relation types and path lengths
  - Does NOT check `max_node_distance_*` parameters (these are for construction only)
  - Stores discovered relations in `relations_dict` for later processing
  - Pure data collection, no graph manipulation
  
- **Construction Phase (Hierarchical Building)**:
  - Builds the actor graph by adding edges from discovered relations
  - Uses `max_node_distance_*` to prevent redundant edges in the actor graph
  - Processes relations hierarchically (leading → neighbor → opposite)
  - Within each type, processes shortest paths first
  - Checks if a path already exists in the actor graph before adding new edges
  - Pure graph building, no relation discovery

- **Clear Interfaces**: Well-defined data structures passed between phases

**CRITICAL**: `max_node_distance_*` parameters are ONLY used during construction phase to check paths in the **actor graph**, NOT during discovery phase to filter map graph paths. This allows the hierarchical construction to work correctly by discovering all potential relations first, then deciding which ones to add based on existing paths.

### 2. Hierarchical Processing
- **Leading/Following**: Highest priority, processed first
- **Neighbor**: Medium priority, processed second  
- **Opposite**: Lowest priority, processed last
- **Shortest-First**: Within each type, process shortest paths first
- **Triangle Prevention**: Longer edges are rejected if a shorter path already exists

### 3. Incremental Graph Updates
- **Immediate Updates**: Graph state updated after every edge addition
- **Path Consistency**: All path checks use current graph state
- **No Race Conditions**: Sequential processing prevents timing issues
- **Edge Count Semantics**: `max_node_distance` refers to number of EDGES, not nodes (len(path) - 1)

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
- **Directional Logic**: Each direction checked **independently** (A→B and B→A can be added separately)
- **Rationale**: Leading/following are directional relationships with different semantics (A follows B ≠ B follows A)

#### `_add_neighbor_edges()`
- **Processing**: Neighbor relations (bidirectional)
- **Sorting**: By abs(path_length) (handles negatives)
- **Path Checking**: Uses max_node_distance_neighbor
- **Directional Logic**: Each direction checked **independently** (A→B and B→A can be added separately)
- **Rationale**: Neighbor relationships can be asymmetric (forward/backward neighbor)

#### `_add_opposite_edges()`
- **Processing**: Opposite relations (bidirectional)
- **Sorting**: By abs(path_length) (handles negatives)
- **Path Checking**: Uses max_node_distance_opposite
- **Distance Limits**: Separate forward/backward limits
- **Bidirectional Logic**: If EITHER direction has an existing path, NEITHER direction is added (prevents triangles)
- **Rationale**: Opposite is inherently symmetric (A opposes B ⟺ B opposes A), so both directions must be treated together

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

## Known Issues and Fixes

### Triangle Pattern in Opposite Relations (FIXED)
**Problem**: Actors forming triangles where all three pairs had opposing edges, even though the longest edge should have been rejected by hierarchical construction.

**Root Cause**: The `_has_path_within_distance()` method was checking `len(path) <= max_distance` (number of nodes) instead of `len(path) - 1 <= max_distance` (number of edges). For a path A → B → C:
- Number of nodes: 3
- Number of edges: 2
- With `max_node_distance_opposite=2`, the check `3 <= 2` failed, so the path wasn't detected
- This caused the third edge to be added, creating a triangle

**Fix Applied**:
1. Changed `_has_path_within_distance()` to check `len(path) - 1 <= max_distance` (lines 382-407)
   - This fix applies to ALL relation types (leading, neighbor, opposite)
   - Updated edge case handling for `max_distance == 0`
2. Updated `_add_opposite_edges()` to properly check both directions together (lines 774-778)
   - If either direction has a path, neither direction is added
   - This is specific to opposite relations due to their symmetric nature

**Result**: Hierarchical construction now works correctly for all relation types. For the opposite triangle case (actors 204, 208, 217):
- Adds 204 ↔ 208 (shortest: 37m)
- Adds 204 ↔ 217 (second: 39m)
- Rejects 208 ↔ 217 (longest: 53m) because path 208 → 204 → 217 already exists with 2 edges ≤ max_node_distance_opposite(2)

**Verification**: Tested across all timesteps - no triangles found in any relation type (following_lead, leading_vehicle, neighbor_vehicle, opposite_vehicle)

### Euclidean Distance Validation
**Enhancement**: Added Euclidean distance checks for all relation types (following, neighbor, opposite) to prevent relations between actors that are geometrically far apart even if lane-based path length suggests they're close. This handles curved roads and complex intersections correctly.

## Testing Strategy

### Unit Testing
- **Individual Methods**: Test each phase independently
- **Edge Cases**: Test with empty data, single actors, etc.
- **Parameter Validation**: Test boundary conditions
- **Triangle Detection**: Verify no opposing triangles exist in output graphs

### Integration Testing
- **End-to-End**: Test complete graph creation pipeline
- **Data Consistency**: Verify graph structure matches input data
- **Performance**: Test with realistic dataset sizes
- **Hierarchical Behavior**: Verify shortest paths are added first and longer redundant paths are rejected

### Regression Testing
- **Graph Properties**: Verify consistent edge/node counts
- **Path Properties**: Ensure shortest paths are maintained
- **Parameter Sensitivity**: Test different distance limit combinations
- **Triangle Prevention**: Ensure no opposing/neighbor/following triangles form
