# Graph Coverage

A graph-based framework for coverage analysis in autonomous driving. This project converts traffic scenes into graph representations and analyzes them using graph neural networks and subgraph pattern matching to measure test coverage of autonomous driving scenarios.

## Overview

The framework enables researchers to:
- **Create Traffic Scene Graphs** - Represent actors (vehicles, pedestrians, cyclists) as nodes and their spatial interactions (following, neighboring, opposite) as edges
- **Build Map Graphs** - Model road networks as graphs with lanes as nodes and connections as edges
- **Learn Graph Embeddings** - Use Graph Isomorphism Networks with Edge attributes (GINE) for traffic scene representation
- **Analyze Coverage** - Extract and identify common subgraph patterns to measure scenario coverage

## Project Structure

```
graph_coverage/
├── src/
│   ├── graph_creator/          # Core graph creation and ML
│   │   ├── ActorGraph.py       # Actor interaction graphs
│   │   ├── MapGraph.py         # Road network graphs
│   │   ├── ActorTimeGraph.py   # Temporal graphs across timesteps
│   │   ├── graph_embeddings.py # GINE model and training utilities
│   │   ├── create_graph.py     # Main graph creation API
│   │   └── models.py           # Pydantic data models
│   └── subgraphs/              # Pattern analysis
│       ├── SubgraphExtractor.py
│       ├── SubgraphIsomorphismChecker.py
│       └── subgraph_types.py
├── scripts/                    # Processing and training scripts
├── configs/                    # Configuration files
│   ├── embeddings/             # Training configurations
│   ├── graph_settings/         # Actor graph parameters
│   └── subgraphs/              # Subgraph extraction configs
├── notebooks/                  # Jupyter notebooks for analysis
├── actor_graphs/               # Generated graph files
└── checkpoints/                # Model checkpoints
```

## Datasets

The framework supports two autonomous driving datasets:

- **Argoverse 2** - Real-world motion forecasting dataset with vehicle, pedestrian, and cyclist trajectories
- **CARLA** - Synthetic traffic scenarios from the open-source autonomous driving simulator

## Key Features

- **Multi-dataset support**: Works with both real (Argoverse 2) and simulated (CARLA) data
- **Temporal modeling**: Tracks actor interactions over time with time-lag attributes
- **Spatial relationships**: Four edge types capture following/leading, neighbor, and opposite vehicle relationships
- **Deep learning integration**: GINE-based graph embeddings with contrastive learning
- **Pattern discovery**: Subgraph isomorphism checking for coverage analysis
- **Flexible configuration**: Parameterizable distance thresholds, timesteps, and model hyperparameters

## Installation

This project uses Poetry for dependency management:

```bash
poetry install
```

## Usage

### Creating Actor Graphs

Generate actor graphs from Argoverse 2 scenarios:
```bash
python scripts/argoverse_actor_graph_creation.py
```

Generate actor graphs from CARLA simulation data:
```bash
python scripts/carla_actor_graph_creation.py
```

### Training Graph Embeddings

Train the GINE model for graph embeddings:
```bash
python scripts/graph_embedding_training.py
```

Configuration is loaded from `configs/embeddings/training_config_optimized.json`.

### Extracting Components

Extract connected components from full graphs:
```bash
python scripts/extract_components_from_full_graphs.py
```

## Configuration

### Graph Settings

Actor graph generation parameters are configured in `configs/graph_settings/`:
- `delta_timestep`: Time interval between graph snapshots
- `lead_vehicle_distance`: Maximum distance for lead vehicle detection
- `neighbor_distance_forward/backward`: Distance thresholds for neighboring vehicles
- `opposite_distance_forward/backward`: Distance thresholds for opposite direction vehicles

### Training Configuration

Model training parameters in `configs/embeddings/training_config_optimized.json`:
- Embedding dimension: 192
- Hidden dimension: 384
- Number of layers: 5
- Optimizer: AdamW with learning rate decay
- Contrastive loss with temperature scaling
- Data augmentation (node/edge noise, edge dropping)

## Technologies

- **PyTorch & PyTorch Geometric** - Deep learning and graph neural networks
- **NetworkX** - Graph data structures and algorithms
- **Shapely** - Geometric operations for spatial relationships
- **Argoverse 2 API** - Real-world dataset access
- **CARLA** - Simulation platform integration

## License

MIT License
