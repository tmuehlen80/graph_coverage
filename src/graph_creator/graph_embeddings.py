import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from enum import Enum

import pickle
import networkx as nx
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool

import torch
import torch.nn.functional as F
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import glob
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset


# Define mappings for one-hot encodings
ACTOR_TYPE_MAPPING = {
    'VEHICLE': 0,
    'PEDESTRIAN': 1,
    'CYCLIST': 2,
    'MOTORCYCLE': 3,
}

EDGE_TYPE_MAPPING = {
    'neighbor_vehicle': 0,
    'opposite_vehicle': 1,
    'same_lane': 2,
    'adjacent_lane': 3,
    'following': 4,
    'intersection': 5,
}




# Helper function to get feature dimensions
def get_feature_dimensions(actor_type_mapping=None, edge_type_mapping=None):
    """Get the dimensions of node and edge features"""
    if actor_type_mapping is None:
        actor_type_mapping = ACTOR_TYPE_MAPPING
    if edge_type_mapping is None:
        edge_type_mapping = EDGE_TYPE_MAPPING

    # Node features: 1 (lon_speed) + num_actor_types (one-hot) + 2 (boolean attributes)
    node_features = 1 + len(actor_type_mapping) + 2

    # Edge features: 1 (path_length) + num_edge_types (one-hot)
    edge_features = 1 + len(edge_type_mapping)

    return node_features, edge_features

def networkx_to_pyg(nx_graph, actor_type_mapping=None, edge_type_mapping=None):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        nx_graph: NetworkX graph with node and edge attributes
        actor_type_mapping: Dict mapping actor types to indices
        edge_type_mapping: Dict mapping edge types to indices
    
    Returns:
        PyTorch Geometric Data object
    """
    
    # Use default mappings if none provided
    if actor_type_mapping is None:
        actor_type_mapping = ACTOR_TYPE_MAPPING
    if edge_type_mapping is None:
        edge_type_mapping = EDGE_TYPE_MAPPING
    
    # Get node mapping (NetworkX nodes might not be sequential integers)
    nodes = list(nx_graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Extract node features
    node_features = []
    for node in nodes:
        node_data = nx_graph.nodes[node]

        # Extract lon_speed (continuous feature)
        lon_speed = node_data.get('lon_speed', 0.0)

        # Extract actor_type and convert to one-hot
        actor_type = node_data.get('actor_type')
        if hasattr(actor_type, 'value'):  # Handle enum
            actor_type_str = actor_type.value
        else:
            actor_type_str = str(actor_type)

        # Get actor type index
        actor_type_idx = actor_type_mapping.get(actor_type_str, 0)  # Default to 0 if unknown

        # Create one-hot encoding for actor type
        num_actor_types = len(actor_type_mapping)
        actor_onehot = [0.0] * num_actor_types
        actor_onehot[actor_type_idx] = 1.0

        # Extract boolean attributes
        lane_change = float(node_data.get('lane_change', False))  # Convert boolean to float
        is_on_intersection = float(node_data.get('is_on_intersection', False))  # Convert boolean to float

        # Combine features: [lon_speed, actor_type_onehot..., lane_change, is_on_intersection]
        node_feature = [lon_speed] + actor_onehot + [lane_change, is_on_intersection]
        node_features.append(node_feature)
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Extract edges and edge features
    edge_index = []
    edge_features = []
    
    for source, target, edge_data in nx_graph.edges(data=True):
        # Map node IDs to indices
        source_idx = node_mapping[source]
        target_idx = node_mapping[target]
        
        edge_index.append([source_idx, target_idx])
        
        # Extract path_length (continuous feature)
        path_length = edge_data.get('path_length', 0.0)
        
        # Extract edge_type and convert to one-hot
        edge_type = edge_data.get('edge_type', 'unknown')
        edge_type_idx = edge_type_mapping.get(edge_type, 0)  # Default to 0 if unknown
        
        # Create one-hot encoding for edge type
        num_edge_types = len(edge_type_mapping)
        edge_onehot = [0.0] * num_edge_types
        edge_onehot[edge_type_idx] = 1.0
        
        # Combine features: [path_length, edge_type_onehot...]
        edge_feature = [path_length] + edge_onehot
        edge_features.append(edge_feature)
    
    # Convert to tensors
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        # Handle graphs with no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1 + len(edge_type_mapping)), dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data



class GraphDataset(Dataset):
    def __init__(self, graph_paths):
        self.graph_paths = graph_paths
        
    def __len__(self):
        return len(self.graph_paths)
    
    def __getitem__(self, idx):
        file_path = self.graph_paths[idx]
        with open(file_path, 'rb') as f:
            nx_graph = pickle.load(f)
        pyg_data = networkx_to_pyg(nx_graph)
        return pyg_data, file_path


# Enhanced model with training capabilities
class TrainableGraphGINE(torch.nn.Module):
    def __init__(self, node_features, edge_features, embedding_dim=128, hidden_dim=64, num_layers=3, num_classes=None):
        """
        GINE model for graph-level embeddings with optional classification
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features  
            embedding_dim: Final embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GINE layers
            num_classes: Number of classes for supervised learning (None for unsupervised)
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # GINE layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            GINEConv(
                torch.nn.Sequential(
                    torch.nn.Linear(node_features, hidden_dim),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_features
            )
        )
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.BatchNorm1d(hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim)
                    ),
                    edge_dim=edge_features
                )
            )
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Graph-level pooling and embedding projection
        self.embedding_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, embedding_dim),  # *3 for mean+max+sum pooling
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Projection head for contrastive learning (unsupervised)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim // 2)
        )
        
        # Classification head (supervised)
        if num_classes is not None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(embedding_dim, num_classes)
            )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Handle empty edge attributes
        if edge_attr.size(0) == 0:
            edge_attr = None
        
        # GINE layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Graph-level pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        
        # Concatenate pooling strategies
        graph_repr = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
        
        # Final embedding
        embeddings = self.embedding_proj(graph_repr)
        
        outputs = {'embeddings': embeddings}
        
        # Add projection for contrastive learning
        outputs['projection'] = self.projection_head(embeddings)
        
        # Add classification if applicable
        if hasattr(self, 'classifier'):
            outputs['logits'] = self.classifier(embeddings)
        
        return outputs
    
    def get_embedding(self, data):
        """Extract embeddings only"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data)
            return outputs['embeddings']


# Contrastive loss
def contrastive_loss(z1, z2, temperature=0.1):
    """InfoNCE contrastive loss"""
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # Labels for positive pairs (diagonal)
    labels = torch.arange(batch_size, device=z1.device)
    
    return F.cross_entropy(sim_matrix, labels)

# Data augmentation for contrastive learning
def augment_graph(data, node_noise=0.1, edge_noise=0.1):
    """Simple graph augmentation with configurable noise"""
    augmented_data = data.clone()
    
    # Add noise to continuous features
    # Node features: first feature is lon_speed
    if augmented_data.x.size(1) > 0 and node_noise > 0:
        augmented_data.x[:, 0] += torch.randn_like(augmented_data.x[:, 0]) * node_noise
    
    # Edge features: first feature is path_length
    if augmented_data.edge_attr.size(0) > 0 and augmented_data.edge_attr.size(1) > 0 and edge_noise > 0:
        augmented_data.edge_attr[:, 0] += torch.randn_like(augmented_data.edge_attr[:, 0]) * edge_noise
    
    return augmented_data

