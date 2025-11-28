import glob
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Try to import DataLoader from loader (newer PyG) or data (older PyG)
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from graph_creator.graph_embeddings import (
    GraphDataset,
    TrainableGraphGINE,
    contrastive_loss,
    augment_graph,
    get_feature_dimensions,
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train():
    # Paths
    carla_pattern = os.path.join(PROJECT_ROOT, "actor_graphs/carla_actor_graph_setting_1_50_50_10_20_20_4_4_4_components_nx/*.pkl")
    argoverse_pattern = os.path.join(PROJECT_ROOT, "actor_graphs/argoverse_actor_graph_setting_1_50_50_10_20_20_4_4_4_components_nx/*.pkl")
    
    print("Looking for graphs...")
    graph_paths = glob.glob(carla_pattern)
    argoverse_graph_paths = glob.glob(argoverse_pattern)
    graph_paths.extend(argoverse_graph_paths)
    
    print(f"Found {len(graph_paths)} graphs")

    # Checkpoints
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    ensure_dir(checkpoint_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    print(f"Using device: {device}")

    # Data Loading
    graph_ds = GraphDataset(graph_paths)
    indices = np.arange(len(graph_ds))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    train_ds = Subset(graph_ds, train_idx)
    test_ds = Subset(graph_ds, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)
    
    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")
    data_loaders = {"train": train_loader, "test": test_loader}

    # Model
    node_dim, edge_dim = get_feature_dimensions()
    # model = TrainableGraphGINE(node_dim, edge_dim, 256, 96, 4).to(device)
    model = TrainableGraphGINE(node_dim, edge_dim, 256, 128, 6).to(device)
    initial_lr = 0.4
    total_losses = {"train": [], "test": [], "lr": [initial_lr]}

    # Initial Evaluation (Pre-training)
    print("Evaluating pre-training loss...")
    for split in data_loaders:
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(data_loaders[split], desc=f"Pre-train {split}"):
                # Batch comes as (data, path), so we take batch[0]
                batch_data = batch[0].to(device)
                aug_batch = augment_graph(batch_data).to(device)
                
                outputs1 = model(batch_data)
                outputs2 = model(aug_batch)
                loss = contrastive_loss(outputs1['projection'], outputs2['projection'])
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loaders[split])
        total_losses[split].append(avg_loss)
    
    print(f"Pre-training losses: {total_losses}")

    # Training Loop
    for i in range(10):
        lr = initial_lr * 0.75**i
        # Optimizer is re-initialized every outer loop in the notebook
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print(f"loop {i}, Learning rate: {lr}")
        
        for epoch in range(12):
            model.train()
            train_loss_sum = 0
            for batch in tqdm(train_loader, desc=f"Loop {i} Epoch {epoch}"):
                batch_data = batch[0].to(device)
                aug_batch = augment_graph(batch_data).to(device)
                
                optimizer.zero_grad()
                outputs1 = model(batch_data)
                outputs2 = model(aug_batch)
                loss = contrastive_loss(outputs1['projection'], outputs2['projection'])
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)
            print(f'Epoch {epoch}, Loss: {avg_train_loss:.4f}')
            
            # Record LR
            total_losses["lr"].append(lr)

            # Validation/Evaluation
            model.eval()
            for split in data_loaders:
                total_loss = 0
                with torch.no_grad():
                    for batch in data_loaders[split]:
                        batch_data = batch[0].to(device)
                        aug_batch = augment_graph(batch_data).to(device)
                        
                        outputs1 = model(batch_data)
                        outputs2 = model(aug_batch)
                        loss = contrastive_loss(outputs1['projection'], outputs2['projection'])
                        total_loss += loss.item()
                
                avg_split_loss = total_loss / len(data_loaders[split])
                total_losses[split].append(avg_split_loss)
            
            print(f"Current Losses - Train: {total_losses['train'][-1]:.4f}, Test: {total_losses['test'][-1]:.4f}")

            # Checkpointing
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_loop{i}_epoch{epoch}.pt")
            torch.save({
                'loop': i,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': total_losses
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()

