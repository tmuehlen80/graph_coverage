import glob
import os
import sys
import pickle
import torch
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# python scripts/graph_embedding_training.py --config configs/embeddings/training_config_default.json

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

def train(config):
    # Setup timestamp and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_checkpoint_dir = os.path.join(PROJECT_ROOT, config['training'].get('checkpoint_dir', 'checkpoints'))
    run_dir = os.path.join(base_checkpoint_dir, timestamp)
    ensure_dir(run_dir)

    # Logger setup
    log_file = os.path.join(run_dir, 'training.log')
    # Configure logger to write to file
    logger.add(log_file, rotation="10 MB")
    logger.info(f"Starting training run: {timestamp}")
    logger.info(f"Run directory: {run_dir}")

    # Save config copy
    config_save_path = os.path.join(run_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved config to {config_save_path}")

    # Paths
    carla_pattern = os.path.join(PROJECT_ROOT, config['data']['carla_pattern'])
    argoverse_pattern = os.path.join(PROJECT_ROOT, config['data']['argoverse_pattern'])
    
    logger.info("Looking for graphs...")
    graph_paths = glob.glob(carla_pattern)
    argoverse_graph_paths = glob.glob(argoverse_pattern)
    graph_paths.extend(argoverse_graph_paths)
    
    logger.info(f"Found {len(graph_paths)} graphs")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data Loading
    graph_ds = GraphDataset(graph_paths)
    indices = np.arange(len(graph_ds))
    
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    batch_size = config['data']['batch_size']

    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)
    
    train_ds = Subset(graph_ds, train_idx)
    test_ds = Subset(graph_ds, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")
    data_loaders = {"train": train_loader, "test": test_loader}

    # Model
    node_dim, edge_dim = get_feature_dimensions()
    
    # Model params (support embedding_dim with fallback to output_dim)
    embedding_dim = config['model'].get('embedding_dim', config['model'].get('output_dim', 128))
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']

    model = TrainableGraphGINE(
        node_dim,
        edge_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    initial_lr = config['training']['initial_lr']
    total_losses = {"train": [], "test": [], "lr": [initial_lr]}

    # Training hyperparams
    weight_decay = config['training'].get('weight_decay', 0.0)
    optimizer_name = config['training'].get('optimizer', 'Adam')
    gradient_clip_norm = config['training'].get('gradient_clip_norm', None)

    # Augmentation and loss hyperparams
    node_noise = config.get('augmentation', {}).get('node_noise', 0.1)
    edge_noise = config.get('augmentation', {}).get('edge_noise', 0.1)
    loss_temperature = config.get('loss', {}).get('temperature', 0.1)

    # Initial Evaluation (Pre-training)
    logger.info("Evaluating pre-training loss...")
    for split in data_loaders:
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(data_loaders[split], desc=f"Pre-train {split}"):
                # Batch comes as (data, path), so we take batch[0]
                batch_data = batch[0].to(device)
                aug_batch = augment_graph(batch_data, node_noise=node_noise, edge_noise=edge_noise).to(device)
                
                outputs1 = model(batch_data)
                outputs2 = model(aug_batch)
                loss = contrastive_loss(outputs1['projection'], outputs2['projection'], temperature=loss_temperature)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loaders[split])
        total_losses[split].append(avg_loss)
    
    logger.info(f"Pre-training losses: {total_losses}")

    # Training Loop
    num_loops = config['training']['num_loops']
    epochs_per_loop = config['training']['epochs_per_loop']
    lr_decay = config['training']['lr_decay']

    for i in range(num_loops):
        lr = initial_lr * (lr_decay ** i)
        # Optimizer is re-initialized every outer loop
        if optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info(f"loop {i}, Learning rate: {lr}, Optimizer: {optimizer_name}, weight_decay: {weight_decay}")
        
        for epoch in range(epochs_per_loop):
            model.train()
            train_loss_sum = 0
            for batch in tqdm(train_loader, desc=f"Loop {i} Epoch {epoch}"):
                batch_data = batch[0].to(device)
                aug_batch = augment_graph(batch_data, node_noise=node_noise, edge_noise=edge_noise).to(device)
                
                optimizer.zero_grad()
                outputs1 = model(batch_data)
                outputs2 = model(aug_batch)
                loss = contrastive_loss(outputs1['projection'], outputs2['projection'], temperature=loss_temperature)
                loss.backward()
                if gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)
            logger.info(f'Epoch {epoch}, Loss: {avg_train_loss:.4f}')
            
            # Record LR
            total_losses["lr"].append(lr)

            # Validation/Evaluation
            model.eval()
            for split in data_loaders:
                total_loss = 0
                with torch.no_grad():
                    for batch in data_loaders[split]:
                        batch_data = batch[0].to(device)
                        aug_batch = augment_graph(batch_data, node_noise=node_noise, edge_noise=edge_noise).to(device)
                        
                        outputs1 = model(batch_data)
                        outputs2 = model(aug_batch)
                        loss = contrastive_loss(outputs1['projection'], outputs2['projection'], temperature=loss_temperature)
                        total_loss += loss.item()
                
                avg_split_loss = total_loss / len(data_loaders[split])
                total_losses[split].append(avg_split_loss)
            
            logger.info(f"Current Losses - Train: {total_losses['train'][-1]:.4f}, Test: {total_losses['test'][-1]:.4f}")
            logger.info(f"Current lr: {lr:.4f}")

            # Checkpointing
            checkpoint_path = os.path.join(run_dir, f"checkpoint_loop{i}_epoch{epoch}.pt")
            torch.save({
                'loop': i,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': total_losses
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Embeddings")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train(config)
