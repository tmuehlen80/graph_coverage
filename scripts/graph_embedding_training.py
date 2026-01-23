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

# python scripts/graph_embedding_training.py --config configs/embeddings/training_config_optimized.json

# Try to import DataLoader from loader (newer PyG) or data (older PyG)
try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))  # For unpickling graphs that reference graph_creator module

from src.graph_creator.graph_embeddings import (
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

    # Optimize DataLoader with workers
    num_workers = config['training'].get('num_workers', 4)
    logger.info(f"Using {num_workers} DataLoader workers")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

    # NEW: Additional training features
    use_warmup = config['training'].get('use_warmup', False)
    warmup_epochs = config['training'].get('warmup_epochs', 3)
    use_cosine_annealing = config['training'].get('use_cosine_annealing', False)
    early_stopping_patience = config['training'].get('early_stopping_patience', None)
    use_mixed_precision = config['training'].get('use_mixed_precision', False)

    # Augmentation and loss hyperparams
    node_noise = config.get('augmentation', {}).get('node_noise', 0.1)
    edge_noise = config.get('augmentation', {}).get('edge_noise', 0.1)
    drop_edge_prob = config.get('augmentation', {}).get('drop_edge_prob', 0.0)
    loss_temperature = config.get('loss', {}).get('temperature', 0.1)

    # NEW: Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
    if use_mixed_precision:
        logger.info("Using mixed precision training")

    # NEW: Early stopping setup
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(run_dir, "best_model.pt")

    # Initial Evaluation (Pre-training)
    logger.info("Evaluating pre-training loss...")
    for split in data_loaders:
        total_loss = 0
        with torch.no_grad():
            batch_count = 0
            # Limit pre-training eval to avoid waiting too long
            PRETRAIN_EVAL_LIMIT = 100
            
            for batch in tqdm(data_loaders[split], desc=f"Pre-train {split}"):
                # Batch comes as (data, path), so we take batch[0]
                batch_data = batch[0].to(device)
                aug_batch = augment_graph(
                    batch_data,
                    node_noise=node_noise,
                    edge_noise=edge_noise,
                    drop_edge_prob=drop_edge_prob
                ).to(device)

                outputs1 = model(batch_data)
                outputs2 = model(aug_batch)
                loss = contrastive_loss(outputs1['projection'], outputs2['projection'], temperature=loss_temperature)
                total_loss += loss.item()
                
                batch_count += 1
                if batch_count >= PRETRAIN_EVAL_LIMIT:
                    logger.info(f"Limited pre-training eval to {PRETRAIN_EVAL_LIMIT} batches")
                    break

        avg_loss = total_loss / len(data_loaders[split])
        total_losses[split].append(avg_loss)

    logger.info(f"Pre-training losses: {total_losses}")

    # Training Loop
    num_loops = config['training']['num_loops']
    epochs_per_loop = config['training']['epochs_per_loop']
    lr_decay = config['training']['lr_decay']

    # Track global epoch for warmup
    global_epoch = 0

    for i in range(num_loops):
        base_lr = initial_lr * (lr_decay ** i)

        # Optimizer is re-initialized every outer loop
        if optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

        # NEW: Cosine annealing scheduler within each loop
        scheduler = None
        if use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs_per_loop,
                eta_min=base_lr * 0.1
            )

        logger.info(f"loop {i}, Base Learning rate: {base_lr}, Optimizer: {optimizer_name}, weight_decay: {weight_decay}")
        if use_cosine_annealing:
            logger.info(f"Using cosine annealing within loop")

        for epoch in range(epochs_per_loop):
            # NEW: Warmup learning rate adjustment
            current_lr = base_lr
            if use_warmup and global_epoch < warmup_epochs:
                warmup_factor = (global_epoch + 1) / warmup_epochs
                current_lr = base_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                logger.info(f"Warmup active: epoch {global_epoch}, warmup_factor: {warmup_factor:.3f}")

            model.train()
            train_loss_sum = 0
            for batch in tqdm(train_loader, desc=f"Loop {i} Epoch {epoch}"):
                batch_data = batch[0].to(device)
                aug_batch = augment_graph(
                    batch_data,
                    node_noise=node_noise,
                    edge_noise=edge_noise,
                    drop_edge_prob=drop_edge_prob
                ).to(device)

                optimizer.zero_grad()

                # NEW: Mixed precision training
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs1 = model(batch_data)
                        outputs2 = model(aug_batch)
                        loss = contrastive_loss(outputs1['projection'], outputs2['projection'], temperature=loss_temperature)

                    scaler.scale(loss).backward()

                    # Unscale before gradient clipping
                    if gradient_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
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

            # NEW: Step cosine annealing scheduler (only after warmup)
            if scheduler is not None and (not use_warmup or global_epoch >= warmup_epochs):
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

            # Record LR
            total_losses["lr"].append(current_lr)

            # Validation/Evaluation
            model.eval()
            for split in data_loaders:
                total_loss = 0
                with torch.no_grad():
                    for batch in data_loaders[split]:
                        batch_data = batch[0].to(device)
                        aug_batch = augment_graph(
                            batch_data,
                            node_noise=node_noise,
                            edge_noise=edge_noise,
                            drop_edge_prob=drop_edge_prob
                        ).to(device)

                        outputs1 = model(batch_data)
                        outputs2 = model(aug_batch)
                        loss = contrastive_loss(outputs1['projection'], outputs2['projection'], temperature=loss_temperature)
                        total_loss += loss.item()

                avg_split_loss = total_loss / len(data_loaders[split])
                total_losses[split].append(avg_split_loss)

            logger.info(f"Current Losses - Train: {total_losses['train'][-1]:.4f}, Test: {total_losses['test'][-1]:.4f}")
            logger.info(f"Current lr: {current_lr:.6f}")

            # NEW: Early stopping check
            if early_stopping_patience is not None:
                if total_losses['test'][-1] < best_test_loss:
                    best_test_loss = total_losses['test'][-1]
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'loop': i,
                        'epoch': epoch,
                        'global_epoch': global_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': total_losses,
                        'best_test_loss': best_test_loss
                    }, best_model_path)
                    logger.info(f"New best test loss: {best_test_loss:.4f} - Saved to {best_model_path}")
                else:
                    patience_counter += 1
                    logger.info(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered - no improvement in validation loss")
                        # Save final checkpoint before stopping
                        checkpoint_path = os.path.join(run_dir, f"checkpoint_loop{i}_epoch{epoch}_final.pt")
                        torch.save({
                            'loop': i,
                            'epoch': epoch,
                            'global_epoch': global_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'losses': total_losses,
                            'early_stopped': True
                        }, checkpoint_path)
                        logger.info(f"Saved final checkpoint to {checkpoint_path}")
                        return

            # Checkpointing
            checkpoint_path = os.path.join(run_dir, f"checkpoint_loop{i}_epoch{epoch}.pt")
            torch.save({
                'loop': i,
                'epoch': epoch,
                'global_epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': total_losses
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            global_epoch += 1

    logger.info("Training completed successfully")
    logger.info(f"Best test loss achieved: {best_test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Embeddings (V2 - Enhanced)")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)

    train(config)
