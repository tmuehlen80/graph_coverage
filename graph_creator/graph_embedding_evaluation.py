"""
Graph Embedding Evaluation Module

This module provides functions to evaluate the quality of graph embeddings
independent of the training noise level. These metrics help compare models
trained with different augmentation settings.

Metrics implemented:
1. Silhouette Score - Clustering quality
2. Nearest Neighbor Consistency - Stability under augmentation
3. Embedding Space Analysis - Distribution statistics
4. Retrieval Metrics - Precision@K for similar graphs
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any


def compute_embeddings(model, data_loader, device: str = 'cuda') -> Tuple[np.ndarray, List[str]]:
    """
    Extract embeddings for all graphs in a data loader.

    Args:
        model: Trained TrainableGraphGINE model
        data_loader: PyTorch Geometric DataLoader
        device: Device to use for computation

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        paths: list of file paths for each graph
    """
    model.eval()
    model.to(device)

    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing embeddings"):
            batch_data, paths = batch
            batch_data = batch_data.to(device)

            outputs = model(batch_data)
            embeddings = outputs['embeddings'].cpu().numpy()

            all_embeddings.append(embeddings)
            all_paths.extend(paths)

    return np.vstack(all_embeddings), all_paths


def nearest_neighbor_consistency(
    model,
    data_loader,
    augment_fn,
    device: str = 'cuda',
    n_augmentations: int = 5,
    k: int = 5,
    noise_levels: List[float] = [0.05, 0.08, 0.1]
) -> Dict[str, float]:
    """
    Measure how consistently a graph's nearest neighbors remain the same
    when the graph is augmented. High consistency = robust embeddings.

    Args:
        model: Trained model
        data_loader: DataLoader with graphs
        augment_fn: Augmentation function (augment_graph)
        device: Computation device
        n_augmentations: Number of augmented versions per graph
        k: Number of nearest neighbors to consider
        noise_levels: Different noise levels to test

    Returns:
        Dictionary with consistency scores for each noise level
    """
    model.eval()
    model.to(device)

    # First, compute original embeddings
    original_embeddings = []
    all_batch_data = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing original embeddings"):
            batch_data = batch[0].to(device)
            all_batch_data.append(batch_data)
            outputs = model(batch_data)
            original_embeddings.append(outputs['embeddings'].cpu())

    original_embeddings = torch.cat(original_embeddings, dim=0)
    original_embeddings = F.normalize(original_embeddings, dim=1)

    # Compute original nearest neighbors
    sim_matrix = torch.mm(original_embeddings, original_embeddings.t())
    # Set diagonal to -inf to exclude self
    sim_matrix.fill_diagonal_(-float('inf'))
    _, original_neighbors = torch.topk(sim_matrix, k=k, dim=1)

    results = {}

    for noise in noise_levels:
        consistencies = []

        for _ in range(n_augmentations):
            augmented_embeddings = []

            with torch.no_grad():
                for batch_data in all_batch_data:
                    aug_batch = augment_fn(batch_data, node_noise=noise, edge_noise=noise).to(device)
                    outputs = model(aug_batch)
                    augmented_embeddings.append(outputs['embeddings'].cpu())

            augmented_embeddings = torch.cat(augmented_embeddings, dim=0)
            augmented_embeddings = F.normalize(augmented_embeddings, dim=1)

            # Compute augmented nearest neighbors
            aug_sim_matrix = torch.mm(augmented_embeddings, original_embeddings.t())
            # Set diagonal to -inf
            aug_sim_matrix.fill_diagonal_(-float('inf'))
            _, aug_neighbors = torch.topk(aug_sim_matrix, k=k, dim=1)

            # Compute overlap between original and augmented neighbors
            overlap = 0
            for i in range(len(original_neighbors)):
                orig_set = set(original_neighbors[i].numpy())
                aug_set = set(aug_neighbors[i].numpy())
                overlap += len(orig_set & aug_set) / k

            consistencies.append(overlap / len(original_neighbors))

        results[f'nn_consistency_noise_{noise}'] = np.mean(consistencies)
        results[f'nn_consistency_std_noise_{noise}'] = np.std(consistencies)

    return results


def clustering_quality(
    embeddings: np.ndarray,
    n_clusters_range: List[int] = [5, 10, 15, 20, 30]
) -> Dict[str, Any]:
    """
    Evaluate embedding quality using clustering metrics.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        n_clusters_range: List of cluster numbers to try

    Returns:
        Dictionary with clustering metrics
    """
    results = {
        'silhouette_scores': {},
        'calinski_harabasz_scores': {},
        'davies_bouldin_scores': {},
        'best_n_clusters': None,
        'best_silhouette': -1
    }

    for n_clusters in tqdm(n_clusters_range, desc="Evaluating clustering"):
        if n_clusters >= len(embeddings):
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        sil_score = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)

        results['silhouette_scores'][n_clusters] = sil_score
        results['calinski_harabasz_scores'][n_clusters] = ch_score
        results['davies_bouldin_scores'][n_clusters] = db_score

        if sil_score > results['best_silhouette']:
            results['best_silhouette'] = sil_score
            results['best_n_clusters'] = n_clusters

    return results


def embedding_space_analysis(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Analyze the distribution and properties of the embedding space.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)

    Returns:
        Dictionary with space analysis metrics
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1)
    normalized = embeddings / (norms[:, np.newaxis] + 1e-8)

    # Compute pairwise cosine similarities
    sim_matrix = np.dot(normalized, normalized.T)
    # Get upper triangle (excluding diagonal)
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    # Compute statistics
    results = {
        'mean_cosine_similarity': float(np.mean(upper_tri)),
        'std_cosine_similarity': float(np.std(upper_tri)),
        'min_cosine_similarity': float(np.min(upper_tri)),
        'max_cosine_similarity': float(np.max(upper_tri)),
        'median_cosine_similarity': float(np.median(upper_tri)),

        # Embedding norm statistics
        'mean_embedding_norm': float(np.mean(norms)),
        'std_embedding_norm': float(np.std(norms)),

        # Dimensionality analysis via PCA
        'effective_dimensionality': compute_effective_dimensionality(embeddings),

        # Uniformity (how well embeddings use the space)
        'uniformity': compute_uniformity(normalized),
    }

    return results


def compute_effective_dimensionality(embeddings: np.ndarray, threshold: float = 0.95) -> int:
    """
    Compute effective dimensionality as number of PCA components
    needed to explain threshold variance.
    """
    pca = PCA()
    pca.fit(embeddings)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumsum, threshold) + 1)


def compute_uniformity(normalized_embeddings: np.ndarray, t: float = 2.0) -> float:
    """
    Compute uniformity loss (Wang & Isola, 2020).
    Lower values indicate more uniform distribution on hypersphere.
    """
    sq_pdist = np.sum((normalized_embeddings[:, None] - normalized_embeddings[None, :]) ** 2, axis=-1)
    # Sample to avoid memory issues
    if len(normalized_embeddings) > 5000:
        idx = np.random.choice(len(normalized_embeddings), 5000, replace=False)
        sq_pdist = sq_pdist[idx][:, idx]

    return float(np.log(np.mean(np.exp(-t * sq_pdist))))


def retrieval_precision_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute retrieval precision@k given ground truth labels.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: Ground truth labels for each embedding
        k_values: List of k values for precision@k

    Returns:
        Dictionary with precision@k for each k
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)

    # Compute similarity matrix
    sim_matrix = np.dot(normalized, normalized.T)
    np.fill_diagonal(sim_matrix, -np.inf)

    results = {}

    for k in k_values:
        if k >= len(embeddings):
            continue

        # Get top-k indices for each sample
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]

        # Compute precision
        precisions = []
        for i, neighbors in enumerate(top_k_indices):
            same_label = np.sum(labels[neighbors] == labels[i])
            precisions.append(same_label / k)

        results[f'precision@{k}'] = float(np.mean(precisions))

    return results


def plot_embedding_visualization(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    title: str = 'Embedding Visualization',
    figsize: Tuple[int, int] = (12, 10),
    sample_size: int = 5000
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE or PCA.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        labels: Optional labels for coloring
        method: 'tsne' or 'pca'
        title: Plot title
        figsize: Figure size
        sample_size: Max samples to visualize (for performance)

    Returns:
        matplotlib Figure
    """
    # Sample if too large
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]
        if labels is not None:
            labels = labels[idx]

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=10)

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title)

    return fig


def plot_similarity_distribution(
    embeddings: np.ndarray,
    title: str = 'Pairwise Cosine Similarity Distribution',
    figsize: Tuple[int, int] = (10, 6),
    sample_size: int = 3000
) -> plt.Figure:
    """
    Plot histogram of pairwise cosine similarities.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        title: Plot title
        figsize: Figure size
        sample_size: Max samples to use

    Returns:
        matplotlib Figure
    """
    # Sample if too large
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]

    # Normalize and compute similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    sim_matrix = np.dot(normalized, normalized.T)

    # Get upper triangle
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(upper_tri, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(upper_tri), color='red', linestyle='--', label=f'Mean: {np.mean(upper_tri):.3f}')
    ax.axvline(np.median(upper_tri), color='green', linestyle='--', label=f'Median: {np.median(upper_tri):.3f}')

    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    return fig


def plot_clustering_metrics(
    clustering_results: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot clustering quality metrics across different numbers of clusters.

    Args:
        clustering_results: Output from clustering_quality()
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Silhouette Score (higher is better)
    n_clusters = list(clustering_results['silhouette_scores'].keys())
    silhouette = list(clustering_results['silhouette_scores'].values())
    axes[0].plot(n_clusters, silhouette, 'bo-')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score (higher = better)')
    axes[0].axvline(clustering_results['best_n_clusters'], color='red', linestyle='--', alpha=0.5)

    # Calinski-Harabasz Score (higher is better)
    ch_scores = list(clustering_results['calinski_harabasz_scores'].values())
    axes[1].plot(n_clusters, ch_scores, 'go-')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Calinski-Harabasz Score')
    axes[1].set_title('Calinski-Harabasz (higher = better)')

    # Davies-Bouldin Score (lower is better)
    db_scores = list(clustering_results['davies_bouldin_scores'].values())
    axes[2].plot(n_clusters, db_scores, 'ro-')
    axes[2].set_xlabel('Number of Clusters')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('Davies-Bouldin (lower = better)')

    plt.tight_layout()
    return fig


def compare_models(
    models_dict: Dict[str, torch.nn.Module],
    data_loader,
    augment_fn,
    device: str = 'cuda'
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models across all evaluation metrics.

    Args:
        models_dict: Dictionary mapping model names to models
        data_loader: DataLoader with evaluation data
        augment_fn: Augmentation function
        device: Computation device

    Returns:
        Dictionary with results for each model
    """
    results = {}

    for name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating model: {name}")
        print('='*50)

        # Compute embeddings
        embeddings, paths = compute_embeddings(model, data_loader, device)

        # Run all evaluations
        model_results = {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1],
        }

        # Embedding space analysis
        print("Running embedding space analysis...")
        model_results['space_analysis'] = embedding_space_analysis(embeddings)

        # Clustering quality
        print("Running clustering analysis...")
        model_results['clustering'] = clustering_quality(embeddings)

        # Nearest neighbor consistency
        print("Running nearest neighbor consistency...")
        model_results['nn_consistency'] = nearest_neighbor_consistency(
            model, data_loader, augment_fn, device
        )

        results[name] = model_results

    return results


def print_comparison_summary(comparison_results: Dict[str, Dict[str, Any]]):
    """Print a formatted summary of model comparison results."""

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    # Header
    models = list(comparison_results.keys())
    print(f"\n{'Metric':<40} | " + " | ".join(f"{m:<15}" for m in models))
    print("-"*80)

    # Embedding space metrics
    metrics = [
        ('mean_cosine_similarity', 'Mean Cosine Similarity'),
        ('std_cosine_similarity', 'Std Cosine Similarity'),
        ('effective_dimensionality', 'Effective Dimensionality'),
        ('uniformity', 'Uniformity (lower=better)'),
    ]

    for key, label in metrics:
        values = [comparison_results[m]['space_analysis'][key] for m in models]
        print(f"{label:<40} | " + " | ".join(f"{v:<15.4f}" for v in values))

    # Clustering metrics
    print("-"*80)
    for m in models:
        best_k = comparison_results[m]['clustering']['best_n_clusters']
        best_sil = comparison_results[m]['clustering']['best_silhouette']
        print(f"Best Silhouette ({m}): {best_sil:.4f} at k={best_k}")

    # NN Consistency
    print("-"*80)
    for noise in [0.05, 0.08, 0.1]:
        key = f'nn_consistency_noise_{noise}'
        if key in comparison_results[models[0]]['nn_consistency']:
            values = [comparison_results[m]['nn_consistency'][key] for m in models]
            print(f"{'NN Consistency (noise=' + str(noise) + ')':<40} | " + " | ".join(f"{v:<15.4f}" for v in values))

    print("="*80)
