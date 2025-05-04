import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import Counter
from typing import Dict, List, Callable, Set, Tuple, Any
import networkx.algorithms.isomorphism as iso
import math
import os

class SubgraphExtractor:
    """
    A class for extracting common subgraphs from a collection of graphs and
    representing each graph using these common subgraphs.
    """
    
    def __init__(self, 
                 subgraph_selection_strategy: str = 'frequency',
                 min_subgraph_size: int = 2,
                 max_subgraph_size: int = None,
                 node_match=None, 
                 edge_match=None):
        """
        Initialize the SubgraphExtractor.
        
        Args:
            subgraph_selection_strategy: Strategy for selecting subgraphs ('frequency', 
                                         'density', 'coverage', 'complexity')
            min_subgraph_size: Minimum number of nodes in a subgraph
            max_subgraph_size: Maximum number of nodes in a subgraph (None for no limit)
            node_match: Function for node matching in isomorphism check
            edge_match: Function for edge matching in isomorphism check
        """
        self.min_subgraph_size = min_subgraph_size
        self.max_subgraph_size = max_subgraph_size
        
        # Default node and edge matchers that consider all attributes
        if node_match is None:
            self.node_match = iso.categorical_node_match(['*'], [None])
        else:
            self.node_match = node_match
            
        if edge_match is None:
            self.edge_match = iso.categorical_edge_match(['edge_type'], [None])
        else:
            self.edge_match = edge_match
            
        # Set the subgraph selection strategy
        self.set_selection_strategy(subgraph_selection_strategy)
        
        self.subgraph_library = {}  # Maps subgraph ID to subgraph
        self.subgraph_frequency = {}  # Maps subgraph ID to frequency
        
    def set_selection_strategy(self, strategy: str):
        """
        Set the strategy for selecting subgraphs.
        
        Args:
            strategy: One of 'frequency', 'density', 'coverage', 'complexity'
        """
        STRATEGIES = {
            'frequency': self._rank_by_frequency,
            'density': self._rank_by_density,
            'coverage': self._rank_by_coverage,
            'complexity': self._rank_by_complexity,
            'size': self._rank_by_size
        }
        
        if strategy not in STRATEGIES:
            raise ValueError(f"Strategy must be one of {list(STRATEGIES.keys())}")
        
        self.selection_strategy = strategy
        self.rank_subgraphs = STRATEGIES[strategy]
    
    def _rank_by_frequency(self, subgraphs: Dict[int, nx.DiGraph]) -> List[int]:
        """Rank subgraphs by frequency (most frequent first)"""
        return sorted(subgraphs.keys(), key=lambda sg_id: self.subgraph_frequency.get(sg_id, 0), reverse=True)
    
    def _rank_by_density(self, subgraphs: Dict[int, nx.DiGraph]) -> List[int]:
        """Rank subgraphs by edge density (densest first)"""
        def density(g):
            n = g.number_of_nodes()
            if n <= 1:
                return 0
            return g.number_of_edges() / (n * (n - 1))
        
        return sorted(subgraphs.keys(), key=lambda sg_id: density(subgraphs[sg_id]), reverse=True)
    
    def _rank_by_coverage(self, subgraphs: Dict[int, nx.DiGraph]) -> List[int]:
        """Rank subgraphs by coverage (number of edges × frequency)"""
        return sorted(subgraphs.keys(), 
                      key=lambda sg_id: subgraphs[sg_id].number_of_edges() * self.subgraph_frequency.get(sg_id, 0), 
                      reverse=True)
    
    def _rank_by_complexity(self, subgraphs: Dict[int, nx.DiGraph]) -> List[int]:
        """Rank subgraphs by structural complexity (more complex first)"""
        def complexity(g):
            # A simple complexity metric: nodes × edges × unique edge types
            edge_types = set()
            for _, _, data in g.edges(data=True):
                edge_types.add(data.get('edge_type', 'default'))
            return g.number_of_nodes() * g.number_of_edges() * len(edge_types)
        
        return sorted(subgraphs.keys(), key=lambda sg_id: complexity(subgraphs[sg_id]), reverse=True)
    
    def _rank_by_size(self, subgraphs: Dict[int, nx.DiGraph]) -> List[int]:
        """Rank subgraphs by size (largest first)"""
        return sorted(subgraphs.keys(), 
                      key=lambda sg_id: (subgraphs[sg_id].number_of_nodes(), subgraphs[sg_id].number_of_edges()), 
                      reverse=True)
    
    def _find_all_connected_subgraphs(self, graph: nx.DiGraph) -> List[nx.DiGraph]:
        """Find all connected subgraphs of the given graph within size constraints"""
        subgraphs = []
        
        for component in nx.weakly_connected_components(graph):
            component_graph = graph.subgraph(component).copy()
            
            # Skip if component is too small
            if len(component) < self.min_subgraph_size:
                continue
                
            # If component is within size limits, add it
            if self.max_subgraph_size is None or len(component) <= self.max_subgraph_size:
                subgraphs.append(component_graph)
                
            # Find all connected subgraphs of appropriate size
            # This is a simplified approach - for very large graphs, this could be optimized
            if self.max_subgraph_size is not None and len(component) > self.max_subgraph_size:
                for k in range(self.min_subgraph_size, min(len(component), self.max_subgraph_size) + 1):
                    for nodes in nx.generators.subset.combinations(component, k):
                        subg = graph.subgraph(nodes).copy()
                        if nx.is_weakly_connected(subg) and subg.number_of_edges() > 0:
                            subgraphs.append(subg)
        
        return subgraphs
    
    def _graph_to_hashable(self, graph: nx.DiGraph) -> frozenset:
        """Convert a graph to a hashable representation for frequency counting"""
        edges = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('edge_type', 'default')
            edges.append((u, v, edge_type))
        
        nodes = []
        for n, data in graph.nodes(data=True):
            # Include node attributes in the hash if needed
            nodes.append(n)
            
        return frozenset(edges)
    
    def _are_isomorphic(self, g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
        """Check if two graphs are isomorphic considering node and edge attributes"""
        return nx.is_isomorphic(g1, g2, node_match=self.node_match, edge_match=self.edge_match)
    
    def extract_subgraphs(self, actor_graphs: List[Any]) -> Dict[int, nx.DiGraph]:
        """
        Extract common subgraphs from actor graphs and represent each graph
        using these common subgraphs.
        
        Args:
            actor_graphs: List of ActorGraph objects
        
        Returns:
            Dict mapping subgraph IDs to subgraph objects
        """
        # Extract all possible connected subgraphs
        all_subgraphs = []
        all_hashable_subgraphs = {}  # Maps hash to actual subgraph
        
        for actor_graph_obj in actor_graphs:
            for t, graph in actor_graph_obj.actor_graphs.items():
                subgraphs = self._find_all_connected_subgraphs(graph)
                for sg in subgraphs:
                    sg_hash = self._graph_to_hashable(sg)
                    all_subgraphs.append(sg_hash)
                    all_hashable_subgraphs[sg_hash] = sg
        
        # Count frequency of each subgraph
        subgraph_counter = Counter(all_subgraphs)
        
        # Create canonical subgraphs to avoid duplicates
        canonical_subgraphs = {}  # ID -> subgraph
        canonical_frequency = {}  # ID -> frequency
        
        # Get unique subgraphs and assign IDs
        sg_id = 1
        for sg_hash, sg in all_hashable_subgraphs.items():
            # Check if this is isomorphic to an existing subgraph
            is_new = True
            for existing_id, existing_sg in canonical_subgraphs.items():
                if self._are_isomorphic(sg, existing_sg):
                    canonical_frequency[existing_id] += subgraph_counter[sg_hash]
                    is_new = False
                    break
            
            if is_new:
                canonical_subgraphs[sg_id] = sg.copy()
                canonical_frequency[sg_id] = subgraph_counter[sg_hash]
                sg_id += 1
        
        self.subgraph_library = canonical_subgraphs
        self.subgraph_frequency = canonical_frequency
        
        # Now decompose each graph into these subgraphs
        self._decompose_graphs(actor_graphs)
        
        return self.subgraph_library
    
    def _decompose_graphs(self, actor_graphs: List[Any]):
        """
        Decompose each graph into subgraphs from the library.
        
        Args:
            actor_graphs: List of ActorGraph objects
        """
        for actor_graph_obj in actor_graphs:
            actor_graph_obj.actor_subgraphs = {}
            
            for t, graph in actor_graph_obj.actor_graphs.items():
                graph_copy = copy.deepcopy(graph)
                actor_graph_obj.actor_subgraphs[t] = []
                
                # Get ordered list of subgraphs based on selection strategy
                ordered_subgraphs = {sg_id: sg for sg_id, sg in 
                                    self.subgraph_library.items()}
                ordered_sg_ids = self.rank_subgraphs(ordered_subgraphs)
                
                # Greedy algorithm: iteratively find and remove subgraphs
                while graph_copy.number_of_edges() > 0:
                    best_match = None
                    best_match_id = None
                    best_match_mapping = None
                    
                    # Try subgraphs in order of selection strategy
                    for sg_id in ordered_sg_ids:
                        sg_template = self.subgraph_library[sg_id]
                        
                        # Skip if subgraph is larger than remaining graph
                        if (sg_template.number_of_nodes() > graph_copy.number_of_nodes() or
                            sg_template.number_of_edges() > graph_copy.number_of_edges()):
                            continue
                        
                        # Use VF2 algorithm to find subgraph isomorphisms
                        matcher = iso.DiGraphMatcher(graph_copy, sg_template, 
                                                    node_match=self.node_match,
                                                    edge_match=self.edge_match)
                        
                        # Find first isomorphism
                        for mapping in matcher.subgraph_isomorphisms_iter():
                            # Inverse mapping (template -> graph)
                            inv_map = {v: k for k, v in mapping.items()}
                            
                            # Extract the matched subgraph from our graph
                            match_nodes = list(inv_map.values())
                            matched_subgraph = graph_copy.subgraph(match_nodes).copy()
                            
                            best_match = matched_subgraph
                            best_match_id = sg_id
                            best_match_mapping = inv_map
                            break
                        
                        if best_match is not None:
                            break
                    
                    if best_match is not None:
                        # Add this subgraph to our representation
                        actor_graph_obj.actor_subgraphs[t].append({
                            'subgraph_id': best_match_id,
                            'nodes': list(best_match_mapping.values()),
                            'node_mapping': best_match_mapping
                        })
                        
                        # Remove these edges from our graph copy
                        graph_copy.remove_edges_from(list(best_match.edges()))
                        
                        # Remove isolated nodes
                        graph_copy.remove_nodes_from(list(nx.isolates(graph_copy)))
                    else:
                        # If no match found, create a new single-edge subgraph
                        if graph_copy.number_of_edges() > 0:
                            edge = list(graph_copy.edges(data=True))[0]
                            u, v, data = edge
                            
                            # Create a new subgraph for this edge
                            new_sg = nx.MultiDiGraph()
                            new_sg.add_node(u)
                            new_sg.add_node(v)
                            new_sg.add_edge(u, v, **data)
                            
                            # Add to library with a new ID
                            next_id = max(self.subgraph_library.keys()) + 1 if self.subgraph_library else 1
                            self.subgraph_library[next_id] = new_sg
                            self.subgraph_frequency[next_id] = 1
                            
                            # Add to our representation
                            actor_graph_obj.actor_subgraphs[t].append({
                                'subgraph_id': next_id,
                                'nodes': [u, v],
                                'node_mapping': {u: u, v: v}
                            })
                            
                            # Remove this edge
                            graph_copy.remove_edge(u, v)
                            
                            # Remove isolated nodes
                            graph_copy.remove_nodes_from(list(nx.isolates(graph_copy)))
    
    def visualize_subgraphs(self, output_dir: str = None, figsize: tuple = (10, 8), max_complete_graphs: int = 50):
        """
        Visualize all subgraphs in the library and complete graphs.
        
        Args:
            output_dir: Directory to save visualizations (None for display only)
            figsize: Figure size for each subplot
            max_complete_graphs: Maximum number of complete graphs to plot
        """
        if not self.subgraph_library:
            print("No subgraphs to visualize. Run extract_subgraphs first.")
            return
        
        # Create output directory if specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine grid size for subplots
        n_subgraphs = len(self.subgraph_library)
        grid_size = math.ceil(math.sqrt(n_subgraphs))
        
        # Create figure with subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 30))
        fig.suptitle(f"Subgraph Library ({self.selection_strategy} strategy)", fontsize=24)
        
        # Flatten axes for easier indexing
        if n_subgraphs > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each subgraph
        for i, (sg_id, sg) in enumerate(sorted(self.subgraph_library.items())):
            if i < len(axes):
                ax = axes[i]
                
                # Draw the graph
                pos = nx.spring_layout(sg)
                nx.draw_networkx_nodes(sg, pos, ax=ax, node_size=800, node_color='lightblue')
                
                # Draw edges with colors based on edge_type
                edge_colors = {}
                for u, v, data in sg.edges(data=True):
                    edge_type = data.get('edge_type', 'default')
                    if edge_type not in edge_colors:
                        edge_colors[edge_type] = len(edge_colors)
                
                # Define specific colors for edge types
                edge_type_colors = {
                    'following_lead': 'blue',
                    'neighbor_vehicle': 'forestgreen',
                    'opposite_vehicle': 'orange',
                    'lead_vehicle': 'red',
                }
                
                # Draw edges by type
                for edge_type, color_idx in edge_colors.items():
                    edges_of_type = [(u, v) for u, v, data in sg.edges(data=True) 
                                    if data.get('edge_type', 'default') == edge_type]
                    
                    # Use specific color if defined, otherwise use red
                    color = edge_type_colors.get(edge_type, 'gray')
                    nx.draw_networkx_edges(sg, pos, ax=ax, edgelist=edges_of_type, 
                                          edge_color=color, 
                                          width=2, arrowsize=15)
                
                nx.draw_networkx_labels(sg, pos, ax=ax, font_size=12)
                
                # Add legend for edge types if there are multiple types
                if len(edge_colors) > 1:
                    ax.legend([plt.Line2D([0], [0], color=color, lw=2) for color in edge_type_colors.values()],
                             list(edge_type_colors.keys()),
                             loc='upper right')
                
                # Set title with frequency information
                freq = self.subgraph_frequency.get(sg_id, 0)
                ax.set_title(f"ID: {sg_id}, Freq: {freq}", fontsize=16)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_subgraphs, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, f'subgraph_library_{self.selection_strategy}.png'), dpi=300, bbox_inches='tight')
            
            # Also save individual subgraphs
            for sg_id, sg in self.subgraph_library.items():
                plt.figure(figsize=(8, 6))
                pos = nx.spring_layout(sg)
                
                nx.draw_networkx_nodes(sg, pos, node_size=500, node_color='lightblue')
                
                # Draw edges with colors based on edge_type
                edge_colors = {}
                for u, v, data in sg.edges(data=True):
                    edge_type = data.get('edge_type', 'default')
                    if edge_type not in edge_colors:
                        edge_colors[edge_type] = len(edge_colors)
                
                # Define specific colors for edge types
                edge_type_colors = {
                    'following_lead': 'blue',
                    'neighbor_vehicle': 'forestgreen',
                    'opposite_vehicle': 'orange',
                    'leading_vehicle': 'red',
                }
                
                # Draw edges by type
                for edge_type, color_idx in edge_colors.items():
                    edges_of_type = [(u, v) for u, v, data in sg.edges(data=True) 
                                    if data.get('edge_type', 'default') == edge_type]
                    
                    # Use specific color if defined, otherwise use red
                    color = edge_type_colors.get(edge_type, 'gray')
                    nx.draw_networkx_edges(sg, pos, edgelist=edges_of_type, 
                                          edge_color=color, 
                                          width=2, arrowsize=20)
                
                nx.draw_networkx_labels(sg, pos, font_size=12)
                
                # Add legend for edge types
                if len(edge_colors) > 1:
                    plt.legend([plt.Line2D([0], [0], color=color, lw=2) for color in edge_type_colors.values()],
                               list(edge_type_colors.keys()),
                               loc='upper right')
                
                freq = self.subgraph_frequency.get(sg_id, 0)
                plt.title(f"Subgraph ID: {sg_id}, Frequency: {freq}", fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'subgraph_{sg_id}_{self.selection_strategy}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        else:
            plt.show()
        
        plt.close()
    
    def _plot_complete_graphs(self, output_dir: str, max_graphs: int):
        """
        Plot complete graphs from the library.
        
        Args:
            output_dir: Directory to save visualizations
            max_graphs: Maximum number of graphs to plot
        """
        # Get all unique graphs from the library
        unique_graphs = set()
        for sg_id, sg in self.subgraph_library.items():
            # Convert graph to a canonical form for comparison
            graph_str = nx.to_edgelist(sg)
            unique_graphs.add((sg_id, sg, graph_str))
        
        # Sort by frequency
        sorted_graphs = sorted(unique_graphs, 
                             key=lambda x: self.subgraph_frequency.get(x[0], 0), 
                             reverse=True)
        
        # Take only the top max_graphs
        top_graphs = sorted_graphs[:max_graphs]
        
        # Define specific colors for edge types
        edge_type_colors = {
            'following_lead': 'blue',
            'neighbor_vehicle': 'forestgreen',
            'opposite_vehicle': 'orange',
            'leading_vehicle': 'red'
        }
        
        # Plot each graph individually
        for sg_id, sg, _ in top_graphs:
            plt.figure(figsize=(20, 20))
            
            # Draw the graph
            pos = nx.spring_layout(sg)
            nx.draw_networkx_nodes(sg, pos, node_size=1000, node_color='lightblue')
            
            # Draw edges with specific colors
            for edge_type in edge_type_colors:
                edges_of_type = [(u, v) for u, v, data in sg.edges(data=True) 
                                if data.get('edge_type', 'default') == edge_type]
                if edges_of_type:
                    nx.draw_networkx_edges(sg, pos, edgelist=edges_of_type,
                                          edge_color=edge_type_colors[edge_type],
                                          width=3, arrowsize=20)
            
            # Draw remaining edges in gray
            remaining_edges = [(u, v) for u, v, data in sg.edges(data=True)
                             if data.get('edge_type', 'default') not in edge_type_colors]
            if remaining_edges:
                nx.draw_networkx_edges(sg, pos, edgelist=remaining_edges,
                                      edge_color='gray', width=3, arrowsize=20)
            
            nx.draw_networkx_labels(sg, pos, font_size=16)
            
            # Add legend
            legend_elements = []
            for edge_type, color in edge_type_colors.items():
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=edge_type))
            legend_elements.append(plt.Line2D([0], [0], color='gray', lw=2, label='other'))
            plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
            
            # Set title with frequency information
            freq = self.subgraph_frequency.get(sg_id, 0)
            plt.title(f"Complete Graph ID: {sg_id}, Frequency: {freq}", fontsize=24)
            plt.axis('off')
            
            # Save the figure
            if output_dir is not None:
                plt.savefig(os.path.join(output_dir, f'complete_graph_{sg_id}.png'), 
                          dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
    
    def print_subgraph_properties(self):
        """Print properties of each subgraph in the library"""
        if not self.subgraph_library:
            print("No subgraphs in the library. Run extract_subgraphs first.")
            return
            
        print("\nSubgraph Library:")
        print("=================")
        
        # Sort by selection strategy
        ordered_sg_ids = self.rank_subgraphs(self.subgraph_library)
        
        for sg_id in ordered_sg_ids:
            sg = self.subgraph_library[sg_id]
            print(f"Subgraph ID: {sg_id}")
            print(f"  Frequency: {self.subgraph_frequency.get(sg_id, 0)}")
            print(f"  Nodes: {sg.number_of_nodes()}")
            print(f"  Edges: {sg.number_of_edges()}")
            
            # Edge type distribution
            edge_types = {}
            for _, _, data in sg.edges(data=True):
                edge_type = data.get('edge_type', 'default')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            print("  Edge types:")
            for edge_type, count in edge_types.items():
                print(f"    - {edge_type}: {count}")
            print()