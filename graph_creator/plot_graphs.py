
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import numpy as np

def plot_lane_map(graph, figsize=(15, 10), intersection_color='red', lane_color='blue', alpha=0.6):
   """
   Plot a map showing lanes as polygons from a NetworkX graph.
   
   Parameters:
   - graph: NetworkX graph with node_info containing lane_polygon
   - figsize: Figure size tuple
   - intersection_color: Color for intersection lanes
   - lane_color: Color for regular lanes
   - alpha: Transparency level
   """
   fig, ax = plt.subplots(figsize=figsize)
   
   # Lists to store all coordinates for setting axis limits
   all_x_coords = []
   all_y_coords = []
   
   # Plot each lane polygon
   for node_id, node_data in graph.nodes(data=True):
       node_info = node_data['node_info']
       lane_polygon = node_info.lane_polygon
       lane_id = node_info.lane_id
       is_intersection = node_info.is_intersection
       
       # Extract coordinates from shapely polygon
       if lane_polygon and hasattr(lane_polygon, 'exterior'):
           coords = list(lane_polygon.exterior.coords)
           x_coords = [coord[0] for coord in coords]
           y_coords = [coord[1] for coord in coords]
           
           # Add to all coordinates for axis limits
           all_x_coords.extend(x_coords)
           all_y_coords.extend(y_coords)
           
           # Choose color based on intersection status
           color = intersection_color if is_intersection else lane_color
           
           # Create polygon patch
           polygon_patch = patches.Polygon(coords, 
                                         facecolor=color, 
                                         edgecolor='black', 
                                         alpha=alpha,
                                         linewidth=0.5)
           ax.add_patch(polygon_patch)
           
           # Add lane ID as text at centroid
           centroid = lane_polygon.centroid
           ax.text(centroid.x, centroid.y, lane_id, 
                  fontsize=8, ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
   
   # Set axis limits and properties
   if all_x_coords and all_y_coords:
       margin = 10  # Add some margin around the map
       ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
       ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
   
   ax.set_aspect('equal')
   ax.grid(True, alpha=0.3)
   ax.set_xlabel('X Coordinate')
   ax.set_ylabel('Y Coordinate')
   ax.set_title('Lane Map')
   
   # Add legend
   # intersection_patch = patches.Patch(color=intersection_color, label='Intersections')
   # lane_patch = patches.Patch(color=lane_color, label='Regular Lanes')
   # ax.legend(handles=[intersection_patch, lane_patch])
   
   plt.tight_layout()
   return fig, ax


def plot_lane_map_advanced(graph, figsize=(15, 10), show_labels=True, 
                         color_by_length=False, cmap='viridis', fig = None, ax = None):
   """
   Advanced lane map plotting with additional features.
   """

   if fig is None or ax is None:
       fig, ax = plt.subplots(figsize=figsize)
   
   all_x_coords = []
   all_y_coords = []
   lengths = []
   
   # Collect all lane lengths for color mapping
   if color_by_length:
       for node_id, node_data in graph.nodes(data=True):
           lengths.append(node_data['node_info'].length)
       
       # Normalize lengths for color mapping
       min_length, max_length = min(lengths), max(lengths)
   
   for i, (node_id, node_data) in enumerate(graph.nodes(data=True)):
       node_info = node_data['node_info']
       lane_polygon = node_info.lane_polygon
       lane_id = node_info.lane_id
       is_intersection = node_info.is_intersection
       length = node_info.length
       
       if lane_polygon and hasattr(lane_polygon, 'exterior'):
           coords = list(lane_polygon.exterior.coords)
           x_coords = [coord[0] for coord in coords]
           y_coords = [coord[1] for coord in coords]
           
           all_x_coords.extend(x_coords)
           all_y_coords.extend(y_coords)
           
           # Determine color
           if color_by_length:
               # Color by length using colormap
               normalized_length = (length - min_length) / (max_length - min_length)
               color = plt.cm.get_cmap(cmap)(normalized_length)
           else:
               # Color by intersection status
               color = 'red' if is_intersection else 'blue'
           
           # Create and add polygon
           polygon_patch = patches.Polygon(coords, 
                                         facecolor=color, 
                                         edgecolor='black', 
                                         alpha=0.6,
                                         linewidth=0.5)
           ax.add_patch(polygon_patch)
           
           # Add labels if requested
           if show_labels:
               centroid = lane_polygon.centroid
               label_text = f"{lane_id}\nL:{length:.1f}m"
               ax.text(centroid.x, centroid.y, label_text, 
                      fontsize=6, ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
   
   # Set axis properties
   if all_x_coords and all_y_coords:
       margin = 10
       ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
       ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
   
   ax.set_aspect('equal')
   ax.grid(True, alpha=0.3)
   ax.set_xlabel('X Coordinate')
   ax.set_ylabel('Y Coordinate')
   ax.set_title('Advanced Lane Map')
   
    # Add colorbar if coloring by length
    #    if color_by_length:
    #        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_length, vmax=max_length))
    #        sm.set_array([])
    #        plt.colorbar(sm, ax=ax, label='Lane Length (m)')
   
   plt.tight_layout()
   return fig, ax




def add_actors_to_map(fig, ax, actor_graph, actor_size=50, show_actor_labels=True, 
                    vehicle_color='green', pedestrian_color='orange', other_color='purple'):
   """
   Add actors to an existing lane map plot.
   
   Parameters:
   - fig, ax: Existing matplotlib figure and axis from plot_lane_map_advanced
   - actor_graph: NetworkX graph with actor information
   - actor_size: Size of actor markers
   - show_actor_labels: Whether to show actor IDs and info
   - vehicle_color: Color for vehicle actors
   - pedestrian_color: Color for pedestrian actors
   - other_color: Color for other actor types
   """
   
   # Color mapping for different actor types
   actor_colors = {
       'VEHICLE': vehicle_color,
       'PEDESTRIAN': pedestrian_color,
       'CYCLIST': 'cyan',
       'MOTORCYCLE': 'magenta'
   }
   
   # Lists to store actor data for plotting
   actor_x = []
   actor_y = []
   actor_colors_list = []
   actor_labels = []
   
   # Process each actor
   for actor_id, actor_data in actor_graph.nodes(data=True):
       lane_id = actor_data['lane_id']
       s_position = actor_data['s']
       xyz_point = actor_data['xyz']
       lon_speed = actor_data['lon_speed']
       actor_type = actor_data['actor_type']
       
       # Extract coordinates from Point Z
       x_coord = xyz_point.x
       y_coord = xyz_point.y
       
       actor_x.append(x_coord)
       actor_y.append(y_coord)
       
       # Determine actor color based on type
       actor_type_str = str(actor_type).split('.')[-1].replace('>', '')  # Extract enum value
       color = actor_colors.get(actor_type_str, other_color)
       actor_colors_list.append(color)
       
       # Create label text
       if show_actor_labels:
           label = f"ID:{actor_id}\nLane:{lane_id}\nSpeed:{lon_speed:.1f}m/s"
           actor_labels.append(label)
   
   # Plot actors as scatter points
   scatter = ax.scatter(actor_x, actor_y, 
                       c=actor_colors_list, 
                       s=actor_size, 
                       alpha=0.8, 
                       edgecolors='black', 
                       linewidth=1,
                       zorder=5)  # Higher zorder to appear on top
   
   # Add actor labels if requested
   if show_actor_labels and actor_labels:
       for i, (x, y, label) in enumerate(zip(actor_x, actor_y, actor_labels)):
           ax.annotate(label, (x, y), 
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=6, ha='left', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
   
    # Update legend to include actor types
    #    handles, labels = ax.get_legend_handles_labels()
    
    #    # Add actor type handles to legend
    #    for actor_type, color in actor_colors.items():
    #        handles.append(patches.Patch(color=color, label=f'{actor_type}s'))
    
    #    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1))
   
   # Update title
   current_title = ax.get_title()
   ax.set_title(f"{current_title} with Actors")
   
   return fig, ax


def add_actor_edges_to_map(fig, ax, actor_graph):
   """
   Add edges from actor graph to the existing map plot.
   
   Parameters:
   - fig, ax: Existing matplotlib figure and axis 
   - actor_graph: NetworkX graph with actor edges
   """
   
   # Color mapping for different edge types
   edge_colors = {
       'opposite_vehicle': 'red',
       'following_lead': 'purple',
       'neighbor_vehicle': 'green',
   }
   default_edge_color = 'gray'
   
   # Process each edge
   for source, target, edge_data in actor_graph.edges(data=True):
       # Get source and target actor positions
       source_xyz = actor_graph.nodes[source]['xyz']
       target_xyz = actor_graph.nodes[target]['xyz']
       
       source_x, source_y = source_xyz.x, source_xyz.y
       target_x, target_y = target_xyz.x, target_xyz.y
       
       # Get edge properties
       edge_type = edge_data.get('edge_type', 'unknown')
       path_length = edge_data.get('path_length', 0)
       
       # Determine edge color and style
       color = edge_colors.get(edge_type, default_edge_color)
       
       # Line style based on path length (negative = dashed, positive = solid)
       linestyle = '--' if path_length < 0 else '-'
       # linewidth = min(max(abs(path_length) / 10, 0.5), 3.0)  # Scale line width by path length
       linewidth = 0.75
       # Draw edge as arrow
       ax.annotate('', xy=(target_x, target_y), xytext=(source_x, source_y),
                  arrowprops=dict(arrowstyle='->', 
                                color=color, 
                                linewidth=linewidth,
                                linestyle=linestyle,
                                alpha=0.7))
   
   # Update title
   current_title = ax.get_title()
   ax.set_title(f"{current_title} with Actor Relationships")
   
   return fig, ax

