import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point as ShapelyPoint
import numpy as np


class MapVisualizer:
    """
    Class for visualizing lane maps with actors, handling label positioning to avoid overlaps.
    """
    
    def __init__(self, fig=None, ax=None, figsize=(15, 10)):
        """
        Initialize the visualizer.
        
        Parameters:
        - fig, ax: Existing matplotlib figure and axis (optional)
        - figsize: Figure size if creating new figure
        """
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig = fig
            self.ax = ax
            
        self.lane_polygons = []  # Store lane polygon patches for collision detection
        self.actor_positions = []  # Store actor (x, y) positions
        self.actor_colors = []
        self.actor_ids = []
        self.text_objects = []
        
    def add_lanes(self, graph, intersection_color='red', lane_color='blue', 
                  alpha=0.6, color_by_length=False, cmap='viridis', show_labels=False):
        """
        Add lane polygons to the map.
        
        Parameters:
        - graph: NetworkX graph with lane information
        - intersection_color: Color for intersection lanes
        - lane_color: Color for regular lanes
        - alpha: Transparency level
        - color_by_length: Color lanes by their length
        - cmap: Colormap for lane coloring
        - show_labels: Whether to show lane ID labels
        """
        all_x_coords = []
        all_y_coords = []
        lengths = []
        
        # Collect all lane lengths for color mapping
        if color_by_length:
            for node_id, node_data in graph.nodes(data=True):
                lengths.append(node_data['node_info'].length)
            
            if lengths:
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
                if color_by_length and lengths:
                    normalized_length = (length - min_length) / (max_length - min_length)
                    color = plt.cm.get_cmap(cmap)(normalized_length)
                else:
                    color = intersection_color if is_intersection else lane_color
                
                # Create and add polygon
                polygon_patch = patches.Polygon(coords, 
                                              facecolor=color, 
                                              edgecolor='black', 
                                              alpha=alpha,
                                              linewidth=0.5)
                self.ax.add_patch(polygon_patch)
                self.lane_polygons.append(lane_polygon)  # Store shapely polygon
                
                # Add labels if requested
                if show_labels:
                    centroid = lane_polygon.centroid
                    label_text = f"{lane_id}\nL:{length:.1f}m"
                    self.ax.text(centroid.x, centroid.y, label_text, 
                               fontsize=6, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Set axis properties
        if all_x_coords and all_y_coords:
            margin = 10
            self.ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
            self.ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        
        return self
    
    def add_actors(self, actor_graph, actor_size=50, 
                   vehicle_color='green', pedestrian_color='orange', other_color='purple'):
        """
        Add actors as dots to the map.
        
        Parameters:
        - actor_graph: NetworkX graph with actor information
        - actor_size: Size of actor markers
        - vehicle_color: Color for vehicle actors
        - pedestrian_color: Color for pedestrian actors
        - other_color: Color for other actor types
        """
        # Color mapping for different actor types
        actor_color_map = {
            'VEHICLE': vehicle_color,
            'PEDESTRIAN': pedestrian_color,
            'CYCLIST': 'cyan',
            'MOTORCYCLE': 'magenta'
        }
        
        # Process each actor
        for actor_id, actor_data in actor_graph.nodes(data=True):
            xyz_point = actor_data['xyz']
            actor_type = actor_data['actor_type']
            
            x_coord = xyz_point.x
            y_coord = xyz_point.y
            
            self.actor_positions.append((x_coord, y_coord))
            
            # Determine actor color based on type
            actor_type_str = str(actor_type).split('.')[-1].replace('>', '')
            color = actor_color_map.get(actor_type_str, other_color)
            self.actor_colors.append(color)
            self.actor_ids.append(actor_id)
        
        # Plot actors as scatter points (dots)
        actor_x = [pos[0] for pos in self.actor_positions]
        actor_y = [pos[1] for pos in self.actor_positions]
        
        # Plot dots without border - just solid colored circles
        self.ax.scatter(actor_x, actor_y, 
                       c=self.actor_colors, 
                       s=actor_size * 2,  # Make dots larger
                       alpha=1.0,  # Fully opaque
                       edgecolors='none',  # No border
                       linewidth=0,  # No border width
                       zorder=8)  # High z-order to be on top of lanes
        
        return self
    
    def add_actor_labels(self, fontsize=None):
        """
        Add actor ID labels with intelligent positioning to avoid overlaps with:
        - Other labels
        - Actor dots
        - Lane polygons (streets)
        
        Parameters:
        - fontsize: Font size for labels (defaults to title font size)
        """
        if not self.actor_ids:
            return self
        
        # Get font size
        if fontsize is None:
            title_obj = self.ax.title
            fontsize = title_obj.get_fontsize() if hasattr(title_obj, 'get_fontsize') else 12
        
        actor_x = [pos[0] for pos in self.actor_positions]
        actor_y = [pos[1] for pos in self.actor_positions]
        
        # Create text objects OFFSET from actor positions initially
        # This ensures they start away from the dots
        for i, (x, y, actor_id) in enumerate(zip(actor_x, actor_y, self.actor_ids)):
            # Start with an offset position
            angle = 2 * np.pi * i / len(self.actor_ids)
            offset = 30  # Initial offset distance
            initial_x = x + offset * np.cos(angle)
            initial_y = y + offset * np.sin(angle)
            
            text = self.ax.text(initial_x, initial_y, str(actor_id), 
                   fontsize=fontsize / 2,  # Half the size
                   ha='center', 
                   va='center',
                   fontweight='normal',  # Not bold
                   color='black',  # Black text
                   bbox=None,  # No border/background
                   zorder=10)  # Very high z-order to be on top
            self.text_objects.append(text)
        
        # Use adjust_text with strong repulsion from lane polygons
        try:
            from adjustText import adjust_text
            
            # Create dense grid of avoid points from lane polygons
            avoid_x = []
            avoid_y = []
            
            for lane_poly in self.lane_polygons:
                coords = list(lane_poly.exterior.coords)
                # Sample EVERY point for better avoidance
                for coord in coords:
                    avoid_x.append(coord[0])
                    avoid_y.append(coord[1])
                
                # Also add interior points by sampling the polygon area
                minx, miny, maxx, maxy = lane_poly.bounds
                for dx in np.linspace(minx, maxx, 5):
                    for dy in np.linspace(miny, maxy, 5):
                        point = ShapelyPoint(dx, dy)
                        if lane_poly.contains(point):
                            avoid_x.append(dx)
                            avoid_y.append(dy)
            
            # Adjust text positions with strong repulsion
            # Don't draw arrows during adjustment - we'll draw them manually after
            adjust_text(self.text_objects, 
                       x=actor_x, 
                       y=actor_y,
                       add_objects=avoid_x + avoid_y if avoid_x else None,
                       arrowprops=None,  # Don't draw arrows during adjustment
                       expand_points=(5.0, 5.0),    # Reduced from 8.0
                       expand_text=(4.0, 4.0),      # Reduced from 6.0
                       force_points=(3.0, 3.0),     # Reduced from 5.0
                       force_text=(3.0, 3.0),       # Reduced from 5.0
                       force_objects=(2.5, 2.5),    # Reduced from 4.0
                       only_move={'points': 'xy', 'text': 'xy', 'objects': 'xy'},
                       lim=2000,                    # Reduced from 3000
                       ax=self.ax)
            
            # Now manually draw connecting lines from actors to edge of labels
            for i, (text, x, y) in enumerate(zip(self.text_objects, actor_x, actor_y)):
                label_x, label_y = text.get_position()
                
                # Calculate direction from actor to label
                dx = label_x - x
                dy = label_y - y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    # Normalize direction
                    dx_norm = dx / distance
                    dy_norm = dy / distance
                    
                    # Estimate text width/height (approximate based on font size)
                    text_offset = fontsize / 3  # Rough estimate of half-width
                    
                    # End line at edge of text, not center
                    end_x = label_x - dx_norm * text_offset
                    end_y = label_y - dy_norm * text_offset
                    
                    self.ax.plot([x, end_x], [y, end_y], 
                               color='black', 
                               linewidth=1.0, 
                               alpha=0.7,
                               zorder=9,
                               linestyle='-')
            
        except ImportError:
            # Fallback: manual positioning with lane avoidance
            self._manual_label_positioning()
        
        return self
    
    def _manual_label_positioning(self):
        """
        Fallback method for positioning labels when adjust_text is not available.
        """
        actor_x = [pos[0] for pos in self.actor_positions]
        actor_y = [pos[1] for pos in self.actor_positions]
        
        for i, (text, x, y) in enumerate(zip(self.text_objects, actor_x, actor_y)):
            # Try multiple angles and distances to find a position not overlapping lanes
            base_angle = 2 * np.pi * i / len(self.text_objects)
            found_position = False
            
            # Try increasing distances
            for offset in [30, 40, 50, 60]:
                # Try multiple angles at each distance
                for angle_offset in range(8):
                    angle = base_angle + (angle_offset * np.pi / 4)
                    new_x = x + offset * np.cos(angle)
                    new_y = y + offset * np.sin(angle)
                    
                    # Check if this position overlaps with any lane
                    point = ShapelyPoint(new_x, new_y)
                    overlaps = False
                    
                    for lane_poly in self.lane_polygons:
                        if lane_poly.contains(point) or lane_poly.distance(point) < 10:
                            overlaps = True
                            break
                    
                    if not overlaps:
                        text.set_position((new_x, new_y))
                        # Draw connecting line ending at edge of text
                        dx = new_x - x
                        dy = new_y - y
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance > 0:
                            dx_norm = dx / distance
                            dy_norm = dy / distance
                            text_offset = 5  # Approximate half-width of text
                            end_x = new_x - dx_norm * text_offset
                            end_y = new_y - dy_norm * text_offset
                            self.ax.plot([x, end_x], [y, end_y], 
                                       color='black', 
                                       linewidth=1.0, 
                                       alpha=0.7,
                                       zorder=9,
                                       linestyle='-')
                        found_position = True
                        break
                
                if found_position:
                    break
            
            # If still no good position found, place it far away
            if not found_position:
                new_x = x + 80 * np.cos(base_angle)
                new_y = y + 80 * np.sin(base_angle)
                text.set_position((new_x, new_y))
                # Draw connecting line ending at edge of text
                dx = new_x - x
                dy = new_y - y
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 0:
                    dx_norm = dx / distance
                    dy_norm = dy / distance
                    text_offset = 5  # Approximate half-width of text
                    end_x = new_x - dx_norm * text_offset
                    end_y = new_y - dy_norm * text_offset
                    self.ax.plot([x, end_x], [y, end_y], 
                               color='black', 
                               linewidth=1.0, 
                               alpha=0.7,
                               zorder=9,
                               linestyle='-')
    
    def add_actor_edges(self, actor_graph):
        """
        Add edges from actor graph to show relationships.
        
        Parameters:
        - actor_graph: NetworkX graph with actor edges
        """
        edge_colors = {
            'opposite_vehicle': 'red',
            'following_lead': 'purple',
            'neighbor_vehicle': 'green',
        }
        default_edge_color = 'gray'
        
        for source, target, edge_data in actor_graph.edges(data=True):
            source_xyz = actor_graph.nodes[source]['xyz']
            target_xyz = actor_graph.nodes[target]['xyz']
            
            source_x, source_y = source_xyz.x, source_xyz.y
            target_x, target_y = target_xyz.x, target_xyz.y
            
            edge_type = edge_data.get('edge_type', 'unknown')
            path_length = edge_data.get('path_length', 0)
            
            color = edge_colors.get(edge_type, default_edge_color)
            linestyle = '--' if path_length < 0 else '-'
            linewidth = 0.75
            
            self.ax.annotate('', xy=(target_x, target_y), xytext=(source_x, source_y),
                           arrowprops=dict(arrowstyle='->', 
                                         color=color, 
                                         linewidth=linewidth,
                                         linestyle=linestyle,
                                         alpha=0.7))
        
        return self
    
    def set_title(self, title):
        """Set the plot title."""
        self.ax.set_title(title)
        return self
    
    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()
        return self
    
    def get_figure(self):
        """Return the figure and axis."""
        return self.fig, self.ax
