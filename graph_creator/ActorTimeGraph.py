from graph_creator.ActorGraph import ActorGraph
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
from shapely.geometry import Point


class ActorTimeGraph:
    def __init__(self, G):
        self.G = G
        self.timestamps = list(G.actor_graphs.keys())
        self.actor_time_graphs = {}

    def create_actor_time_graphs(self):
        for i in range(1, len(self.timestamps)):
            # create a 
            current_graph = self.G.actor_graphs[self.timestamps[i]].copy()

            # update the node attributes to include the time step
            for node in current_graph.nodes:
                interim = current_graph.nodes(data=True)[node].copy()
                current_graph.nodes[node].clear()
                current_graph.nodes[node].update({0: interim})
            
            # add all nodes from the previous timestep graph, if not already in the current graph
            for node in self.G.actor_graphs[self.timestamps[i - 1]].nodes:
                if node not in current_graph.nodes:
                    current_graph.add_node(node)
                    interim = current_graph.nodes(data=True)[node].copy()
                    current_graph.nodes[node].clear()
                    current_graph.nodes[node].update({1: interim})
                    # print(current_graph.nodes(data=True)[node])
            
            # for all nodes in the current timestep, add the node attributes from the previous timestep (if the node existed there)
            for node in current_graph.nodes:
                if node in self.G.actor_graphs[self.timestamps[i - 1]].nodes:
                    interim = self.G.actor_graphs[self.timestamps[i - 1]].nodes(data=True)[node].copy()
                    current_graph.nodes[node].update({1: interim})
                    # print(current_graph.nodes(data=True)[node])
            
            # now update all current edges with a timestep attribute
            for edge in current_graph.edges(data=True):
                edge[2]["time_lag"] = 0
            
            # and finally, add the edges from the previous timestep
            for edge in self.G.actor_graphs[self.timestamps[i - 1]].edges(data=True):
                # edge = list(ag_carla.actor_graphs[ag_timestamps[i - 1]].edges(data=True))[1]
                interim = edge[2].copy()
                interim["time_lag"] = 1
                current_graph.add_edge(edge[0], edge[1], **interim)
                
            self.actor_time_graphs[self.timestamps[i]] = current_graph

        self.actor_time_components = {}
        # print("instance.actor_graphs.keys(): ", instance.actor_graphs.keys())
        for key, value in self.actor_time_graphs.items():
            components = list(nx.weakly_connected_components(value))
            subgraphs = [value.subgraph(c).copy() for c in components]
            self.actor_time_components[key] = subgraphs








