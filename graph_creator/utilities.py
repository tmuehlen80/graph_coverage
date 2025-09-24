import pandas as pd
import networkx as nx
from typing import Tuple

def make_node_edge_df(graph: nx.Graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Make a dataframe from a networkx graph.
    
    Args:
        graph: networkx graph

    Returns:
        df_nodes: dataframe of nodes
        df_edges: dataframe of edges
    """
    df_nodes = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    df_nodes.reset_index(inplace=True)
    df_nodes.rename(columns={'index': 'node_id'}, inplace=True)
    df_edges = pd.DataFrame([(u, v, d) for u, v, d in graph.edges(data=True)], columns=['source', 'target', 'attributes'])
    df_edges = pd.concat([df_edges[['source', 'target']], pd.json_normalize(df_edges['attributes'])], axis=1)
    return df_nodes, df_edges
