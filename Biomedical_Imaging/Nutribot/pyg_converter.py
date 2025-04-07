# pyg_converter.py
import torch
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
from networkx.convert_matrix import from_pandas_edgelist

def nx_to_pyg(G):
    # 1. Relabel nodes to integers for PyG
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_mapping)

    # 2. Build edge_index
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    # 3. Build node feature matrix
    node_features = []
    for node in G.nodes(data=True):
        features = []
        for key in ["spicy", "sweet", "vegan", "protein"]:  # modify according to your data
            features.append(node[1].get(key, 0))
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # 4. Build PyG data object
    data = Data(x=x, edge_index=edge_index)

    return data, node_mapping

# Example use
# from graph_builder import G
# pyg_data, node_map = nx_to_pyg(G)
