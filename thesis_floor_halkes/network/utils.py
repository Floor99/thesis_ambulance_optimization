import numpy as np
import torch 
import osmnx as ox
from torch_geometric.data import Data
import networkx as nx

def clean_numeric(val, default=50.0):
    if isinstance(val, list):
        val = val[0]  # If it's a list, pick the first value
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_graph_from_network(lat, lon, dist):
    G = ox.graph_from_point((lat,lon), dist = dist, network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True)
    G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)
    
    nodes, edges = ox.graph_to_gdfs(G)
    
    # fix edges 
    edges = edges[['maxspeed', 'length']]
    # edges['maxspeed'] = edges['maxspeed'].fillna(5).astype(float)
    edges['maxspeed'] = edges['maxspeed'].apply(lambda x: clean_numeric(x, default=50.0))
    edges['length'] = edges['length'].apply(lambda x: clean_numeric(x, default=30.0))  # 30m as a reasonable default)
    edges['maxspeed'] = edges['maxspeed'].fillna(50.0)
    edges['length'] = edges['length'].fillna(30.0)
    
    
    # Extract u and v (start and end nodes) from edges
    edges = edges.reset_index()  # if u, v, key are in the index
    u = torch.tensor(edges['u'].values, dtype=torch.long)
    v = torch.tensor(edges['v'].values, dtype=torch.long)

    # Manually duplicate edges in reverse direction to ensure undirected behavior
    edge_index = torch.cat([
        torch.stack([u, v], dim=0),
        torch.stack([v, u], dim=0)
    ], dim=1)

    # Edge attributes: duplicate for reverse direction
    edge_features = edges[['length', 'maxspeed']]
    edge_attr = torch.tensor(edge_features.values, dtype=torch.float)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    
    
    # fix nodes
    nodes = nodes[['y', 'x']]
    nodes.loc[:, 'traffic_light']= 1
    # 2. Set traffic light status: randomly 0 (red) or 1 (green)
    nodes['traffic_light_status'] = np.random.choice([0, 1], size=len(nodes))
    # 3. Set waiting time (in seconds): random between 0 and 60 seconds
    nodes['waiting_time_seconds'] = np.random.randint(0, 61, size=len(nodes))
    node_features = nodes[['traffic_light', 'traffic_light_status', 'waiting_time_seconds']]

    # Convert to tensor
    x = torch.tensor(node_features.values, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # create static and dynamic Data
    static_node_idx = [0]
    dynamic_node_idx = [1, 2]
    static_edge_idx = [0,1]

    x_static = data.x[:, static_node_idx].clone()
    x_dynamic = data.x[:, dynamic_node_idx].clone()

    if hasattr(data, 'edge_attr'):
        e_static = data.edge_attr[:, static_edge_idx].clone()
    else: 
        e_static = e_dynamic = None

    data_static = Data(
        x = data.x.new_tensor(x_static),
        edge_index= data.edge_index,
        edge_attr=(e_static if e_static is not None else None)
    )

    data_dynamic = Data(
        x=data.x.new_tensor(x_dynamic),
        edge_index=data.edge_index,)
    
    return G, data_static, data_dynamic   
    

    
    
    