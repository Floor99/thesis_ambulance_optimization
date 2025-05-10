import osmnx as ox 
import networkx as nx

def get_coords_nodes_from_osmnx(place):
    G = ox.graph_from_place(place, network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True)
    G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)
    
    nodes, edges = ox.graph_to_gdfs(G)
    
    nodes = nodes[['y', 'x']].reset_index(names='node_id')
    nodes = nodes.rename(columns={'y': 'lat', 'x': 'lon'})
    
    return nodes

def clean_numeric(val, default=50.0):
    if isinstance(val, list):
        val = val[0]  # If it's a list, pick the first value
    try:
        return float(val)
    except (ValueError, TypeError):
        return default 
    
def get_edge_features_from_osmnx(place):
    G = ox.graph_from_place(place, network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True)
    G = G.to_undirected()
    G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)
    
    nodes, edges = ox.graph_to_gdfs(G)
    
    edges = edges[['maxspeed', 'length']]
    edges['maxspeed'] = edges['maxspeed'].apply(lambda x: clean_numeric(x, default=50.0))
    edges['length'] = edges['length'].apply(lambda x: clean_numeric(x, default=30.0))  # 30m as a reasonable default)
    edges['maxspeed'] = edges['maxspeed'].fillna(50.0)
    edges['length'] = edges['length'].fillna(30.0)
    
    return edges


if __name__ == "__main__":
    # df = get_coords_nodes_from_osmnx("Helmond, Netherlands")
    # df.to_parquet("data/processed/coords_nodes_helmond.parquet", index=False)
    
    df2 = get_edge_features_from_osmnx("Helmond, Netherlands")
    df2 = df2.reset_index()
    df2.to_parquet("data/processed/edge_features_helmond.parquet", index=False)
    