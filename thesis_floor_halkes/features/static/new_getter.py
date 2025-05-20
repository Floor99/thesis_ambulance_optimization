import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import osmnx as ox

from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.create_training_data import get_sub_graph, get_timeseries_subgraph
from thesis_floor_halkes.utils.haversine import haversine


def get_static_data_object_subgraph(
    timeseries_subgraph_path: str,
    edge_features_path: str, 
    G_cons_path: str,
    G_pt_cons_path: str,
    start_node: int = None,
    end_node: int = None,
        ):
    
    timeseries = pd.read_parquet(timeseries_subgraph_path)
    
    # add start and end node to static data object
    if start_node is None:
        start_node = np.random.choice(timeseries.node_id.values).item()
    if end_node is None:
        end_node = np.random.choice(timeseries.node_id.values).item()
        while start_node == end_node:
            end_node = np.random.choice(timeseries.node_id.values).item()
    
    timeseries["start_node"] = start_node
    timeseries["end_node"] = end_node
    
    # compute distance-to-goal for each node
    goal_coords = timeseries.loc[
        timeseries["node_id"] == end_node, ["lat", "lon"]
    ].iloc[0]
    timeseries["distance_to_goal_meters"] = timeseries.apply(
        lambda row: haversine((row.lat, row.lon), (goal_coords.lat, goal_coords.lon)),
        axis=1,
    )
    
    edge_features = pd.read_parquet(edge_features_path)
    # edge_features = edge_features.reset_index()
    edge_index = (
        torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long)
        .t()
        .contiguous()
    )
    
    edge_attr = torch.tensor(edge_features[["length", "maxspeed"]].values, dtype=torch.float)
    
    node_features = timeseries.drop_duplicates(subset=["node_id"]).copy()
    node_features = torch.tensor(
        node_features[["lat", "lon", "has_light", "distance_to_goal_meters"]].values,
        dtype=torch.float,
    )
    
    static_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    
    static_data.start_node = start_node
    static_data.end_node = end_node
    
    G_cons = ox.io.load_graphml(G_cons_path)
    G_pt_cons = ox.io.load_graphml(G_pt_cons_path)
    
    static_data.G_sub = G_cons
    static_data.G_pt = G_pt_cons
    
    deduped_node_ids = timeseries.drop_duplicates(
        subset=["node_id"]
    ).copy()
    node_id_mapping = dict(
        zip(deduped_node_ids["node_id"], deduped_node_ids["osmid_original"])
    )
    static_data.node_id_mapping = node_id_mapping
    
    static_data.timeseries = timeseries
    
    return static_data
    

def collect_static_data_objects(
    base_dir: str = "data/training_data",
    start_node: int = None,
    end_node: int = None,
):
    static_list = []
    for sub in os.listdir(base_dir):
        subpath = os.path.join(base_dir, sub)
        if not os.path.isdir(subpath):
            continue
        
        timeseries_path = os.path.join(subpath, "timeseries.parquet")
        edge_features_path = os.path.join(subpath, "edge_features.parquet")
        G_cons_path = os.path.join(subpath, "G_cons.graphml")
        G_pt_cons_path = os.path.join(subpath, "G_pt_cons.graphml")
        
        static_object = get_static_data_object_subgraph(
            timeseries_subgraph_path=timeseries_path,
            edge_features_path=edge_features_path,
            G_cons_path=G_cons_path,
            G_pt_cons_path=G_pt_cons_path,
            start_node=start_node,
            end_node=end_node,
        )
        
        static_list.append(static_object)
        
    return static_list


if __name__ == "__main__":
    static_list = collect_static_data_objects()
    print(static_list)