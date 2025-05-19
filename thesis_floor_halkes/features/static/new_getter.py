import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.merge_sub_timeseries import pipeline
from thesis_floor_halkes.utils.haversine import haversine


def get_static_data_object_subgraph(
    timeseries_subgraph_path: str,
    edge_features_path: str, 
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
    edge_index = (
        torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long)
        .t()
        .contiguous()
    )
    
    edge_attr = torch.tensor(edge_features[["maxspeed", "length"]].values, dtype=torch.float)
    
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
    
    G_sub, _ = get_timeseries_subgraph()
    _, G_pt = get_timeseries_subgraph()
    
    static_data.G_sub = G_sub
    static_data.G_pt = G_pt
    
    deduped_node_ids = timeseries.drop_duplicates(
        subset=["node_id_y"]
    ).copy()
    node_id_mapping = dict(
        zip(deduped_node_ids["node_id_y"], deduped_node_ids["osmid_original"])
    )
    static_data.node_id_mapping = node_id_mapping
    
    return static_data
    

