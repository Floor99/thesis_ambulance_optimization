import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import osmnx as ox

from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.create_training_data import get_sub_graph, get_timeseries_subgraph
from thesis_floor_halkes.utils.haversine import haversine
from torch_geometric.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long, device=device)
        .t()
        .contiguous()
    )
    
    edge_attr = torch.tensor(edge_features[["length", "maxspeed"]].values, dtype=torch.float, device=device)
    
    node_features = timeseries.drop_duplicates(subset=["node_id"]).copy()
    node_features = torch.tensor(
        node_features[["lat", "lon", "has_light", "distance_to_goal_meters"]].values,
        dtype=torch.float,
        device=device,
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
    # static_data.node_id_mapping = node_id_mapping
    
    static_data.timeseries = timeseries
    
    return static_data
    

# def collect_static_data_objects(
#     base_dir: str = "data/training_data",
#     start_node: int = None,
#     end_node: int = None,
# ):
#     static_list = []
#     for sub in os.listdir(base_dir):
#         subpath = os.path.join(base_dir, sub)
#         if not os.path.isdir(subpath):
#             continue
        
#         timeseries_path = os.path.join(subpath, "timeseries.parquet")
#         edge_features_path = os.path.join(subpath, "edge_features.parquet")
#         G_cons_path = os.path.join(subpath, "G_cons.graphml")
#         G_pt_cons_path = os.path.join(subpath, "G_pt_cons.graphml")
        
#         static_object = get_static_data_object_subgraph(
#             timeseries_subgraph_path=timeseries_path,
#             edge_features_path=edge_features_path,
#             G_cons_path=G_cons_path,
#             G_pt_cons_path=G_pt_cons_path,
#             start_node=start_node,
#             end_node=end_node,
#         )
        
#         static_list.append(static_object)
        
#     return static_list

def collect_static_data_objects(
    base_dir: str = "data/training_data",
    subgraph_dirs=None,
    num_pairs_per_graph: int = 5,
    seed: int = 42,
):
    static_list = []
    rng = np.random.default_rng(seed)
    if subgraph_dirs is None:
        subgraph_dirs = [d for d in os.listdir(base_dir) if d.startswith("subgraph_")]
    for sub in subgraph_dirs:
        subpath = os.path.join(base_dir, sub)
        if not os.path.isdir(subpath):
            continue

        timeseries_path = os.path.join(subpath, "timeseries.parquet")
        edge_features_path = os.path.join(subpath, "edge_features.parquet")
        G_cons_path = os.path.join(subpath, "G_cons.graphml")
        G_pt_cons_path = os.path.join(subpath, "G_pt_cons.graphml")

        # Load timeseries once to get node ids
        timeseries = pd.read_parquet(timeseries_path)
        node_ids = timeseries["node_id"].unique()

        for _ in range(num_pairs_per_graph):
            start_node = rng.choice(node_ids)
            end_node = rng.choice(node_ids)
            while start_node == end_node:
                end_node = rng.choice(node_ids)

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

import os
import random

def split_subgraphs(base_dir, train_frac=0.7, val_frac=0.15, seed=42):
    subgraphs = [d for d in os.listdir(base_dir) if d.startswith("subgraph_")]
    random.seed(seed)
    random.shuffle(subgraphs)
    n = len(subgraphs)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = subgraphs[:n_train]
    val = subgraphs[n_train:n_train+n_val]
    test = subgraphs[n_train+n_val:]
    return train, val, test


# class StaticDataObjectSet(Dataset):
#     def __init__(self, base_dir: str, transform=None):
#         super(StaticDataObjectSet, self).__init__(transform=transform)
#         self.data_objects = collect_static_data_objects(base_dir=base_dir)
        
#     def len(self):
#         return len(self.data_objects)
    
#     def get(self, idx):
#         return self.data_objects[idx]

class StaticDataObjectSet(Dataset):
    def __init__(self, base_dir: str, subgraph_dirs=None, num_pairs_per_graph=5, seed=42, transform=None):
        super(StaticDataObjectSet, self).__init__(transform=transform)
        self.data_objects = collect_static_data_objects(
            base_dir=base_dir,
            subgraph_dirs=subgraph_dirs,
            num_pairs_per_graph=num_pairs_per_graph,
            seed=seed,
        )

    def len(self):
        return len(self.data_objects)

    def get(self, idx):
        return self.data_objects[idx]


if __name__ == "__main__":
    static_list = collect_static_data_objects()
    print(static_list)