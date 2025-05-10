import pandas as pd 
import torch
import numpy as np
from torch_geometric.data import Data


node_df = pd.read_parquet("data/processed/node_features.parquet")
edge_df = pd.read_parquet("data/processed/edge_features_helmond.parquet")

def build_static_data_helmond(node_df, edge_df):
    static_node_df = (
        node_df
        .drop_duplicates(subset='node_id', keep='first')
        .sort_values('node_id')
    )

    static_vals = static_node_df[['lat','lon','has_light']]\
                    .to_numpy(dtype=np.float32)   # already a single ndarray
    static_x = torch.from_numpy(static_vals)  

    edge_index = torch.tensor([
        edge_df['u'].values,
        edge_df['v'].values
    ], dtype=torch.long)

    edge_attr = torch.tensor(
        edge_df[['length', 'maxspeed']].values, dtype=torch.float32
    )

    static_data = Data(
        x = static_x,
        edge_index = edge_index, 
        edge_attr = edge_attr
    )

    return static_data
    

def build_dynamic_data_helmond(node_df, edge_df):
    
