from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
import osmnx as ox

from thesis_floor_halkes.utils.haversine import haversine

ox.settings.bidirectional_network_types = ['drive', 'walk', 'bike']

# class StaticFeatureGetter(ABC):
#     """
#     Abstract base class for static feature getters.
#     """

#     @abstractmethod
#     def get_static_features(self):
#         """
#         Abstract method to get static features.
#         """
#         pass


# class StaticFeatureFromListGetter(StaticFeatureGetter):
#     """
#     Static feature getter that retrieves features from a list.
#     """

#     def __init__(self, static_features: list[Data]):
#         """
#         Initialize the static feature getter with a list of static features.

#         Args:
#             static_features: List of Data objects with static features.
#         """
#         pass


# import pandas as pd
# import torch
# import numpy as np
# import osmnx as ox
# import networkx as nx
# from torch_geometric.data import Dataset
# from thesis_floor_halkes.data_processing.open_street_map_coords import (
#     create_osmnx_subgraph_from_coordinates,
#     create_node_coordinates_dataframe_from_osmnx_graph,
# )


# def get_static_data_object(
#     time_series_df_path: str,
#     edge_df_path: str,
#     dist: int = 500,
#     start_node: int = None,
#     end_node: int = None,
#     seed: int = None,
# ):
#     # get full time series dataframe
#     time_series = pd.read_parquet(time_series_df_path)

#     # get full edge dataframe
#     # edge_df = pd.read_parquet(edge_df_path)

#     # select a random location from full timeseries dataframe
#     location = time_series.sample(1, random_state=seed)
#     lat = location.iloc[0]["lat"]
#     lon = location.iloc[0]["lon"]

#     # create subgraph based on random location
#     graph = create_osmnx_subgraph_from_coordinates(lat, lon, dist=dist)
    
#     edge_features = get_edge_features_from_osmnx(graph).reset_index()

#     # filter time series dataframe based on nodes from subgraph
#     subgraph_nodes_df = create_node_coordinates_dataframe_from_osmnx_graph(graph)
#     filtered_time_series_df = time_series.merge(
#         subgraph_nodes_df, left_on=["lat", "lon"], right_on=["lat", "lon"], how="inner"
#     )
#     print(f"{filtered_time_series_df= }")
#     # subset edge dataframe based on nodes from subgraph
#     # filtered_edge_df = edge_df[
#     #     edge_df["u"].isin(filtered_time_series_df.node_id)
#     #     & edge_df["v"].isin(filtered_time_series_df.node_id)
#     # ]
    
#     # build edge index
#     edge_index = (
#         torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long)
#         .t()
#         .contiguous()
#     )

#     # build static edge attributes
#     edge_attr = torch.tensor(
#         edge_features[["length", "maxspeed"]].values, dtype=torch.float
#     )

#     # build static node features
#     node_features = filtered_time_series_df.drop_duplicates(subset=["node_id_y"]).copy()
#     node_features = torch.tensor(
#         node_features[["lat", "lon", "has_light"]].values, dtype=torch.float
#     )

#     # build static data object
#     static_data = Data(
#         x=node_features,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#     )

#     # add filtered time series dataframe to static data object
#     static_data.filtered_time_series_df = filtered_time_series_df

#     # add start and end node to static data object
#     if start_node is None:
#         # randomly select start node
#         start_node = np.random.choice(filtered_time_series_df.node_id_y.values)
#     if end_node is None:
#         # randomly select end node
#         end_node = np.random.choice(filtered_time_series_df.node_id_y.values)
#         while start_node == end_node:
#             end_node = np.random.choice(
#                 filtered_time_series_df.node_id_y.values
#             )
#     static_data.start_node = start_node
#     static_data.end_node = end_node

#     # add filtered edge dataframe to static data object
#     static_data.filtered_edge_df = edge_features

#     return static_data


# def create_osmnx_subgraph_from_coordinates(lat, lon, dist):
#     G = ox.graph_from_point(
#         (lat, lon),
#         dist=dist,
#         network_type="drive",
#         simplify=True,
#         retain_all=False,
#         truncate_by_edge=True,
#     )
#     # G = G.to_undirected()
#     # G = nx.convert_node_labels_to_integers(G, ordering="sorted", first_label=0)

#     return G


# def create_node_coordinates_dataframe_from_osmnx_graph(G):
#     G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)    
#     nodes, _ = ox.graph_to_gdfs(G)

#     nodes = nodes[["y", "x"]]  # .reset_index(names='node_id')
#     nodes = nodes.rename(columns={"y": "lat", "x": "lon"})
#     return nodes

# def clean_numeric(val, default=50.0):
#     if isinstance(val, list):
#         val = val[0]  # If it's a list, pick the first value
#     try:
#         return float(val)
#     except (ValueError, TypeError):
#         return default 
    
# def get_edge_features_from_osmnx(G):
#     G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)    
#     _, edges = ox.graph_to_gdfs(G)
    
#     edges = edges[['maxspeed', 'length']]
#     edges['maxspeed'] = edges['maxspeed'].apply(lambda x: clean_numeric(x, default=50.0))
#     edges['length'] = edges['length'].apply(lambda x: clean_numeric(x, default=30.0))  # 30m as a reasonable default)
#     edges['maxspeed'] = edges['maxspeed'].fillna(50.0)
#     edges['length'] = edges['length'].fillna(30.0)
    
#     return edges


# # def get_static_data_object(final_node_df, final_edge_df):
# #     node_for_coords_graph = final_node_df.sample(1)
# #     lat = node_for_coords_graph.iloc[0]["lat"]
# #     lon = node_for_coords_graph.iloc[0]["lon"]

# #     graph = create_osmnx_subgraph_from_coordinates(lat, lon, dist=50)
# #     print("Edge INDEX: ", graph.edges())
# #     nodes = create_node_coordinates_dataframe_from_osmnx_graph(graph)
# #     df_filtered = final_node_df.merge(nodes, on=["lat", "lon"], how="inner")

# #     nodes_idx = df_filtered.set_index("node_id_x")
# #     mask_u = final_edge_df["u"].isin(nodes_idx.index)
# #     mask_v = final_edge_df["v"].isin(nodes_idx.index)
# #     filtered_edges = final_edge_df[mask_u & mask_v]

# #     # 1) Build your cleaned & sorted node list
# #     static_node_df = df_filtered.drop_duplicates(
# #         subset="node_id_x", keep="first"
# #     ).sort_values("node_id_x")

# #     # 2) Create a mapping old_id → new_id (0…N-1)
# #     orig_ids = static_node_df["node_id_x"].to_numpy()
# #     old2new = {old: new for new, old in enumerate(orig_ids)}

# #     # 3) Map your edges into the new index space
# #     u_new = filtered_edges["u"].map(old2new).to_numpy()
# #     v_new = filtered_edges["v"].map(old2new).to_numpy()

# #     # 4) Build your tensors
# #     x = torch.from_numpy(
# #         static_node_df[["lat", "lon", "has_light"]].to_numpy(dtype=np.float32)
# #     )

# #     edge_index = torch.from_numpy(np.vstack([u_new, v_new])).long()
# #     edge_attr = torch.from_numpy(
# #         filtered_edges[["length", "maxspeed"]].to_numpy(dtype=np.float32)
# #     )

# #     orig_id = torch.from_numpy(orig_ids).long()

# #     # # 4) Remap u/v → contiguous 0…N-1 and make them torch tensors
# #     # u = torch.tensor(filtered_edges['u'].map(old2new).tolist(), dtype=torch.long)
# #     # v = torch.tensor(filtered_edges['v'].map(old2new).tolist(), dtype=torch.long)

# #     # # 5) Duplicate edges for undirected traversal
# #     # edge_index = torch.cat([
# #     #     torch.stack([u, v], dim=0),
# #     #     torch.stack([v, u], dim=0)
# #     # ], dim=1)  # shape: (2, 2E)

# #     # # 6) Edge attributes (length, maxspeed), duplicated as well
# #     # edge_attr_single = torch.tensor(
# #     #     filtered_edges[['length','maxspeed']].values,
# #     #     dtype=torch.float32
# #     # )  # shape: (E, 2)
# #     # edge_attr = torch.cat([edge_attr_single, edge_attr_single], dim=0)  # (2E, 2)

# #     # # 7) Node features x = [lat, lon, has_light]
# #     # x = torch.tensor(
# #     #     static_node_df[['lat','lon','has_light']].values,
# #     #     dtype=torch.float32
# #     # )  # shape: (N, 3)

# #     # 5) If you ever need original IDs later, stick them on:
# #     static_data = Data(
# #         x=x,
# #         edge_index=edge_index,
# #         edge_attr=edge_attr,
# #         orig_id=orig_id,
# #     )

# #     num_nodes = static_data.x.size(0)
# #     static_data.start_node = torch.randint(0, num_nodes, (1,)).item()
# #     static_data.end_node = torch.randint(0, num_nodes, (1,)).item()

# #     while static_data.start_node == static_data.end_node:
# #         static_data.end_node = torch.randint(0, num_nodes, (1,)).item()
# #     static_data.sub_nodes_df = df_filtered

# #     return static_data


# class StaticDataSet(Dataset):
#     def __init__(self, num_graphs: int):
#         super().__init__()
#         self.num_graphs = num_graphs
#         self.data_list = [
#             get_static_data_object(
#                 "data/processed/node_features.parquet",
#                 "data/processed/edge_features_helmond.parquet",
#                 dist=100,
#             )
#             for _ in range(num_graphs)
#         ]

#     def len(self):
#         return self.num_graphs

#     def get(self, idx: int):
#         return self.data_list[idx]

# import networkx as nx
# import pandas as pd 
# import numpy as np

# def get_static_data_object(
#     time_series_df_path: str,
#     edge_df_path: str,
#     dist: int = 500,
#     start_node: int = None,
#     end_node: int = None,
#     seed: int = None,
# ):
#     # get full time series dataframe
#     time_series = pd.read_parquet(time_series_df_path)
    
#     # get full edge dataframe
#     # edge_df = pd.read_parquet(edge_df_path)
    
#     # select a random location from full timeseries dataframe
#     location = time_series.sample(1, random_state=seed)
#     lat = location.iloc[0]["lat"]
#     lon = location.iloc[0]["lon"]
#     lat = 51.4657
#     lon = 5.661920
    
#     # create subgraph based on random location
#     graph = create_osmnx_subgraph_from_coordinates(lat, lon, dist=dist)
    
#     edge_features = get_edge_features_from_osmnx(graph).reset_index()
#     print(f"{edge_features= }")
#     # filter time series dataframe based on nodes from subgraph
#     subgraph_nodes_df = create_node_coordinates_dataframe_from_osmnx_graph(graph)
#     print(f"{subgraph_nodes_df= }")
#     filtered_time_series_df = time_series.merge(
#         subgraph_nodes_df, left_on=["lat", "lon"], right_on=["lat", "lon"], how="inner"
#     )
#     print(f"{filtered_time_series_df= }")
#     # subset edge dataframe based on nodes from subgraph
#     # filtered_edge_df = edge_features[
#     #     edge_features["u"].isin(filtered_time_series_df.node_id)
#     #     & edge_features["v"].isin(filtered_time_series_df.node_id)
#     # ]
#     # print(filtered_edge_df)
    
#     # build edge index
#     edge_index = torch.tensor(
#         edge_features[["u", "v"]].values, dtype=torch.long
#     ).t().contiguous()

#     # build static edge attributes
#     edge_attr = torch.tensor(
#         edge_features[["length", "maxspeed"]].values, dtype=torch.float
#     )

#     # build static node features
#     node_features = filtered_time_series_df.drop_duplicates(subset=["node_id_y"]).copy()
#     node_features = torch.tensor(
#         node_features[["lat", "lon", "has_light"]].values, dtype=torch.float
#     )
#     print(f"{node_features= }")
    
#     # build static data object
#     static_data = Data(
#         x=node_features,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#     )
    
#     # add filtered time series dataframe to static data object
#     static_data.filtered_time_series_df = filtered_time_series_df
    
#     # add start and end node to static data object
#     if start_node is None:
#         # randomly select start node
#         start_node = np.random.choice(filtered_time_series_df.node_id_y.values).item()
#     if end_node is None:
#         # randomly select end node
#         end_node = np.random.choice(filtered_time_series_df.node_id_y.values).item()
#         while start_node == end_node:
#             end_node = np.random.choice(filtered_time_series_df.node_id_y.values).item()
#     static_data.start_node = start_node
#     static_data.end_node = end_node

#     # add filtered edge dataframe to static data object
#     static_data.filtered_edge_df = edge_features
    
#     return static_data


# def create_osmnx_subgraph_from_coordinates(lat,lon, dist):
#     G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True)
#     # G = G.to_undirected()
#     # G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)
    
#     return G

# def create_node_coordinates_dataframe_from_osmnx_graph(G):
#     G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)
#     nodes, _ = ox.graph_to_gdfs(G)
    
#     nodes = nodes[['y', 'x']].reset_index(names='node_id')
#     nodes = nodes.rename(columns={'y': 'lat', 'x': 'lon'})
#     return nodes

# def clean_numeric(val, default=50.0):
#     if isinstance(val, list):
#         val = val[0]  # If it's a list, pick the first value
#     try:
#         return float(val)
#     except (ValueError, TypeError):
#         return default 

# def get_edge_features_from_osmnx(G):
#     G = nx.convert_node_labels_to_integers(G, ordering='sorted', first_label=0)    
#     _, edges = ox.graph_to_gdfs(G)
    
#     edges = edges[['maxspeed', 'length']]
#     edges['maxspeed'] = edges['maxspeed'].apply(lambda x: clean_numeric(x, default=50.0))
#     edges['length'] = edges['length'].apply(lambda x: clean_numeric(x, default=30.0))  # 30m as a reasonable default)
#     edges['maxspeed'] = edges['maxspeed'].fillna(50.0)
#     edges['length'] = edges['length'].fillna(30.0)
    
#     return edges

import pandas as pd
import torch
import numpy as np
import osmnx as ox
import networkx as nx
from torch_geometric.data import Data

from thesis_floor_halkes.features.graph.graph_generator import create_osmnx_sub_graph_only_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph, plot_sub_graph_in_and_out_nodes_helmond
ox.settings.bidirectional_network_types = ['drive', 'walk', 'bike']

def get_static_data_object(
    time_series_df_path: str,
    dist: int = 500,
    start_node: int = None,
    end_node: int = None,
    seed: int = None,
):
        # get full time series dataframe
    time_series = pd.read_parquet(time_series_df_path)
    
        # select a random location from full timeseries dataframe
    location = time_series.sample(1, random_state=seed)
    lat = location.iloc[0]["lat"]
    lon = location.iloc[0]["lon"]
    # lat = 51.473609
    # lon = 5.738671
    
        # create subgraph based on random location inside Helmond
    graph, G_pt = create_osmnx_sub_graph_only_inside_helmond(lat, lon, dist, time_series)
    
        # get node features from subgraph with sorted node ids
    subgraph_nodes_df = get_node_features_subgraph(graph)
    print(f"{subgraph_nodes_df= }")
    
        # add start and end node to static data object
    if start_node is None:
        start_node = np.random.choice(subgraph_nodes_df.node_id.values).item()
    if end_node is None:
        end_node = np.random.choice(subgraph_nodes_df.node_id.values).item()
        while start_node == end_node:
            end_node = np.random.choice(subgraph_nodes_df.node_id.values).item()
    subgraph_nodes_df["start_node"] = start_node
    subgraph_nodes_df["end_node"] = end_node

        # compute distance-to-goal for each node
    goal_coords = subgraph_nodes_df.loc[subgraph_nodes_df["node_id"] == end_node, ["lat", "lon"]].iloc[0]
    subgraph_nodes_df["distance_to_goal_meters"] = subgraph_nodes_df.apply(
        lambda row: haversine(
            (row.lat, row.lon), (goal_coords.lat, goal_coords.lon)
        ), axis=1
    )
    
        # filter time series dataframe based on nodes from subgraph
    filtered_time_series_df = time_series.merge(
        subgraph_nodes_df, left_on=["lat", "lon"], right_on=["lat", "lon"], how="inner"
    )
    print(f"{filtered_time_series_df= }")
    
        # get edge features from subgraph with sorted node ids
    edge_features = get_edge_features_subgraph(graph).reset_index()
    
        # build edge index
    edge_index = torch.tensor(
        edge_features[["u", "v"]].values, dtype=torch.long
    ).t().contiguous()

        # build static edge attributes
    edge_attr = torch.tensor(
        edge_features[["length", "maxspeed"]].values, dtype=torch.float
    )

        # build static node features based on node_id_y (sorted)
    node_features = filtered_time_series_df.drop_duplicates(subset=["node_id_y"]).copy()
    node_features = torch.tensor(
        node_features[["lat", "lon", "has_light", "distance_to_goal_meters"]].values, dtype=torch.float
    )
    
        # build static data object
    static_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    
        # add filtered time series dataframe to static data object
    static_data.filtered_time_series_df = filtered_time_series_df
    
        # add start and end node to static data object
    static_data.start_node = start_node
    static_data.end_node = end_node

    plot_sub_graph_in_and_out_nodes_helmond(graph, G_pt)
    static_data.G_sub = graph
    static_data.G_pt = G_pt
    deduped_node_ids = filtered_time_series_df.drop_duplicates(subset=["node_id_y"]).copy()
    node_id_mapping = dict(zip(deduped_node_ids["node_id_y"], deduped_node_ids["node_id_x"]))
    print(node_id_mapping)
    static_data.old_node_id = node_id_mapping
    # static_data.old_node_id = filtered_time_series_df["node_id_x"].values
    print(f"{static_data= }")
    print(f"{static_data.old_node_id= }")
        
    return static_data

if __name__ == "__main__":
    get_static_data_object(
        time_series_df_path="data/processed/node_features.parquet",
        dist=50,
        seed=1,
    )
