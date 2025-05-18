from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
import osmnx as ox
import pandas as pd
import torch
import numpy as np

from thesis_floor_halkes.features.static.graph_utils import create_osmnx_sub_graph_only_inside_helmond_from_bbox
from thesis_floor_halkes.utils.haversine import haversine
from thesis_floor_halkes.features.graph.graph_generator import (
    create_osmnx_sub_graph_only_inside_helmond,
    get_edge_features_subgraph,
    get_node_features_subgraph,
    plot_sub_graph_in_and_out_nodes_helmond,
)

ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]


def get_static_data_object(
    time_series_df_path: str,
    dist: int = 500,
    start_node: int = None,
    end_node: int = None,
    seed: int = None,
):
    # get full time series dataframe
    time_series = pd.read_parquet(time_series_df_path)

    # # select a random location from full timeseries dataframe
    # location = time_series.sample(1, random_state=seed)
    # lat = location.iloc[0]["lat"]
    # lon = location.iloc[0]["lon"]
    # # lat = 51.473609
    # # lon = 5.738671

    # # create subgraph based on random location inside Helmond
    # graph, G_pt = create_osmnx_sub_graph_only_inside_helmond(
    #     lat, lon, dist, time_series
    # )
    
    graph, G_pt = create_osmnx_sub_graph_only_inside_helmond_from_bbox(5.67676, 51.47096, 5.70173, 51.47922, time_series)
    graph, G_pt = create_osmnx_sub_graph_only_inside_helmond(51.474744, 5.679176, dist = 500, timeseries_df=time_series)
    # get node features from subgraph with sorted node ids
    subgraph_nodes_df = get_node_features_subgraph(graph)
    print(f"{subgraph_nodes_df= }")
    
    start_node = 1
    end_node = 3


    # add start and end node to static data object
    # if start_node is None:
    #     start_node = np.random.choice(subgraph_nodes_df.node_id.values).item()
    # if end_node is None:
    #     end_node = np.random.choice(subgraph_nodes_df.node_id.values).item()
    #     while start_node == end_node:
    #         end_node = np.random.choice(subgraph_nodes_df.node_id.values).item()
    
    subgraph_nodes_df["start_node"] = start_node
    subgraph_nodes_df["end_node"] = end_node

    # compute distance-to-goal for each node
    goal_coords = subgraph_nodes_df.loc[
        subgraph_nodes_df["node_id"] == end_node, ["lat", "lon"]
    ].iloc[0]
    subgraph_nodes_df["distance_to_goal_meters"] = subgraph_nodes_df.apply(
        lambda row: haversine((row.lat, row.lon), (goal_coords.lat, goal_coords.lon)),
        axis=1,
    )

    # filter time series dataframe based on nodes from subgraph
    filtered_time_series_df = time_series.merge(
        subgraph_nodes_df, left_on=["lat", "lon"], right_on=["lat", "lon"], how="inner"
    )
    
    # print(filtered_time_series_df[filtered_time_series_df['has_light']==1].drop_duplicates(subset='node_id_x'))
    # check_df = filtered_time_series_df[filtered_time_series_df['node_id_y']==17]
    # print('AAAAAAAAA\n', check_df[check_df['timestamp'] == '2024-01-31 08:45:00'], '\n\n')
    # print('AAAAAAAAA\n', check_df[check_df['timestamp'] == '2024-01-31 09:00:00'], '\n\n')
    # print('AAAAAAAAA\n', check_df[check_df['timestamp'] == '2024-01-31 09:15:00'], '\n\n')
    # print('AAAAAAAAA\n', check_df[check_df['timestamp'] == '2024-01-31 09:30:00'], '\n\n')
    # print('AAAAAAAAA\n', check_df[check_df['timestamp'] == '2024-01-31 09:45:00'], '\n\n')

    # get edge features from subgraph with sorted node ids
    edge_features = get_edge_features_subgraph(graph).reset_index()
    print(f"{edge_features= }")
    print(f"{edge_features.isna().sum()= }")

    # build edge index
    edge_index = (
        torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long)
        .t()
        .contiguous()
    )

    # build static edge attributes
    edge_attr = torch.tensor(
        edge_features[["length", "maxspeed"]].values, dtype=torch.float
    )

    # build static node features based on node_id_y (sorted)
    node_features = filtered_time_series_df.drop_duplicates(subset=["node_id_y"]).copy()
    node_features = torch.tensor(
        node_features[["lat", "lon", "has_light", "distance_to_goal_meters"]].values,
        dtype=torch.float,
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
    deduped_node_ids = filtered_time_series_df.drop_duplicates(
        subset=["node_id_y"]
    ).copy()
    node_id_mapping = dict(
        zip(deduped_node_ids["node_id_y"], deduped_node_ids["node_id_x"])
    )
    static_data.node_id_mapping = node_id_mapping
    # static_data.node_id_mapping = filtered_time_series_df["node_id_x"].values

    return static_data


if __name__ == "__main__":
    get_static_data_object(
        time_series_df_path="data/processed/node_features.parquet",
        dist=50,
        seed=1,
    )
