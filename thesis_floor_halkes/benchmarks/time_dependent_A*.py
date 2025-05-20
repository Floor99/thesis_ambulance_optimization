import heapq
import pandas as pd

from thesis_floor_halkes.features.static.new_getter import get_static_data_object_subgraph
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix

def time_dependent_a_star(
    static_data,
    # dynamic_feature_getter,
    # dynamic_node_idx,
    static_node_idx,
    static_edge_idx,
    start_time_stamp,
):
    timeseries = static_data.timeseries.copy()
    start_node = static_data.start_node
    print(f"Start node: {start_node}")
    
    timeseries_start = timeseries[timeseries['node_id'] == start_node]
    time_stamps = sorted(timeseries_start["timestamp"].unique())
    print(f"Time stamps:\n {time_stamps}\n")
    ts0 = pd.to_datetime(start_time_stamp)
    t0 = time_stamps.index(ts0)
    T = len(time_stamps)
    
    adj = build_adjecency_matrix(static_data.num_nodes, static_data)
    end_node = static_data.end_node
    
    max_speed = static_data.edge_attr[:, static_edge_idx['speed']].max().item()
    dist_to_goal = static_data.x[:, static_node_idx['dist_to_goal']]
    print(f"Max speed: {max_speed}")
    print(f"Distance to goal: {dist_to_goal}")
    
    
    return 


if __name__ == "__main__":
    static_data = get_static_data_object_subgraph(
        timeseries_subgraph_path="data/training_data/subgraph_0/timeseries.parquet",
        edge_features_path="data/training_data/subgraph_0/edge_features.parquet",
        G_cons_path="data/training_data/subgraph_0/G_cons.graphml",
        G_pt_cons_path="data/training_data/subgraph_0/G_pt_cons.graphml",
    )
    
    dynamic_node_idx = {
        "status": 0,
        "wait_time": 1,
        "current_node": 2,
        "visited_nodes": 3,
    }

    static_node_idx = {
        "lat": 0,
        "lon": 1,
        "has_light": 2,
        "dist_to_goal": 3,
    }

    static_edge_idx = {
        "length": 0,
        "speed": 1,
    }
    
    time_dependent_a_star(static_data, static_node_idx, static_edge_idx,"2024-01-31 08:30:00")