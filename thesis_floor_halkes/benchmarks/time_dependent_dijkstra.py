from torch_geometric.data import Data
import pandas as pd
import heapq
import torch

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.graph.graph_generator import create_osmnx_sub_graph_only_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph, plot_sub_graph_in_and_out_nodes_helmond
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.haversine import haversine



def get_static_data_object(
    time_series_df_path: str,
    start_node: int = None,
    end_node: int = None,
    seed: int = None,
):
    # get full time series dataframe
    time_series = pd.read_parquet(time_series_df_path)
    
    graph, G_pt = create_osmnx_sub_graph_only_inside_helmond(51.474744, 5.679176, dist = 50, timeseries_df=time_series)
    subgraph_nodes_df = get_node_features_subgraph(graph)
    print(f"Subgraph nodes\n: {subgraph_nodes_df}\n")
    
    start_node = 0
    end_node = 3

    subgraph_nodes_df["start_node"] = start_node
    subgraph_nodes_df["end_node"] = end_node
    print(f"Subgraph nodes with start and end node:\n {subgraph_nodes_df}\n")

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
    print(f"Filtered time series dataframe:\n {filtered_time_series_df}\n")

    # get edge features from subgraph with sorted node ids
    edge_features = get_edge_features_subgraph(graph).reset_index()
    print(f"Edge features:\n {edge_features}\n")

    # build edge index
    edge_index = (
        torch.tensor(edge_features[["u", "v"]].values, dtype=torch.long)
        .t()
        .contiguous()
    )
    print(f"Edge index:\n {edge_index}\n")

    # build static edge attributes
    edge_attr = torch.tensor(
        edge_features[["length", "maxspeed"]].values, dtype=torch.float
    )
    print(f"Edge attributes:\n {edge_attr}\n")

    # build static node features based on node_id_y (sorted)
    node_features = filtered_time_series_df.drop_duplicates(subset=["node_id_y"]).copy()
    print(f"Node features:\n {node_features}\n")
    node_features = torch.tensor(
        node_features[["lat", "lon", "has_light", "distance_to_goal_meters"]].values,
        dtype=torch.float,
    )
    print(f"Node features tensor:\n {node_features}\n")

    # build static data object
    static_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    print(f"Static data:\n {static_data}\n")

    # add filtered time series dataframe to static data object
    static_data.filtered_time_series_df = filtered_time_series_df

    # add start and end node to static data object
    static_data.start_node = start_node
    static_data.end_node = end_node

    fig, ax = plot_sub_graph_in_and_out_nodes_helmond(graph, G_pt)
    
    static_data.G_sub = graph
    static_data.G_pt = G_pt
    
    deduped_node_ids = filtered_time_series_df.drop_duplicates(
        subset=["node_id_y"]
    ).copy()
    node_id_mapping = dict(
        zip(deduped_node_ids["node_id_y"], deduped_node_ids["node_id_x"])
    )
    
    static_data.node_id_mapping = node_id_mapping
    print(f"End static data object:\n {static_data}\n")
    return static_data



def time_dependent_dijkstra(
    static_data, 
    dynamic_feature_getter, 
    dynamic_node_idx,
    static_node_idx,
    static_edge_idx,
    start_time_stamp,
):
    df = static_data.filtered_time_series_df.copy()
    start_node = static_data.start_node
    print(f"Start node: {start_node}")
    
    df_start = df[df['node_id_y'] == start_node]    
    print(f"start df:\n {df_start}\n")
    time_stamps = sorted(df_start["timestamp"].unique())
    print(f"Time stamps:\n {time_stamps}\n")
    ts0 = pd.to_datetime(start_time_stamp)
    t0 = time_stamps.index(ts0)

    T = len(time_stamps)
    
    print(f"INIT start node: {start_node}, t0: {t0}, T: {T}")
    
    adj = build_adjecency_matrix(static_data.num_nodes, static_data)
    print(f"Adjacency matrix:\n {adj}\n")
    start = static_data.start_node
    end = static_data.end_node
    
    class _TmpEnv:
        pass 
    tmp = _TmpEnv()
    tmp.static_data = static_data
    tmp.static_node_idx = static_node_idx
    
    heap = [(0.0, start, t0, [start])]
    print(f"Heap: {heap}")
    best = {(start, t0): 0.0}

    while heap: 
        cost, node, t_idx, path = heapq.heappop(heap)
        print(f"[POP] cost={cost:.3f}, node={node}, t_idx={t_idx}, path={path}")
        if cost > best.get((node, t_idx), float("inf")):
            print("       → skipping stale entry")
            continue
        if node == end:
            print(f"[FOUND] cost={cost:.3f}, path={path}")
            return cost, path
        next_t = t_idx + 1
        if next_t >= T:
            print(f"       → next_t={next_t} ≥ T={T}, no further expansion")
            continue
        print(f"       → expanding at time index {next_t} (ts={time_stamps[next_t]})")
    
        dyn = dynamic_feature_getter.get_dynamic_features(
            environment = tmp, 
            traffic_light_idx = static_node_idx["has_light"],
            current_node = node, 
            visited_nodes = path,
            time_step = next_t, 
            sub_node_df = df,
        )
        wait_times = dyn.x[:, dynamic_node_idx["wait_time"]]
        
        for nbr, eidx in adj[node]:
            length = static_data.edge_attr[eidx, static_edge_idx["length"]]
            speed = static_data.edge_attr[eidx, static_edge_idx["speed"]]
            travel_time = length / (speed / 3.6)
            wait = wait_times[nbr].item()
            
            ts = time_stamps[next_t]
            orig_wait_series = df[
                (df["node_id_y"] == nbr) & (df["timestamp"] == ts)
            ]
            print(f"  Original wait series:\n {orig_wait_series}\n")
            print(f"node = {nbr} @ ts={ts}"
                  f"> dyn.get() = {wait:.2f}s ")
            
            
            new_cost = cost + travel_time + wait
            
            print(
                    f"         edge {node}→{nbr}: travel={travel_time:.2f}, "
                    f"wait={wait:.2f} → new_cost={new_cost:.2f}"
                )
            
            key = (nbr, next_t)
            if new_cost < best.get(key, float("inf")):
                best[key] = new_cost
                heapq.heappush(heap, (new_cost, nbr, next_t, path + [nbr]))
                print(f"           • pushed (cost={new_cost:.2f}, node={nbr}, t={next_t})")
        
    return None, float("inf")  # No path found
        
        
    

if __name__ == "__main__":
    time_series_path = "data/processed/node_features_expanded.parquet"
    static_data = get_static_data_object(time_series_path)
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
    
    dynamic_feature_getter = DynamicFeatureGetterDataFrame()
    
    cost, route = time_dependent_dijkstra(static_data, dynamic_feature_getter, 
                            dynamic_node_idx, static_node_idx, 
                            static_edge_idx,
                            "2024-01-31 08:30:00")
    
    print(f"TD Dijkstra cost: {cost}, route: {route}")
    
