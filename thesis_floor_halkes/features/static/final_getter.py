import json
import os
from pathlib import Path
from typing import List
import pandas as pd
import osmnx as ox
import torch
from torch_geometric.data import Data

from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, consolidate_edges_to_single, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.expand_to_min_subgraph import expand_wait_times
from thesis_floor_halkes.data_processing.merge_sub_timeseries import merge_timeseries_pipeline
from thesis_floor_halkes.utils.haversine import haversine

def get_random_locations_with_traffic_lights(
        nodes_traffic_lights_path: str, 
        seed=42, 
        num_samples=10
    ):
    traffic_lights = pd.read_parquet(nodes_traffic_lights_path)
    traffic_lights_coords = traffic_lights[['lat', 'lon']].drop_duplicates()

    random_locations = traffic_lights_coords.sample(num_samples, random_state=seed)

    lijst_locations = [(row['lat'].item(), row['lon'].item()) for _, row in random_locations.iterrows()]

    return lijst_locations

def get_random_timestamp(timeseries_path: str, 
                         seed=42):
    df = pd.read_parquet(timeseries_path)
    random_timestamp = df.sample(1, random_state=seed).index[0]
    return random_timestamp


def create_sub_network(nodes_helmond_path,
                  lat,
                  lon,
                  node_features_path,
                  edge_features_path,
                  G_cons_path,
                  G_pt_cons_path,
                  dist,
                  ):
    
    G_sub, G_pt = create_subgraph_inside_helmond(nodes_helmond_path, lat, lon, dist=dist)
    G_cons = consilidate_subgraph(G_sub)
    G_pt_cons = consilidate_subgraph(G_pt)
    
    G_cons = consolidate_edges_to_single(G_cons, weight_attr="length", agg="min")
    
    node_features = get_node_features_subgraph(G_cons)
    edge_features = get_edge_features_subgraph(G_cons)
    
    ox.io.save_graphml(G_cons, G_cons_path)
    ox.io.save_graphml(G_pt_cons, G_pt_cons_path)
    
    node_features.to_parquet(node_features_path)
    edge_features.to_parquet(edge_features_path)
    
    return G_cons, G_pt_cons

def get_timeseries_sub_network( node_features_path, 
                                meta_path, 
                                measurement_path,
                                output_path_timeseries,
                                threshold,
                                ):
    
    timeseries_subgraph = merge_timeseries_pipeline(meta_path, 
                                   node_features_path,
                                   measurement_path,
                                   threshold)
    
    expanded_timeseries = expand_wait_times(timeseries_subgraph, num_peaks=2,
                                            amp_frac=0.1, sigma=1.0)
    
    expanded_timeseries.to_parquet(output_path_timeseries, index=False)
    
    return expanded_timeseries

def generate_train_data(
    nodes_helmond_path: str,
    meta_path: str,
    measurement_path: str,
    nodes_traffic_lights_path: str,
    base_output_dir: str = "data/training_data",
    num_samples: int = 5,
    threshold: float = 25,
    dist: int = 600,
    seed: int = 0
):
    base = Path(base_output_dir)
    successfully_created = 0
    seed = seed
    random_nodes_with_traffic_lights = get_random_locations_with_traffic_lights(nodes_traffic_lights_path=nodes_traffic_lights_path,
                                                                                seed = seed, 
                                                                                num_samples=num_samples,)
    
    coord_dict = {}
    
    for coord in random_nodes_with_traffic_lights:       
    
        try:
            # Create subgraph directory
            subdir = base / f"network_{successfully_created}"
            subdir.mkdir(parents=True, exist_ok=True)

            
            node_features_path = subdir / "node_metadata.parquet"
            edge_features_path = subdir / "edge_features.parquet"
            output_path_timeseries = subdir / "timeseries.parquet"
            G_cons_path = subdir / "G_cons.graphml"
            G_pt_cons_path = subdir / "G_pt_cons.graphml"

            # Build subgraph
            G_cons, _ = create_sub_network(
                nodes_helmond_path=nodes_helmond_path,
                node_features_path=str(node_features_path),
                edge_features_path=str(edge_features_path),
                lat= coord[0],
                lon= coord[1],
                G_cons_path= G_cons_path,
                G_pt_cons_path= G_pt_cons_path,
                dist=dist,
            )
            
            # Generate timeseries
            ts = get_timeseries_sub_network(
                node_features_path=str(node_features_path),
                meta_path=meta_path,
                measurement_path=measurement_path,
                output_path_timeseries=str(output_path_timeseries),
                threshold=threshold,
            )

            print(f"‣ Finished subgraph {successfully_created}:")
            print(f"   • nodes: {G_cons.number_of_nodes()}  edges: {G_cons.number_of_edges()}")
            print(f"   • timeseries shape: {ts.shape}")
            coord_dict.update({coord: successfully_created})
            print(coord_dict)

            successfully_created += 1

        except OverflowError as e:
            print(f"Error creating subgraph {successfully_created}: {e}")
            successfully_created += 1
            continue
        
    with open(base_output_dir, 'w') as fp:
        json.dump(coord_dict, fp, indent=4)



import numpy as np

def get_static_data_object_subgraph(
    timeseries_subgraph_path: str,
    edge_features_path: str, 
    G_cons_path: str,
    G_pt_cons_path: str,
    start_node: int = None,
    end_node: int = None,
    device: torch.device = torch.device("cpu")
    
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
    edge_features = edge_features.reset_index()
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


def create_and_save_static_data_objects(
    root_dir: str,
    objects_per_network: int = 5,
    device: torch.device = torch.device("cpu")
) -> List[str]:
    
    saved_paths = []

    # Loop through all network subdirectories
    for network_id in os.listdir(root_dir):
        network_path = os.path.join(root_dir, network_id)
        if not os.path.isdir(network_path):
            continue

        # Define expected file paths
        timeseries_path = os.path.join(network_path, "timeseries.parquet")
        edge_features_path = os.path.join(network_path, "edge_features.parquet")
        G_cons_path = os.path.join(network_path, "G_cons.graphml")
        G_pt_cons_path = os.path.join(network_path, "G_pt_cons.graphml")

        for _ in range(objects_per_network):
            try:
                static_data = get_static_data_object_subgraph(
                    timeseries_path,
                    edge_features_path,
                    G_cons_path,
                    G_pt_cons_path
                )
                static_data = static_data.to(device)

                # Construct filename
                filename = f"{network_id}_{static_data.start_node}_{static_data.end_node}.pt"
                file_path = os.path.join(network_path, filename)

                # Save the object
                torch.save(static_data, file_path)
                saved_paths.append(file_path)
                print(f"Saved: {file_path}")

            except Exception as e:
                print(f"Error in network '{network_id}': {e}")

    return saved_paths



if __name__ == "__main__":
    generate_train_data(
        nodes_traffic_lights_path="data/processed/traffic_lights_in_helmond.parquet",
        nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
        meta_path="data/processed/intersection_metadata.csv",
        measurement_path="data/processed/intersection_measurements_31_01_24.csv",)
    
    create_and_save_static_data_objects(
        root_dir="data/training_data",
        objects_per_network=2,
        device=torch.device("cpu")
    )