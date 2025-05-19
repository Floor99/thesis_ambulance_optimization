from pathlib import Path
import pandas as pd

from thesis_floor_halkes.data_processing.create_subgraph import consilidate_subgraph, create_subgraph_inside_helmond, get_edge_features_subgraph, get_node_features_subgraph
from thesis_floor_halkes.data_processing.expand_to_min_subgraph import expand_wait_times
from thesis_floor_halkes.data_processing.merge_sub_timeseries import merge_timeseries_pipeline



def get_timeseries_subgraph(nodes_helmond_path, 
                                     node_features_path, 
                                     edge_features_path,
                                     meta_path, 
                                     measurement_path,
                                     output_path_timeseries,
                                     threshold,
                                     dist, 
                                     seed,
                                     ):
    nodes_helmond = pd.read_parquet(nodes_helmond_path)
    random_location = nodes_helmond.sample(1, random_state=seed)
    lat = random_location.iloc[0]["lat"]
    lon = random_location.iloc[0]["lon"]
    
    G_sub, G_pt = create_subgraph_inside_helmond(nodes_helmond_path, lat, lon, dist=dist)
    G_cons = consilidate_subgraph(G_sub)
    node_features = get_node_features_subgraph(G_cons)
    edge_features = get_edge_features_subgraph(G_cons)
    
    node_features.to_parquet(node_features_path)
    edge_features.to_parquet(edge_features_path)
    
    timeseries_subgraph = merge_timeseries_pipeline(meta_path, 
                                   node_features_path,
                                   measurement_path,
                                   threshold)
    
    expanded_timeseries = expand_wait_times(timeseries_subgraph, num_peaks=2,
                                            amp_frac=0.1, sigma=1.0)
    
    expanded_timeseries.to_parquet(output_path_timeseries, index=False)
    
    return expanded_timeseries, G_sub, G_pt


def generate_train_data(
    nodes_helmond_path: str,
    meta_path: str,
    measurement_path: str,
    base_output_dir: str = "data/training_data",
    n_subgraphs: int = 10,
    threshold: float = 25,
    dist: int = 100,
):
    base = Path(base_output_dir)
    for i in range(n_subgraphs):
        subdir = base / f"subgraph_{i}"
        subdir.mkdir(parents=True, exist_ok=True)

        node_features_path = subdir / "node_metadata.parquet"
        edge_features_path = subdir / "edge_features.parquet"
        output_path_timeseries = subdir / "timeseries.parquet"

        # Call your existing function
        ts, G_sub, G_pt = get_timeseries_subgraph(
            nodes_helmond_path=nodes_helmond_path,
            node_features_path=str(node_features_path),
            edge_features_path=str(edge_features_path),
            meta_path=meta_path,
            measurement_path=measurement_path,
            output_path_timeseries=str(output_path_timeseries),
            threshold=threshold,
            dist=dist,
            seed=i,
        )

        print(f"‣ Finished subgraph {i}:")
        print(f"   • nodes: {G_sub.number_of_nodes()}  edges: {G_sub.number_of_edges()}")
        print(f"   • timeseries shape: {ts.shape}")


if __name__ == "__main__":
    generate_train_data(
        nodes_helmond_path="data/processed_new/helmond_nodes.parquet",
        meta_path="data/processed/intersection_metadata.csv",
        measurement_path="data/processed/intersection_measurements_31_01_24.csv",
        n_subgraphs=3,
    )
