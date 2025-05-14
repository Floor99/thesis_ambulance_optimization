# import os
# import torch 
# from torch_geometric.data import Data, Dataset

# from thesis_floor_halkes.features.static.getter import get_static_data_object


# class StaticDataSet(Dataset):
#     def __init__(self, num_graphs: int):
#         super().__init__()
#         self.num_graphs = num_graphs
#         self.data_list = [
#             get_static_data_object(
#                 time_series_df_path = "data/processed/node_features.parquet",
#                 dist=100,
#                 seed = i,
#             )
#             for i in range(num_graphs)
#         ]

#     def len(self):
#         return self.num_graphs

#     def get(self, idx: int):
#         return self.data_list[idx]
        

# import torch
# from thesis_floor_halkes.features.graph.graph_generator import get_static_data_object

# # 1) pick exactly 5 seeds
# seeds = [0, 1, 2, 3, 4]

# # 2) generate your 5 graphs
# data_list = [
#     get_static_data_object(
#         time_series_df_path="data/processed/node_features.parquet",
#         dist=500,
#         seed=seed,
#     )
#     for seed in seeds
# ]

# # 3) save to a .pt file
# torch.save(data_list, "data/processed/static_graphs.pt")
# print("Saved 5 static graphs to data/processed/static_graphs.pt")





import os
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Dataset
from thesis_floor_halkes.features.graph.graph_generator import plot_sub_graph_in_and_out_nodes_helmond
from thesis_floor_halkes.features.static.getter import get_static_data_object
from torch.serialization import add_safe_globals
import torch_geometric.data.data as geom_data

add_safe_globals([geom_data.DataEdgeAttr])

class StaticListDataset(Dataset):
    def __init__(
        self,
        ts_path: str = "data/processed/node_features.parquet",
        dists: int | list[int] = 100,
        seeds: list[int] | None = None,
        cache_path: str = "data/processed/static_graphs.pt",
    ):
        """
        A small fixed set of graphs, cached to disk so they're identical every run.
        - ts_path: your merged time-series Parquet
        - dist: radius for subgraph
        - seeds: list of RNG seeds; one graph per seed
        - cache_path: where to save/load the list of Data objects
        """
        super().__init__()
        # if no explicit seeds, pick first 5 ints
        self.seeds = seeds if seeds is not None else list(range(5))
        N = len(self.seeds)

        # 2) Normalize dists → a list of length N
        if isinstance(dists, int):
            self.dists = [dists] * N
        else:
            if len(dists) != N:
                raise ValueError(f"len(dists) ({len(dists)}) must match len(seeds) ({N})")
            self.dists = dists
        
        self.cache_path = cache_path

        if os.path.exists(self.cache_path):
            # 1) load the exact same list every time
            self.data_list = torch.load(self.cache_path, weights_only=False)
        else:
            # 2) generate once, then cache
            self.data_list = []
            for seed, dist in zip(self.seeds, self.dists):
                data = get_static_data_object(
                    time_series_df_path=ts_path,
                    dist=dist,
                    seed=seed,
                )
                self.data_list.append(data)

            # make sure folder exists
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self.data_list, self.cache_path)

    def len(self):
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]
    

if __name__ == "__main__":
    dataset = StaticListDataset(
    ts_path="data/processed/node_features.parquet",
    seeds=[0,1,2,3,4],
    dists=[200, 300, 400, 500, 600],   # each graph has its own radius
    )

    # 2) iterate, collect info
    summaries = []
    for idx, data in enumerate(dataset):
        summaries.append({
            "idx"       : idx,
            "nodes"     : data.num_nodes,
            "edges"     : data.num_edges,
            "start_node": data.start_node,
            "end_node"  : data.end_node,
            "gsub"     : data.G_sub,
        })

    # 3) print at the end
    print("=== Summary of all 5 graphs ===")
    for s in summaries:
        print(
            f"Graph {s['idx']}: "
            f"{s['nodes']} nodes, "
            f"{s['edges']} edges, "
            f"start={s['start_node']}, "
            f"end={s['end_node']}"
        )
        

    out_dir = "data/plots/subgraphs"
    os.makedirs(out_dir, exist_ok=True)

    for idx, data in enumerate(dataset):
        ax = plot_sub_graph_in_and_out_nodes_helmond(data.G_sub, data.G_pt)
        ax.set_title(f"Helmond Subgraph #{idx}")
        # save with distinct filename
        filename = os.path.join(out_dir, f"helmond_subgraph_{idx}.png")
        ax.figure.savefig(filename, dpi=300)
        plt.close(ax.figure)
        
    

                