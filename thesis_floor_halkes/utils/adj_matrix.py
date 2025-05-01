from torch_geometric.data import Data

def build_adjecency_matrix(num_nodes:int, data: Data):
    adj_matrix = {i: [] for i in range(num_nodes)}
    for idx, (u, v) in enumerate(data.edge_index.t().tolist()):
        adj_matrix[u].append((v, idx))

    return adj_matrix
