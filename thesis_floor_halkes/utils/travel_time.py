from torch_geometric.data import Data

# def calculate_edge_travel_time(data: Data,):
#     lengths = data.edge_attr[:, 0]
#     speeds_kmh = data.edge_attr[:, 1]
#     speeds_ms = speeds_kmh / 3.6
#     edge_travel_times = lengths / speeds_ms
    
#     return edge_travel_times

def calculate_edge_travel_time(data: Data, edge_index:int, length_feature_idx: int, speed_feature_idx: int):
    length = data.edge_attr[edge_index, length_feature_idx]
    speeds_kmh = data.edge_attr[edge_index, speed_feature_idx]
    speed_ms = speeds_kmh / 3.6
    edge_travel_time = length / speed_ms
    return edge_travel_time