import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

########################################### GRAPH GENERATOR #########################################################
import torch
from torch_geometric.data import Data

class GraphGenerator:
    """
    Generates a random dummy graph with purely static node features:
      - is_traffic_light: 0 or 1
      - light_status:     0=red, 1=green (only meaningful if is_traffic_light=1)
      - waiting_time:     random float seconds

    Edge features:
      - length:      random float meters
      - max_speed:   random float km/h

    Ensures the graph is fully connected and no node is isolated.
    """
    def __init__(self,
                 num_nodes: int = 10,
                 edge_prob: float = 0.3,
                 max_wait: float = 30.0,
                 min_length: float = 100.0,
                 max_length: float = 1000.0,
                 min_speed: float = 30.0,
                 max_speed: float = 100.0):
        self.num_nodes   = num_nodes
        self.edge_prob   = edge_prob
        self.max_wait    = max_wait
        self.min_length  = min_length
        self.max_length  = max_length
        self.min_speed   = min_speed
        self.max_speed   = max_speed

    def generate(self) -> Data:
        # 1) Sample random undirected edges
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if torch.rand(1).item() < self.edge_prob:
                    edges += [(i, j), (j, i)]

        # 2) Guarantee each node has at least one neighbor
        for i in range(self.num_nodes):
            if not any(u == i for u, _ in edges):
                j = torch.randint(0, self.num_nodes - 1, (1,)).item()
                if j >= i: j += 1
                edges += [(i, j), (j, i)]

        # 3) Force full connectivity via a simple spanning‐tree pass
        connected = {0}
        for i in range(1, self.num_nodes):
            if not any((u == i and v in connected) or (v == i and u in connected)
                       for u, v in edges):
                k = next(iter(connected))
                edges += [(i, k), (k, i)]
            connected.add(i)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 4) Static node features
        # is_light     = torch.randint(0, 2, (self.num_nodes, 1), dtype=torch.float)
        is_light     = torch.ones((self.num_nodes, 1), dtype=torch.float) # all nodes are traffic lights    
        # Only nodes with a light get a random status; others default to red (0)
        light_status = (torch.randint(0, 2, (self.num_nodes, 1), dtype=torch.float)
                        * is_light)

        # 5) Static waiting-time (still static for now)
        waiting_time = torch.rand((self.num_nodes, 1), dtype=torch.float) * self.max_wait

        # 6) Stack features into x: [is_light, light_status, waiting_time]
        x = torch.cat([is_light, light_status, waiting_time], dim=1)  # shape [N×3]

        # 7) Edge attributes: length [m], max_speed [km/h]
        num_edges = edge_index.size(1)
        lengths = torch.rand((num_edges, 1)) * (self.max_length - self.min_length) + self.min_length
        speeds  = torch.rand((num_edges, 1)) * (self.max_speed - self.min_speed) + self.min_speed
        edge_attr = torch.cat([lengths, speeds], dim=1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


##################################################### TRAFFIC GRAPH SET ##########################################################

class TrafficGraphSet(Dataset):
    def __init__(self, base_graph, wait_tensor, status_tensor, time_blocks=96):
        """
        Creates a dataset of graphs, each representing the network at a different time block,
        with constant topology and edge features, and varying node wait times.
        Also added the varying status of the traffic lights.

        Args:
            base_graph (Data): A PyTorch Geometric Data object representing the fixed network.
            wait_tensor (Tensor): Shape [num_nodes, time_blocks] with wait times.
            time_blocks (int): Number of time intervals (default 96 for 15-min blocks).
        """
        num_nodes = wait_tensor.size(0)
        assert wait_tensor.shape == (num_nodes, time_blocks), "wait_tensor shape mismatch"
        assert status_tensor.shape == (num_nodes, time_blocks), "status_tensor shape mismatch"
        

        self.graphs = []
        self.time_blocks = time_blocks
        self.wait_tensor = wait_tensor
        self.status_tensor = status_tensor

        for t in range(time_blocks):
            wait_times = wait_tensor[:, t]
            status = status_tensor[:, t]
            # graph = graph_from_template(base_graph, wait_times)
            graph = base_graph.clone()
            graph.x[:, 2] = wait_times
            graph.x[:, 1] = status 
            graph.time_block = t  # optional: tag graph with time block
            graph.time_string = f"{(t * 15) // 60:02d}:{(t * 15) % 60:02d}"
            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
    

################################################## RANDOM GRAPH DATASET ##########################################################

class RandomGraphDataset(Dataset):
    """
    Returns a new random graph on each __getitem__:
      - num_nodes randomly chosen in [min_nodes, max_nodes]
      - edge_prob randomly chosen in [min_prob, max_prob]
      - Node features (traffic_light, status, waiting_time) random
      - Edge features (length, speed) random
    """
    def __init__(
        self,
        num_graphs: int,
        min_nodes:   int = 5,
        max_nodes:   int = 15,
        min_prob:    float = 0.2,
        max_prob:    float = 0.8,
        **generator_kwargs
    ):
        """
        Args:
            num_graphs: how many total graphs this dataset will pretend to have.
            min_nodes / max_nodes: range for random node counts.
            min_prob / max_prob: range for random edge probabilities.
            generator_kwargs: any other DummyGraphGenerator args (waiting‐time bounds, etc.).
        """
        self.num_graphs = num_graphs
        self.min_nodes  = min_nodes
        self.max_nodes  = max_nodes
        self.min_prob   = min_prob
        self.max_prob   = max_prob
        self.gen_kwargs = generator_kwargs

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        # 1) Randomize size & density
        num_nodes = torch.randint(self.min_nodes, self.max_nodes+1, (1,)).item()
        edge_prob = torch.rand(1).item() * (self.max_prob - self.min_prob) + self.min_prob

        # 2) Generate the graph
        data = GraphGenerator(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            **self.gen_kwargs
        ).generate()

        # 3) (Optionally) tag it
        data.graph_id = idx
        return data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

############################################ DUMMY GRAPH WITH DYNAMIC FEATURES #################################################
import torch
from torch_geometric.data import Data

class DummyGraphGeneratorDynamic:
    """
    Generates a random dummy graph with node and edge features for testing.
    Static node features:
      - is_traffic_light: 0 or 1
      - light_status:     0=red, 1=green
    Dynamic node feature:
      - waiting_time:     random float seconds
    Edge features:
      - length:           random float meters
      - max_speed:        random float km/h
    Ensures the graph is fully connected and no node is isolated.
    """
    def __init__(self,
                 num_nodes: int = 10,
                 edge_prob: float = 0.3,
                 max_wait: float = 30.0,
                 min_length: float = 100.0,
                 max_length: float = 1000.0,
                 min_speed: float = 30.0,
                 max_speed: float = 100.0):
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.max_wait  = max_wait
        self.min_length = min_length
        self.max_length = max_length
        self.min_speed  = min_speed
        self.max_speed  = max_speed

    def generate(self) -> Data:
        # 1) Sample random undirected edges
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if torch.rand(1).item() < self.edge_prob:
                    edges += [(i, j), (j, i)]

        # 2) Guarantee each node has at least one neighbor
        for i in range(self.num_nodes):
            if not any(u == i for u, _ in edges):
                j = torch.randint(0, self.num_nodes - 1, (1,)).item()
                if j >= i:
                    j += 1
                edges += [(i, j), (j, i)]

        # 3) Force full connectivity via a spanning‑tree pass
        connected = {0}
        for i in range(1, self.num_nodes):
            if not any((u == i and v in connected) or (v == i and u in connected)
                       for u, v in edges):
                k = next(iter(connected))
                edges += [(i, k), (k, i)]
            connected.add(i)

        # 4) Build edge_index
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 5) Static node features: is_light, light_status
        is_light     = torch.randint(0, 2, (self.num_nodes, 1), dtype=torch.float)
        light_status = torch.randint(0, 2, (self.num_nodes, 1), dtype=torch.float)

        # 6) Dynamic node feature: waiting_time
        waiting_time = torch.rand((self.num_nodes, 1), dtype=torch.float) * self.max_wait

        # 7) Combine into x: [is_light, light_status, waiting_time]
        x = torch.cat([is_light, light_status, waiting_time], dim=1)
        
        # 8) Edge attributes: length [m], max_speed [km/h]
        num_edges = edge_index.size(1)
        lengths = torch.rand((num_edges, 1)) * (self.max_length - self.min_length) + self.min_length
        speeds  = torch.rand((num_edges, 1)) * (self.max_speed - self.min_speed) + self.min_speed
        edge_attr = torch.cat([lengths, speeds], dim=1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
