import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch 
from torch_geometric.data import Data
import pandas as pd


def plot_graph(data, save_path: str = 'graph.png',
               with_labels: bool = True,
               node_size: int = 300,
               traffic_colors: dict = None):
    """
    Plot a PyTorch Geometric Data graph and save it as a PNG.

    Args:
        data (torch_geometric.data.Data): expects
            data.x[:,0] == is_traffic_light (0/1)
            data.x[:,1] == light_status    (0=red, 1=green)
        save_path (str): where to dump the PNG
        with_labels (bool): whether to draw node indices
        node_size (int): how big the circles should be
        traffic_colors (dict): override default colors:
            { 'green':..., 'red':..., 'none':... }
    """
    # Default color mapping
    traffic_colors = traffic_colors or {
        'green': 'palegreen',
        'red':   'tomato',
        'none':  'grey'
    }

    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)

    # Build color list
    node_colors = []
    if data.x is not None and data.x.size(1) >= 2:
        is_lights   = data.x[:, 0].tolist()
        statuses    = data.x[:, 1].tolist()
        for has_light, status in zip(is_lights, statuses):
            if int(has_light) == 1:
                # there is a light: choose green or red
                node_colors.append(
                    traffic_colors['green'] if int(status) == 1
                    else traffic_colors['red']
                )
            else:
                # no traffic light → grey
                node_colors.append(traffic_colors['none'])
    else:
        # fallback: all grey
        node_colors = [traffic_colors['none']] * data.num_nodes

    # Draw
    plt.figure(figsize=(8,6))
    nx.draw(G, pos,
            with_labels=with_labels,
            node_color=node_colors,
            node_size=node_size,
            edge_color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



def create_fixed_graph():
    """
    Creates a fixed graph of 5 nodes with edges:
      1-2, 1-3, 2-3, 3-5, 2-4, 4-5
    Returns a torch_geometric.data.Data object with:
      - x: [num_nodes, 2] feature matrix (zeros)
      - edge_index: [2, num_edges] tensor of undirected edges
      - edge_attr: [num_edges, 2] dummy edge attributes (zeros)
    Node indices are 0-based internally (user description 1-based).
    """
    # Define edges in 1-based indexing
    edges_1b = [(1,2), (1,3), (2,3), (3,5), (2,4), (4,5)]
    # Convert to 0-based
    edges = [(u-1, v-1) for u, v in edges_1b]
    # Make undirected: include both directions
    edge_index = torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).t().contiguous()

    # Dummy node features (e.g., [traffic_light, waiting_time])
    num_nodes = 5
    x = torch.zeros((num_nodes, 2), dtype=torch.float)

        # Dummy edge attributes (e.g., [length, max_speed]) with non-zero values
    # length between 100 and 1000 m, speed between 30 and 100 km/h
    num_edges = edge_index.size(1)
    lengths = torch.rand((num_edges, 1), dtype=torch.float) * 900 + 100
    speeds  = torch.rand((num_edges, 1), dtype=torch.float) * 70  + 30
    edge_attr = torch.cat([lengths, speeds], dim=1)


    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



def load_feature_tensor_from_csv(csv_path, num_nodes,
                                 feature,
                                 days: list[str] = None,
                                 time_blocks: int = 96):
    """
    Loads a (num_nodes x (time_blocks * num_days)) tensor from a CSV with columns:
      ID, timestamp, <feature>
    If `days` is None, uses all dates in the file; otherwise only those date-strings.
    """
    import pandas as pd
    import torch

    df = pd.read_csv(csv_path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M")
    df = df.sort_values(['ID','timestamp'])

    # Extract the unique dates (YYYY-MM-DD) in chronological order
    df['date'] = df['timestamp'].dt.date.astype(str)
    all_dates = sorted(df['date'].unique())
    if days is None:
        use_dates = all_dates
    else:
        # accept either a single string or list
        if isinstance(days, str):
            days = [days]
        use_dates = [d for d in all_dates if d in days]

    num_days = len(use_dates)
    total_blocks = time_blocks * num_days

    # assign each row a day-index and intra-day block
    df = df[df['date'].isin(use_dates)].copy()
    # map date to its index in use_dates
    date_to_idx = {d:i for i,d in enumerate(use_dates)}
    df['day_idx'] = df['date'].map(date_to_idx)
    df['intrablock'] = (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute) // (24*60/time_blocks)
    # global block index
    df['block_idx'] = df['day_idx'] * time_blocks + df['intrablock']

    # Prepare output tensor
    full = torch.zeros((num_nodes, total_blocks), dtype=torch.float)

    # build ID→row mapping
    df['ID'] = df['ID'].astype(str)
    unique_ids = sorted(df['ID'].unique())[:num_nodes]
    id_to_index = {node_id: idx for idx, node_id in enumerate(unique_ids)}

    # fill in averages per (ID, block_idx)
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    grouped = df.groupby(['ID','block_idx'])[feature].mean().reset_index()

    for _, row in grouped.iterrows():
        nid = str(row['ID'])
        b   = int(row['block_idx'])
        val = float(row[feature])
        if nid in id_to_index and 0 <= b < total_blocks:
            full[id_to_index[nid], b] = val

    return full  # [num_nodes, time_blocks * num_days]


