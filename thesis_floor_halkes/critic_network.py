import torch.nn as nn
import torch.nn.functional as F
from model_dynamic_attention import StaticGATEncoder, DynamicGATConvEncoder
from torch_geometric.nn import global_mean_pool

class CriticNetwork(nn.Module):
    def __init__(self, in_static, in_dynamic, hidden_size, edge_attr_dim):
        super().__init__()

        # Static encoder (with edge attributes)
        self.static_encoder = StaticGATEncoder(
            in_channels=in_static,
            hidden_size=hidden_size,
            out_size=hidden_size,
            num_layers=2,
            heads=4,
            dropout=0.2,
            edge_attr_dim=edge_attr_dim
        )

        # Dynamic encoder (no edge attributes)
        self.dynamic_encoder = DynamicGATConvEncoder(
            in_channels=in_dynamic,
            hidden_size=hidden_size,
            heads=1,
            dropout=0.2
        )

        # Value head: predicts scalar value from graph embedding
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch  # required for pooling over multiple graphs

        x_static = x[:, :1]
        x_dynamic = x[:, 1:]

        h_static = self.static_encoder(x_static, edge_index, edge_attr)
        h_dynamic = self.dynamic_encoder(x_dynamic, edge_index)

        h_combined = h_static + h_dynamic

        graph_embeddings = global_mean_pool(h_combined, batch)  # shape: [batch_size, hidden_size]
        return self.value_head(graph_embeddings).squeeze(-1)    # shape: [batch_size]

