import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single scalar
        )

    def forward(self, node_embeddings):
        """
        node_embeddings: Tensor of shape [num_nodes, embed_dim]
        """
        pooled = node_embeddings.mean(dim=0)  # [embed_dim]
        return self.critic(pooled).squeeze(-1)