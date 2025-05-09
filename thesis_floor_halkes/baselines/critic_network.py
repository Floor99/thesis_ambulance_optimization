import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # self.critic = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)  # Outputs a single scalar
        # )

    def forward(self, embedding, hidden_dim=128):
        embedding = embedding.view(embedding.shape[0] * embedding.shape[1])
        input_dim = embedding.shape[0]
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single scalar
        )
        value = self.critic(embedding).squeeze(-1)
        return value
