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

    def eval(self, embeddings, returns):
        """
        Args:
            states: list of state objects (with .graph_embedding)
            returns: torch.Tensor of shape [T]
        Returns:
            baselines: torch.Tensor of shape [T]
            loss_b: scalar MSE loss
        """
        # Extract pooled embeddings (e.g., mean node embedding)
        graph_embeddings = [entry['graph']for entry in embeddings]
        graph_embeddings = torch.stack(graph_embeddings)  # [T, D]
        baseline_preds = self.critic(graph_embeddings).squeeze(-1)  # [T]
        loss_b = F.mse_loss(baseline_preds, returns)
        return baseline_preds, loss_b
