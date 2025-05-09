import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class FixedContext(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(3*embed_dim, embed_dim)
        
    def forward(self, final_node_embeddings, current_idx, end_idx):
        graph_embedding = final_node_embeddings.mean(dim=0) # (batch_size, embed_dim)
        
        current_node_embedding = final_node_embeddings[current_idx, :]  # (batch_size, embed_dim)
        end_node_embedding = final_node_embeddings[end_idx, :]  # (batch_size, embed_dim)
        
        context = torch.cat([graph_embedding, current_node_embedding, end_node_embedding], dim=-1)  # (batch_size, 3*embed_dim)
        
        return context

class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.project_context = nn.Linear(3*embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, 
                context_vector, 
                node_embeddings, 
                invalid_action_mask: torch.Tensor | None = None):    
        
        n_nodes = node_embeddings.size(0)
        projected_query = self.project_context(context_vector).unsqueeze(0)  # [batch_size, embed_dim]
        query = projected_query.unsqueeze(0) # reshape to [seq_length, batch_size, embed_dim] 
        
        valid_indices = (~invalid_action_mask).nonzero(as_tuple=True)[0]  # (k,)
        valid_keys = node_embeddings[valid_indices].unsqueeze(1)          # (k, 1, d_h)
        
        attn_output, attn_weights = self.attn(query, valid_keys, valid_keys, key_padding_mask=None) # [1, batch_size, n_nodes]
        
        logits = attn_weights.squeeze(0).squeeze(0) # [n_nodes, ] 
        
        probs = F.softmax(logits, dim=-1)  
        dist = torch.distributions.Categorical(probs)
        sampled_idx = dist.sample()
        action = valid_indices[sampled_idx].item()
        log_prob = dist.log_prob(sampled_idx)
        
        return action, log_prob