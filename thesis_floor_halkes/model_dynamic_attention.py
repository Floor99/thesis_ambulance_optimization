import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GAT, GATConv
from typing import NamedTuple, List


class StaticGATEncoder(nn.Module):
    """
    High-level GAT encoder for static features using PyG's GAT model.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_size: int = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        edge_attr_dim: int = None
    ):
        super().__init__()
        out_size = out_size or hidden_size
        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_size,
            out_channels=out_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_attr_dim
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim] or None
        returns: [N, out_size]
        """
        return self.gat(x, edge_index, edge_attr=edge_attr)

class DynamicGATEncoder(nn.Module):
    """
    High-level GAT encoder for dynamic features using PyG's GAT model.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_size: int = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        out_size = out_size or hidden_size
        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_size,
            out_channels=out_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_attr_dim] or None
        returns: [N, out_size]
        """
        return self.gat(x, edge_index, edge_attr=edge_attr)




class DynamicGATConvEncoder(nn.Module):
    """
    Two-layer GAT encoder for dynamic features (no edge attributes).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        heads: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_size,
            heads=heads,
            concat=False,
            dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_size,
            hidden_size,
            heads=1,
            concat=False,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dyn: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_dyn: [N, in_channels]
        edge_index: [2, E]
        returns: [N, hidden_size]
        """
        h = self.dropout(x_dyn)
        h = F.elu(self.conv1(h, edge_index))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index))
        return h


class FixedContext(NamedTuple):
    """
    Cached static embeddings for the current episode: 
    static_embedding: [N, H]
    """
    static_embedding: torch.Tensor
    # K_s: torch.Tensor 
    # V_s: torch.Tensor
    
    def __getitem__(self, idx):
        return FixedContext(
            static_embedding=self.static_embedding[idx],
            # K_s=self.K_s[idx],
            # V_s=self.V_s[idx],
            )
    
    
class AttentionDecoderChat(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

        # Output scoring head: maps [context || node_embedding] → logit
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, node_embeddings: torch.Tensor, 
                current_node_idx: int, 
                invalid_action_mask: torch.Tensor | None =None):
        N = node_embeddings.size(0)
        device = node_embeddings.device

        # --- 1. Project Q, K, V ---
        query = self.query_proj(node_embeddings[current_node_idx]).unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        keys = self.key_proj(node_embeddings).unsqueeze(1)     # [N, 1, D]
        values = self.value_proj(node_embeddings).unsqueeze(1) # [N, 1, D]

        # --- 2. Compute attention context ---
        attn_output, _ = self.attn(query, keys, values, key_padding_mask=None)  # [1, 1, D]
        context = attn_output.squeeze(0).squeeze(0)  # [D]

        # --- 3. Expand context and concat with node embeddings ---
        context_expanded = context.expand(N, -1)  # [N, D]
        concat = torch.cat([context_expanded, node_embeddings], dim=-1)  # [N, 2D]

        # --- 4. Score each node ---
        logits = self.out_proj(concat).squeeze(-1)  # [N]

        # --- 5. Mask invalid actions ---
        if invalid_action_mask is not None:
            logits[invalid_action_mask] = float('-1e9')

        # --- 6. Sample action ---
        probs = F.softmax(logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, logits
    
    
    
    
    
    
# class FixedContext(NamedTuple):
#     """
#     Cached static context for attention decoding:
#       - h_static: [N, H]
#       - K_s, V_s, L_s: [N, H] static projections
#     """
#     h_static: torch.Tensor
#     K_s: torch.Tensor
#     V_s: torch.Tensor
#     L_s: torch.Tensor

#     def __getitem__(self, idx):
#         return FixedContext(
#             h_static=self.h_static[idx],
#             K_s=self.K_s[idx],
#             V_s=self.V_s[idx],
#             L_s=self.L_s[idx]
#         )

    
        
    # def forward(self, node_embeddings, current_node_idx, invalid_action_mask=None):
    #     # Project query (current node embedding)
    #     query = self.query_proj(node_embeddings[current_node_idx])
    #     query = query.unsqueeze(0).unsqueeze(0)  # [1, 1, D]

    #     # Project keys/values (all nodes)
    #     keys = self.key_proj(node_embeddings).unsqueeze(1)   # [N, 1, D]
    #     values = self.value_proj(node_embeddings).unsqueeze(1)

    #     if invalid_action_mask is not None:
    #         print(f"invalid_action_mask shape = {invalid_action_mask.shape}")
    #         invalid_action_mask = invalid_action_mask.unsqueeze(0)  # [1, N]

    #     # Run attention
    #     _, attn_weights = self.attn(query, keys, values, key_padding_mask=invalid_action_mask)
    #     logits = attn_weights.squeeze(0).squeeze(0)  # [N]

    #     # Masked logits will have very low values — so softmax still works correctly
    #     probs = F.softmax(logits, dim=-1)
        
    #     # Sample action and compute log-prob
    #     dist = torch.distributions.Categorical(probs)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)

    #     return action.item(), log_prob, logits
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# import math
# from typing import List

# class AttentionDecoder2(nn.Module):
#     """
#     Attention-based decoder for pre-concatenated [static || dynamic] embeddings.
#     Assumes input is [N, 2H], where 2H = concat(static, dynamic).
#     """
#     def __init__(self, embed_dim: int):  # embed_dim = 2 * H
#         super().__init__()
#         self.embed_dim = embed_dim

#         self.project_kvl = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
#         self.project_q   = nn.Linear(embed_dim, embed_dim, bias=False)

#         self.out_mlp = nn.Sequential(
#             nn.Linear(2 * embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 1)
#         )

#     def forward(
#         self,
#         node_embeddings: torch.Tensor,  # [N, 2H]
#         current_node: int,
#         valid_actions: List[int]
#     ):
#         N = node_embeddings.size(0)
#         device = node_embeddings.device

#         # Project all nodes into K, V, L
#         kvl = self.project_kvl(node_embeddings)  # [N, 3 * 2H]
#         K, V, _ = kvl.chunk(3, dim=-1)  # Each [N, 2H]

#         # Build query from current node embedding
#         q = self.project_q(node_embeddings[current_node])  # [2H]

#         # Compute scaled attention scores
#         compat = (K @ q) / math.sqrt(self.embed_dim)  # [N]

#         # Mask invalid actions
#         mask = torch.zeros(N, dtype=torch.bool, device=device)
#         mask[valid_actions] = True
#         compat[~mask] = float('-1e9')

#         # Attention weights and context
#         alpha = F.softmax(compat, dim=0)           # [N]
#         context = (alpha.unsqueeze(1) * V).sum(0)  # [2H]

#         # Combine context and query, compute final score
#         fuse = torch.cat([node_embeddings[current_node], context], dim=-1)  # [4H]
#         logit = self.out_mlp(fuse).squeeze(-1)  # scalarcoul

#         # Create full logit vector with masked entries
#         logits_all = torch.full((N,), float('-1e9'), device=device)
#         for idx in valid_actions:
#             logits_all[idx] = logit

#         probs = F.softmax(logits_all, dim=0)
#         dist = Categorical(probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         return action.item(), log_prob



# class AttentionDecoder(nn.Module):
#     """
#     Attention-based decoder that:
#       - Caches static K/V/L once per episode
#       - Recomputes dynamic projections each step
#       - Masks invalid actions
#     """
#     def __init__(self, hidden_size: int):
#         super().__init__()
#         self.H = hidden_size
#         # project static embeddings --> [K||V||L]
#         self.project_static_kvl = nn.Linear(hidden_size, 3*hidden_size, bias=False)
#         # project dynamic embeddings --> [K||V||L]
#         self.project_dyn_kvl    = nn.Linear(hidden_size, 3*hidden_size, bias=False)
#         # project current-node embedding --> query
#         self.project_q          = nn.Linear(hidden_size, hidden_size, bias=False)
#         # MLP for final score
#         self.out_mlp            = nn.Sequential(
#             nn.Linear(2*hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
#         self.fixed_ctx: FixedContext = None

#     def clear_cache(self):
#         """Clear cached static projections."""
#         self.fixed_ctx = None

#     def precompute(self, h_static: torch.Tensor):
#         """Compute and cache static K/V/L projections once per episode."""
#         kvl = self.project_static_kvl(h_static)  # [N, 3H]
#         K_s, V_s, L_s = kvl.chunk(3, dim=-1)     # each [N, H]
#         self.fixed_ctx = FixedContext(h_static, K_s, V_s, L_s)

#     def forward(self, h_dynamic: torch.Tensor, current_node: int, valid_actions: List[int]):
#         if self.fixed_ctx is None:
#             raise RuntimeError("Static context not precomputed. Call precompute() first.")
#         fixed = self.fixed_ctx
#         N = h_dynamic.size(0)
#         device = h_dynamic.device

#         # dynamic projections
#         kvl_d = self.project_dyn_kvl(h_dynamic)  # [N, 3H]
#         K_d, V_d, L_d = kvl_d.chunk(3, dim=-1)   # each [N, H]

#         # combine static + dynamic
#         K = fixed.K_s + K_d
#         V = fixed.V_s + V_d

#         # build query from current node
#         h_cur = fixed.h_static[current_node] + h_dynamic[current_node]
#         q     = self.project_q(h_cur)

#         # attention scores
#         compat = (K @ q) / math.sqrt(self.H)  # [N]

#         # mask invalid
#         mask = torch.zeros(N, dtype=torch.bool, device=device)
#         mask[valid_actions] = True
#         compat[~mask] = float('-1e9')

#         # attention weights & context
#         alpha   = F.softmax(compat, dim=0)           # [N]
#         context = (alpha.unsqueeze(1) * V).sum(dim=0)  # [H]

#         # score fusion
#         fuse  = torch.cat([h_cur, context], dim=-1)  # [2H]
#         logit = self.out_mlp(fuse).squeeze(-1)       # scalar

#         # build masked logits and sample
#         logits_all = torch.full((N,), float('-1e9'), device=device)
#         for idx in valid_actions:
#             logits_all[idx] = logit
#         probs  = F.softmax(logits_all, dim=0)
#         dist   = Categorical(probs)
#         choice = dist.sample()
#         return choice.item(), dist.log_prob(choice)


# class PolicyNetworkGATDynamicAttention(nn.Module):
#     """
#     Full policy: static GAT, dynamic GATConv, and attention decoder.
#     """
#     def __init__(
#         self,
#         in_static: int,
#         in_dyn: int,
#         hidden_size: int,
#         static_layers: int = 2,
#         static_heads: int = 4,
#         dyn_heads: int = 1,
#         dropout: float = 0.2,
#         edge_attr_dim: int = None
#     ):
#         super().__init__()
#         self.in_static = in_static
#         # static GAT encoder
#         self.static_enc = GATModelEncoderStatic(
#             in_channels=in_static,
#             hidden_size=hidden_size,
#             out_size=hidden_size,
#             num_layers=static_layers,
#             heads=static_heads,
#             dropout=dropout,
#             edge_attr_dim=edge_attr_dim
#         )
#         # dynamic GAT encoder
#         self.dyn_enc = DynamicGATEncoder(
#             in_channels=in_dyn,
#             hidden_size=hidden_size,
#             heads=dyn_heads,
#             dropout=dropout
#         )
#         # attention decoder
#         self.decoder = AttentionDecoder(hidden_size)

#     def clear_static_cache(self):
#         self.decoder.clear_cache()

#     def forward(self, data, current_node: int, valid_actions: List[int]):
#         x   = data.x
#         x_s = x[:, :self.in_static]  # static features
#         x_d = x[:, self.in_static:]  # dynamic features
#         ei  = data.edge_index
#         ea  = data.edge_attr

#         # precompute static at episode start
#         if self.decoder.fixed_ctx is None:
#             h_s = self.static_enc(x_s, ei, ea)
#             self.decoder.precompute(h_s)

#         # dynamic each step
#         h_d = self.dyn_enc(x_d, ei)

#         # decode action
#         return self.decoder(h_d, current_node, valid_actions)

################# only attention weight 
# class AttentionDecoderChat(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.query_proj = nn.Linear(embed_dim, embed_dim)
#         self.key_proj = nn.Linear(embed_dim, embed_dim)
#         self.value_proj = nn.Linear(embed_dim, embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    
#     def forward(self, node_embeddings, current_node_idx, invalid_action_mask=None):
#         # Project query (current node embedding)
#         query = self.query_proj(node_embeddings[current_node_idx])
#         query = query.unsqueeze(0).unsqueeze(0)
#         # [1, 1, D]
#         # Project keys/values (all nodes)
#         keys = self.key_proj(node_embeddings).unsqueeze(1)   # [N, 1, D]
#         values = self.value_proj(node_embeddings).unsqueeze(1)
#         # [N, 1, D]
#         if invalid_action_mask is not None:
#             # print(f"invalid_action_mask shape = {invalid_action_mask.shape}")
#             invalid_action_mask = invalid_action_mask.squeeze(0)
#         # [1, N]
#         # Run attention
#         _, attn_weights = self.attn(query, keys, values, key_padding_mask=None)
#         logits = attn_weights.squeeze(0).squeeze(0)  # [N]
#         logits[invalid_action_mask] = float('-1e9')
        
#         # Masked logits will have very low values — so softmax still works correctly
#         probs = F.softmax(logits, dim=-1)
#         # Sample action and compute log-prob
#         dist = torch.distributions.Categorical(probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action.item(), log_prob, logits
        