# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from torch_geometric.nn import GAT, GATConv
# from typing import NamedTuple, List

# # # --- Dynamic GAT encoder (no edge attributes) ---
# # class DynamicGATEncoder(nn.Module):
# #     def __init__(self, in_channels: int, hidden_size: int, heads: int = 1, dropout: float = 0.2):
# #         super().__init__()
# #         self.conv1 = GATConv(in_channels, hidden_size, heads=heads, concat=False, dropout=dropout)
# #         self.elu1  = nn.ELU()
# #         self.conv2 = GATConv(hidden_size, hidden_size, heads=1, concat=False, dropout=dropout)
# #         self.elu2  = nn.ELU()

# #     def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
# #         h = self.elu1(self.conv1(x, edge_index))
# #         h = self.elu2(self.conv2(h, edge_index))
# #         return h  # [N x H]


# # --- Fixed static context container ---
# class FixedContext(NamedTuple):
#     h_static: torch.Tensor  # [N x H]
#     K_s: torch.Tensor       # [N x H]
#     V_s: torch.Tensor       # [N x H]
#     L_s: torch.Tensor       # [N x H]

#     def __getitem__(self, idx):
#         return FixedContext(
#             h_static=self.h_static[idx],
#             K_s=self.K_s[idx],
#             V_s=self.V_s[idx],
#             L_s=self.L_s[idx]
#         )


# # --- Attention‐based decoder ---
# class AttentionDecoder(nn.Module):
#     def __init__(self, hidden_size: int):
#         super().__init__()
#         self.H = hidden_size
#         # project static embeddings --> [K||V||L]
#         self.project_static_kvl = nn.Linear(hidden_size, 3*hidden_size, bias=False)
#         # project dynamic embeddings --> [K||V||L]
#         self.project_dyn_kvl    = nn.Linear(hidden_size, 3*hidden_size, bias=False)
#         # project fused current node embedding --> query
#         self.project_q          = nn.Linear(hidden_size, hidden_size, bias=False)
#         # final MLP to score fused context
#         self.out_mlp            = nn.Sequential(
#             nn.Linear(2*hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
#         # placeholder for static context
#         self.fixed_ctx: FixedContext = None

#     def clear_cache(self):
#         """Clear cached static projections before a new episode."""
#         self.fixed_ctx = None

#     def precompute(self, h_static: torch.Tensor):
#         """Compute and cache static K/V/L projections (once per episode)."""
#         # h_static: [N x H]
#         kvl = self.project_static_kvl(h_static)  # [N x 3H]
#         K_s, V_s, L_s = kvl.chunk(3, dim=-1)     # each [N x H]
#         self.fixed_ctx = FixedContext(h_static, K_s, V_s, L_s)

#     def forward(self, h_dynamic: torch.Tensor, current_node: int, valid_actions: List[int]):
#         """
#         h_dynamic: [N x H] dynamic embeddings
#         current_node: index of current node
#         valid_actions: list of legal neighbor indices
#         Returns (action, log_prob)
#         """
#         if self.fixed_ctx is None:
#             raise RuntimeError("Static context not precomputed. Call precompute() first.")
#         fixed = self.fixed_ctx
#         N = h_dynamic.size(0)
#         device = h_dynamic.device

#         # 1) dynamic projections
#         kvl_d = self.project_dyn_kvl(h_dynamic)   # [N x 3H]
#         K_d, V_d, L_d = kvl_d.chunk(3, dim=-1)    # each [N x H]

#         # 2) combine static + dynamic
#         K = fixed.K_s + K_d  # [N x H]
#         V = fixed.V_s + V_d  # [N x H]
#         # L_unused = fixed.L_s + L_d  # optionally for direct logit scoring

#         # 3) build query from fused current node
#         h_cur = fixed.h_static[current_node] + h_dynamic[current_node]  # [H]
#         q     = self.project_q(h_cur)                                   # [H]

#         # 4) compute raw attention scores over all nodes
#         compat = (K @ q) / math.sqrt(self.H)  # [N]

#         # 5) mask invalid actions
#         mask = torch.zeros(N, dtype=torch.bool, device=device)
#         mask[valid_actions] = True
#         compat[~mask] = float('-1e9')

#         # 6) attention weights and context
#         alpha   = F.softmax(compat, dim=0)          # [N]
#         context = (alpha.unsqueeze(1) * V).sum(dim=0)  # [H]

#         # 7) fuse and score
#         fuse  = torch.cat([h_cur, context], dim=-1)  # [2H]
#         logit = self.out_mlp(fuse).squeeze(-1)       # scalar

#         # 8) build logits over N nodes, fill only valids
#         logits_all = torch.full((N,), float('-1e9'), device=device)
#         for idx in valid_actions:
#             logits_all[idx] = logit

#         # 9) sample action
#         probs  = F.softmax(logits_all, dim=0)
#         dist   = Categorical(probs)
#         choice = dist.sample()
#         return choice.item(), dist.log_prob(choice)


# # --- Overall policy combining GATs and the AttentionDecoder ---
# class PolicyNetworkGATDynamicAttention(nn.Module):
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
#         self.in_static   = in_static
#         # static GAT encoder
#         self.static_enc = GAT(
#             in_channels=in_static,
#             hidden_channels=hidden_size,
#             out_channels=hidden_size,
#             num_layers=static_layers,
#             heads=static_heads,
#             dropout=dropout,
#             edge_dim=edge_attr_dim
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

#     def forward(self, data: torch.Tensor, current_node: int, valid_actions: List[int]):
#         # split features
#         x = data.x
#         x_s = x[:, :self.in_static]    # static features
#         x_d = x[:, self.in_static:]    # dynamic features
#         ei  = data.edge_index
#         ea  = data.edge_attr

#         # 1) precompute static once per episode
#         if self.decoder.fixed_ctx is None:
#             h_s = self.static_enc(x_s, ei, ea)  # [N x H]
#             self.decoder.precompute(h_s)

#         # 2) dynamic embedding every step
#         h_d = self.dyn_enc(x_d, ei)    # [N x H]

#         # 3) attention decode
#         return self.decoder(h_d, current_node, valid_actions)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GAT, GATConv
from typing import NamedTuple, List


class GATModelEncoderStatic(nn.Module):
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
    Cached static context for attention decoding:
      - h_static: [N, H]
      - K_s, V_s, L_s: [N, H] static projections
    """
    h_static: torch.Tensor
    K_s: torch.Tensor
    V_s: torch.Tensor
    L_s: torch.Tensor

    def __getitem__(self, idx):
        return FixedContext(
            h_static=self.h_static[idx],
            K_s=self.K_s[idx],
            V_s=self.V_s[idx],
            L_s=self.L_s[idx]
        )


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder that:
      - Caches static K/V/L once per episode
      - Recomputes dynamic projections each step
      - Masks invalid actions
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.H = hidden_size
        # project static embeddings --> [K||V||L]
        self.project_static_kvl = nn.Linear(hidden_size, 3*hidden_size, bias=False)
        # project dynamic embeddings --> [K||V||L]
        self.project_dyn_kvl    = nn.Linear(hidden_size, 3*hidden_size, bias=False)
        # project current-node embedding --> query
        self.project_q          = nn.Linear(hidden_size, hidden_size, bias=False)
        # MLP for final score
        self.out_mlp            = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.fixed_ctx: FixedContext = None

    def clear_cache(self):
        """Clear cached static projections."""
        self.fixed_ctx = None

    def precompute(self, h_static: torch.Tensor):
        """Compute and cache static K/V/L projections once per episode."""
        kvl = self.project_static_kvl(h_static)  # [N, 3H]
        K_s, V_s, L_s = kvl.chunk(3, dim=-1)     # each [N, H]
        self.fixed_ctx = FixedContext(h_static, K_s, V_s, L_s)

    def forward(self, h_dynamic: torch.Tensor, current_node: int, valid_actions: List[int]):
        if self.fixed_ctx is None:
            raise RuntimeError("Static context not precomputed. Call precompute() first.")
        fixed = self.fixed_ctx
        N = h_dynamic.size(0)
        device = h_dynamic.device

        # dynamic projections
        kvl_d = self.project_dyn_kvl(h_dynamic)  # [N, 3H]
        K_d, V_d, L_d = kvl_d.chunk(3, dim=-1)   # each [N, H]

        # combine static + dynamic
        K = fixed.K_s + K_d
        V = fixed.V_s + V_d

        # build query from current node
        h_cur = fixed.h_static[current_node] + h_dynamic[current_node]
        q     = self.project_q(h_cur)

        # attention scores
        compat = (K @ q) / math.sqrt(self.H)  # [N]

        # mask invalid
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[valid_actions] = True
        compat[~mask] = float('-1e9')

        # attention weights & context
        alpha   = F.softmax(compat, dim=0)           # [N]
        context = (alpha.unsqueeze(1) * V).sum(dim=0)  # [H]

        # score fusion
        fuse  = torch.cat([h_cur, context], dim=-1)  # [2H]
        logit = self.out_mlp(fuse).squeeze(-1)       # scalar

        # build masked logits and sample
        logits_all = torch.full((N,), float('-1e9'), device=device)
        for idx in valid_actions:
            logits_all[idx] = logit
        probs  = F.softmax(logits_all, dim=0)
        dist   = Categorical(probs)
        choice = dist.sample()
        return choice.item(), dist.log_prob(choice)


class PolicyNetworkGATDynamicAttention(nn.Module):
    """
    Full policy: static GAT, dynamic GATConv, and attention decoder.
    """
    def __init__(
        self,
        in_static: int,
        in_dyn: int,
        hidden_size: int,
        static_layers: int = 2,
        static_heads: int = 4,
        dyn_heads: int = 1,
        dropout: float = 0.2,
        edge_attr_dim: int = None
    ):
        super().__init__()
        self.in_static = in_static
        # static GAT encoder
        self.static_enc = GATModelEncoderStatic(
            in_channels=in_static,
            hidden_size=hidden_size,
            out_size=hidden_size,
            num_layers=static_layers,
            heads=static_heads,
            dropout=dropout,
            edge_attr_dim=edge_attr_dim
        )
        # dynamic GAT encoder
        self.dyn_enc = DynamicGATEncoder(
            in_channels=in_dyn,
            hidden_size=hidden_size,
            heads=dyn_heads,
            dropout=dropout
        )
        # attention decoder
        self.decoder = AttentionDecoder(hidden_size)

    def clear_static_cache(self):
        self.decoder.clear_cache()

    def forward(self, data, current_node: int, valid_actions: List[int]):
        x   = data.x
        x_s = x[:, :self.in_static]  # static features
        x_d = x[:, self.in_static:]  # dynamic features
        ei  = data.edge_index
        ea  = data.edge_attr

        # precompute static at episode start
        if self.decoder.fixed_ctx is None:
            h_s = self.static_enc(x_s, ei, ea)
            self.decoder.precompute(h_s)

        # dynamic each step
        h_d = self.dyn_enc(x_d, ei)

        # decode action
        return self.decoder(h_d, current_node, valid_actions)
