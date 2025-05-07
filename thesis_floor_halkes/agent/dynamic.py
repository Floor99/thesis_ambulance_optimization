from thesis_floor_halkes.agent.base import Agent
import torch.nn as nn
import torch

from thesis_floor_halkes.model_dynamic_attention import FixedContext
from thesis_floor_halkes.state import State


class DynamicAgent(Agent):
    """
    A dynamic agent that adapts its behavior based on the environment.
    """

    def __init__(
        self,
        static_encoder: nn.Module,
        dynamic_encoder: nn.Module,
        decoder: nn.Module,
        baseline: nn.Module = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        """
        Initialize the dynamic agent with static and dynamic encoders and a decoder.

        Args:
            static_encoder: The static encoder for processing static features.
            dynamic_encoder: The dynamic encoder for processing dynamic features.
            decoder: The decoder for generating actions.
        """
        self.static_encoder = static_encoder
        self.dynamic_encoder = dynamic_encoder
        self.fixed_context = None
        self.decoder = decoder
        self.baseline = baseline
        self.gamma = gamma
        
        params = (
            list(static_encoder.parameters()) +
            list(dynamic_encoder.parameters()) +
            list(decoder.parameters())
        )
        
        if baseline is not None:
            params += list(baseline.parameters()) # get_learnable_parameters() ?!
        
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        self.states = []  # list van states en dan telkens in select de laatste state pakken?
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.penalties = []
        self.baselines = []
        self.embeddings = []
        self.routes = [] # list with list of integers --> all routes of training session
        self.current_route = []
        
    def _embed_graph(self, data, graph_type="static"):
        if graph_type == "static":
            x_static = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            return self.static_encoder(x_static, edge_index, edge_attr)
        elif graph_type == "dynamic":
            x_dynamic = data.x
            edge_index = data.edge_index
            return self.dynamic_encoder(x_dynamic, edge_index)
        else:
            raise ValueError("Invalid graph type. Use 'static' or 'dynamic'.")

    def select_action(self, state: State):
        # static_embedding = self._embed_graph(state.static_data, graph_type="static")
        if self.fixed_context is None:
            static_embedding = self._embed_graph(state.static_data, graph_type="static")
            self.fixed_context = FixedContext(static_embedding)
        else: 
            static_embedding = self.fixed_context.static_embedding
            
        dynamic_embedding = self._embed_graph(state.dynamic_data, graph_type="dynamic")
        final_embedding = torch.cat(
            (static_embedding, dynamic_embedding), dim=1
        )                                                       # overwegen om naar + ipv cat te doen
        
        graph_embedding = final_embedding.mean(dim=0).detach()  # mean pooling over nodes
        
        self.embeddings.append({"static": static_embedding, 
                        "dynamic": dynamic_embedding, 
                        "final": final_embedding,
                        "graph": graph_embedding})
        
        invalid_action_mask = self._get_action_mask(
            state.valid_actions, state.static_data.num_nodes
        )
        
        action, action_log_prob, _ = self.decoder(
            final_embedding,
            current_node_idx=state.current_node,
            invalid_action_mask=invalid_action_mask,
        )
        
        if not self.current_route: 
            self.current_route.append(state.start_node)
        
        self.current_route.append(action)
        print(f"Current route: {self.current_route}")

        return action, action_log_prob
    
    def _get_action_mask(self, valid_actions: list[int], num_nodes: int) -> torch.Tensor:
        """
        Create a mask for valid actions.
        """
        action_mask = torch.ones(num_nodes, dtype=torch.bool)
        action_mask[valid_actions] = 0
        return action_mask
    
    def store_baseline(self, baseline):
        self.baselines.append(baseline)

    def store_state(self, state: State):
        self.states.append(state)

    def store_penalty(self, penalty: float):
        self.penalties.append(penalty)

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def store_action(self, action):
        self.actions.append(action)

    def store_action_log_prob(self, log_prob):
        self.action_log_probs.append(log_prob)

    def finish_episode(self):
        # 1) Compute raw discounted returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # 2) Evaluate baseline(s) # Use the collected states to compute baselines
        if self.baseline is not None:
            baselines, loss_b = self.baseline.eval(self.embeddings, returns)
            advantages = returns - baselines.detach() # detach????
        else: 
            advantages = returns
            loss_b = 0
        
        # 4) Policy loss
        log_probs_tensor = torch.stack(self.action_log_probs)
        policy_loss = -(log_probs_tensor * advantages).mean()

        # 5) Total loss = policy + baseline loss
        total_loss = policy_loss + loss_b
        
        self.routes.append(self.current_route.copy())
            
        # return loss, policy_loss, loss_b #if self.baseline is not None else loss, policy_loss
        return total_loss, policy_loss, loss_b
            
    def reset(self):
        self.action_log_probs.clear()
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.penalties.clear()
        self.baselines.clear()
        self.embeddings.clear()
        self.fixed_context = None
        self.current_route.clear()
        
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
