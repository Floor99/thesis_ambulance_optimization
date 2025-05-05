from thesis_floor_halkes.agent.base import Agent
import torch.nn as nn
import torch

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
        self.decoder = decoder
        self.baseline = baseline

        self.states = []  # list van states en dan telkens in select de laatste state pakken?
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.penalties = []
        self.baselines = []  # --> is dit nodig?

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
        static_embedding = self._embed_graph(state.static_data, graph_type="static")
        dynamic_embedding = self._embed_graph(state.dynamic_data, graph_type="dynamic")
        final_embedding = torch.cat(
            (static_embedding, dynamic_embedding), dim=1
        )  # overwegen om naar + ipv cat te doen
        
        invalid_action_mask = self._get_action_mask(
            state.valid_actions, state.static_data.num_nodes
        )
        
        action, action_log_prob, _ = self.decoder(
            final_embedding,
            current_node_idx=state.current_node,
            invalid_action_mask=invalid_action_mask,
        )
        
        # action, action_log_prob = self.decoder(
        #     final_embedding, 
        #     state.current_node,
        #     state.valid_actions,
        # )

        return action, action_log_prob
    
    def _get_action_mask(self, valid_actions: list[int], num_nodes: int) -> torch.Tensor:
        """
        Create a mask for valid actions.
        """
        action_mask = torch.ones(num_nodes, dtype=torch.bool)
        action_mask[valid_actions] = 0
        return action_mask

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

        # 2) Evaluate baseline(s)
        if self.baseline is not None:
            # Use the collected states to compute baselines
            baselines, loss_b = self.baseline.eval(self.states, returns)

        # 3) Compute advantages
        advantages = returns - baselines

        # 4) Actor loss
        log_probs_tensor = torch.stack(self.action_log_probs)
        actor_loss = -(log_probs_tensor * advantages.detach()).mean()

        # 5) Total loss = actor + baseline loss
        loss = actor_loss + loss_b  # if self.baseline is not None else actor_loss

        # 6) Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # 7) Reset                 --> dit dan in reset() ? dan moet je dat ook wel weer aanroepen in train()
        # self.action_log_probs.clear()  --> of hier _reset() aanroepen?
        # self.rewards.clear()
        # self.states.clear()

        return actor_loss, loss_b  # if self.baseline is not None else actor_loss

    def reset(self):
        self.action_log_probs.clear()
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.penalties.clear()
        self.baselines.clear()

    def update(self):
        pass
