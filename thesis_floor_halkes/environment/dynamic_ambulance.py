from typing import List
import torch
from torch_geometric.data import Data, Dataset

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter
from thesis_floor_halkes.penalties.calculator import PenaltyCalculator
from thesis_floor_halkes.state import State
from utils.adj_matrix import build_adjecency_matrix
from utils.travel_time import calculate_edge_travel_time
from thesis_floor_halkes.environment.base import Environment

class DynamicEnvironment(Environment):
    """
    Environment for ambulance routing with dynamic waiting times and light status.

    Static:
      - traffic_lights: bool vector [N] indicating where lights exist
      - edge_index: connectivity
      - edge_attr: [E×2] length, max_speed
    Dynamic (re-sampled each step):
      - light_status: bool vector [N] (0=red,1=green)
      - waiting_times: float vector [N] seconds

    Reward: negative (travel_time + wait_time), plus bonuses/penalties.
    """
    def __init__(
        self, 
        static_dataset: list[Data]|Dataset,
        dynamic_feature_getter: DynamicFeatureGetter,
        penalty_calculator: PenaltyCalculator,
        max_steps: int = 30,
        ):
        self.static_dataset = static_dataset 
        self.dynamic_feature_getter = dynamic_feature_getter
        self.penalty_calculator = penalty_calculator
        self.max_steps = max_steps
        
        self.reset()
        
    
    def reset(self):
        
        if isinstance(self.static_dataset, Dataset):
            nr_of_graphs = self.static_dataset.len()
            graph_idx = torch.randint(0, nr_of_graphs, (1,)).item()
            self.static_data = self.static_dataset.get(graph_idx)
        elif isinstance(self.static_dataset, list):
            nr_of_graphs = len(self.static_dataset)
            graph_idx = torch.randint(0, nr_of_graphs, (1,)).item()
            self.static_data = self.static_dataset[graph_idx]
        else:
            raise ValueError("Dataset must be a list of Data objects or a Dataset object.")
        
        # self.start_node = self.data.start_node
        # self.end_node = self.data.end_node 
        # self.num_nodes = self.data.num_nodes
        # self.current_node = self.start_node
        
        # adjacency matrix
        # self.adjecency_matrix = build_adjecency_matrix(self.num_nodes, self.data)
        self.steps_taken = 0
        self.terminated = False
        self.truncated = False

        return self._get_state()
    
    def _get_state(self, action=None):
        """
        Get the current state of the environment.
        """
        # resample dynamic features
        dynamic_features = self.dynamic_feature_getter.get_dynamic_features(self, traffic_light_idx=0, max_wait = 10.0)
        # get static features
        static_features = self.static_data
        # get current node
        if action is not None:
            current_node = action
        else:
            current_node = self.static_data.start_node

        # get adjecency matrix
        adjecency_matrix = build_adjecency_matrix(self.static_data.num_nodes, self.static_data)
        
        # get valid actions
        valid_actions = self.get_valid_actions(adjecency_matrix)
        
        # get visited nodes
        visited_nodes = self.update_visited_nodes(action)
        
        state = State(
            static_data=static_features,
            dynamic_data=dynamic_features,
            start_node=self.static_data.start_node,
            end_node=self.static_data.end_node,
            num_nodes=self.static_data.num_nodes,
            current_node=current_node,
            visited_nodes=visited_nodes,
            valid_actions=valid_actions
        )
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        """
        old_state = self.states[-1] if self.states else None
        
        if self.steps_taken >= self.max_steps:
            self.truncated = True
            return old_state, reward, self.terminated, self.truncated, {}
        self.steps_taken += 1
        
        # Check if action is valid
        if action not in old_state.valid_actions:
            raise ValueError(f"Invalid action {action} from node {new_state.current_node}.")
        
        new_state = self._get_state(action)
        
        # Compute the travel time 
        edge_idx = next(idx for (v, idx) in self.adjecency_matrix[new_state.current_node] if v == action)
        travel_time_edge = calculate_edge_travel_time(
            self.static_data, 
            edge_index=edge_idx, 
            length_feature_idx=0, 
            speed_feature_idx=1
        )

        # Compute the reward 
        penalty = self.penalty_calculator.calculate_penalty(self, action)
        reward = - travel_time_edge + penalty        
        
        return new_state, reward, self.terminated, self.truncated, {}

    def get_valid_actions(self, adj_matrix: dict[int, list[tuple[int, int]]]) -> list[int]:
        """
        Return valid actions (neighbors) based on the adjacency matrix.
        """
        return [v for v, _ in adj_matrix[self.current_node]]
    
    def update_visited_nodes(self, action):
        # TODO: check if this is correct
        if not self.states.visited_nodes:
            return [self.state.start_node]
        return self.states[-1].visited_nodes.append(action)