from typing import List
import torch
from torch_geometric.data import Data, Dataset

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter, RandomDynamicFeatureGetter
from thesis_floor_halkes.penalties.calculator import PenaltyCalculator, RewardModifierCalculator
from thesis_floor_halkes.state import State
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.travel_time import calculate_edge_travel_time
from thesis_floor_halkes.environment.base import Environment
from pprint import pprint

class DynamicEnvironment(Environment):
    def __init__(
        self, 
        static_dataset: list[Data]|Dataset,
        dynamic_feature_getter: RandomDynamicFeatureGetter,
        reward_modifier_calculator: RewardModifierCalculator,
        max_steps: int = 30,
        ):
        self.static_dataset = static_dataset 
        self.dynamic_feature_getter = dynamic_feature_getter
        self.reward_modifier_calculator = reward_modifier_calculator
        self.max_steps = max_steps
    
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

        self.steps_taken = 0
        self.terminated = False
        self.truncated = False
        
        self.adjecency_matrix = build_adjecency_matrix(self.static_data.num_nodes, self.static_data)
        
        self.states = []
        init_state = self._get_state()
        self.states.append(init_state)
        
        return init_state
    
    def _get_state(self, action=None):
        """
        Get the current state of the environment.
        """
        
        if action is not None:
            current_node = action
            previous_visited_nodes = self.states[-1].visited_nodes
            visited_nodes = self.update_visited_nodes(previous_visited_nodes, current_node)
            
        else:
            current_node = self.static_data.start_node
            visited_nodes = [self.static_data.start_node]
            
        # resample dynamic features
        dynamic_features = self.dynamic_feature_getter.get_dynamic_features(environment=self, 
                                                                            traffic_light_idx=0, 
                                                                            current_node = current_node, 
                                                                            visited_nodes = visited_nodes, 
                                                                            max_wait = 10.0)
        # get static features
        static_features = self.static_data
        
        # get valid actions
        valid_actions = self.get_valid_actions(self.adjecency_matrix, current_node, visited_nodes)
        
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
        old_state = self.states[-1]
        
        if self.steps_taken >= self.max_steps:
            self.truncated = True
            return old_state, reward, self.terminated, self.truncated, {}
        
        self.steps_taken += 1
        
        
        new_state = self._get_state(action)
        
        self.states.append(new_state)
        
        # Check if action is valid
        if action not in old_state.valid_actions:
            raise ValueError(f"Invalid action {action} from node {new_state.current_node}.")
        
        # Compute the travel time 
        edge_idx = next(idx for (v, idx) in self.adjecency_matrix[old_state.current_node] if v == action)
        travel_time_edge = calculate_edge_travel_time(
            self.static_data, 
            edge_index=edge_idx, 
            length_feature_idx=0, 
            speed_feature_idx=1
        )

        # Compute the reward 
        penalty = self.reward_modifier_calculator.calculate_total(visited_nodes = old_state.visited_nodes,
                                                            action= action,
                                                            current_node= new_state.current_node,
                                                            end_node= new_state.end_node,
                                                            environment = self)

        reward = - travel_time_edge + penalty        
        
        if new_state.current_node == new_state.end_node:
            self.terminated = True

        if self.steps_taken >= self.max_steps:
            self.truncated = True
        
        return new_state, reward, self.terminated, self.truncated, {}

    def get_valid_actions(self, adj_matrix: dict[int, list[tuple[int, int]]], current_node:int, visited_nodes) -> list[int]:
        """
        Return valid actions (neighbors) based on the adjacency matrix.
        """
        # return [v for v, _ in adj_matrix[current_node]]
        return [v for v, _ in adj_matrix[current_node] if v not in visited_nodes]
    
    def update_visited_nodes(self, prev_visited_nodes:list[int], action):
        return prev_visited_nodes + [action]