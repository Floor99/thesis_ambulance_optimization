from typing import List
import torch
from torch_geometric.data import Data, Dataset

from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter
from thesis_floor_halkes.penalties.calculator import PenaltyCalculator
from thesis_floor_halkes.state import State
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.travel_time import calculate_edge_travel_time
from thesis_floor_halkes.environment.base import Environment
from pprint import pprint

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
        
        # self.reset()
        
    
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
        print("Initial state:")
        print(f"{init_state.start_node= } {init_state.end_node= } {init_state.current_node= } {init_state.visited_nodes= } {init_state.valid_actions= }")

        return init_state
    
    def _get_state(self, action=None):
        """
        Get the current state of the environment.
        """
        # resample dynamic features
        dynamic_features = self.dynamic_feature_getter.get_dynamic_features(environment=self, traffic_light_idx=0, max_wait = 10.0)
        # get static features
        static_features = self.static_data
        
        if action is not None:
            current_node = action
            previous_visited_nodes = self.states[-1].visited_nodes
            visited_nodes = self.update_visited_nodes(previous_visited_nodes, current_node)
            
        else:
            current_node = self.static_data.start_node
            visited_nodes = [self.static_data.start_node]
            
        
        # get valid actions
        valid_actions = self.get_valid_actions(self.adjecency_matrix, current_node)
        
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
        print(f"Step {self.steps_taken}: {action= }")
        
        
        new_state = self._get_state(action)
        print("Old state:")
        print(f"{old_state.current_node= } {old_state.visited_nodes= } {old_state.valid_actions= }")
        
        print("New state:")
        print(f"{new_state.current_node= } {new_state.visited_nodes= } {new_state.valid_actions= }")
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
        # penalty = self.penalty_calculator.calculate_penalty(self, action)
        reward = - travel_time_edge #+ penalty        
        
        return new_state, reward, self.terminated, self.truncated, {}

    def get_valid_actions(self, adj_matrix: dict[int, list[tuple[int, int]]], current_node:int) -> list[int]:
        """
        Return valid actions (neighbors) based on the adjacency matrix.
        """
        return [v for v, _ in adj_matrix[current_node]]
    
    def update_visited_nodes(self, prev_visited_nodes:list[int], action):
        # # TODO: check if this is correct
        # if not self.states.visited_nodes:
        #     return [self.states.start_node]
        # return self.state[-1].visited_nodes.append(action)
        
        # prev_visited_nodes = self.states[-1].visited_nodes
        # if not prev_visited_nodes:
        #     return [self.states[-1].start_node]
        return prev_visited_nodes + [action]