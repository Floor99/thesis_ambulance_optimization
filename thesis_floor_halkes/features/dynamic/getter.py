from abc import ABC, abstractmethod
from typing import List

import torch
from torch_geometric.data import Data

from thesis_floor_halkes.environment.base import Environment


class DynamicFeatureGetter(ABC):
    """
    Abstract base class for dynamic feature getters.
    """
    @abstractmethod
    def get_dynamic_features(self, 
                             environment: Environment,
                             traffic_light_idx: int,
                             max_wait: float = 10.0) -> Data:
        """
        Abstract method to get dynamic features.
        """
        pass

class RandomDynamicFeatureGetter(DynamicFeatureGetter):
    """
    Random dynamic feature getter that generates random features.
    """
    def get_dynamic_features(self, environment:Environment, traffic_light_idx:int, current_node:int, visited_nodes:List[int], max_wait: float = 10.0):
        """
        Generate random dynamic features.
        """
        torch.manual_seed(1) # for reproducibility
        torch.cuda.manual_seed(1) # for reproducibility
        num_nodes = environment.static_data.num_nodes
        traffic_lights = environment.static_data.x[:, traffic_light_idx].bool()
        rand_bits = torch.randint(0, 2, (num_nodes,), dtype=torch.bool,) # get random bits
        light_status = rand_bits & traffic_lights # if there is a traffic light, set status from random bits
        waiting_times = torch.rand(num_nodes) * max_wait # set random waiting time
        
        is_current_node = torch.zeros(num_nodes)
        is_current_node[current_node] = 1.0 # set current node to 1.0
        
        is_visited = torch.zeros(num_nodes)
        is_visited[visited_nodes] = 1.0
        
        x = torch.stack([
            light_status.to(torch.float),          # dynamic feature
            waiting_times,                          # dynamic feature
            is_current_node,
            is_visited,
        ], dim=1)
        
        data = Data(
            x=x,
            edge_index=environment.static_data.edge_index,)
        return data
