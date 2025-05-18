from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import torch
from torch_geometric.data import Data

from thesis_floor_halkes.environment.base import Environment


class DynamicFeatureGetter(ABC):
    """
    Abstract base class for dynamic feature getters.
    """

    @abstractmethod
    def get_dynamic_features(
        self, environment: Environment, traffic_light_idx: int, max_wait: float = 10.0
    ) -> Data:
        """
        Abstract method to get dynamic features.
        """
        pass


class RandomDynamicFeatureGetter(DynamicFeatureGetter):
    """
    Random dynamic feature getter that generates random features.
    """

    def get_dynamic_features(
        self,
        environment: Environment,
        traffic_light_idx: int,
        current_node: int,
        visited_nodes: List[int],
        max_wait: float = 10.0,
    ):
        """
        Generate random dynamic features.
        """
        torch.manual_seed(1)  # for reproducibility
        torch.cuda.manual_seed(1)  # for reproducibility
        num_nodes = environment.static_data.num_nodes
        traffic_lights = environment.static_data.x[:, traffic_light_idx].bool()
        rand_bits = torch.randint(
            0,
            2,
            (num_nodes,),
            dtype=torch.bool,
        )  # get random bits
        light_status = (
            rand_bits & traffic_lights
        )  # if there is a traffic light, set status from random bits
        waiting_times = torch.rand(num_nodes) * max_wait  # set random waiting time

        is_current_node = torch.zeros(num_nodes)
        is_current_node[current_node] = 1.0  # set current node to 1.0

        is_visited = torch.zeros(num_nodes)
        is_visited[visited_nodes] = 1.0

        x = torch.stack(
            [
                light_status.to(torch.float),  # dynamic feature
                waiting_times,  # dynamic feature
                is_current_node,
                is_visited,
            ],
            dim=1,
        )

        data = Data(
            x=x,
            edge_index=environment.static_data.edge_index,
        )
        return data


class DynamicFeatureGetterDataFrame(DynamicFeatureGetter):
    def __init__(
        self,
    ):
        pass

    def get_dynamic_features(
        self,
        environment: Environment,
        traffic_light_idx: int,
        current_node: int,
        visited_nodes: List[int],
        time_step: int,
        sub_node_df: pd.DataFrame,
    ) -> Data:
        self.timestamps = sorted(sub_node_df["timestamp"].unique())
        t = self.timestamps[time_step]
        df_t = sub_node_df[sub_node_df["timestamp"] == t].sort_values("node_id_y")

        wait_times = torch.tensor(df_t["wait_time"].values, dtype=torch.float)
        # print(f"wait_times: {wait_times= }")
        num_nodes = environment.static_data.num_nodes

        has_light = environment.static_data.x[:, traffic_light_idx].bool()
        rand_bits = torch.randint(0, 2, (num_nodes,), dtype=torch.bool)
        light_status = (rand_bits & has_light).to(torch.float)
        # set light status to red
        light_status[~has_light] = 0.0

        is_current_node = torch.zeros(num_nodes)
        is_current_node[current_node] = 1.0

        is_visited = torch.zeros(num_nodes)
        is_visited[visited_nodes] = 1.0

        x = torch.stack(
            [
                light_status,
                wait_times,
                is_current_node,
                is_visited,
            ],
            dim=1,
        )

        return Data(
            x=x,
            edge_index=environment.static_data.edge_index,
        )
