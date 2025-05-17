from typing import List
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import random

from thesis_floor_halkes.features.dynamic.getter import (
    DynamicFeatureGetter,
    RandomDynamicFeatureGetter,
)
from thesis_floor_halkes.features.static.getter import get_static_data_object
from thesis_floor_halkes.penalties.calculator import (
    PenaltyCalculator,
    RewardModifierCalculator,
)
from thesis_floor_halkes.penalties.revisit_node_penalty import (
    AggregatedStepPenalty,
    WaitTimePenalty,
)
from thesis_floor_halkes.state import State
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.utils.travel_time import calculate_edge_travel_time
from thesis_floor_halkes.environment.base import Environment


class DynamicEnvironment(Environment):
    def __init__(
        self,
        static_dataset: list[Data] | Dataset,
        dynamic_feature_getter: RandomDynamicFeatureGetter,
        reward_modifier_calculator: RewardModifierCalculator,
        max_steps: int = 30,
        start_timestamp: str | pd.Timestamp = None,
    ):
        self.static_dataset = static_dataset
        self.dynamic_feature_getter = dynamic_feature_getter
        self.reward_modifier_calculator = reward_modifier_calculator
        self.max_steps = max_steps
        # self.time_stamps = sorted(self.dynamic_feature_getter.df['timestamp'].unique())
        self.start_timestamp = start_timestamp

    def reset(self):
        self.steps_taken = 0
        self.terminated = False
        self.truncated = False
        self.time_stamps = sorted(
            self.static_data.filtered_time_series_df["timestamp"].unique()
        )
        self.adjecency_matrix = build_adjecency_matrix(
            self.static_data.num_nodes, self.static_data
        )

        if self.start_timestamp is not None:
            self.start_timestamp = pd.to_datetime(self.start_timestamp)
            if self.start_timestamp not in self.time_stamps:
                raise ValueError(
                    f"start_timestamp {self.start_timestamp} not in dynamic data"
                )
            self.current_time_idx = self.time_stamps.index(self.start_timestamp)
        else:
            T = len(self.time_stamps)
            max_start = max(0, T - 1 - self.max_steps)
            self.current_time_idx = random.randrange(0, max_start)
            self.start_timestamp = self.time_stamps[self.current_time_idx]

        self.states = []
        init_state = self._get_state()
        self.states.append(init_state)

        self.step_travel_time_route = []
        self.step_modifier_contributions = []

        return init_state

    def _get_state(self, action=None):
        """
        Get the current state of the environment.
        """

        if action is not None:
            current_node = action
            previous_visited_nodes = self.states[-1].visited_nodes
            visited_nodes = self.update_visited_nodes(
                previous_visited_nodes, current_node
            )

        else:
            current_node = self.static_data.start_node
            visited_nodes = [self.static_data.start_node]

        sub_node_df = self.static_data.filtered_time_series_df
        # resample dynamic features
        dynamic_features = self.dynamic_feature_getter.get_dynamic_features(
            environment=self,
            traffic_light_idx=0,
            current_node=current_node,
            visited_nodes=visited_nodes,
            time_step=self.current_time_idx,
            sub_node_df=sub_node_df,
        )

        # get static features
        static_features = self.static_data

        # get valid actions
        valid_actions = self.get_valid_actions(
            self.adjecency_matrix, current_node, visited_nodes
        )

        state = State(
            static_data=static_features,
            dynamic_data=dynamic_features,
            start_node=self.static_data.start_node,
            end_node=self.static_data.end_node,
            num_nodes=self.static_data.num_nodes,
            current_node=current_node,
            visited_nodes=visited_nodes,
            valid_actions=valid_actions,
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
        self.current_time_idx += 1  # bounds by end timestamp toevoegen!

        new_state = self._get_state(action)

        self.states.append(new_state)

        # Check if action is valid
        if action not in old_state.valid_actions:
            raise ValueError(
                f"Invalid action {action} from node {new_state.current_node}."
            )

        # Compute the travel time
        edge_idx = next(
            idx
            for (v, idx) in self.adjecency_matrix[old_state.current_node]
            if v == action
        )
        travel_time_edge = calculate_edge_travel_time(
            self.static_data,
            edge_index=edge_idx,
            length_feature_idx=0,
            speed_feature_idx=1,
        )

        self.step_travel_time_route.append(travel_time_edge)

        # Compute the reward
        penalty = self.reward_modifier_calculator.calculate_total(
            visited_nodes=old_state.visited_nodes,
            action=action,
            current_node=new_state.current_node,
            end_node=new_state.end_node,
            valid_actions=new_state.valid_actions,
            environment=self,
        )

        self.modifier_contributions = (
            self.reward_modifier_calculator.store_modifier_per_step(
                visited_nodes=old_state.visited_nodes,
                action=action,
                current_node=new_state.current_node,
                end_node=new_state.end_node,
                valid_actions=new_state.valid_actions,
                environment=self,
            )
        )

        self.modifier_contributions.update({"step": self.steps_taken})
        self.step_modifier_contributions.append(self.modifier_contributions)

        reward = -travel_time_edge + penalty

        if new_state.valid_actions == []:
            self.truncated = True

        if new_state.current_node == new_state.end_node:
            self.terminated = True

        if self.steps_taken >= self.max_steps:
            self.truncated = True

        if self.terminated or self.truncated:
            agg = AggregatedStepPenalty(name="aggregated_step_penalty", penalty=-3.0)
            agg_value = agg(environment=self)
            reward += agg_value

        return new_state, reward, self.terminated, self.truncated, {}

    # def get_valid_actions(self, adj_matrix: dict[int, list[tuple[int, int]]], current_node:int, visited_nodes) -> list[int]:
    #     """
    #     Return valid actions (neighbors) based on the adjacency matrix.
    #     """
    #     # return [v for v, _ in adj_matrix[current_node]]
    #     return [v for v, _ in adj_matrix[current_node] if v not in visited_nodes]

    def get_valid_actions(
        self,
        adj_matrix: dict[int, list[tuple[int, int]]],
        current_node: int,
        visited_nodes: set[int],
    ) -> list[int]:
        """
        Return valid actions (neighbors) based on the adjacency matrix,
        masking out dead-ends (nodes whose only neighbor is `current_node`)
        except when that node is the goal.
        """
        valid = []
        goal = self.static_data.end_node

        for v, _ in adj_matrix[current_node]:
            # 1. skip visited
            if v in visited_nodes:
                continue

            # 2. always allow stepping onto the goal
            if v == goal:
                valid.append(v)
                continue

            # 3. check if v has any other neighbor besides current_node
            #    (i.e. degree > 1)
            has_other = any(u != current_node for u, _ in adj_matrix[v])
            if has_other:
                valid.append(v)

        return valid

    def update_visited_nodes(self, prev_visited_nodes: list[int], action):
        return prev_visited_nodes + [action]
