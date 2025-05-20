import json
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import osmnx as ox
import os
from torch.nn.utils import clip_grad_norm_

from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.graph.graph_generator import plot_with_route
from thesis_floor_halkes.features.static.getter import get_static_data_object
from thesis_floor_halkes.features.static.static_dataset import StaticListDataset
from thesis_floor_halkes.model.decoder import AttentionDecoder, FixedContext
from thesis_floor_halkes.model.encoders import StaticGATEncoder, DynamicGATEncoder
from thesis_floor_halkes.penalties.calculator import RewardModifierCalculator
from thesis_floor_halkes.penalties.revisit_node_penalty import (
    AggregatedStepPenalty,
    CloserToGoalBonus,
    DeadEndPenalty,
    GoalBonus,
    HigherSpeedBonus,
    PenaltyPerStep,
    RevisitNodePenalty,
    WaitTimePenalty,
)
from thesis_floor_halkes.baselines.critic_network import CriticBaseline
from thesis_floor_halkes.train.mlflow_utils import (
    log_agent_checkpoint,
    log_episode_artifacts,
    log_full_models,
    log_gradient_norms,
)
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.agent.dynamic import DynamicAgent
from thesis_floor_halkes.utils.reward_logger import RewardLogger
from thesis_floor_halkes.benchmarks.simulate_dijkstra import simulate_dijkstra_path_cost
from thesis_floor_halkes.utils.plot_graph import plot_graph

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

os.makedirs("checkpoints", exist_ok=True)
ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]
mlflow.set_experiment("dynamic_ambulance_training")


# ==== Load the static dataset ====
dataset = StaticListDataset(
    ts_path="data/processed/node_features_expanded.parquet",
    seeds=[0, 1, 2, 3, 4],
    dists=[200, 300, 400, 500, 600],  # each graph has its own radius
)

# dataset = [get_static_data_object(
#     time_series_df_path="data/processed/node_features_expanded.parquet",
#     dist = 1000,
#     seed = 42)]

# ==== Reward Modifiers ====
revisit_penalty = RevisitNodePenalty(name="Revisit Node Penalty", penalty=-50.0)
penalty_per_step = PenaltyPerStep(name="Penalty Per Step", penalty=-5)
goal_bonus = GoalBonus(name="Goal Bonus", bonus=100.0)
dead_end_penalty = DeadEndPenalty(name="Dead End Penalty", penalty=-100.0)
waiting_time_penalty = WaitTimePenalty(name="Waiting Time Penalty")
higher_speed_bonus = HigherSpeedBonus(name="Higher Speed Bonus", bonus=20.0)
aggregated_step_penalty = AggregatedStepPenalty(
    name="Aggregated Step Penalty", penalty=-10.0
)
closer_to_goal_bonus = CloserToGoalBonus(name="Closer To Goal Bonus", bonus=1.0)

reward_modifier_calculator = RewardModifierCalculator(
    modifiers=[
        revisit_penalty,
        penalty_per_step,
        goal_bonus,
        waiting_time_penalty,
        dead_end_penalty,
        higher_speed_bonus,
        closer_to_goal_bonus,
    ],
    weights=[
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
)

# ==== Environment and Agent ====
env = DynamicEnvironment(
    static_dataset=dataset,
    dynamic_feature_getter=DynamicFeatureGetterDataFrame(),
    reward_modifier_calculator=reward_modifier_calculator,
    max_steps=3,
    # start_timestamp = '2024-01-31 08:30:00',
)

hidden_size = 64
input_dim = hidden_size * 2
learning_rate = 0.001
num_epochs = 2

static_encoder = StaticGATEncoder(
    in_channels=4, hidden_size=hidden_size, edge_attr_dim=2, num_layers=4
)
dynamic_encoder = DynamicGATEncoder(
    in_channels=4, hidden_size=hidden_size, num_layers=4
)
decoder = AttentionDecoder(embed_dim=hidden_size * 2, num_heads=4)
fixed_context = FixedContext(embed_dim=hidden_size * 2)
baseline = CriticBaseline(hidden_size * 2, hidden_dim=128)

use_joint_optimization = False  # or False
baseline_weight = 0.001  # or 0.1 if critic loss dominates
gamma = 0.99
agent = DynamicAgent(
    static_encoder=static_encoder,
    dynamic_encoder=dynamic_encoder,
    decoder=decoder,
    fixed_context=fixed_context,
    baseline=baseline,
    gamma=gamma,
)
if use_joint_optimization:
    # One optimizer for all trainable components
    optimizer = torch.optim.Adam(
        list(agent.static_encoder.parameters())
        + list(agent.dynamic_encoder.parameters())
        + list(agent.decoder.parameters())
        + list(agent.baseline.parameters()),
        lr=learning_rate,
    )
else:
    # Separate optimizers
    policy_optimizer = torch.optim.Adam(
        list(agent.static_encoder.parameters())
        + list(agent.dynamic_encoder.parameters())
        + list(agent.decoder.parameters()),
        lr=learning_rate,
    )
    baseline_optimizer = torch.optim.Adam(agent.baseline.parameters(), lr=learning_rate)


policy_parameters = [
    {"params": agent.static_encoder.parameters()},
    {"params": agent.dynamic_encoder.parameters()},
    {"params": agent.decoder.parameters()},
]
baseline_parameters = [
    {"params": agent.baseline.parameters()},
]

logger = RewardLogger(smooth_window=20)
torch.autograd.set_detect_anomaly(True)

# ==== Training Loop with MLFlow Tracking ====
with mlflow.start_run():
    mlflow.log_params(
        {
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "max_steps": env.max_steps,
            "num_epochs": num_epochs,
            "decoder_type": "AttentionDecoder",
            "encoder_type": "GATEncoder",
        }
    )

    for epoch in range(num_epochs):
        print(f"\n === Epoch {epoch} ===")
        episode_infos = []

        for graph_idx, static_data in enumerate(dataset):
            print(f"\n\n === Graph {graph_idx} ===")
            # if graph_idx == 1:
            #     break
            env.static_data = static_data
            total_reward = 0
            state = env.reset()
            entropies = []

            for step in range(env.max_steps):
                print(f"\n{step= }")
                action, action_log_prob, entropy = agent.select_action(state)
                entropies.append(entropy.item())

                embedding = agent.embeddings[-1]["final"]
                embedding_for_critic = embedding.detach()  # .clone()#.requires_grad_()
                baseline_value = agent.baseline(embedding_for_critic)

                new_state, reward, terminated, truncated, _ = env.step(action)

                agent.store_state(new_state)
                agent.store_action_log_prob(action_log_prob)
                agent.store_action(action)
                agent.store_reward(reward)
                agent.store_baseline_value(baseline_value)
                agent.store_entropy(entropy)

                total_reward += reward
                state = new_state

                if terminated or truncated:
                    break

            step_id = epoch * len(dataset.data_list) + graph_idx

            policy_loss, baseline_loss = agent.finish_episode()
            total_loss = policy_loss + baseline_weight * baseline_loss

            if use_joint_optimization:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            else:
                policy_optimizer.zero_grad()
                baseline_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                baseline_loss.backward()
                policy_optimizer.step()
                baseline_optimizer.step()

            ### MLflow logging
            avg_entropy = sum(entropies) / len(entropies)
            final_node = agent.current_route[-1]
            goal_node = env.static_data.end_node
            reached_goal = int(final_node == goal_node)
            total_travel_time = sum(env.step_travel_time_route)
            total_travel_time_including_waittime = total_travel_time + sum(
                [
                    step["Waiting Time Penalty"]
                    for step in env.step_modifier_contributions
                ]
            )
            reward_modifier_contributions = [
                {k: float(v) for k, v in step_contrib.items()}
                for step_contrib in env.step_modifier_contributions
            ]

            episode_info = {
                "epoch": int(epoch),
                "graph_idx": int(graph_idx),
                "start_time": str(env.start_timestamp),
                "start_node": int(env.static_data.start_node),
                "end_node": int(env.static_data.end_node),
                "reached_goal": int(reached_goal),
                "total_travel_time": float(total_travel_time),
                "total_travel_time_including_waittime": float(
                    total_travel_time_including_waittime
                ),
                "total_reward": float(total_reward),
                "avg_entropy": float(avg_entropy),
                "policy_loss": float(policy_loss.item()),
                "baseline_loss": float(baseline_loss.item()),
                "route": [int(n) for n in agent.current_route],
                "reward_modifier_contributions": reward_modifier_contributions,
            }
            episode_infos.append(episode_info)

            orig_ids_route = [
                env.static_data.node_id_mapping[i] for i in agent.current_route
            ]
            route = np.array(orig_ids_route).tolist()
            fig, _ = plot_with_route(
                env.static_data.G_sub,
                env.static_data.G_pt,
                route,
                goal_node=env.static_data.node_id_mapping[goal_node],
            )

            # Log the metrics
            mlflow.log_metrics(
                {
                    "reward": total_reward,
                    "policy_loss": policy_loss.item(),
                    "baseline_loss": baseline_loss.item(),
                    "avg_entropy": avg_entropy,
                    "reached_goal": reached_goal,
                    "total_travel_time": total_travel_time,
                },
                step=epoch * len(dataset) + graph_idx,
            )

            # Log artifacts
            log_episode_artifacts(episode_info, epoch, graph_idx, fig)

            ### Reset agent and environment for next episode
            agent.reset()
            env.reward_modifier_calculator.reset()

        # Log the agent state dict
        log_agent_checkpoint(agent, epoch)

    # Log full models
    log_full_models(agent)
