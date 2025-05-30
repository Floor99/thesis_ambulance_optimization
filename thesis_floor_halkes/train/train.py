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
    ts_path="data/processed/node_features.parquet",
    seeds=[0, 1, 2, 3, 4],
    dists=[200, 300, 400, 500, 600],  # each graph has its own radius
)

# dataset = [get_static_data_object(
#     time_series_df_path="data/processed/node_features.parquet",
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
    max_steps=50,
    # start_timestamp = '2024-01-31 08:30:00',
)

hidden_size = 64
input_dim = hidden_size * 2
learning_rate = 0.001
num_epochs = 4

static_encoder = StaticGATEncoder(
    in_channels=4, hidden_size=hidden_size, edge_attr_dim=2, num_layers=4
)
dynamic_encoder = DynamicGATEncoder(
    in_channels=4, hidden_size=hidden_size, num_layers=4
)
decoder = AttentionDecoder(embed_dim=hidden_size * 2, num_heads=4)
fixed_context = FixedContext(embed_dim=hidden_size * 2)
baseline = CriticBaseline()

agent = DynamicAgent(
    static_encoder=static_encoder,
    dynamic_encoder=dynamic_encoder,
    decoder=decoder,
    fixed_context=fixed_context,
    baseline=baseline,
)
agent.routes.clear()

policy_parameters = [
    {"params": agent.static_encoder.parameters()},
    {"params": agent.dynamic_encoder.parameters()},
    {"params": agent.decoder.parameters()},
]
baseline_parameters = [
    {"params": agent.baseline.parameters()},
]
policy_optimizer = torch.optim.Adam(policy_parameters, lr=learning_rate)
baseline_optimizer = torch.optim.Adam(baseline_parameters, lr=learning_rate)


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

        for idx, static_data in enumerate(dataset):
            print(f"\n\n === Graph {idx} ===")
            env.static_data = static_data
            total_reward = 0
            state = env.reset()

            entropies = []

            for step in range(env.max_steps):
                print(f"\n{step= }")
                action, action_log_prob, entropy = agent.select_action(state)
                entropies.append(entropy.item())

                embedding = agent.embeddings[-1]["final"]
                embedding_for_critic = embedding.detach().clone().requires_grad_()
                baseline_value = agent.baseline(embedding_for_critic, hidden_dim=128)

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
                    print("terminated")
                    break

            print(f"{agent.current_route= }")
            policy_loss, baseline_loss = agent.finish_episode()
            policy_optimizer.zero_grad()
            baseline_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            baseline_loss.backward()
            baseline_optimizer.step()

            step_id = epoch * len(dataset.data_list) + idx
            avg_entropy = sum(entropies) / len(entropies)
            final_node = agent.current_route[-1]
            goal_node = env.static_data.end_node
            reached_goal = int(final_node == goal_node)
            total_travel_time = sum(
                env.step_travel_time_route
            )  # + sum([step['Waiting Time Penalty'] for step in env.])
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

            # MLflow logging
            mlflow.log_metric("reward", total_reward, step=step_id)
            mlflow.log_metric("policy_loss", policy_loss.item(), step=step_id)
            mlflow.log_metric("baseline_loss", baseline_loss.item(), step=step_id)
            mlflow.log_metric(
                "avg_entropy", sum(entropies) / len(entropies), step=step_id
            )
            mlflow.log_metric("reached_goal", reached_goal, step=step_id)
            mlflow.log_metric("total_travel_time", total_travel_time, step=step_id)

            # route_path = f"checkpoints/route_epoch_{epoch}_graph_{idx}.txt"
            # with open(route_path, "w") as f:
            #     f.write("->".join(map(str, agent.current_route)))
            # mlflow.log_artifact(route_path)

            episode_infos.append(
                {
                    "epoch": int(epoch),
                    "graph_idx": int(idx),
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
            )

            # Plot the route in graph
            orig_ids_route = [
                env.static_data.node_id_mapping[i] for i in agent.current_route
            ]
            route = np.array(orig_ids_route).tolist()
            fig, ax = plot_with_route(
                env.static_data.G_sub,
                env.static_data.G_pt,
                route,
                goal_node=env.static_data.node_id_mapping[goal_node],
            )
            mlflow.log_figure(fig, f"plots/route_epoch_{epoch}_graph_{idx}.png")
            # plt.close(fig)
            # fig.savefig(f"data/plots/subgraph_route_graph_{idx}_epoch_{epoch}.png", dpi=300)

            agent.reset()
            env.reward_modifier_calculator.reset()

            num_nodes = state.static_data.x.size(0)
            logger.log(total_reward, num_nodes)
            print(f"\n  -- Graph {idx} reward: {total_reward:.2f}")

            # logger.summary()

        # Save JSON summary of all episodes in this epoch
        mlflow.log_dict(episode_infos, f"checkpoints/episode/info_epoch{epoch}.json")
        # episode_json_path = f"checkpoints/episode_info_epoch{epoch}.json"
        # with open(episode_json_path, "w") as f:
        #     json.dump(episode_infos, f, indent=2)
        # mlflow.log_artifact(episode_json_path)

        # Save .pt checkpoint
        checkpoint_path = f"checkpoints/agent_epoch_{epoch}.pt"
        torch.save(
            {
                "static_encoder": agent.static_encoder.state_dict(),
                "dynamic_encoder": agent.dynamic_encoder.state_dict(),
                "decoder": agent.decoder.state_dict(),
                "baseline": agent.baseline.state_dict(),
            },
            checkpoint_path,
        )
        mlflow.log_artifact(checkpoint_path)

    # Log full models
    mlflow.pytorch.log_model(agent.static_encoder, "static_encoder_model")
    mlflow.pytorch.log_model(agent.dynamic_encoder, "dynamic_encoder_model")
    mlflow.pytorch.log_model(agent.decoder, "decoder_model")
    mlflow.pytorch.log_model(agent.baseline, "baseline_model")

    logger.plot()
    plot_graph(state.dynamic_data)
