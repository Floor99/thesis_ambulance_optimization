import json
import random
import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
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
# from thesis_floor_halkes.features.static.getter import get_static_data_object
from thesis_floor_halkes.features.static.new_getter import collect_static_data_objects
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
    NoSignalIntersectionPenalty,
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


torch.autograd.set_detect_anomaly(True)


# ==== Training Loop with MLFlow Tracking ====
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs("checkpoints", exist_ok=True)
    ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]
    mlflow.set_experiment("dynamic_ambulance_training")

    # ==== Load the static dataset ====
    # dataset = StaticListDataset(
    #     ts_path="data/processed/node_features_expanded.parquet",
    #     # seeds=[0, 1, 2, 3, 4],
    #     seeds=[5, 6, 7, 8, 9],
    #     dists=[200, 300, 400, 500, 600],  # each graph has its own radius
    # )

    dynamic_node_idx = {
        "status": 0,
        "wait_time": 1,
        "current_node": 2,
        "visited_nodes": 3,
    }

    static_node_idx = {
        "lat": 0,
        "lon": 1,
        "has_light": 2,
        "dist_to_goal": 3,
    }

    static_edge_idx = {
        "length": 0,
        "speed": 1,
    }

    # dataset = [
    #     get_static_data_object(
    #         time_series_df_path="data/processed/node_features_expanded.parquet",
    #         dist = 1000,
    #         seed=1,
    #     )
    # ]
    
    dataset = collect_static_data_objects(
        base_dir = "data/training_data",
    )

    # ==== Reward Modifiers ====
    revisit_penalty = RevisitNodePenalty(
        name="Revisit Node Penalty", penalty=cfg.reward_mod.revisit_penalty_value
    )
    penalty_per_step = PenaltyPerStep(
        name="Penalty Per Step", penalty=cfg.reward_mod.penalty_per_step_value
    )
    goal_bonus = GoalBonus(name="Goal Bonus", bonus=cfg.reward_mod.goal_bonus_value)
    dead_end_penalty = DeadEndPenalty(
        name="Dead End Penalty", penalty=cfg.reward_mod.dead_end_penalty_value
    )
    waiting_time_penalty = WaitTimePenalty(name="Waiting Time Penalty")
    higher_speed_bonus = HigherSpeedBonus(
        name="Higher Speed Bonus", bonus=cfg.reward_mod.higher_speed_bonus_value
    )
    closer_to_goal_bonus = CloserToGoalBonus(
        name="Closer To Goal Bonus", bonus=cfg.reward_mod.closer_to_goal_bonus_value
    )
    no_signal_intersection_penalty = NoSignalIntersectionPenalty(
        name="No Signal Intersection Penalty", penalty=cfg.reward_mod.no_signal_intersection_penalty_value
    )

    reward_modifier_calculator = RewardModifierCalculator(
        modifiers=[
            # revisit_penalty,
            penalty_per_step,
            goal_bonus,
            waiting_time_penalty,
            dead_end_penalty,
            higher_speed_bonus,
            closer_to_goal_bonus,
            no_signal_intersection_penalty,
        ],
        weights=[
            # 1.0,
            1.0,
            1.0,
            1.0,
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
        max_steps=cfg.training.max_steps,
        start_timestamp="2024-01-31 08:30:00",
        dynamic_node_idx=dynamic_node_idx,
        static_node_idx=static_node_idx,
        static_edge_idx=static_edge_idx,
    )

    encoder_output_dim = cfg.stat_enc.out_size
    static_encoder = StaticGATEncoder(
        in_channels=4,
        hidden_size=cfg.stat_enc.hidden_size,
        edge_attr_dim=2,
        num_layers=cfg.stat_enc.num_layers,
        heads=cfg.stat_enc.num_heads,
        dropout=cfg.stat_enc.dropout,
        out_size=encoder_output_dim,
    )
    dynamic_encoder = DynamicGATEncoder(
        in_channels=4,
        hidden_size=cfg.dyn_enc.hidden_size,
        num_layers=cfg.dyn_enc.num_layers,
        heads=cfg.dyn_enc.num_heads,
        dropout=cfg.dyn_enc.dropout,
        out_size=encoder_output_dim,
    )
    
    decoder = AttentionDecoder(embed_dim=encoder_output_dim * 2, num_heads=cfg.decoder.num_heads)
    fixed_context = FixedContext(embed_dim=encoder_output_dim * 2)
    baseline = CriticBaseline(encoder_output_dim * 2, hidden_dim=cfg.baseline.hidden_size)
    
    gamma = cfg.reinforce.discount_factor
    agent = DynamicAgent(
        static_encoder=static_encoder,
        dynamic_encoder=dynamic_encoder,
        decoder=decoder,
        fixed_context=fixed_context,
        baseline=baseline,
        gamma=gamma,
        entropy_coeff=cfg.reinforce.entropy_coeff,
    )

    use_joint_optimization = True  # or False
    baseline_weight = cfg.reinforce.baseline_loss_coeff if baseline is not None else 0.0

    if use_joint_optimization and baseline is not None:
        # One optimizer for all trainable components
        optimizer = torch.optim.Adam(
            [
            {"params": agent.static_encoder.parameters(), "lr": cfg.stat_enc.learning_rate},
            {"params": agent.dynamic_encoder.parameters(), "lr": cfg.dyn_enc.learning_rate},
            {"params": agent.decoder.parameters(), "lr": cfg.decoder.learning_rate},
            {"params": agent.baseline.parameters(), "lr": cfg.baseline.learning_rate},
            ]
        )
    else:
        # Separate optimizers
        policy_optimizer = torch.optim.Adam(
            [
            {"params": agent.static_encoder.parameters(), "lr": cfg.stat_enc.learning_rate},
            {"params": agent.dynamic_encoder.parameters(), "lr": cfg.dyn_enc.learning_rate},
            {"params": agent.decoder.parameters(), "lr": cfg.decoder.learning_rate},
            ]
        )
        if baseline is not None:
            baseline_optimizer = torch.optim.Adam(
                agent.baseline.parameters(), lr=cfg.baseline.learning_rate
            )


    with mlflow.start_run():
        # mlflow.log_params(
        #     {
        #         "learning_rate": learning_rate,
        #         "hidden_size": hidden_size,
        #         "max_steps": env.max_steps,
        #         "num_epochs": num_epochs,
        #         "decoder_type": "AttentionDecoder",
        #         "encoder_type": "GATEncoder",
        #     }
        # )

        success_history = []
        rolling_window = 20

        num_epochs = cfg.training.num_epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            for graph_idx, static_data in enumerate(dataset):
                env.static_data = static_data
                total_reward = 0
                state = env.reset()
                print(env.states[0].dynamic_data.x)
                entropies = []
                step_log = []

                for step in range(env.max_steps):
                    action, action_log_prob, entropy = agent.select_action(state)
                    entropies.append(entropy.item())

                    if baseline is not None:
                        embedding = agent.embeddings[-1]["final"]
                        embedding_for_critic = embedding.detach()
                        baseline_value = agent.baseline(embedding_for_critic)

                    new_state, reward, terminated, truncated, _ = env.step(action)
                    print(f"Action: {action}")

                    new_static_data_x = new_state.static_data.x
                    new_dynamic_data_x = new_state.dynamic_data.x

                    agent.store_state(new_state)
                    agent.store_action_log_prob(action_log_prob)
                    agent.store_action(action)
                    agent.store_reward(reward)
                    if baseline is not None:
                        agent.store_baseline_value(baseline_value)
                    agent.store_entropy(entropy)

                    total_reward += reward
                    state = new_state
                    
                    step_information = {
                        "Step": step,
                        "Action": action,
                        "Static Features": new_static_data_x[action],
                        "Dynamic Features": new_dynamic_data_x[action],                        
                    }
                    step_log.append(step_information)
                    
                    if terminated or truncated:
                        break

                # step_id = epoch * len(dataset.data_list) + graph_idx
                step_id = epoch * len(dataset) + graph_idx

                policy_loss, baseline_loss = agent.finish_episode()
                total_loss = policy_loss + (baseline_weight * baseline_loss)

                if use_joint_optimization and baseline is not None:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                else:
                    policy_optimizer.zero_grad()
                    if baseline is not None:
                        baseline_optimizer.zero_grad()
                        policy_loss.backward(retain_graph=True)
                        baseline_loss.backward()
                        policy_optimizer.step()
                        baseline_optimizer.step()
                    else:
                        policy_loss.backward()
                        policy_optimizer.step()

                ### MLflow logging
                avg_entropy = sum(entropies) / len(entropies)
                final_node = agent.current_route[-1]
                goal_node = env.static_data.end_node
                reached_goal = int(final_node == goal_node)
                total_travel_time = sum(env.step_travel_time_route)
                total_travel_time_including_waittime = total_travel_time - sum(
                    [
                        step["Waiting Time Penalty"]
                        for step in env.step_modifier_contributions
                    ]
                )
                total_total_travel_time = total_travel_time_including_waittime - sum(
                    [
                        step["No Signal Intersection Penalty"]
                        for step in env.step_modifier_contributions
                    ]
                )
                reward_modifier_contributions = [
                    {k: float(v) for k, v in step_contrib.items()}
                    for step_contrib in env.step_modifier_contributions
                ]

                if reached_goal == 1:
                    success_history.append(1)
                else:
                    success_history.append(0)

                if len(success_history) >= rolling_window:
                    success_rate = np.mean(success_history[-rolling_window:] * 100)
                    mlflow.log_metric("success_rate", success_rate, step=step_id)

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
                    "total_total_travel_time": float(total_total_travel_time),
                    "total_reward": float(total_reward),
                    "avg_entropy": float(avg_entropy),
                    "policy_loss": float(policy_loss.item()),
                    "baseline_loss": float(baseline_loss.item()) if baseline else 0,
                    "route": [int(n) for n in agent.current_route],
                    "reward_modifier_contributions": reward_modifier_contributions,
                }

                # orig_ids_route = [
                #     env.static_data.node_id_mapping[i] for i in agent.current_route
                # ]
                # route = np.array(route).tolist()
                route = agent.current_route
                fig, _ = plot_with_route(
                    env.static_data.G_sub,
                    env.static_data.G_pt,
                    route,
                    # goal_node=env.static_data.node_id_mapping[goal_node],
                    goal_node = env.static_data.end_node,
                )

                # Log the metrics
                mlflow.log_metrics(
                    {
                        "reward": total_reward,
                        "policy_loss": policy_loss.item(),
                        "baseline_loss": baseline_loss.item() if baseline else 0,
                        "total_loss": total_loss.item(),
                        "avg_entropy": avg_entropy,
                        "reached_goal": reached_goal,
                        "total_travel_time": total_travel_time,
                    },
                    step=epoch * len(dataset) + graph_idx,
                )

                # Log artifacts
                log_episode_artifacts(episode_info, step_log, epoch, graph_idx, fig)

                ### Reset agent and environment for next episode
                agent.reset()
                env.reward_modifier_calculator.reset()

            # Log the agent state dict
            log_agent_checkpoint(agent, epoch)

        # Log full models
        log_full_models(agent)
    return total_total_travel_time


if __name__ == "__main__":
    main()
