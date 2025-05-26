from dataclasses import dataclass
import datetime
import json
import math
import random
import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import osmnx as ox
import os
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import random_split

from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.graph.graph_generator import plot_with_route
# from thesis_floor_halkes.features.static.getter import get_static_data_object
from thesis_floor_halkes.features.static.new_getter import collect_static_data_objects, StaticDataObjectSet, split_subgraphs
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
    flatten_dict,
    get_nested,
    log_agent_checkpoint,
    log_episode_artifacts,
    log_full_models,
    log_gradient_norms,
    log_latest_best_params_to_mlflow,
)
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.agent.dynamic import DynamicAgent
from thesis_floor_halkes.utils.episode import finish_episode
from thesis_floor_halkes.utils.reward_logger import RewardLogger
from thesis_floor_halkes.benchmarks.simulate_dijkstra import simulate_dijkstra_path_cost
from thesis_floor_halkes.utils.plot_graph import plot_graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # For testing purposes, use CPU

print(f"Using device: {device}")


torch.autograd.set_detect_anomaly(True)


# ==== Training Loop with MLFlow Tracking ====
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs("checkpoints", exist_ok=True)
    # ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]
    sweep_id = os.environ.get("SWEEP_ID", "no_sweep_id")
    mlflow.set_experiment("dynamic_ambulance_training")
    
    base_dir = "data/training_data/small_subgraphs"
    train_dirs, val_dirs, test_dirs = split_subgraphs(base_dir, train_frac=1, val_frac=0.15, seed=42)
    
    train_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=train_dirs, num_pairs_per_graph = 1, seed = 42)
    # train_set.data_objects = train_set.data_objects[:1]
    # val_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=val_dirs, num_pairs_per_graph = 5, seed = 42)
    # test_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=test_dirs, num_pairs_per_graph = 5, seed = 42)
    
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=4, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=4, shuffle=True)
    

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
    
    

    # ==== Reward Modifiers ====
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
            penalty_per_step,
            goal_bonus,
            waiting_time_penalty,
            dead_end_penalty,
            higher_speed_bonus,
            closer_to_goal_bonus,
            no_signal_intersection_penalty,
        ],
        weights=[
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
        static_dataset=train_set,
        dynamic_feature_getter=DynamicFeatureGetterDataFrame(),
        reward_modifier_calculator=reward_modifier_calculator,
        max_steps=cfg.training.max_steps,
        # start_timestamp="2024-01-31 08:30:00",
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
    ).to(device)
    dynamic_encoder = DynamicGATEncoder(
        in_channels=4,
        hidden_size=cfg.dyn_enc.hidden_size,
        num_layers=cfg.dyn_enc.num_layers,
        heads=cfg.dyn_enc.num_heads,
        dropout=cfg.dyn_enc.dropout,
        out_size=encoder_output_dim,
    ).to(device)
    
    decoder = AttentionDecoder(embed_dim=encoder_output_dim * 2, num_heads=cfg.decoder.num_heads).to(device)
    fixed_context = FixedContext(embed_dim=encoder_output_dim * 2).to(device)
    baseline = CriticBaseline(encoder_output_dim * 2, hidden_dim=cfg.baseline.hidden_size).to(device)
    
    gamma = cfg.reinforce.discount_factor
    agent = DynamicAgent(
        static_encoder=static_encoder,
        dynamic_encoder=dynamic_encoder,
        decoder=decoder,
        fixed_context=fixed_context,
        baseline=baseline,
    )

    use_joint_optimization = True  # or False

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
        for epoch in range(cfg.training.num_epochs):
            print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
            batch_infos = []
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                episode_infos = []
                skip_episode = False

                for episode in range(batch.num_graphs):
                    print(f"Episode {episode + 1}/{batch.num_graphs} in batch {batch_idx + 1}/{len(train_loader)}")
                    
                    static_data = batch.get_example(episode)
                    env.static_data = static_data
                    state = env.reset()
                    agent.reset()
                    
                    episode_info = {
                        "rewards": [],
                        "action_log_probs": [],
                        "entropies": [],
                        "baseline_values": [],
                        "actions": [],
                        "states": [],
                        "reached_goal": False,
                        "total_loss": None,
                        "policy_loss": None,
                        "baseline_loss": None,
                        "entropy_loss": None,
                    }
                    
                    for step in range(cfg.training.max_steps):
                        # print(f"Step {step + 1}/{cfg.training.max_steps}")
                        
                        if len(env.states[-1].valid_actions) == 0:
                            print(f"Graph {episode} has no valid actions. Skipping episode.")
                            skip_episode = True
                            break
                        
                        action, log_prob, entropy = agent.select_action(state)
                        baseline_value = agent.baseline(agent.final_embedding)
                        next_state, reward, terminated, truncated, _ = env.step(action)

                        episode_info["rewards"].append(reward)
                        episode_info["action_log_probs"].append(log_prob)
                        episode_info["entropies"].append(entropy)
                        episode_info["baseline_values"].append(baseline_value)
                        episode_info["actions"].append(action)
                        episode_info["states"].append(state)
                        episode_info["reached_goal"] = True if terminated else False
                        
                        state = next_state
                        if terminated or truncated:
                            break
                    if skip_episode:
                        continue
                    
                    total_loss, policy_loss, baseline_loss, entropy_loss = finish_episode(
                        rewards=episode_info["rewards"],
                        action_log_probs=episode_info["action_log_probs"],
                        entropies=episode_info["entropies"],
                        baseline_weight=cfg.reinforce.baseline_loss_coeff,
                        baseline_values= episode_info["baseline_values"],
                        gamma=gamma,
                        entropy_coeff=cfg.reinforce.entropy_coeff,
                    )
                    episode_info["total_loss"] = total_loss
                    episode_info["policy_loss"] = policy_loss
                    episode_info["baseline_loss"] = baseline_loss
                    episode_info["entropy_loss"] = entropy_loss
                    episode_infos.append(episode_info)
                    
                    # Log metrics to MLFlow
                    mlflow.log_metrics(
                        {
                            "total_loss": total_loss.clone().detach().cpu().item(),
                            "policy_loss": policy_loss.clone().detach().cpu().item(),
                            "baseline_loss": baseline_loss.clone().detach().cpu().item(),
                            "entropy_loss": entropy_loss.clone().detach().cpu().item(),
                            "reached_goal": int(episode_info["reached_goal"]),
                            "num_steps": len(episode_info["rewards"]),
                            "reward": sum(episode_info["rewards"]),
                        },
                        step=epoch * len(train_set) + batch_idx * train_loader.batch_size + episode,
                    )
                    
                    episode_infos.append(episode_info)
                
                
                batch_mean_total_loss = torch.mean(torch.stack(
                    [info["total_loss"] for info in episode_infos if info["total_loss"] is not None]
                ))
                batch_mean_policy_loss = torch.mean(torch.stack(
                    [info["policy_loss"] for info in episode_infos if info["policy_loss"] is not None]
                ))
                batch_mean_baseline_loss = torch.mean(torch.stack(
                    [info["baseline_loss"] for info in episode_infos if info["baseline_loss"] is not None]
                ))
                batch_mean_entropy_loss = torch.mean(torch.stack(
                    [info["entropy_loss"] for info in episode_infos if info["entropy_loss"] is not None]
                ))
                
                if use_joint_optimization:
                    optimizer.zero_grad()
                    batch_mean_total_loss.backward()
                    # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                    optimizer.step()
                else:
                    if baseline is not None:
                        policy_optimizer.zero_grad()
                        baseline_optimizer.zero_grad()
                        batch_mean_policy_loss.backward(retain_graph=True)
                        batch_mean_baseline_loss.backward()
                        # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                        policy_optimizer.step()
                        baseline_optimizer.step()
                    else:
                        policy_optimizer.zero_grad()
                        batch_mean_policy_loss.backward()
                        # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                        policy_optimizer.step()
                
                batch_info = {
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                    "batch_mean_policy_loss": batch_mean_policy_loss.clone().detach().cpu().item(),
                    "batch_mean_baseline_loss": batch_mean_baseline_loss.clone().detach().cpu().item() if baseline is not None else None,
                    "batch_mean_total_loss": batch_mean_total_loss.clone().detach().cpu().item() if use_joint_optimization else None,
                    "batch_mean_entropy_loss": batch_mean_entropy_loss.clone().detach().cpu().item(),
                    "episode_infos": episode_infos,
                    "batch_mean_reward": np.mean([np.sum(info["rewards"]) for info in episode_infos]),
                    "batch_success_rate": np.mean([1 if info["reached_goal"] else 0 for info in episode_infos]),
                }
                
                # log batch metrics to MLFlow
                mlflow.log_metrics(
                    {
                        "batch_mean_policy_loss": batch_info["batch_mean_policy_loss"],
                        "batch_mean_baseline_loss": batch_info["batch_mean_baseline_loss"],
                        "batch_mean_entropy_loss": batch_info["batch_mean_entropy_loss"],
                        "batch_mean_total_loss": batch_info["batch_mean_total_loss"],
                        "batch_mean_reward": batch_info["batch_mean_reward"],
                        "batch_success_rate": batch_info["batch_success_rate"],
                    },
                    step=epoch * math.ceil(len(train_set) / train_loader.batch_size) + batch_idx,
                )
                batch_infos.append(batch_info)
            
            # epoch metrics to MLFlow
            epoch_mean_policy_loss = np.mean([batch_info["batch_mean_policy_loss"] for batch_info in batch_infos])
            epoch_mean_baseline_loss = np.mean([batch_info["batch_mean_baseline_loss"] for batch_info in batch_infos if batch_info["batch_mean_baseline_loss"] is not None])
            epoch_mean_total_loss = np.mean([batch_info["batch_mean_total_loss"] for batch_info in batch_infos if batch_info["batch_mean_total_loss"] is not None])
            epoch_mean_entropy_loss = np.mean([batch_info["batch_mean_entropy_loss"] for batch_info in batch_infos])
            epoch_mean_reward = np.mean([batch_info["batch_mean_reward"] for batch_info in batch_infos])
            epoch_success_rate = np.mean([batch_info["batch_success_rate"] for batch_info in batch_infos])
            mlflow.log_metrics(
                {
                    "epoch_mean_policy_loss": epoch_mean_policy_loss,
                    "epoch_mean_baseline_loss": epoch_mean_baseline_loss,
                    "epoch_mean_total_loss": epoch_mean_total_loss,
                    "epoch_mean_entropy_loss": epoch_mean_entropy_loss,
                    "epoch_mean_reward": epoch_mean_reward,
                    "epoch_success_rate": epoch_success_rate,
                },
                step=epoch,
            )

if __name__ == "__main__":
    sweep_id = f"optuna_sweep_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    os.environ["SWEEP_ID"] = sweep_id
    main()
    if os.environ.get("HYDRA_JOB_NAME") == 'multirun':
        log_latest_best_params_to_mlflow()
    
    