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
from thesis_floor_halkes.features.static.new_getter import (
    collect_static_data_objects,
    split_subgraphs,
    get_static_data_object_subgraph,
)
from thesis_floor_halkes.features.static.final_getter import StaticDataObjectSet
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
# device = torch.device("cpu")  # For testing purposes, use CPU

print(f"Using device: {device}")


torch.autograd.set_detect_anomaly(True)

from hydra.core.hydra_config import HydraConfig


# ==== Training Loop with MLFlow Tracking ====
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs("checkpoints", exist_ok=True)
    # ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    # mlflow.set_experiment("dynamic_ambulance_training")

    train_base_dir = "data/training_data/networks"
    val_base_dir = "data/validation_data/"
    train_set = StaticDataObjectSet(
        base_dir=train_base_dir,
    )
    train_set = train_set[:2]
    # train_set.data_objects = train_set.data_objects[:1]
    val_set = StaticDataObjectSet(base_dir=val_base_dir)
    # test_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=test_dirs, num_pairs_per_graph = 5, seed = 42)

    train_loader = DataLoader(
        train_set, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size, shuffle=True)
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
        name="No Signal Intersection Penalty",
        penalty=cfg.reward_mod.no_signal_intersection_penalty_value,
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

    decoder = AttentionDecoder(
        embed_dim=encoder_output_dim * 2, num_heads=cfg.decoder.num_heads
    ).to(device)
    fixed_context = FixedContext(embed_dim=encoder_output_dim * 2).to(device)
    baseline = CriticBaseline(
        encoder_output_dim * 2, hidden_dim=cfg.baseline.hidden_size
    ).to(device)

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
                {
                    "params": agent.static_encoder.parameters(),
                    "lr": cfg.stat_enc.learning_rate,
                },
                {
                    "params": agent.dynamic_encoder.parameters(),
                    "lr": cfg.dyn_enc.learning_rate,
                },
                {"params": agent.decoder.parameters(), "lr": cfg.decoder.learning_rate},
                {
                    "params": agent.baseline.parameters(),
                    "lr": cfg.baseline.learning_rate,
                },
            ]
        )
    else:
        # Separate optimizers
        policy_optimizer = torch.optim.Adam(
            [
                {
                    "params": agent.static_encoder.parameters(),
                    "lr": cfg.stat_enc.learning_rate,
                },
                {
                    "params": agent.dynamic_encoder.parameters(),
                    "lr": cfg.dyn_enc.learning_rate,
                },
                {"params": agent.decoder.parameters(), "lr": cfg.decoder.learning_rate},
            ]
        )
        if baseline is not None:
            baseline_optimizer = torch.optim.Adam(
                agent.baseline.parameters(), lr=cfg.baseline.learning_rate
            )
    from hydra.types import RunMode

    is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
    parent_run_id = os.environ.get("MLFLOW_PARENT_RUN_ID")
    with mlflow.start_run(
        run_name="optuna_trial" if is_sweep else "single_run",
        nested=bool(parent_run_id),
    ):
        if parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        mlflow.set_tag("is_sweep", str(is_sweep))
        param_overrides = HydraConfig.get().overrides.task
        param_overrides = [param.split("=") for param in param_overrides]
        for k, v in param_overrides:
            if k.startswith("hydra."):
                continue
            if isinstance(v, str) and v.isdigit():
                v = int(v)
            elif isinstance(v, str) and v.replace(".", "", 1).isdigit():
                v = float(v)
            mlflow.log_param(k, v)

        for epoch in range(cfg.training.num_epochs):
            print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
            batch_infos = []
            agent.static_encoder.train()
            agent.dynamic_encoder.train()
            agent.decoder.train()
            agent.baseline.train() if baseline is not None else None

            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                episode_infos = []
                skip_episode = False

                for episode in range(batch.num_graphs):
                    print(
                        f"Episode {episode + 1}/{batch.num_graphs} in batch {batch_idx + 1}/{len(train_loader)}"
                    )
                    static_data = batch.get_example(episode)
                    static_data.start_node = int(static_data.start_node.item())
                    static_data.end_node = int(static_data.end_node.item())
                    env.static_data = static_data
                    state = env.reset()
                    agent.reset()

                    step_info = {}

                    for step in range(cfg.training.max_steps):
                        # print(f"Step {step + 1}/{cfg.training.max_steps}")

                        if len(env.states[-1].valid_actions) == 0:
                            print(
                                f"Graph {episode} has no valid actions. Skipping episode."
                            )
                            skip_episode = True
                            break

                        next_state, terminated, truncated, step_info = execute_step(
                            env, agent, state, step_info
                        )
                        state = next_state
                        if terminated or truncated:
                            break

                    if skip_episode:
                        continue

                    total_loss, policy_loss, baseline_loss, entropy_loss = (
                        finish_episode(
                            rewards=step_info["rewards"],
                            action_log_probs=step_info["action_log_probs"],
                            entropies=step_info["entropies"],
                            baseline_weight=cfg.reinforce.baseline_loss_coeff,
                            baseline_values=step_info["baseline_values"],
                            gamma=gamma,
                            entropy_coeff=cfg.reinforce.entropy_coeff,
                        )
                    )

                    episode_info = record_episode_info(
                        step_info, total_loss, policy_loss, baseline_loss, entropy_loss
                    )
                    episode_infos.append(episode_info)

                    episode_step_id = (
                        episode
                        + batch_idx * train_loader.batch_size
                        + epoch * len(train_set)
                    )
                    mlflow_log_episode_metrics(
                        episode_info,
                        epoch_id=epoch,
                        batch_id=batch_idx,
                        episode_id=episode,
                        graph_id=env.static_data.graph_id,
                        step_id=episode_step_id,
                        prefix="TRAIN",
                    )

                batch_info = record_batch_info(episode_infos)
                batch_infos.append(batch_info)

                if use_joint_optimization:
                    optimizer.zero_grad()
                    batch_info["mean_total_loss"].backward()
                    # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                    optimizer.step()
                else:
                    if baseline is not None:
                        policy_optimizer.zero_grad()
                        baseline_optimizer.zero_grad()
                        batch_info["mean_policy_loss"].backward(retain_graph=True)
                        batch_info["mean_baseline_loss"].backward()
                        # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                        policy_optimizer.step()
                        baseline_optimizer.step()
                    else:
                        policy_optimizer.zero_grad()
                        batch_info["mean_policy_loss"].backward()
                        # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                        policy_optimizer.step()

                batch_step_id = (
                    epoch * math.ceil(len(train_set) / train_loader.batch_size)
                    + batch_idx
                )
                mlflow_log_batch_metrics(batch_info, batch_step_id, prefix="TRAIN")

            epoch_info = record_epoch_info(batch_infos)
            epoch_step_id = epoch + 1
            mlflow_log_epoch_metrics(epoch_info, epoch_step_id, prefix="TRAIN")

            agent.static_encoder.eval()
            agent.dynamic_encoder.eval()
            agent.decoder.eval()
            agent.baseline.eval() if baseline is not None else None

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    print(f"Validation Batch {batch_idx + 1}/{len(val_loader)}")
                    batch = batch.to(device)
                    episode_infos = []
                    skip_episode = False

                    for episode in range(batch.num_graphs):
                        static_data = batch.get_example(episode)
                        static_data.start_node = int(static_data.start_node.item())
                        static_data.end_node = int(static_data.end_node.item())
                        env.static_data = static_data
                        state = env.reset()
                        agent.reset()

                        for step in range(cfg.training.max_steps):
                            if len(env.states[-1].valid_actions) == 0:
                                print(
                                    f"Graph {episode} has no valid actions. Skipping episode."
                                )
                                skip_episode = True
                                break

                            next_state, terminated, truncated, step_info = execute_step(
                                env, agent, state, step_info
                            )
                            state = next_state

                            if terminated or truncated:
                                break
                        if skip_episode:
                            continue

                        total_loss, policy_loss, baseline_loss, entropy_loss = (
                            finish_episode(
                                rewards=step_info["rewards"],
                                action_log_probs=step_info["action_log_probs"],
                                entropies=step_info["entropies"],
                                baseline_weight=cfg.reinforce.baseline_loss_coeff,
                                baseline_values=step_info["baseline_values"],
                                gamma=gamma,
                                entropy_coeff=cfg.reinforce.entropy_coeff,
                            )
                        )

                        episode_info = record_episode_info(
                            step_info,
                            total_loss,
                            policy_loss,
                            baseline_loss,
                            entropy_loss,
                        )
                        episode_infos.append(episode_info)

                    batch_info = record_batch_info(episode_infos)
                    batch_infos.append(batch_info)

            epoch_info = record_epoch_info(batch_infos)
            mlflow_log_epoch_metrics(
                epoch_info,
                epoch_step_id,
                prefix="VAL",
            )

            # Clear cache, cpu memory, GPU memory and garbage collector
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return 0


def mlflow_log_epoch_metrics(
    epoch_info: dict,
    epoch_step_id: int,
    prefix: str = None,
    exclude_metrics: list = None,
):
    if exclude_metrics is not None:
        assert isinstance(exclude_metrics, list), "exclude_metrics should be a list"
        for metric in exclude_metrics:
            if metric in epoch_info:
                del epoch_info[metric]
            else:
                raise KeyError(f"Metric {metric} not found in epoch_info")
    if prefix is None:
        prefix = "EPOCH_"
    else:
        prefix = f"{prefix}_EPOCH_"

    epoch_info = {f"{prefix}{k}": v for k, v in epoch_info.items()}
    mlflow.log_metrics(
        epoch_info,
        step=epoch_step_id,
    )


def record_epoch_info(batch_infos: list):
    epoch_info = {}
    epoch_info["mean_policy_loss"] = torch.mean(
        torch.stack([batch_info["mean_policy_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_baseline_loss"] = torch.mean(
        torch.stack([batch_info["mean_baseline_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_entropy_loss"] = torch.mean(
        torch.stack([batch_info["mean_entropy_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_total_loss"] = torch.mean(
        torch.stack([batch_info["mean_total_loss"] for batch_info in batch_infos])
    )
    epoch_info["mean_reward"] = np.mean(
        [batch_info["mean_reward"] for batch_info in batch_infos]
    )
    epoch_info["success_rate"] = np.mean(
        [batch_info["success_rate"] for batch_info in batch_infos]
    )
    epoch_info["mean_travel_time"] = torch.mean(
        torch.stack([batch_info["mean_travel_time"] for batch_info in batch_infos])
    )
    print(f"Mean Epoch travel time: {epoch_info['mean_travel_time']}")
    return epoch_info


def execute_step(env, agent, state, step_info):
    action, log_prob, entropy = agent.select_action(state)
    baseline_value = agent.baseline(agent.final_embedding)
    next_state, reward, terminated, truncated, _ = env.step(action)

    step_info = record_step_info(
        step_info,
        env,
        state,
        action,
        log_prob,
        entropy,
        baseline_value,
        reward,
        terminated,
    )

    return next_state, terminated, truncated, step_info


def record_batch_info(episode_infos):
    batch_info = {}
    batch_info["mean_total_loss"] = torch.mean(
        torch.stack(
            [
                info["total_loss"]
                for info in episode_infos
                if info["total_loss"] is not None
            ]
        )
    )
    batch_info["mean_policy_loss"] = torch.mean(
        torch.stack(
            [
                info["policy_loss"]
                for info in episode_infos
                if info["policy_loss"] is not None
            ]
        )
    )
    batch_info["mean_baseline_loss"] = torch.mean(
        torch.stack(
            [
                info["baseline_loss"]
                for info in episode_infos
                if info["baseline_loss"] is not None
            ]
        )
    )
    batch_info["mean_entropy_loss"] = torch.mean(
        torch.stack(
            [
                info["entropy_loss"]
                for info in episode_infos
                if info["entropy_loss"] is not None
            ]
        )
    )
    batch_info["mean_reward"] = np.mean(
        [np.sum(info["rewards"]) for info in episode_infos]
    )
    batch_info["success_rate"] = np.mean(
        [1 if info["reached_goal"] else 0 for info in episode_infos]
    )
    batch_info["mean_travel_time"] = torch.mean(
        torch.stack(
            [
                torch.sum(torch.tensor(info["step_travel_time_route"]))
                for info in episode_infos
            ]
        )
    )
    return batch_info


def mlflow_log_batch_metrics(
    batch_info, batch_step_id, prefix=None, exclude_metrics: list = None
):
    if exclude_metrics is not None:
        assert isinstance(exclude_metrics, list), "exclude_metrics should be a list"
        for metric in exclude_metrics:
            if metric in batch_info:
                del batch_info[metric]
            else:
                raise KeyError(f"Metric {metric} not found in batch_info")

    if prefix is None:
        prefix = "BATCH_"
    else:
        prefix = f"{prefix}_BATCH_"

    batch_info = {
        "policy_loss": batch_info["mean_policy_loss"].clone().detach().cpu().item(),
        "baseline_loss": batch_info["mean_baseline_loss"].clone().detach().cpu().item(),
        "entropy_loss": batch_info["mean_entropy_loss"].clone().detach().cpu().item(),
        "total_loss": batch_info["mean_total_loss"].clone().detach().cpu().item(),
        "reward": batch_info["mean_reward"],
        "success_rate": batch_info["success_rate"],
    }
    batch_info = {f"{prefix}{k}": v for k, v in batch_info.items()}

    mlflow.log_metrics(
        batch_info,
        step=batch_step_id,
    )


def mlflow_log_episode_metrics(
    episode_info,
    epoch_id,
    batch_id,
    episode_id,
    graph_id,
    step_id,
    prefix=None,
    exclude_metrics: list = None,
):
    if exclude_metrics is not None:
        assert isinstance(exclude_metrics, list), "exclude_metrics should be a list"
        for metric in exclude_metrics:
            if metric in episode_info:
                del episode_info[metric]
            else:
                raise KeyError(f"Metric {metric} not found in episode_info")

    if prefix is None:
        prefix = "EPISODE_"
    else:
        prefix = f"{prefix}_EPISODE_"

    episode_metrics = {
        f"total_loss": episode_info["total_loss"].clone().detach().cpu().item(),
        f"policy_loss": episode_info["policy_loss"].clone().detach().cpu().item(),
        f"baseline_loss": episode_info["baseline_loss"].clone().detach().cpu().item(),
        f"entropy_loss": episode_info["entropy_loss"].clone().detach().cpu().item(),
        f"reached_goal": int(episode_info["reached_goal"]),
        f"num_steps": len(episode_info["rewards"]),
        f"reward": sum(episode_info["rewards"]),
    }
    episode_metrics = {f"{prefix}{k}": v for k, v in episode_metrics.items()}

    mlflow.log_metrics(
        episode_metrics,
        step=step_id,
    )

    table = pd.DataFrame(episode_info[f"penalty_contributions"])
    table["policy_loss"] = episode_metrics[f"{prefix}policy_loss"]
    table["baseline_loss"] = episode_metrics[f"{prefix}baseline_loss"]
    table["entropy_loss"] = episode_metrics[f"{prefix}entropy_loss"]
    table["total_loss"] = episode_metrics[f"{prefix}total_loss"]
    table["total_reward"] = episode_metrics[f"{prefix}reward"]
    table["reached_goal"] = episode_metrics[f"{prefix}reached_goal"]
    table["num_steps"] = episode_metrics[f"{prefix}num_steps"]
    table["route"] = episode_info[f"route"]
    table["success"] = 1 if episode_info[f"reached_goal"] else 0
    mlflow.log_table(
        data=table,
        artifact_file=f"{prefix}epoch_{epoch_id}/batch_{batch_id}/episode_{episode_id}_{graph_id}.json",
    )


def record_episode_info(
    step_info, total_loss, policy_loss, baseline_loss, entropy_loss
):
    episode_info = {}
    episode_info["total_loss"] = total_loss
    episode_info["policy_loss"] = policy_loss
    episode_info["baseline_loss"] = baseline_loss
    episode_info["entropy_loss"] = entropy_loss

    episode_info = episode_info | step_info
    return episode_info


def record_step_info(
    step_info, env, state, action, log_prob, entropy, baseline_value, reward, terminated
):
    if not step_info:
        step_info = {
            "rewards": [],
            "action_log_probs": [],
            "entropies": [],
            "baseline_values": [],
            "route": [],
            "states": [],
            "step_travel_time_route": [],
            "penalty_contributions": [],
        }
    step_info["rewards"].append(reward.clone().detach().cpu().item())
    step_info["action_log_probs"].append(log_prob)
    step_info["entropies"].append(entropy)
    step_info["baseline_values"].append(baseline_value)
    step_info["route"].append(action)
    step_info["states"].append(state)
    step_info["reached_goal"] = True if terminated else False
    step_info["step_travel_time_route"].append(env.step_travel_time_route[-1])
    step_info["penalty_contributions"].append(env.modifier_contributions)
    return step_info


if __name__ == "__main__":
    sweep_id = f"optuna_sweep_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    os.environ["SWEEP_ID"] = sweep_id

    # Always assume we're in single run; actual check is inside `main()`
    mlflow.set_experiment("dynamic_ambulance_training")
    with mlflow.start_run(run_name="optuna_sweep_parent") as parent_run:
        os.environ["MLFLOW_PARENT_RUN_ID"] = parent_run.info.run_id
        main()
