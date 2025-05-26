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
from thesis_floor_halkes.utils.reward_logger import RewardLogger
from thesis_floor_halkes.benchmarks.simulate_dijkstra import simulate_dijkstra_path_cost
from thesis_floor_halkes.utils.plot_graph import plot_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")


torch.autograd.set_detect_anomaly(True)


# ==== Training Loop with MLFlow Tracking ====
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs("checkpoints", exist_ok=True)
    ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]
    sweep_id = os.environ.get("SWEEP_ID", "no_sweep_id")
    mlflow.set_experiment("dynamic_ambulance_training")

    # ==== Load the static dataset and dataloader ====
    # dataset = collect_static_data_objects(
    #     base_dir = "data/training_data",
    # )
    
    base_dir = "data/training_data"
    train_dirs, val_dirs, test_dirs = split_subgraphs(base_dir, train_frac=0.7, val_frac=0.15, seed=42)
    
    
    # dataset = StaticDataObjectSet(base_dir="data/training_data")
    # n = len(dataset)
    # train_size = int(0.7 * n)
    # val_size = int(0.15 * n)
    # test_size = n - train_size - val_size
    
    # train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    
    train_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=train_dirs, num_pairs_per_graph = 2, seed = 42)
    # train_set.data_objects = [train_set.data_objects[0]]
    # val_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=val_dirs, num_pairs_per_graph = 5, seed = 42)
    # test_set = StaticDataObjectSet(base_dir=base_dir, subgraph_dirs=test_dirs, num_pairs_per_graph = 5, seed = 42)
    
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=False)
    # val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
    # test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    
    # dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

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
    # higher_speed_bonus = HigherSpeedBonus(
    #     name="Higher Speed Bonus", bonus=cfg.reward_mod.higher_speed_bonus_value
    # )
    # closer_to_goal_bonus = CloserToGoalBonus(
    #     name="Closer To Goal Bonus", bonus=cfg.reward_mod.closer_to_goal_bonus_value
    # )
    # no_signal_intersection_penalty = NoSignalIntersectionPenalty(
    #     name="No Signal Intersection Penalty", penalty=cfg.reward_mod.no_signal_intersection_penalty_value
    # )

    reward_modifier_calculator = RewardModifierCalculator(
        modifiers=[
            penalty_per_step,
            goal_bonus,
            waiting_time_penalty,
            dead_end_penalty,
            # higher_speed_bonus,
            # closer_to_goal_bonus,
            # no_signal_intersection_penalty,
        ],
        weights=[
            1.0,
            1.0,
            1.0,
            1.0,
            # 1.0,
            # 1.0,
            # 1.0,
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
        mlflow.set_tag("sweep_id", sweep_id)
        log_hydra_config_to_mlflow(cfg)

        num_epochs = cfg.training.num_epochs
        num_batches_per_epoch = math.ceil(len(train_set) / train_loader.batch_size)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            batch_metrics_objects = []
            batch_policy_loss = []
            batch_baseline_loss = []
            for batch_idx, batch in enumerate(train_loader):
                
                
                episode_metrics_objects = []
                episode_artifacts_objects = []
                for graph_idx in range(batch.num_graphs):
                    print(f"Graph {graph_idx + 1}/{batch.num_graphs} of Batch {batch_idx + 1}/{len(train_loader)}")
                    episode_step_id = epoch * len(train_set) + batch_idx * train_loader.batch_size + graph_idx
                    
                    
                    static_data = batch.get_example(graph_idx)
                    env.static_data = static_data
                    
                    state = env.reset()
                    skip_episode = False

                    for step in range(env.max_steps):
                        # print(env.states[-1].valid_actions)
                        if len(env.states[-1].valid_actions) == 0:
                            print(f"Graph {graph_idx} has no valid actions. Skipping episode.")
                            skip_episode = True
                            break               
                        action, action_log_prob, entropy = agent.select_action(state)
                        new_state, reward, terminated, truncated, _ = env.step(action)

                        agent.store_state(new_state)
                        agent.store_action_log_prob(action_log_prob)
                        agent.store_action(action)
                        agent.store_reward(reward)
                        agent.store_entropy(entropy)
                        if baseline is not None:
                            baseline_value = get_baseline_value(agent)
                            agent.store_baseline_value(baseline_value)

                        state = new_state
                        
                        if terminated or truncated:
                            break
                            
                    if skip_episode:
                        continue
                    policy_loss, baseline_loss, entropy_loss = agent.finish_episode()
                    total_loss = policy_loss + (baseline_weight * baseline_loss)
                    
                    # Log the episode metrics
                    episode_metrics, episode_artifacts = create_and_log_episode_data_to_mlflow(
                        step_id=episode_step_id,
                        episode_id=graph_idx,
                        batch_id=batch_idx,
                        epoch_id=epoch,
                        policy_loss=policy_loss,
                        baseline_loss=baseline_loss,
                        entropy_loss=entropy_loss,
                        total_loss=total_loss,
                        env=env,
                        agent=agent,
                        static_data=static_data,
                    )
                    episode_metrics_objects.append(episode_metrics)
                    episode_artifacts_objects.append(episode_artifacts)
                    
                    
                    
                    # if use_joint_optimization:
                    #     batch_policy_loss.append(total_loss)
                    # else:
                    #     batch_policy_loss.append(policy_loss)
                    #     batch_baseline_loss.append(baseline_loss)

                    ### Reset agent and environment for next episode
                    agent.reset()
                    env.reward_modifier_calculator.reset()
                
                batch_step_id = epoch * num_batches_per_epoch + batch_idx
                batch_metrics = create_and_log_batch_data_to_mlflow(
                    epidode_metrics_objects=episode_metrics_objects,
                    batch_id=batch_step_id,
                )
                batch_metrics_objects.append(batch_metrics)
                
                # def print_param_norms(module, label):
                #     print(f"{label} param norms:")
                #     for name, param in module.named_parameters():
                #         print(f"  {name}: {param.data.norm().item()}")
                #     print_param_norms(agent.static_encoder, "Static Encoder")
                #     print_param_norms(agent.dynamic_encoder, "Dynamic Encoder")
                #     print_param_norms(agent.decoder, "Decoder")
                #     print_param_norms(agent.baseline, "Baseline")
                
                # prev_params = {name: param.clone() for name, param in agent.static_encoder.named_parameters()}
                if use_joint_optimization and baseline is not None:
                    batch_total_loss = batch_metrics.batch_total_loss
                    # print("Batch total loss:", batch_total_loss)
                    optimizer.zero_grad()
                    batch_total_loss.backward()
                    # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                    
                    # # 5) Check gradient norms
                    # print("Gradient norms BEFORE optimizer.step():")
                    # for name, param in agent.static_encoder.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"  {name:20s} | grad norm = {param.grad.norm():.4f}")
                    # print("batch_policy_loss grad_fn:", batch_total_loss.grad_fn)
                    # print("batch_baseline_loss grad_fn:", batch_baseline_loss.grad_fn)
                    optimizer.step()
                    # # 7) Measure parameter changes
                    # print("\nParameter change magnitudes AFTER optimizer.step():")
                    # for name, param in agent.static_encoder.named_parameters():
                    #     delta = (param.data - prev_params[name]).norm()
                    #     print(f"  {name:20s} | Δ param = {delta:.6f}")
                else:
                    batch_policy_loss = batch_metrics.batch_policy_loss
                    batch_baseline_loss = batch_metrics.batch_baseline_loss
                    if baseline is not None:
                        policy_optimizer.zero_grad()
                        baseline_optimizer.zero_grad()
                        batch_policy_loss.backward(retain_graph=True)
                        batch_baseline_loss.backward()
                        # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                        policy_optimizer.step()
                        baseline_optimizer.step()
                    else:
                        policy_optimizer.zero_grad()
                        batch_policy_loss.backward()
                        # clip_grad_norm_(agent.parameters(), max_norm=cfg.training.max_grad_norm)
                        policy_optimizer.step()
                # agent.reset()
                # env.reward_modifier_calculator.reset()
                
                # print_param_norms(agent.static_encoder, "Static Encoder")
                # print_param_norms(agent.dynamic_encoder, "Dynamic Encoder")
                # print_param_norms(agent.decoder, "Decoder")
                # print_param_norms(agent.baseline, "Baseline")
                        
                        
            # Log the agent state dict
            log_agent_checkpoint(agent, epoch)
            
            # Log epoch metrics
            epoch_metrics = create_and_log_epoch_data_to_mlflow(
                batch_metrics_objects=batch_metrics_objects,
                epoch_id=epoch,
            )

        # Log full models
        log_full_models(agent)
        # episode_travel_times = [episode.travel_time for episode in episode_metrics_objects]
        # return np.mean([travel_time.cpu() for travel_time in episode_travel_times])
        episode_rewards = [sum(episode.rewards) for episode in episode_metrics_objects]
        return np.mean([reward.cpu() for reward in episode_rewards])

@dataclass
class EpisodeMetrics:
    step_id: int
    episode_id: int
    batch_id: int
    epoch_id: int
    travel_time: float
    policy_loss: float
    baseline_loss: float
    entropy_loss: float
    total_loss: float
    rewards: list[float]
    entropies: list[float]
    reached_goal: int
    baseline_values: list[torch.tensor] | None = None
    
    def __post_init__(self):
        self.mean_baseline_value = torch.mean(torch.stack(self.baseline_values))

@dataclass
class EpisodeArtifacts:
    step_id: int
    episode_id: int
    batch_id: int
    epoch_id: int
    route: list[int]
    reward_modifiers: list[dict]
    step_log: list[dict]
    graph_plot: plt.Figure
    travel_times: list[float]
    static_data: Data
    
    def __post_init__(self):
        fig, _ = plot_with_route(
            self.static_data.G_sub,
            self.route,
            goal_node = self.static_data.end_node,
        )
        self.graph_plot = fig


@dataclass
class BatchMetrics:
    batch_id: int
    batch_travel_time: float
    batch_reward: float
    batch_policy_loss: float
    batch_baseline_loss: float
    batch_entropy_loss: float
    batch_total_loss: float
    batch_baseline_values: float
    batch_success_rate: float

@dataclass
class EpochMetrics:
    epoch_id: int
    epoch_travel_time: float
    epoch_reward: float
    epoch_policy_loss: float
    epoch_baseline_loss: float
    epoch_entropy_loss: float
    epoch_total_loss: float
    epoch_baseline_values: float
    epoch_success_rate: float


def create_and_log_episode_data_to_mlflow(
    step_id,
    episode_id,
    batch_id,
    epoch_id,
    policy_loss,
    baseline_loss,
    entropy_loss,
    total_loss,
    env,
    agent,
    static_data,
):
    reached_goals = has_agent_reached_goal(env, agent)
    episode_metrics = EpisodeMetrics(
                        step_id=step_id,
                        episode_id=episode_id,
                        batch_id=batch_id,
                        epoch_id=epoch_id,
                        travel_time=sum(env.step_travel_time_route),
                        policy_loss=policy_loss.clone(),
                        baseline_loss=baseline_loss.clone(),
                        entropy_loss=entropy_loss.clone(),
                        total_loss=total_loss,
                        rewards=[reward.clone() for reward in agent.rewards],
                        entropies=[entropy.clone() for entropy in agent.entropies],
                        reached_goal=reached_goals,
                        baseline_values=[baseline_value.clone() for baseline_value in agent.baseline_values],
                    )
    fig, _ = plot_with_route(
        static_data.G_sub,
        agent.current_route,
        goal_node=static_data.end_node,
    )
    episode_artifacts = EpisodeArtifacts(
                        step_id=step_id,
                        episode_id=episode_id,
                        batch_id=batch_id,
                        epoch_id=epoch_id,
                        route=agent.current_route,
                        reward_modifiers=env.reward_modifier_calculator.modifier_contributions,
                        step_log=create_step_log(agent),
                        travel_times=env.step_travel_time_route,
                        static_data=static_data,
                        graph_plot=fig,
                    )
    log_episode_to_mlflow_(episode_metrics, episode_artifacts)
    return episode_metrics, episode_artifacts


def log_episode_to_mlflow_(
    episode_metrics: EpisodeMetrics,
    episode_artifacts: EpisodeArtifacts
):
    mlflow.log_metrics(
        {
            "episode_travel_time": episode_metrics.travel_time,
            "episode_reward": sum(episode_metrics.rewards),
            "episode_policy_loss": episode_metrics.policy_loss,
            "episode_baseline_loss": episode_metrics.baseline_loss,
            "episode_entropy_loss": episode_metrics.entropy_loss,
            "episode_total_loss": episode_metrics.total_loss,
            "episode_entropy": sum(episode_metrics.entropies),
            "episode_baseline_value": sum(episode_metrics.baseline_values),
            "episode_reached_goal": episode_metrics.reached_goal,
        },
        step=episode_metrics.step_id,
    )
    # table = {
    #     "route": episode_artifacts.route,
    #     "rewards": [reward.clone().cpu().item() for reward in episode_metrics.rewards],
    #     "reward_modifiers": episode_artifacts.reward_modifiers,
    #     "entropies": [entropy.clone().cpu().item() for entropy in episode_metrics.entropies],
    #     "baseline_values": [baseline_value.clone().cpu().item() for baseline_value in episode_metrics.baseline_values],
    #     "travel_times": [travel_time.clone().cpu().item() for travel_time in episode_artifacts.travel_times],
    # }
    # mlflow.log_table(
    #     table,
    #     artifact_file=f"episode_{episode_metrics.episode_id}_metrics.json",
    # )
    
    epoch_id = episode_artifacts.epoch_id
    batch_id = episode_artifacts.batch_id
    episode_id = episode_artifacts.episode_id
    graph_folder = f"epoch_{epoch_id:03}/batch_{batch_id:03}/graph_{episode_id:02}"
    mlflow.log_figure(episode_artifacts.graph_plot, f"{graph_folder}/graph.png")        
    

def create_and_log_batch_data_to_mlflow(
    epidode_metrics_objects: list[EpisodeMetrics],
    batch_id: int,
):
    batch_travel_time = torch.mean(torch.stack([metrics.travel_time for metrics in epidode_metrics_objects]))
    batch_reward = torch.mean(torch.stack([torch.sum(torch.tensor(metrics.rewards)) for metrics in epidode_metrics_objects]))
    batch_policy_loss = torch.mean(torch.stack([metrics.policy_loss for metrics in epidode_metrics_objects]))   
    batch_baseline_loss = torch.mean(torch.stack([metrics.baseline_loss for metrics in epidode_metrics_objects]))
    batch_entropy_loss = torch.mean(torch.stack([metrics.entropy_loss for metrics in epidode_metrics_objects]))
    batch_total_loss = torch.mean(torch.stack([metrics.total_loss for metrics in epidode_metrics_objects]))
    
    # batch_baseline_values = torch.mean(torch.stack([metrics.mean_baseline_value for metrics in epidode_metrics_objects]))
    batch_baseline_values = torch.mean(torch.stack([torch.sum(torch.tensor(metrics.baseline_values)) for metrics in epidode_metrics_objects]))
    
    batch_success_rate = np.mean([metrics.reached_goal for metrics in epidode_metrics_objects])
    batch_metrics = BatchMetrics(
        batch_id=batch_id,
        batch_travel_time=batch_travel_time,
        batch_reward=batch_reward,
        batch_policy_loss=batch_policy_loss,
        batch_baseline_loss=batch_baseline_loss,
        batch_entropy_loss=batch_entropy_loss,
        batch_total_loss=batch_total_loss,
        batch_baseline_values=batch_baseline_values,
        batch_success_rate=batch_success_rate,
    )
    log_batch_data_to_mlflow(batch_metrics)
    return batch_metrics

def log_batch_data_to_mlflow(
    batch_metrics: BatchMetrics,
):
    mlflow.log_metrics(
        {
            "batch_travel_time": batch_metrics.batch_travel_time,
            "batch_reward": batch_metrics.batch_reward,
            "batch_policy_loss": batch_metrics.batch_policy_loss,
            "batch_baseline_loss": batch_metrics.batch_baseline_loss,
            "batch_entropy_loss": batch_metrics.batch_entropy_loss,
            "batch_total_loss": batch_metrics.batch_total_loss,
            "batch_baseline_values": batch_metrics.batch_baseline_values,
            "batch_success_rate": batch_metrics.batch_success_rate,
        },
        step=batch_metrics.batch_id,
    )

def create_and_log_epoch_data_to_mlflow(
    batch_metrics_objects: list[BatchMetrics],
    epoch_id: int,
):
    epoch_travel_time = torch.mean(torch.stack([metrics.batch_travel_time for metrics in batch_metrics_objects]))
    epoch_reward = torch.mean(torch.stack([metrics.batch_reward for metrics in batch_metrics_objects]))
    epoch_policy_loss = torch.mean(torch.stack([metrics.batch_policy_loss for metrics in batch_metrics_objects]))
    epoch_baseline_loss = torch.mean(torch.stack([metrics.batch_baseline_loss for metrics in batch_metrics_objects]))
    epoch_entropy_loss = torch.mean(torch.stack([metrics.batch_entropy_loss for metrics in batch_metrics_objects]))
    epoch_total_loss = torch.mean(torch.stack([metrics.batch_total_loss for metrics in batch_metrics_objects]))
    epoch_baseline_values = torch.mean(torch.stack([metrics.batch_baseline_values for metrics in batch_metrics_objects]))
    epoch_success_rate = np.mean([metrics.batch_success_rate for metrics in batch_metrics_objects])
    
    
    epoch_metrics = EpochMetrics(
        epoch_id=epoch_id,
        epoch_travel_time=epoch_travel_time,
        epoch_reward=epoch_reward,
        epoch_policy_loss=epoch_policy_loss,
        epoch_baseline_loss=epoch_baseline_loss,
        epoch_entropy_loss=epoch_entropy_loss,
        epoch_total_loss=epoch_total_loss,
        epoch_baseline_values=epoch_baseline_values,
        epoch_success_rate=epoch_success_rate,
    )
    log_epoch_data_to_mlflow(epoch_metrics)
    return epoch_metrics

def log_epoch_data_to_mlflow(
    epoch_metrics: EpochMetrics,
):
    mlflow.log_metrics(
        {
            "epoch_travel_time": epoch_metrics.epoch_travel_time,
            "epoch_reward": epoch_metrics.epoch_reward,
            "epoch_policy_loss": epoch_metrics.epoch_policy_loss,
            "epoch_baseline_loss": epoch_metrics.epoch_baseline_loss,
            "epoch_entropy_loss": epoch_metrics.epoch_entropy_loss,
            "epoch_total_loss": epoch_metrics.epoch_total_loss,
            "epoch_baseline_values": epoch_metrics.epoch_baseline_values,
            "epoch_success_rate": epoch_metrics.epoch_success_rate,
        },
        step=epoch_metrics.epoch_id,
    )



def has_agent_reached_goal(env, agent):
    final_node = agent.current_route[-1]
    goal_node = env.static_data.end_node
    reached_goal = int(final_node == goal_node)
    return reached_goal

def get_baseline_value(agent):
    embedding = agent.embeddings[-1]["final"]
    embedding_for_critic = embedding#.detach()
    baseline_value = agent.baseline(embedding_for_critic)
    return baseline_value

def log_hydra_config_to_mlflow(cfg):
    params = OmegaConf.to_container(cfg, resolve=True)
    flat_params = flatten_dict(params)
    mlflow.log_params(flat_params)

def create_step_log(agent):
    step_log = []
    for step in range(len(agent.states)):
        step_info = {
            "Step": step,
            "Action": agent.actions[step],
            "Static Features": agent.states[step].static_data.x[agent.actions[step]],#.cpu().numpy().tolist(),
            "Dynamic Features": agent.states[step].dynamic_data.x[agent.actions[step]],#.cpu().numpy().tolist(),
        }
        step_log.append(step_info)
    return step_log

if __name__ == "__main__":
    sweep_id = f"optuna_sweep_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    os.environ["SWEEP_ID"] = sweep_id
    main()
    if os.environ.get("HYDRA_JOB_NAME") == 'multirun':
        log_latest_best_params_to_mlflow()
