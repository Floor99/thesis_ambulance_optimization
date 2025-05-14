import random
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import osmnx as ox 
import os

from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetterDataFrame
from thesis_floor_halkes.features.static.getter import get_static_data_object
from thesis_floor_halkes.features.static.static_dataset import StaticListDataset
from thesis_floor_halkes.model.decoder import AttentionDecoder, FixedContext
from thesis_floor_halkes.model.encoders import StaticGATEncoder, DynamicGATEncoder
from thesis_floor_halkes.penalties.calculator import RewardModifierCalculator
from thesis_floor_halkes.penalties.revisit_node_penalty import (AggregatedStepPenalty, CloserToGoalBonus, 
                                                                DeadEndPenalty, GoalBonus, HigherSpeedBonus, 
                                                                PenaltyPerStep, RevisitNodePenalty, 
                                                                WaitTimePenalty)
from thesis_floor_halkes.baselines.critic_network import CriticBaseline
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.agent.dynamic import DynamicAgent
from thesis_floor_halkes.utils.reward_logger import RewardLogger
from thesis_floor_halkes.utils.simulate_dijkstra import simulate_dijkstra_path_cost
from thesis_floor_halkes.utils.plot_graph import plot_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("checkpoints", exist_ok=True)
ox.settings.bidirectional_network_types = ['drive', 'walk', 'bike']
mlflow.set_experiment("dynamic_ambulance_training")


# data = GraphGenerator(
#     num_nodes = 15,
#     edge_prob = 0.5,
#     max_wait = 10.0,
# ).generate()

# dataset = RandomGraphPytorchDataset(
#     num_graphs = 1,
#     min_nodes = 4059,
#     max_nodes = 4059,
#     min_prob = 0.3,
#     max_prob = 0.7,
# )

dataset =     dataset = StaticListDataset(
    ts_path="data/processed/node_features.parquet",
    seeds=[0,1,2,3,4],
    dists=[200, 300, 400, 500, 600],   # each graph has its own radius
    )

# dataset = [get_static_data_object(time_series_df_path="data/processed/node_features.parquet",
    # dist=1000,
    # seed=5)]

revisit_penalty = RevisitNodePenalty(name="Revisit Node Penalty", penalty = -50.0)
penalty_per_step = PenaltyPerStep(name="Penalty Per Step", penalty = -10)
goal_bonus = GoalBonus(name="Goal Bonus", bonus = 50.0)
dead_end_penalty = DeadEndPenalty(name="Dead End Penalty", penalty = -50.0)
waiting_time_penalty = WaitTimePenalty(name="Waiting Time Penalty")
higher_speed_bonus = HigherSpeedBonus(name="Higher Speed Bonus", bonus = 10.0)
aggregated_step_penalty = AggregatedStepPenalty(name="Aggregated Step Penalty", penalty = -1.0)
closer_to_goal_bonus = CloserToGoalBonus(name="Closer To Goal Bonus", bonus = 1.0)

reward_modifier_calculator = RewardModifierCalculator(
        modifiers = [revisit_penalty, 
                     penalty_per_step, 
                     goal_bonus, 
                     waiting_time_penalty, 
                     dead_end_penalty, 
                     higher_speed_bonus,
                     closer_to_goal_bonus,
                     ],
        weights = [3.0, 
                   1.0, 
                   2.0, 
                   1.0, 
                   1.0, 
                   1.0,
                   1.0,
                   ],
    )


env = DynamicEnvironment(
    static_dataset = dataset,
    dynamic_feature_getter = DynamicFeatureGetterDataFrame(),
    reward_modifier_calculator = reward_modifier_calculator,
    max_steps = 5,
    # start_timestamp = '2024-01-31 08:30:00',
)

hidden_size = 64
input_dim = hidden_size * 2
static_encoder = StaticGATEncoder(in_channels=4, hidden_size=hidden_size, edge_attr_dim=2, num_layers=4).to(device)
dynamic_encoder = DynamicGATEncoder(in_channels=4, hidden_size=hidden_size, num_layers=4).to(device)
decoder = AttentionDecoder(embed_dim=hidden_size * 2, num_heads=4).to(device)
fixed_context = FixedContext(embed_dim=hidden_size * 2).to(device)
baseline = CriticBaseline().to(device)

agent = DynamicAgent(
    static_encoder= static_encoder,
    dynamic_encoder= dynamic_encoder,
    decoder= decoder,
    fixed_context=fixed_context,
    baseline=baseline,
)
agent.routes.clear()

learning_rate = 0.001
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

logger = RewardLogger(smooth_window = 20)
torch.autograd.set_detect_anomaly(True)

num_epochs = 1

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "max_steps": env.max_steps,
        "num_epochs": num_epochs,
        "decoder_type": "AttentionDecoder",
        "encoder_type": "GATEncoder",
    })

    for epoch in range(num_epochs):
        print(f"\n === Epoch {epoch} ===")
        
        for idx, static_data in enumerate(dataset.data_list):
            print(f"\n\n === Graph {idx} ===")
            static_data = static_data.to(device)
            env.static_data = [static_data]
            
            total_reward = 0 
            
            state = env.reset()
            state.static_data = static_data.to(device)
            state.dynamic_data = state.dynamic_data.to(device)

            for step in range(env.max_steps):
                print(f"\n{step= }")
                action, action_log_prob, entropy = agent.select_action(state)
                
                embedding = agent.embeddings[-1]["final"]
                embedding_for_critic = embedding.detach().clone().requires_grad_()
                baseline_value = agent.baseline(embedding_for_critic, hidden_dim=128)
                
                new_state, reward, terminated, truncated, _ = env.step(action)
                new_state.static_data = new_state.static_data.to(device)
                new_state.dynamic_data = new_state.dynamic_data.to(device)
                
                agent.store_state(new_state)
                agent.store_action_log_prob(action_log_prob)
                agent.store_action(action)
                agent.store_reward(reward)
                agent.store_baseline_value(baseline_value)
                agent.store_entropy(entropy)
                
                total_reward += reward
                state = new_state
                
                if terminated or truncated:
                    print('terminated')
                    break
            print(f"{agent.current_route= }")
            policy_loss, baseline_loss = agent.finish_episode()
            policy_optimizer.zero_grad()
            baseline_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            policy_optimizer.step()

            baseline_loss.backward()
            baseline_optimizer.step()
            agent.reset()
            
            num_nodes = state.static_data.x.size(0)
            logger.log(total_reward,num_nodes)
            print(f"\n  -- Graph {idx} reward: {total_reward:.2f}")
            mlflow.log_metric("reward", total_reward, step=epoch * len(dataset.data_list) + idx)

            # logger.summary()
            
    logger.plot()
    torch.save({
        'static_encoder': agent.static_encoder.state_dict(),
        'dynamic_encoder': agent.dynamic_encoder.state_dict(),
        'decoder': agent.decoder.state_dict(),
        'baseline': agent.baseline.state_dict(),
    }, f"checkpoints/agent_epoch_{epoch}.pt")
    mlflow.log_artifact(f"checkpoints/agent_epoch_{epoch}.pt")

    plot_graph(state.dynamic_data)

