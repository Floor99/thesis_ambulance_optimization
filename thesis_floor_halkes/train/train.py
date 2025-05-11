import random

import pandas as pd
import torch
from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.batch.graph_data_batch import GraphGenerator, RandomGraphPytorchDataset
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter, DynamicFeatureGetterDataFrame, RandomDynamicFeatureGetter
from thesis_floor_halkes.features.static.getter import get_static_data_object
from thesis_floor_halkes.model.decoder import AttentionDecoder, FixedContext
from thesis_floor_halkes.model.encoders import StaticGATEncoder, DynamicGATEncoder
from thesis_floor_halkes.penalties.calculator import RewardModifierCalculator
from thesis_floor_halkes.penalties.revisit_node_penalty import DeadEndPenalty, GoalBonus, PenaltyPerStep, RevisitNodePenalty, WaitTimePenalty
from thesis_floor_halkes.baselines.critic_network import CriticBaseline
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.agent.dynamic import DynamicAgent
# from thesis_floor_halkes.model_dynamic_attention import DynamicGATEncoder, StaticGATEncoder, DynamicGATConvEncoder, AttentionDecoderChat
from thesis_floor_halkes.utils.reward_logger import RewardLogger
from thesis_floor_halkes.utils.simulate_dijkstra import simulate_dijkstra_path_cost
from thesis_floor_halkes.utils.plot_graph import plot_graph

import osmnx as ox 
ox.settings.bidirectional_network_types = ['drive', 'walk', 'bike']
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

# dataset = StaticDataSet(2)
dataset = [get_static_data_object(time_series_df_path="data/processed/node_features.parquet",
    edge_df_path="data/processed/edge_features_helmond.parquet",
    dist=20,
    seed=5)]

revisit_penalty = RevisitNodePenalty(name="Revisit Node Penalty", penalty = -50.0)
penalty_per_step = PenaltyPerStep(name="Penalty Per Step", penalty = -10)
goal_bonus = GoalBonus(name="Goal Bonus", bonus = 50.0)
dead_end_penalty = DeadEndPenalty(name="Dead End Penalty", penalty = -500.0)
waiting_time_penalty = WaitTimePenalty(name="Waiting Time Penalty")


reward_modifier_calculator = RewardModifierCalculator(
        modifiers = [revisit_penalty, penalty_per_step, goal_bonus, waiting_time_penalty],
        weights = [3.0, 1.0, 2.0, 1],
    )

# final_node_df = pd.read_parquet("data/processed/node_features.parquet")
# final_edge_df = pd.read_parquet("data/processed/edge_features_helmond.parquet")
# node_df = get_static_data_object(final_node_df, final_edge_df)

# env = DynamicEnvironment(
#     static_dataset = dataset,
#     dynamic_feature_getter = RandomDynamicFeatureGetter(),
#     reward_modifier_calculator = reward_modifier_calculator,
#     max_steps = 30,
# )

env = DynamicEnvironment(
    static_dataset = dataset,
    dynamic_feature_getter = DynamicFeatureGetterDataFrame(),
    reward_modifier_calculator = reward_modifier_calculator,
    max_steps = 30,
    start_timestamp = '2024-01-31 08:30:00',
)

hidden_size = 64
input_dim = hidden_size * 2
static_encoder = StaticGATEncoder(in_channels=3, hidden_size=hidden_size, edge_attr_dim=2, num_layers=4)
dynamic_encoder = DynamicGATEncoder(in_channels=4, hidden_size=hidden_size, num_layers=4)
decoder = AttentionDecoder(embed_dim=hidden_size * 2, num_heads=4)
fixed_context = FixedContext(embed_dim=hidden_size * 2)
# baseline_input_dim = input_dim * 2
# baseline = CriticBaseline(input_dim=baseline_input_dim)
baseline = CriticBaseline()

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

for graph in dataset:
    for episode in range(1):
        total_reward = 0 
        print('\n\n NEW GRAPH')
        state = env.reset()
        for step in range(20):
            print(f"\n{step= }")
            action, action_log_prob, entropy = agent.select_action(state)
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
                print('terminated')
                break

        policy_loss, baseline_loss = agent.finish_episode()
        policy_optimizer.zero_grad()
        baseline_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_optimizer.step()

        baseline_loss.backward()
        baseline_optimizer.step()
        agent.reset()
        # agent.backprop_model(policy_optimizer, policy_loss)
        # agent.backprop_model(baseline_optimizer, baseline_loss)
        num_nodes = state.static_data.x.size(0)
        logger.log(total_reward,num_nodes)
        
        if episode % 1 == 0:
            print(f"{episode= }")
            logger.summary()
        
logger.plot()
plot_graph(state.dynamic_data)

