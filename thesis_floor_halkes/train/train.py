import random
from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.batch.graph_data_batch import GraphGenerator, RandomGraphPytorchDataset
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter, RandomDynamicFeatureGetter
from thesis_floor_halkes.penalties.calculator import RewardModifierCalculator
from thesis_floor_halkes.penalties.revisit_node_penalty import DeadEndPenalty, GoalBonus, PenaltyPerStep, RevisitNodePenalty, WaitTimePenalty
from thesis_floor_halkes.baselines.critic_network import CriticBaseline
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.agent.dynamic import DynamicAgent
from thesis_floor_halkes.model_dynamic_attention import DynamicGATEncoder, StaticGATEncoder, DynamicGATConvEncoder, AttentionDecoderChat
from thesis_floor_halkes.utils.reward_logger import RewardLogger
from thesis_floor_halkes.utils.simulate_dijkstra import simulate_dijkstra_path_cost


data = GraphGenerator(
    num_nodes = 15,
    edge_prob = 0.5,
    max_wait = 10.0,
).generate()

dataset = RandomGraphPytorchDataset(
    num_graphs = 2,
    min_nodes = 5,
    max_nodes = 5,
    min_prob = 0.2,
    max_prob = 0.8,
)

revisit_penalty = RevisitNodePenalty(name="Revisit Node Penalty", penalty = -1.0)
penalty_per_step = PenaltyPerStep(name="Penalty Per Step", penalty = -0.1)
goal_bonus = GoalBonus(name="Goal Bonus", bonus = 5.0)
dead_end_penalty = DeadEndPenalty(name="Dead End Penalty", penalty = -5.0)
waiting_time_penalty = WaitTimePenalty(name="Waiting Time Penalty")


reward_modifier_calculator = RewardModifierCalculator(
        modifiers = [revisit_penalty, penalty_per_step, goal_bonus, waiting_time_penalty],
        weights = [1.0, 1.0, 1.0, 1.0],
    )

env = DynamicEnvironment(
    static_dataset = dataset,
    dynamic_feature_getter = RandomDynamicFeatureGetter(),
    reward_modifier_calculator = reward_modifier_calculator,
    max_steps = 30,
)

hidden_size = 64
input_dim = hidden_size * 2
static_encoder = StaticGATEncoder(in_channels=1, hidden_size=hidden_size, edge_attr_dim=2)
dynamic_encoder = DynamicGATEncoder(in_channels=4, hidden_size=hidden_size)
decoder = AttentionDecoderChat(embed_dim=hidden_size * 2, num_heads=4)

baseline = CriticBaseline(input_dim=input_dim)

agent = DynamicAgent(
    static_encoder= static_encoder,
    dynamic_encoder= dynamic_encoder,
    decoder= decoder,
    baseline=baseline,
)
agent.routes.clear()

logger = RewardLogger(smooth_window = 20)

for graph in dataset:
    for episode in range(30):
        total_reward = 0 
        print('\n NEW GRAPH')
        state = env.reset()
        for step in range(4):
            action, action_log_prob = agent.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_state(new_state)
            agent.store_action_log_prob(action_log_prob)
            agent.store_action(action)
            agent.store_reward(reward)
            total_reward += reward
            state = new_state
            if terminated or truncated:
                break

        total_loss, policy_loss, value_loss = agent.finish_episode()
        agent.reset()
        agent.update(total_loss)
        num_nodes = state.static_data.x.size(0)
        logger.log(total_reward,num_nodes)
        
        if episode % 1 == 0:
            print(f"{episode= }")
            logger.summary()
        
logger.plot()


