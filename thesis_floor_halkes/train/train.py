import random
from thesis_floor_halkes.environment.dynamic_ambulance import DynamicEnvironment
from thesis_floor_halkes.batch.graph_data_batch import GraphGenerator, RandomGraphPytorchDataset
from thesis_floor_halkes.features.dynamic.getter import DynamicFeatureGetter, RandomDynamicFeatureGetter
from thesis_floor_halkes.penalties.calculator import PenaltyCalculator
from thesis_floor_halkes.utils.adj_matrix import build_adjecency_matrix
from thesis_floor_halkes.agent.dynamic import DynamicAgent
from thesis_floor_halkes.model_dynamic_attention import GATModelEncoderStatic, DynamicGATEncoder, AttentionDecoderChat, AttentionDecoder2


data = GraphGenerator(
    num_nodes = 15,
    edge_prob = 0.5,
    max_wait = 10.0,
).generate()

dataset = RandomGraphPytorchDataset(
    num_graphs = 4,
    min_nodes = 5,
    max_nodes = 5,
    min_prob = 1,
    max_prob = 1,
)

env = DynamicEnvironment(
    static_dataset = dataset,
    dynamic_feature_getter = RandomDynamicFeatureGetter(),
    penalty_calculator = PenaltyCalculator,
    max_steps = 30,
)

hidden_size = 64
static_encoder = GATModelEncoderStatic(in_channels=1, hidden_size=hidden_size, edge_attr_dim=2)
dynamic_encoder = DynamicGATEncoder(in_channels=2, hidden_size=hidden_size)
decoder = AttentionDecoderChat(embed_dim=hidden_size * 2, num_heads=4)
# decoder = AttentionDecoder2(embed_dim=hidden_size * 2)

agent = DynamicAgent(
    static_encoder= static_encoder,
    dynamic_encoder= dynamic_encoder,
    decoder= decoder,
)



for graph in dataset: 
    print('NEW GRAPH')
    state = env.reset()
    for step in range(30):
        action, action_log_prob = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)


