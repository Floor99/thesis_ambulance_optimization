import networkx as nx

from thesis_floor_halkes.environment.base import Environment


def simulate_dijkstra_path_cost(env: Environment):
    """
    Simulate following the Dijkstra (shortest static path) in the environment.
    Returns the true dynamic cost (i.e., reward) of this path.

    NOTE: This assumes you call env.reset() just before.
    """
    # 1. Convert to NetworkX graph using static edge attributes
    edge_index = env.static_data.edge_index
    edge_weights = env.static_data.edge_attr[:, 0]  # static length (feature 0)

    G = nx.DiGraph()
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        w = float(edge_weights[i])
        G.add_edge(u, v, weight=w)

    try:
        dijkstra_path = nx.shortest_path(
            G,
            source=env.static_data.start_node,
            target=env.static_data.end_node,
            weight="weight",
        )
    except nx.NetworkXNoPath:
        print("No path found by Dijkstra.")
        return float("inf")

    # 2. Step through the environment using this path
    total_cost = 0.0
    env.reset()  # Reset to fresh dynamic state
    current_state = env.states[-1]

    for next_node in dijkstra_path[1:]:  # skip current_node (start)
        if env.terminated or env.truncated:
            break
        current_state, reward, terminated, truncated, _ = env.step(next_node)
        total_cost += -reward  # reward is negative cost

    return total_cost
