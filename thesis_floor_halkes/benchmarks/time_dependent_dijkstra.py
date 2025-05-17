import pandas as pd
import networkx as nx
from time import time


edge_df = pd.read_parquet("data/processed/edge_features_helmond.parquet")
node_df = pd.read_parquet("data/processed/node_features.parquet")

# edge_df = edge_df.merge(node_df[['node_id', 'wait_time', 'has_light', 'timestamp']],
#                         left_on='v', right_on='node_id', how='left')

# edge_df['travel_time'] = edge_df['length'] / (edge_df['maxspeed']/3.6)

# def compute_total_cost(row):
#     wait = row['wait_time'] if row['has_light'] else 0
#     return row['travel_time'] + wait

# edge_df['time_cost'] = edge_df.apply(compute_total_cost, axis=1)

# sub_df = edge_df[edge_df['timestamp']=='2024-01-31 09:00:00']
# sub_df = sub_df[sub_df['has_light'] == 1][:10]
# # print(sub_df)


# G = nx.DiGraph()
# for _, row in sub_df.iterrows():
#     G.add_edge(row['u'], row['v'], weight=row['time_cost'])

# # print(G.edges)

# origin = 42663481
# destination = 6359798222

# start_time = time()
# path = nx.dijkstra_path(G, source=origin, target=destination, weight='weight')
# total_time = nx.dijkstra_path_length(G, source=origin, target=destination, weight='weight')
# end_time = time()
# print(f"Path: {path}")
# print(f"Total time: {total_time}")
# print(f"Time taken: {end_time - start_time} seconds")

edge_df = edge_df[:100]

import pandas as pd
import heapq


import pandas as pd

# Small edge dataframe (u -> v)
edge_df = pd.DataFrame(
    {
        "u": [0, 0, 1, 2],
        "v": [1, 2, 3, 3],
        "length": [100, 200, 150, 100],  # meters
        "speed": [
            50,
            50,
            30,
            60,
        ],  # meters per minute (converted to minutes for testing)
    }
)

# Small node dataframe with time-dependent wait times
node_df = pd.DataFrame(
    {
        "node_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "timestamp": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "wait_time": [
            10,
            30,
            60,
            10,
            10,
            20,
            30,
            5,
            20,
        ],  # in seconds (can treat as minutes here for simplicity)
        "traffic_light_status": [
            "green",
            "red",
            "red",
            "green",
            "red",
            "red",
            "green",
            "red",
            "green",
        ],
    }
)

# Convert wait_time to minutes for consistency (optional)
node_df["wait_time"] = node_df["wait_time"] / 60


def dynamic_time_dependent_dijkstra(node_df, edge_df, source, target, start_time):
    # Preprocess: build edge lookup and dynamic wait-time tables
    node_df_grouped = node_df.groupby(["node_id", "timestamp"])[
        ["wait_time", "traffic_light_status"]
    ].first()

    # Priority queue: (cumulative_time, current_node, path_taken)
    queue = [(start_time, source, [])]

    visited = dict()  # node -> earliest_arrival_time

    while queue:
        current_time, current_node, path = heapq.heappop(queue)

        # Stop if we reached the target
        if current_node == target:
            return path + [current_node], current_time - start_time

        # Skip if this node was visited at an earlier or same time
        if current_node in visited and current_time >= visited[current_node]:
            continue
        visited[current_node] = current_time

        # Look at all outgoing edges from this node
        outgoing_edges = edge_df[edge_df["u"] == current_node]
        for _, row in outgoing_edges.iterrows():
            next_node = row["v"]
            travel_time = row["length"] / (row["speed"] / 3.6)  # in minutes

            # Calculate arrival time at next node
            arrival_time = current_time + travel_time
            arrival_minute = int(arrival_time)

            # Get wait time and traffic light status at arrival time
            wait_key = (next_node, arrival_minute)
            if wait_key in node_df_grouped.index:
                print("aaaaaaaaaaaaaaaa")
                wait_info = node_df_grouped.loc[wait_key]
                wait_time = (
                    wait_info["wait_time"]
                    if wait_info["traffic_light_status"] == "red"
                    else 0
                )
            else:
                wait_time = 0  # assume no wait if data missing
            print(
                f"  Edge to {next_node}: travel={travel_time:.2f}, arrival={arrival_time:.2f}, wait={wait_time:.2f}, status={wait_info['traffic_light_status'] if wait_key in node_df_grouped.index else 'unknown'}"
            )

            total_time = travel_time + wait_time
            heapq.heappush(
                queue, (current_time + total_time, next_node, path + [current_node])
            )

    return None, float("inf")  # No path found


print(edge_df)

path, total_travel_time = dynamic_time_dependent_dijkstra(
    node_df=node_df,
    edge_df=edge_df,
    source=0,
    target=3,
    start_time=0,  # e.g., 8:00 AM = 480 minutes since midnight
)

print(f"Path: {path}")
print(f"Total Travel Time: {total_travel_time:.2f} minutes")
