import warnings
from matplotlib import pyplot as plt
import osmnx as ox
import networkx as nx
from shapely import LineString
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd

ox.settings.bidirectional_network_types = ["drive", "walk", "bike"]

timeseries_df = pd.read_parquet("data/processed/node_features.parquet")

# def create_osmnx_sub_graph_only_inside_helmond(lat, lon, dist, timeseries_df):
#     unique_timeseries = (timeseries_df.loc[:, ['node_id', 'lat', 'lon']]
#                            .drop_duplicates(subset=['node_id'])
#                            .set_index('node_id'))

#     G_pt = ox.graph_from_point(
#         (lat, lon),
#         dist=dist,
#         network_type='drive',
#         simplify=True,
#     )

#     common_nodes = set(G_pt.nodes()).intersection(unique_timeseries.index)

#     G_sub = G_pt.subgraph(common_nodes).copy()

#     largest_connected_component = max(nx.weakly_connected_components(G_sub), key=len)
#     G_sub = G_sub.subgraph(largest_connected_component).copy()

#     return G_sub, G_pt


def create_osmnx_sub_graph_only_inside_helmond(lat, lon, dist, timeseries_df):
    unique_timeseries = (
        timeseries_df.loc[:, ["node_id", "lat", "lon"]]
        .drop_duplicates(subset=["node_id"])
        .set_index("node_id")
    )
    print(unique_timeseries)
    G_pt = ox.graph_from_point(
        (lat, lon),
        dist=dist,
        network_type="drive",
        simplify=True,
    )

    common_nodes = set(G_pt.nodes()).intersection(unique_timeseries.index)
    print(common_nodes)
    G_sub = G_pt.subgraph(common_nodes).copy()
    G_sub = ox.truncate.largest_component(G_sub, strongly=False)

    return G_sub, G_pt


def plot_sub_graph_in_and_out_nodes_helmond(G_sub, G_pt, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # split nodes into inside vs. outside
    all_nodes = set(G_pt.nodes())
    inside_nodes = set(G_sub.nodes())
    outside_nodes = all_nodes - inside_nodes

    # build a position dict (lon,lat) for plotting
    pos = {nid: (data["x"], data["y"]) for nid, data in G_pt.nodes(data=True)}

    # fig, ax = plt.subplots(figsize=(8,8))

    # draw all edges faintly
    ox.plot_graph(
        G_pt,
        ax=ax,
        show=False,
        close=False,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.5,
        bgcolor="white",
    )

    # overlay nodes
    ax.scatter(
        [pos[n][0] for n in outside_nodes],
        [pos[n][1] for n in outside_nodes],
        c="red",
        s=10,
        label="outside DF1",
    )
    ax.scatter(
        [pos[n][0] for n in inside_nodes],
        [pos[n][1] for n in inside_nodes],
        c="blue",
        s=10,
        label="inside DF1",
    )

    # ax.set_title("OSMnx pull vs. DF1-filtered nodes")
    ax.legend()
    # plt.savefig("data/plots/helmond_subgraph.png", dpi=300)

    return ax


def get_node_features_subgraph(G_sub):
    G_sub = nx.convert_node_labels_to_integers(G_sub, ordering="sorted", first_label=0)
    nodes, _ = ox.graph_to_gdfs(G_sub)
    nodes = nodes[["y", "x"]].reset_index(names="node_id")
    nodes = nodes.rename(columns={"y": "lat", "x": "lon"})
    return nodes


def clean_numeric(val, default=50.0):
    if isinstance(val, list):
        val = val[0]  # If it's a list, pick the first value
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_edge_features_subgraph(G_sub):
    G_sub = nx.convert_node_labels_to_integers(G_sub, ordering="sorted", first_label=0)
    _, edges = ox.graph_to_gdfs(G_sub)
    print(f"{edges= }")

    edges = edges[["maxspeed", "length", "geometry"]]
    edges["maxspeed"] = edges["maxspeed"].apply(
        lambda x: clean_numeric(x, default=50.0)
    )
    edges["length"] = edges["length"].apply(
        lambda x: clean_numeric(x, default=30.0)
    )  # 30m as a reasonable default)
    edges["maxspeed"] = edges["maxspeed"].fillna(50.0)
    edges["length"] = edges["length"].fillna(30.0)
    return edges


def plot_with_route(G_sub, G_pt, route=None, ax=None, goal_node=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # draw base graph
    ox.plot_graph(
        G_pt,
        ax=ax,
        show=False,
        close=False,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.5,
        bgcolor="white",
    )

    # draw route in graph
    if route is not None:
        reached = goal_node is not None and route[-1] == goal_node
        color = "tab:green" if reached else "tab:red"
        ox.plot_graph_route(
            G_pt,
            route,
            route_color=color,
            route_linewidth=3,
            ax=ax,
            show=False,
            close=False,
            orig_dest_node_size=0,
        )

    if route:
        start_node = route[0]
        if start_node in G_pt.nodes():
            x0 = G_pt.nodes[start_node]["x"]
            y0 = G_pt.nodes[start_node]["y"]
            ax.scatter(x0, y0, c="green", s=100, marker="*", label="start")

    if goal_node is not None:
        if goal_node in G_pt.nodes():
            x1 = G_pt.nodes[goal_node]["x"]
            y1 = G_pt.nodes[goal_node]["y"]
            ax.scatter(x1, y1, c="red", s=100, marker="*", label="goal")

    # then your inside/outside scatters...
    pos = {nid: (data["x"], data["y"]) for nid, data in G_pt.nodes(data=True)}
    inside = set(G_sub.nodes())
    outside = set(G_pt.nodes()) - inside

    ax.scatter(
        [pos[n][0] for n in outside],
        [pos[n][1] for n in outside],
        c="red",
        s=10,
        label="outside DF1",
    )
    ax.scatter(
        [pos[n][0] for n in inside],
        [pos[n][1] for n in inside],
        c="blue",
        s=10,
        label="inside DF1",
    )

    ax.legend()

    return fig, ax


if __name__ == "__main__":
    G_sub, G_pt = create_osmnx_sub_graph_only_inside_helmond(
        51.473609, 5.738671, 1000, timeseries_df
    )
    print(f"{G_sub= }")

    get_node_features_subgraph(G_sub)
    get_edge_features_subgraph(G_sub)
