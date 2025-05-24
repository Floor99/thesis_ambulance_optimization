import networkx as nx
from networkx import MultiDiGraph


def get_recursive_dead_end_nodes(G: MultiDiGraph) -> set:
    """
    Recursively remove dead-end nodes (degree == 1) and return the set of all such nodes.
    These are nodes that cannot be part of any loop or through path.
    """
    G = G.copy()
    G_undirected = G.to_undirected()
    dead_end_nodes = set()

    while True:
        # Find current dead ends
        leaves = [n for n, d in G_undirected.degree() if d == 1]
        if not leaves:
            break
        dead_end_nodes.update(leaves)
        G.remove_nodes_from(leaves)
        G_undirected = G.to_undirected()

    return dead_end_nodes


def get_recursive_dead_end_nodes(G, target_node):
    """
    Recursively remove dead ends (nodes with degree 1) **excluding**
    any subgraph that contains the target_node.
    Returns a set of nodes that are recursive dead ends.
    """
    G = G.copy()
    G_undirected = G.to_undirected()
    dead_ends = set()
    
    # Nodes that are protected from pruning
    protected_nodes = set()
    
    while True:
        # Find all current leaf nodes (degree == 1)
        leaves = [n for n in G_undirected.nodes if G_undirected.degree(n) == 1]
        
        # If no more leaves, we're done
        if not leaves:
            break

        # Separate leaves into those that are in the same component as the target
        to_remove = []
        for leaf in leaves:
            if nx.has_path(G_undirected, leaf, target_node):
                # If target is in the same component, don't remove this path
                component = nx.node_connected_component(G_undirected, leaf)
                if target_node in component:
                    protected_nodes.update(component)
                else:
                    to_remove.append(leaf)
            else:
                to_remove.append(leaf)

        if not to_remove:
            break

        G.remove_nodes_from(to_remove)
        G_undirected = G.to_undirected()
        dead_ends.update(to_remove)

    return dead_ends



def get_trap_neighbors_with_target(G: MultiDiGraph, start_node: int, target_node: int) -> set:
    """
    Identify neighbors of start_node that lead into trap-like neighborhoods.
    Do NOT mask any neighbor if it leads to the target_node.
    """
    trap_neighbors = []
    G_no_start = G.copy()
    G_no_start.remove_node(start_node)

    for neighbor in G.successors(start_node):
        if neighbor not in G_no_start:
            continue

        # Step 1: Get all nodes reachable from the neighbor (without going through start_node)
        reachable = nx.descendants(G_no_start, neighbor)
        reachable.add(neighbor)

        # Step 2: If target_node is reachable, do not mask this neighbor
        if target_node in reachable:
            continue  # This is a valid path

        # Step 3: Check if there is any exit from this neighborhood (excluding back to start_node)
        exit_found = False
        for node in reachable:
            for succ in G.successors(node):
                if succ not in reachable and succ != start_node:
                    exit_found = True
                    break
            if exit_found:
                break

        if not exit_found:
            trap_neighbors.append(neighbor)

    return set(trap_neighbors)
