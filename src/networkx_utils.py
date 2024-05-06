### IMPORTS AND STATIC STUFF ###

import networkx as nx
from pathlib import Path
import sys


def create_graph_from_subgraphsets(G, subgraph_sets):
    """
    Creates a new undirected graph by combining specified subgraphs from an existing graph.

    Parameters:
    - G (networkx.Graph): The original graph from which subgraphs will be extracted.
    - subgraph_sets (list of lists): A list of subgraph sets, where each set contains nodes representing a subgraph.

    Returns:
    - res_graph (networkx.Graph): A new undirected graph formed by combining the specified subgraphs from the original graph.
    The resulting graph includes all nodes and edges present in the selected subgraphs.
    """

    res_graph = nx.Graph()
    for g in subgraph_sets:
        subgraph = G.subgraph(g)
        res_graph.add_nodes_from(subgraph.nodes)
        res_graph.add_edges_from(subgraph.edges)

    return res_graph


def collect_init_subgraphs(G, size_cutoff):
    """
    Collects initial subgraphs from graph based on a specified size cutoff.

    Parameters:
    - G (networkx.Graph): The connected graph from which subgraphs will be collected.
    - size_cutoff (int): The minimum size (number of nodes) for a subgraph to be considered.

    Returns:
    - okay_subgraphs (list of sets): A list of subgraphs that meet the size criterion.
    - subgraphs_to_trim (list of networkx.Graph): A list of subgraphs that need to be further trimmed
    to meet the size criterion. Each subgraph is represented as a networkx.Graph object.
    """

    subgraphs = list(nx.connected_components(G))
    okay_subgraphs, subgraphs_to_trim = [], []

    for i, subgraph in enumerate(subgraphs):

        subgraph_size = len(subgraph)

        print(f"Subgraph {i + 1}: {len(subgraph)} nodes")

        if subgraph_size >= size_cutoff:
            subgraphs_to_trim.append(create_graph_from_subgraphsets(G, [subgraph]))
        else:
            okay_subgraphs.append(subgraph)

    return okay_subgraphs, subgraphs_to_trim


def trim_graph_into_subgraphs(G, size_cutoff, priority_nodes=set(),
                              remove_multnodes=1, priority_limit=10):
    """
    Trims a graph into subgraphs based on a size cutoff. 

    Parameters:
    - G: networkx graph object
    - size_cutoff: The maximum size allowed for a subgraph.
    - remove_multnodes: Number of bottleneck nodes to remove at each iteration. If None, removes only one.
                        Only removing one at the time is slower but more precise. 

    Returns:
    A list of subgraphs obtained after trimming the input graph.
    """

    origin_gsize = len(G)
    new_gsize = 0
    # compute subgraphs.
    trimmed_subgraphs = []
    subgraphs = list(nx.connected_components(G))
    subgraph_sizes = [len(subgraph) for subgraph in subgraphs]
    check = [s <= size_cutoff for s in subgraph_sizes]
    flag = all(check)

    while flag == False:
        # identifiy and remove bottleneck node
        betweenness_centrality = nx.betweenness_centrality(G)
        central_sort_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)

        # algorithm:
        # If
        # we want to add stuff, if it is not in priority nodes
        # or if we have gone beyond top 5

        if priority_nodes:

            bottleneck_nodes = []
            nr_nodes = len(central_sort_nodes)
            i = 0

            # if collected all nodes and not gone beyond priority limit
            while len(bottleneck_nodes) < remove_multnodes and priority_limit > i:
                bottleneck_node = central_sort_nodes[i]
                # not in priority nodes
                if bottleneck_node not in priority_nodes:
                    bottleneck_nodes.append(bottleneck_node)
                    print(f"Found non-priorty bottleneck node at {i}/{nr_nodes}")

                i += 1

            # add priority nodes anyway, if we went beyond limt
            diff = remove_multnodes - len(bottleneck_nodes)
            print(bottleneck_nodes)
            print(diff)
            if diff:
                nodes_to_add = [central_sort_nodes[n] for n in range(i) if
                                central_sort_nodes[n] not in bottleneck_nodes]
                bottleneck_nodes += nodes_to_add[:diff]
            print(bottleneck_nodes)
            G.remove_nodes_from(bottleneck_nodes)

        else:
            bottleneck_nodes = central_sort_nodes[:remove_multnodes]
            G.remove_nodes_from(bottleneck_nodes)

        # check new subgraphs
        subgraphs = list(nx.connected_components(G))
        subgraph_sizes = [len(subgraph) for subgraph in subgraphs]
        print(f"Current subgraph sizes {subgraph_sizes}")

        # check subgraph condiditons
        N = len(subgraphs)
        check = [s <= size_cutoff for s in subgraph_sizes]
        flag = all(check)
        anyflag = any(check)

        # we are done
        if flag:
            print("We done!")

        # we can simplify the graph, before continueing
        elif anyflag:
            keep_subgraphs = []
            for i in range(N):
                if check[i]:
                    trimmed_subgraphs.append(subgraphs[i])
                else:
                    keep_subgraphs.append(subgraphs[i])
            # recreate graph from remaining subgraphs
            G = create_graph_from_subgraphsets(G, keep_subgraphs)
            print(f"New total graph size {len(G)}")

        # we will continue
        else:
            print("Continuing")

    # adding the remaining subgraphs
    for subgraph in subgraphs: trimmed_subgraphs.append(subgraph)

    print("Sizes of subgraphs after trimming{}")
    for i, subgraph in enumerate(trimmed_subgraphs):
        size = len(subgraph)
        print(f"Subgraph {i} size {size}")
        new_gsize += size

    print(f"Nodes trimmed {origin_gsize} -> {new_gsize}")

    return trimmed_subgraphs
