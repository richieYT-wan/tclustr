### IMPORTS AND STATIC STUFF ###

import networkx as nx
from pprint import pprint
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_mst_from_distance_matrix(distance_matrix, label_col='peptide', index_col='raw_index', algorithm='kruskal'):
    """
    Given a labelled distance matrix, create the Graph and minimum spanning tree associated
    Args:
        distance_matrix:
        label_col:
        index_col:
        algorithm:

    Returns:

    """
    values = distance_matrix[distance_matrix.index].values
    labels = distance_matrix[label_col].values
    raw_indices = distance_matrix[index_col].values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    G = nx.Graph(values)
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['peptide'] = labels[i]
        G.nodes[node]['raw_index'] = raw_indices[i]
    tree = nx.minimum_spanning_tree(G, algorithm=algorithm)
    # For now, return all values as they may be useful later for analysis
    return tree, G, labels, encoded_labels, label_encoder, raw_indices


def prune_by_distance(tree, distance_threshold, prune_node=True, labels=None, verbose=False):
    """
    Given a minimum spanning tree, prune it based on a distance threshold
    Args:
        tree:
        distance_threshold:
        prune_node: Bool, remove nodes that have become isolated after edge-pruning
        labels:
        verbose:

    Returns:

    """
    tree_pruned = tree.copy()
    removed_edges = []
    removed_nodes = []
    for u, v in tree_pruned.edges():
        weight = tree_pruned.get_edge_data(u, v)['weight']
        if weight >= distance_threshold:  # assuming weight is a distance and not a similarity
            if labels is not None:
                l1 = labels[u]
                l2 = labels[v]
                same_label = l1 == l2 if labels is not None else np.nan
            removed_edges.append((u, v, round(weight, 4), same_label))
            tree_pruned.remove_edge(u, v)
    if prune_node:
        nodes_to_remove = list(nx.isolates(tree_pruned))
        tree_pruned.remove_nodes_from(nodes_to_remove)
    if verbose:
        print(f'***\nbefore pruning : {len(tree.edges())} edges, {len(tree.nodes())} nodes\n***')
        print(f'***\nafter pruning : {len(tree_pruned.edges())} edges, {len(tree_pruned.nodes())} nodes\n***')
        print(len(removed_edges), 'removed edges',
              f'\n{np.array([x[3] for x in removed_edges]).mean():.2%} of the removed edges were actually same-class')
    return tree_pruned, removed_edges, removed_nodes


def draw_mst_spring(tree, color_map, title=None, iterations=300, threshold=1e-5, scale=0.9, k=0.05, dim=2, seed=13):
    sns.set_style('darkgrid')
    # Visualize the graph and the minimum spanning tree
    f, a = plt.subplots(1, 1, figsize=(8, 8))
    pos = nx.spring_layout(tree, iterations=iterations, threshold=threshold, seed=seed, scale=scale, k=k, dim=dim)
    node_colors = [color_map[tree.nodes[node]['peptide']] for node in tree.nodes()]
    nx.draw_networkx_nodes(tree, pos, node_color=node_colors, node_size=52.5,
                           ax=a)  # nx.draw_networkx_edges(G, pos, edge_color="grey")
    nx.draw_networkx_labels(tree, pos, font_size=4, font_color='w', font_weight='semibold', font_family="monospace")
    nx.draw_networkx_edges(tree, pos, edge_color="k", width=0.75)
    # Create a legend
    legend_labels = {k: color_map[k] for k in sorted(color_map.keys())}  # Sorted keys for consistent order
    patches = [Patch(color=color, label=f'{label}') for label, color in legend_labels.items()]
    plt.legend(handles=patches, title='Node Classes', loc='best', borderaxespad=0.)
    # Find bottleneck score ; with centrality AND connectivity
    if title is not None:
        a.set_title(title)
    plt.axis("off")
    plt.show()


def betweenness_cut(tree, cut_threshold, which='node', verbose=False):
    """
    cuts a minimum spanning tree based on a betweenness centrality threshold on either node or edge centrality
    Args:
        tree:
        cut_threshold:
        which:
        verbose:

    Returns:

    """
    tree_cut = tree.copy()
    # For now, compute both betweenness centrality and see how slow it is.
    edge_betweenness = nx.edge_betweenness_centrality(tree_cut)
    node_betweenness = nx.betweenness_centrality(tree_cut)
    # Sort it just for the printing // return ; These are list of tuples [(edge or node, centrality)]
    sorted_edges = sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True)
    sorted_nodes = sorted(node_betweenness.items(), key=lambda item: item[1], reverse=True)

    if which == 'edge':
        # remove edges that exceed the centrality threshold
        edges_to_remove = [x for x in sorted_edges if x[1] > cut_threshold]
        # Take first element (edge) to cut
        tree_cut.remove_edges_from([x[0] for x in edges_to_remove])
        # identify and remove nodes that now have no edges (became singletons)
        nodes_to_remove = [x for x in sorted_nodes if x[0] in list(nx.isolates(tree_cut))]
        tree_cut.remove_nodes_from([x[0] for x in nodes_to_remove])

    elif which == 'node':
        # remove nodes that exceed the centrality threshold
        nodes_to_remove = [x for x in sorted_nodes if x[1] > cut_threshold]
        # TODO: OLD definition to remove
        # edges_to_remove = [x for x in tree_cut.edges() if any([x[0] in nodes_to_remove or x[1] in nodes_to_remove])
        tree_cut.remove_nodes_from([x[0] for x in nodes_to_remove])
        # Further identify nodes that have become singletons and must be removed
        isolated_nodes = list(nx.isolates(tree_cut))
        tree_cut.remove_nodes_from(isolated_nodes)
        nodes_to_remove.extend([x for x in sorted_nodes if x[0] in isolated_nodes])

        # Identify the edges that have been cut and their centrality
        list_nodes = [x[0] for x in nodes_to_remove]
        edges_to_remove = [x for x in sorted_edges if any([x[0][0] in list_nodes or x[0][1] in list_nodes])]


    else:
        raise ValueError(f'`which` must be either "node" or "edge". Got {which} instead')

    if verbose:
        print(f'N edges, M nodes')
        print(f'\tbefore cutting:\t{len(list(tree.edges()))},\t{len(list(tree.nodes()))}')
        print(f'\tafter cutting:\t{len(list(tree_cut.edges()))},\t{len(list(tree_cut.nodes()))}')
        print('N components before cutting:\t', len(list(nx.connected_components(tree))))
        print('N components after cutting:\t', len(list(nx.connected_components(tree_cut))))
        print('\nEdges removed:')
        pprint(edges_to_remove)
        print('\nNodes removed:')
        pprint(nodes_to_remove)

    return tree_cut, edges_to_remove, nodes_to_remove


# JOAKIM FCTS
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
