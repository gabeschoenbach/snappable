from IPython.display import display, clear_output
from networkx.algorithms.tree.mst import SpanningTreeIterator
# from gerrychain import Partition
# from gerrychain.grid import Grid
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle
import math
import time

##### Visualization functions #####

def draw(graph, delay=0, erase=True, edge_colors=None, node_colors=None):
    '''
    A way to visualize a graph.
    Input: Networkx graph.
    Output: Plots graph, then waits for a specified delay.
    '''
    (x_dim, y_dim) = get_dim_of_graph(graph)
    size = 0.5 * (x_dim + y_dim)
    plt.figure(figsize=(y_dim,x_dim)) # this is needed to keep aspect ratio correct
    plt.tight_layout()
    if edge_colors is not None:
        edge_colors = [graph[u][v]['colors'] for u,v,c in graph.edges(data=True)]
    else:
        edge_colors = ['black' for u,v in graph.edges()]
    nx.draw(graph,
            pos={(x,y): (x, y) for x,y in graph.nodes()},
            width=2,
            with_labels=False,
            node_size=80*size,
            edge_color=edge_colors,
            node_color=node_colors,
            # **kwargs,
           )
    plt.show()
    if erase:
        time.sleep(delay)
        clear_output(wait=True)
    return

def draw_plan(partition, delay=0):
    graph = partition.graph
    graph = nx.create_empty_copy(graph)
    edges = []
    for part, subgraph in partition.subgraphs.items():
        for edge in subgraph.edges():
            edges.append(edge)
    graph.update(edges=edges)
    draw(graph, delay=delay)
    return

##### Balance edge functions #####

def find_balance_edge(tree, d=0):
    '''
    Returns a set of all edges that, if cut, would result in precisely two connected components
    each with size within d of half the number of nodes in the tree.
    '''
    ideal_size = len(tree) / 2
    balance_edges = set()
    split_graph = tree.copy()
    for (u, v) in tree.edges:
        split_graph.remove_edge(u, v)
        components = nx.connected_components(split_graph)
        for component in components:
            if len(component) >= ideal_size - d and len(component) <= ideal_size + d: # clean up
                balance_edges.add((u,v))
                if d == 0:
                    return balance_edges
            break
        split_graph.add_edge(u, v)
    return balance_edges

##### Sampling STs functions #####

def uniform_random_spanning_tree(graph, choice=random.choice):
    '''
    Parker's implementation of Wilson's UST algorithm.
    Input: Networkx graph.
    Output: A UST on the graph.
    '''
    root = choice(list(graph.nodes))
    tree_nodes = set([root])
    next_node = {root: None}
    for node in graph.nodes:
        u = node
        while u not in tree_nodes:
            next_node[u] = choice(list(nx.neighbors(graph, u)))
            u = next_node[u]
        u = node
        while u not in tree_nodes:
            tree_nodes.add(u)
            u = next_node[u]

    G = nx.Graph()
    for node in tree_nodes:
        if next_node[node] is not None:
            G.add_edge(node, next_node[node])

    # DEBUG (10/13): did we produce a valid spanning tree?
    assert len(G.nodes) == len(graph.nodes)
    assert len(G.edges) == len(G.nodes) - 1
    assert nx.number_connected_components(G) == 1

    G = make_ST_from_tup(tup(G))

    return G

def random_minimum_spanning_tree(graph):
    '''
    Assign edge weights to a graph uniformly at random, then draw a
    minimum spanning tree using networkx's algo.
    Input: Networkx graph.
    Output: An MST on the graph.
    '''
    for (u, v, w) in graph.edges(data=True):
        w['weight'] = random.uniform(0,1)
    MST = nx.minimum_spanning_tree(graph)

    MST = make_ST_from_tup(tup(MST))
    return MST

##### Enumeration helper functions #####

def generate_grid_graph(N):
    g = nx.grid_graph(dim=(N,N))
    # nx.relabel_nodes(g, {node: (N * node[0]) + node[1] + 1 for node in g.nodes}, copy=False)
    for edge in g.edges:
        u, v = edge
        g[u][v]['weight'] = 1
    return g

def generate_ST(dim, seed=0):
    '''
    Returns a spanning tree on a graph of dimension=dim, drawn uniformly at random.
    '''
    random.seed(seed)
    graph = nx.grid_graph(dim=dim)
    ST = uniform_random_spanning_tree(graph)
    return ST

def get_dim_of_graph(graph):
    '''
    Returns the dimensions of a graph.
    '''
    x_dim = max([u for (u,v) in graph.nodes()]) + 1
    y_dim = max([v for (u,v) in graph.nodes()]) + 1
    return (y_dim, x_dim) # think about why we have to switch...?

def tup(ST):
    '''
    Sorts and tuplifies the edges of a graph, so that we can check if
    a spanning tree (uniquely defined by its edges) has already been seen.
    '''
    return tuple(sorted(ST.edges()))

def get_dim_of_tup(t):
    '''
    Returns the dimensions of the graph whose edges had given us
    the input, a tup.
    '''
    max_x1s = max([x1 for ((x1,y1), (x2,y2)) in t])
    max_x2s = max([x2 for ((x1,y1), (x2,y2)) in t])
    max_y1s = max([y1 for ((x1,y1), (x2,y2)) in t])
    max_y2s = max([y2 for ((x1,y1), (x2,y2)) in t])
    return (max(max_y1s, max_y2s) + 1, max(max_x1s, max_x2s) + 1)

def make_ST_from_tup(t):
    '''
    Turn the tup back into a spanning tree.
    '''
    ST = nx.grid_graph(dim=get_dim_of_tup(t))
    ST = nx.create_empty_copy(ST)
    ST.update(edges = t)
    return ST

def find_empty_edges(ST):
    '''
    Returns the list of edges that do not exist in the given spanning tree.
    '''
    dim = get_dim_of_graph(ST)
    graph = nx.grid_graph(dim=dim)
    all_edges = graph.edges()
    graph = nx.create_empty_copy(graph)
    graph.update(edges = ST.edges())
    return [edge for edge in all_edges if edge not in ST.edges()]

def summarize(i, neighbors, neighboring_STs):
    '''
    Used midway through the enumeration function to update us
    on how many k-neighbors and unique spanning trees have been found.
    '''
    total = 0
    for k, v in neighbors.items():
        print(f"{k}-neighbors: {len(v)}")
        total += len(v)
    print(f"{i}-neighbors: {len(neighboring_STs)}")
    total += len(neighboring_STs)
    print("---")
    print(f"{total} unique STs")
    return

##### Enumeration functions #####

def find_neighboring_STs(ST, seen_STs):
    '''
    Given a spanning tree, returns a list of neighboring spanning trees.
    '''
    neighboring_STs = []

    empty_edges = find_empty_edges(ST)
    for edge in ST.edges():
        ST.remove_edge(edge[0], edge[1]) # should I make a copy or can I use ST?
        for empty_edge in empty_edges:
            ST.add_edge(empty_edge[0], empty_edge[1])
            try:
                nx.find_cycle(ST)
            except:
                if tup(ST) not in seen_STs:
                    neighboring_STs.append(tup(ST)) # this might get slow!
                else:
                    x = 1 # this is where we should count repeats
            ST.remove_edge(empty_edge[0], empty_edge[1])
        ST.add_edge(edge[0], edge[1])
    return neighboring_STs

def enumerate_STs(dim, seed=0):
    '''
    Enumerates all spanning trees of dimension dim.
    Benchmarks:
      -- Up to 3x3 enumeration is nearly instantaneous
      -- 3x4 enumeration takes ~30s.
      -- 4x4 enumeration takes ~35 minutes.
    '''
    seed = generate_ST(dim, seed)
#     seed = make_ST_from_tup(tup(seed))

    neighbors = {
        0:[tup(seed)]
    }

    seen_STs = set()
    for i in range(1,20):
        for ST_tup in neighbors[i-1]:
            seen_STs.add(ST_tup)

        neighboring_STs = []
        for ST_tup in neighbors[i-1]:
            ST = make_ST_from_tup(ST_tup)
            neighboring_STs += find_neighboring_STs(ST, seen_STs)
            neighboring_STs = list(set(neighboring_STs))

            clear_output(wait=True)
            summarize(i, neighbors, neighboring_STs)

        neighbors[i] = neighboring_STs

        if len(neighboring_STs) == 0:
            return neighbors

    return neighbors

##### Counting the number of spanning trees

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    '''
    Removes user-specified rows/columns from a CSR,
    used for find_NST(), taken from https://tinyurl.com/y3szqual.
    Input: csr_matrix (from scipy.sparse)
    Output: csr_matrix
    '''
    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

def find_NST(graph):
    '''
    NEED TO CHANGE THIS TO PARKER'S FUNCTION!
    To find the number of spanning trees on a graph, build the
    Laplacian matrix (will be a sparse matrix), delete the first
    row and column, and take the determinant.
    Input: Networkx graph.
    Output: Int.
    '''
    lap = nx.laplacian_matrix(graph)
    lap = delete_from_csr(lap, row_indices=[0], col_indices=[0])
    lap = lap.toarray()
    NST = round(np.linalg.det(lap))
    return NST

##### Distributions of spanning tree algos #####

def unpack_enumerated_STs(neighbors):
    '''
    Given the neighbors dictionary, returns a list of tup'ed STs.
    '''
    STs = []
    for STs_list in neighbors.values():
        for ST in STs_list:
            STs.append(ST)
    return STs

def sample_STs(STs, funct, num_trials):
    '''
    Given the list of STs, either 'UST' or 'MST', and a number of trials,
    this returns a list of each occurence of an ST, labeled by its index in STs
    '''
    data = []
    for _ in tqdm(range(num_trials)):
        graph = nx.grid_graph(dim=get_dim_of_tup(STs[0]))
        if funct == "UST":
            ST = uniform_random_spanning_tree(graph)
        elif funct == "MST":
            ST = random_minimum_spanning_tree(graph)
        else:
            print("Error: function must be either 'UST' or 'MST'")
        idx = STs.index(tup(ST))
        data.append(idx)
    return data

def plot_sampled_STs(STs, data, save=None):
    '''
    Given the list of STs and the data returned by sample_STs(), this plots
    the distribution and optionally saves it.
    '''
    fig, ax = plt.subplots(figsize=(16,3))

    dim = tree_utils.get_dim_of_tup(STs[0])
    ax.set_title(f"Distribution of STs on {dim[0]}x{dim[1]} grid -- {len(data)} samples", fontsize=20)
    ax.hist(data,
               alpha=1,
               bins=np.arange(0, len(STs)+1),
               density=True)
    ax.grid()
    if save is not None:
        plt.savefig(save, dpi=200)
    plt.show()
    return

def plot_sampled_STs(STs, data, save=None):
    fig, ax = plt.subplots(figsize=(16,3))

    dim = get_dim_of_tup(STs[0])
    ax.set_title(f"Distribution of STs on {dim[0]}x{dim[1]} grid -- {len(data)} samples", fontsize=20)
    ax.hist(data,
               alpha=1,
               bins=np.arange(0, len(STs)+1),
               density=True)
    ax.grid()
    if save is not None:
        plt.savefig(save, dpi=200)
    plt.show()
    return

##### Enumerating ominos #####

def load_partitions(n):
    '''
    Read in Zach Schutzman's enumeration of all possible
    ominoes of an nXn grid and turn it into a dataframe, where
    each row is an assignment of nodes to districts.
    Currently, only have 3x3 - 6x6 stored in the enumerations folder.
    '''
    parts_path = f"./enumerations/{n}x{n}_{n}.txt"
    df = pd.read_csv(parts_path, header=None)
    return df

def make_assignment_dicts(n):
    '''
    Returns a list of every possible node-to-district
    assignment on the nXn grid.
    '''
    assignment_dicts = []
    grid = Grid((n,n))
    nodes = grid.graph.nodes()
    df = load_partitions(n)
    for i in range(len(df)):
        assignment_dict = {}
        partition = df.iloc[i].values.tolist()
        for i, node in enumerate(nodes):
            assignment_dict[node] = partition[i]
        assignment_dicts.append(assignment_dict)
    return assignment_dicts

def make_partitions(n):
    '''
    Returns a list of every possible partition
    on the nXn grid. This is nearly instantaneous for n = 3-5,
    and takes about 60s for n=6.
    '''
    partitions = []
    assignment_dict = make_assignment_dicts(n)
    grid = Grid((n,n))
    for i in range(len(assignment_dict)):
        partition = Partition(grid.graph, assignment_dict[i])
        partitions.append(partition)
    return partitions

def sp_score(partition):
    '''
    Take the product of the number of spanning trees in each part of the partition.
    '''
    prod = 1
    for part, subgraph in partition.subgraphs.items():
        NST = find_NST(subgraph)
        prod *= NST
    return prod

def cut_edges(partition):
    '''
    For each edge in the graph, it is a cut edge if the
    adjacent nodes are in different parts of the partition.
    '''
    cut_edges = 0
    for edge in partition.graph.edges():
        if partition.assignment[edge[0]] != partition.assignment[edge[1]]:
            cut_edges += 1
    return cut_edges

def compute_weights(partitions):
    '''
    Returns a list of spanning tree scores for each
    partition in the list of partitions.
    '''
    weights = []
    for partition in tqdm(partitions):
        sp = sp_score(partition)
        weights.append(sp)
    return weights

def compute_cut_edges(partitions):
    '''
    Returns a list of the cut edges for each
    partition in the list of partitions.
    '''
    ces = []
    for partition in tqdm(partitions):
        ce = cut_edges(partition)
        ces.append(ce)
    return ces

### Grid-Graph experiments
def generate_grid_graph(dims, queen=False):
    g = nx.grid_graph(dim=dims)
    # nx.relabel_nodes(g, {node: (N * node[0]) + node[1] + 1 for node in g.nodes}, copy=False)
    for edge in g.edges:
        u, v = edge
        g[u][v]['weight'] = 1

    if queen:
        for i in range(dims[1]-1):
            for j in range(dims[0]):
                if j<(dims[0]-1):
                    g.add_edge((i,j),(i+1,j+1))
                    g[(i,j)][(i+1,j+1)]['weight'] = 1
                if j >0:
                    g.add_edge((i,j),(i+1,j-1))
                    g[(i,j)][(i+1,j-1)]['weight'] = 1
    return g

def enumerate_all_trees(g):
    trees = SpanningTreeIterator(g)
    trees = []
    for tree in tqdm(SpanningTreeIterator(g)):
        trees.append(tree)
    return trees

def sample_trees(g, func, N):
    trees = []
    for _ in tqdm(range(N)):
        tree = func(g)
        trees.append(tree)
    return trees

def make_tree_dict(trees, ds):
    tree_dict = {}
    for tree in tqdm(trees):
        d_dict = {}
        for d in ds:
            d_dict[d] = find_balance_edge(tree, d)
        tree_dict[tree] = d_dict
    return tree_dict

def make_counting_dict(g, tree_dict, ds):
    counting_dict = {}
    for d in ds:
        counting_dict[d] = {e: 0 for e in g.edges}
        for tree in tqdm(tree_dict):
            balance_edges = tree_dict[tree][d]
            for balance_edge in balance_edges:
                counting_dict[d][balance_edge] += 1
    return counting_dict


## Plotting functions
# # how often is this edge snappable?
# plt.figure(figsize=(dims[1], dims[0])) # this is needed to keep aspect ratio correct
# plt.tight_layout()
# nx.draw(g,
#         pos={(x,y): (x, y) for x,y in g.nodes()},
#         width=2,
#         with_labels=False,
#         node_size=80,
#         edge_color=[counting_dict[0][e] for e in g.edges()],
#         edge_cmap=plt.cm.Reds
#        )
# sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)#, norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# plt.colorbar(sm, fraction=0.046, pad=0.01)
# plt.show()

# # how often is this edge used in a spanning tree?
# counter = {}
# for edge in g.edges: # initialize
#     counter[edge] = 0

# for tree in tqdm(trees):
#     for edge in tree.edges:
#         counter[edge] += 1

# plt.figure(figsize=(dims[1], dims[0])) # this is needed to keep aspect ratio correct
# plt.tight_layout()
# nx.draw(g,
#         pos={(x,y): (x, y) for x,y in g.nodes()},
#         width=2,
#         with_labels=False,
#         node_size=80,
#         edge_color=[counter[e] for e in g.edges()],
#         edge_cmap=plt.cm.Reds
#        )
# sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)#, norm=plt.Normalize(vmin=0, vmax=1))
# sm.set_array([])
# plt.colorbar(sm, fraction=0.046, pad=0.01)
# plt.show()

def sample_edge_from_counts(counting_dict):
    """
    """
    edges = list(counting_dict[0].keys())
    weights = np.array([])
    total = sum(counting_dict[0].values())
    for edge in edges:
        weights = np.append(weights, counting_dict[0][edge]/ total)

    return edges[np.random.choice(len(edges), p=weights)]

def expand_frontier(g, node, cc, other_cc, cc_frontier):
    """
        you give me a node that you just added to my cc.
        i'll add its neighbors to my frontier.
        but i can't add those neighbors that are in the other cc.
        and i cant add those neighbors whose addition creates a loop.
    """
    valid_neighbors = set().union(g.neighbors(node)).difference(cc.nodes).difference(other_cc.nodes)

    for neighbor in valid_neighbors:
        cc.add_edge(node, neighbor)
        if nx.is_tree(cc):
            cc_frontier.add_edge(node, neighbor)
        cc.remove_node(neighbor)

def sample_from_frontier(cc, frontier):
    """ Samples an edge from the frontier and returns it as
        (new_node, old_node).
        ``old_node" is the node that already exists in the connecting
        component, and ``new_node" is the node that is being added to
        the connecting component.
    """
    u, v = random.choice(list(frontier.edges))
    new_node = None
    old_node = None

    if u in cc.nodes:
        new_node = v
        old_node = u
    else:
        new_node = u
        old_node = v

    return new_node, old_node

def sample_from_frontier_to_cc(g, frontier, cc, other_frontier, other_cc):
    """
    """
    new_node, old_node = sample_from_frontier(cc, frontier)

    cc.add_edge(new_node, old_node)
    expand_frontier(g, new_node, cc, other_cc, frontier)
    frontier.remove_edge(new_node, old_node) # do i need to remove the node too? this might cause bugs...
    if new_node in other_frontier.nodes:
        other_frontier.remove_node(new_node)


def get_furthest_bottleneck_node(g, small_cc, big_cc, src_node, snapped_edge):
    # get adjacents
    adjacents = []
    for node in small_cc.nodes:
        for neighbor in g.neighbors(node):
            if neighbor in snapped_edge:
                continue
            if neighbor in big_cc.nodes:
                adjacents.append(neighbor)

    # now get the distance to the adjacents
    max_node = None
    max_dist = 0
    for node in adjacents:
        pl = nx.shortest_path_length(g, source=src_node, target=node)
        if pl > max_dist:
            max_dist = pl
            max_node = node

    return max_node

def break_bottleneck(g, small_cc, big_cc, snapped_edge,
                     small_cc_frontier, big_cc_frontier):
    """
    """
    src_node = None
    for node in snapped_edge:
        if node in big_cc.nodes:
            src_node = node

    max_node = get_furthest_bottleneck_node(g, small_cc, big_cc,
                                            src_node, snapped_edge)
    remove_node(max_node, big_cc, big_cc_frontier, src_node)
    add_node_to_cc(g, small_cc, max_node)
    expand_frontier(g, max_node, small_cc, big_cc, small_cc_frontier)

def remove_node(node, cc, cc_frontier, src_node):
    """
    """
    cc.remove_node(node)
    cc_frontier.remove_node(node)

    # for the case of multiple connected components
    comps = nx.connected_components(cc)

    to_remove = set()
    for comp in comps:
        if src_node not in comp:
            to_remove = to_remove.union(comp)

    for n in to_remove:
        cc.remove_node(n)
        cc_frontier.remove_node(n)


def add_node_to_cc(g, cc, node):
    """
    """
    candidates = []
    for neighbor in g.neighbors(node):
        if neighbor in cc.nodes:
            candidates.append(neighbor)

    choice = random.choice(candidates)
    cc.add_edge(node, choice)

def initialize(g, edge):
    """ Given a graph and a snapped edge, it initializes the connected
        components and their frontiers, and returns them.
    """
    u, v = edge

    cc1 = nx.Graph()
    cc2 = nx.Graph()

    cc1.add_node(u)
    cc2.add_node(v)

    cc1_frontier = nx.Graph()
    cc2_frontier = nx.Graph()

    expand_frontier(g, u, cc1, cc2, cc1_frontier)
    expand_frontier(g, v, cc2, cc1, cc2_frontier)

    return cc1, cc2, cc1_frontier, cc2_frontier

def is_bottlenecked(cc, cc_frontier, target_size):
    """ Returns True if a connected component cannot grow anymore.
        This happens when there are no edges in the frontier but
        the connected component hasn't met the target_size yet.
    """
    return len(cc_frontier.edges) == 0 and len(cc.nodes) < target_size

def trees_correct_sizes(cc1, cc2, target_size):
    """ Returns True when both the argument connected components are
        equal to target_size.
        Currently only works for exact bisection.
    """
    return len(cc1.nodes) == target_size and len(cc2.nodes) == target_size
