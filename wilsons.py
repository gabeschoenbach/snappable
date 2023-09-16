import tree_utils as tu
import networkx as nx
import numpy as np
import random

def wilsons(g, init_root=None):
    if init_root is None:
        root = randomly_pick_root(g)
    else:
        root = init_root

    tree = nx.Graph()
    tree.add_node(root)
    stacks = initialize_stacks(g, root)

    while tree.number_of_nodes() < g.number_of_nodes():
        src = pick_leaf(g, tree)
        lerw(src, g, tree, stacks)
        clear_tree_from_stacks(tree, stacks)

    return tree

def randomly_pick_root(g):
    return random.choice(list(g.nodes))

def pick_leaf(g, tree):
    candidates = list(set(g.nodes()) - set(tree.nodes()))
    return random.choice(candidates)

def erase_singletons(branch):
    """ Removes the nodes that do not have any edges from the networkx graph ``branch".
    """
    # remove the singletons after erasing the loop
    to_remove = []
    for node in branch.nodes:
        if len(list(branch.neighbors(node))) == 0:
            to_remove.append(node)
    for node in to_remove:
        branch.remove_node(node)

def erase_loop_if_exists(branch, stacks, g):
    """
    """
    try:
        cycle = nx.find_cycle(branch)

        for (u, v) in cycle:
            branch.remove_edge(u, v)
            pop(u, stacks, g)
        erase_singletons(branch)

    except nx.exception.NetworkXNoCycle:
        pass

def loop_exists(g):
    """ Returns True if a cycle exists in graph g.
    """
    try:
        cycle = nx.find_cycle(g)
        return True
    except nx.exception.NetworkXNoCycle:
        return False

def pop(node, stacks, g):
    stacks[node] = random.choice(list(g.neighbors(node)))

def initialize_stacks(g, root):
    stacks = dict()
    for node in g.nodes():
        if node == root:
            continue
        pop(node, stacks, g)
    return stacks

def clear_tree_from_stacks(tree, stacks):
    for node in tree.nodes():
        if node in stacks.keys():
            stacks.pop(node)

def lerw(src, g, tree, stacks):
    curr = src
    branch = nx.Graph()
    reached_root = False

    while reached_root is False:
        nxt = stacks[curr]

        # handle the case of small two edge cycles
        if branch.has_edge(curr, nxt):
            branch.remove_edge(curr, nxt)
            erase_singletons(branch)

            pop(curr, stacks, g)
            pop(nxt, stacks, g)

            curr = nxt
            continue


        branch.add_edge(curr, nxt)
        erase_loop_if_exists(branch, stacks, g)
        curr = nxt

        if nxt in tree.nodes():
            reached_root = True

    tree.update(branch.edges)

def add_neighbors_to_frontier(node, g, tree, frontier):
    """ Adds the neighbors of `node" to the frontier.
    """
    for neighbor in g.neighbors(node):
        if neighbor not in tree.nodes():
            frontier.add(neighbor)

def remove_node_from_frontier(node, frontier):
    if node in frontier:
        frontier.remove(node)

def add_frontier_to_tree(g, tree, frontier, stacks):
    """ Adds the nodes in the frontier that are pointing
        towards the tree.

        Returns True if nodes were added to the tree, otherwise False.
    """
    pop_small_cycles_in_frontier(g, frontier, stacks)

    new_edges = []

    for node in frontier:
        if stacks[node] in tree.nodes:
            new_edges.append((node, stacks[node]))

    for (u, v) in new_edges:
        tree.add_edge(u, v)

    return (len(new_edges) > 0)

def get_frontier(g, tree):
    """
    """
    frontier = set()
    for node in tree.nodes:
        for neighbor in g.neighbors(node):
            frontier.add(neighbor)
    frontier = frontier.difference(tree.nodes)
    return frontier

def pop_small_cycles_in_frontier(g, frontier, stacks):
    """
    """
    small_cycle = []
    for node in frontier:
        if (stacks[node] in frontier) and (stacks[stacks[node]] == node):
            small_cycle.append(node)
            small_cycle.append(stacks[node])
    pop_nodes(g, small_cycle, stacks)


def pop_cycles_in_frontier(g, frontier, stacks):
    """ recursively pop cycles from frontier until the frontier has no cycles.
        it returns either a deadlocked frontier, or a frontier
    """
    graph = nx.DiGraph()
    for node in frontier:
        graph.add_edge(node, stacks[node])

    cycles_popped = False
    while loop_exists(graph):
        cycle_nodes = set()

        cycle = nx.find_cycle(graph)
        for (u, v) in cycle:
            cycle_nodes.add(u)
            cycle_nodes.add(v)

        pop_nodes(g, cycle_nodes, stacks)
        graph = nx.DiGraph()
        for node in frontier:
            graph.add_edge(node, stacks[node])
        cycles_popped = True

    return cycles_popped

def pop_nodes(g, nodes, stacks):
    for node in nodes:
        pop(node, stacks, g)

def rev_wilsons(g, init_root=None):
    if init_root is None:
        root = randomly_pick_root(g)
    else:
        root = init_root

    tree = nx.Graph()
    tree.add_node(root)
    stacks = initialize_stacks(g, root)


    while tree.number_of_nodes() < g.number_of_nodes():
        frontier = get_frontier(g, tree)
        if not add_frontier_to_tree(g, tree, frontier, stacks):
            ## deadlock frontier!
#             if pop_cycles_in_frontier(g, frontier, stacks):
#                 continue
#             src = pick_leaf(g, tree)
            src = random.choice(list(frontier))
            lerw(src, g, tree, stacks)
            # clear_tree_from_stacks(tree, stacks)

    return tree

############
# funcs written for agglomerative tree experiment
def agg_get_root(g, forest):
    """
    """
    forest_nodes = set()
    for tree in forest:
        for node in tree.nodes:
            forest_nodes.add(node)

    candidates = list(set(g.nodes()) - set(forest_nodes))
    return random.choice(candidates)

def agg_add_frontier_to_tree(g, tree, frontier, stacks, roots):
    """ Adds the nodes in the frontier that are pointing
        towards the tree.

        Returns True if nodes were added to the tree, otherwise False.
    """
    # pop_small_cycles_in_frontier(g, frontier, stacks)

    new_edges = []

    for node in frontier:
        if node in roots:
            continue
        if stacks[node] in tree.nodes:
            new_edges.append((node, stacks[node]))

    for (u, v) in new_edges:
        tree.add_edge(u, v)

    return (len(new_edges) > 0)

def agg_draw_forest(forest):
    t = nx.Graph()
    for tree in forest:
        for (u, v) in tree.edges():
            t.add_edge(u, v)
        for node in tree.nodes():
            t.add_node(node)
    tu.draw(t)

def agg_get_cut_edges(g, forest):
    """ returns list of cut edges but with each edge listed twice.
    """
    cut_edges = []
    for idx, tree in enumerate(forest):
        for node in tree.nodes:
            for neighbor in g.neighbors(node):
                if neighbor not in tree.nodes:
                    cut_edges.append((node, neighbor))
    return cut_edges

def agglomerative(g):
    forest = []

    # start with first tree
    root = agg_get_root(g, forest)
    tree = nx.Graph()
    tree.add_node(root)
    stacks = initialize_stacks(g, root)
    forest_nodes = 0
    roots = set([root])

    while True:
        frontier = get_frontier(g, tree)
        if not agg_add_frontier_to_tree(g, tree, frontier, stacks, roots):
            ## deadlock frontier!
            forest.append(tree)
            forest_nodes += tree.number_of_nodes()
            roots.add(root)

            if forest_nodes == g.number_of_nodes():
                break

            root = agg_get_root(g, forest)
            tree = nx.Graph()
            tree.add_node(root)


    return forest

# forest = agglomerative(g)
# agg_draw_forest(forest)

def get_tree_indices(cut_edge, forest):
    """
    """
    u, v = cut_edge
    u_idx = None
    v_idx = None

    for idx, tree in enumerate(forest):
        if u in tree.nodes:
            u_idx = idx
        if v in tree.nodes:
            v_idx = idx

    assert(u_idx != v_idx)
    assert(u_idx is not None)
    assert(v_idx is not None)

    return u_idx, v_idx

def merge(forest, cut_edge):
    """ get the index of the trees where the cut edge is
        make a new tree that merges and with the cut edge
        remove the trees from the forest list
        add new tree to forest
    """
    u_idx, v_idx = get_tree_indices(cut_edge, forest)

    new_tree = nx.Graph()
    for u, v in forest[u_idx].edges:
        new_tree.add_edge(u, v)
    for u, v in forest[v_idx].edges:
        new_tree.add_edge(u, v)
    new_tree.add_edge(cut_edge[0], cut_edge[1])

    forest.pop(max([u_idx, v_idx]))
    forest.pop(min([u_idx, v_idx]))

    forest.append(new_tree)

def agglomerate_till_tree(forest):
    """
    """
    while len(forest) > 1:
        cut_edges = agg_get_cut_edges(g, forest)
        cut_edge = random.choice(cut_edges)
        merge(forest, cut_edge)

    return forest[0]

def main_agg(g):
    forest = agglomerative(g)
    tree = agglomerate_till_tree(forest)
    return tree

def sample_trees_agg(num_trials):
    data = []
    for _ in tqdm(range(num_trials)):
        tree = main_agg(g)

        idx = all_trees.index(tu.tup(tree))
        data.append(idx)
    return data

""" Bipartition algorithm functions
"""
def break_loop_randomly(tree):
    """
    """
    cycle = nx.find_cycle(tree)
    throwaway = random.choice(cycle)
    tree.remove_edge(throwaway[0], throwaway[1])

def bipartition_wilsons(g_1, g_2, cut_set):
    """
    """
    trees = [wilsons(g_1), wilsons(g_2)]
    cut_edge = random.choice(cut_set)

    tree = nx.Graph()
    for edge in trees[0].edges:
        tree.add_edge(edge[0], edge[1])
    for edge in trees[1].edges:
        tree.add_edge(edge[0], edge[1])

    if cut_edge == "all":
        # add all edges in cutset
        for edge in cut_set:
            if edge == "all":
                continue
            else:
                tree.add_edge(edge[0], edge[1])
        break_loop_randomly(tree)
    else:
        tree.add_edge(cut_edge[0], cut_edge[1])

    return tree

## these are for 2x4
# set_1 = [((0, 0), (1, 0)), ((0, 0), (0, 1)), ((0, 1), (1, 1)),  ((1, 0), (1, 1))]
# set_2 = [((2, 0), (3, 0)), ((2, 0), (2, 1)), ((2, 1), (3, 1)), ((3, 0), (3, 1))]
# cut_set = [((1, 0), (2, 0)), ((1, 1), (2, 1)), "all"]

# g_1 = g.edge_subgraph(set_1)
# g_2 = g.edge_subgraph(set_2)

"""
"""
