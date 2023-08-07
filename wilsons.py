import networkx as nx
import numpy as np
import random

def wilsons(g):
    root = randomly_pick_root(g)

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
    """ Returns True if cycle exists in graph g.
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

    tree = tree.update(branch.edges)
