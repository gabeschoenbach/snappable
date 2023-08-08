# frontier.py

def get_frontier(g, tree):
    """
    """
    frontier = set()
    for node in tree.nodes:
        for neighbor in g.neighbors(node):
            frontier.add(neighbor)
    frontier = frontier.difference(tree.nodes)
    return frontier

def remove_node_from_frontier(node, frontier):
    if node in frontier:
        frontier.remove(node)

def add_frontier_to_tree(tree, frontier, stacks):
    """ Adds the nodes in the frontier that are pointing
        towards the tree.

        Then, clears the tree from the stacks.

        Returns True if nodes were added to the tree, otherwise False.
    """
    new_edges = []

    for node in frontier:
        if stacks[node] in tree.nodes:
            new_edges.append((node, stacks[node]))

    for (u, v) in new_edges:
        tree.add_edge(u, v)

    return (len(new_edges) > 0)

def lerw_rev_wilson(src, g, tree, stacks, frontier):
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

    # add the branch's frontier to the tree's frontier
    for node in branch.nodes:
        add_neighbors_to_frontier(node, g, tree, frontier)
    for node in branch.nodes:
        remove_node_from_frontier(node, frontier)

def pop_nodes(g, nodes, stacks):
    for node in nodes:
        pop(node, stacks, g)

def add_neighbors_to_frontier(node, g, tree, frontier):
    """ Adds the neighbors of `node" to the frontier.
    """
    for neighbor in g.neighbors(node):
        if neighbor not in tree.nodes():
            frontier.add(neighbor)

def add_frontier_to_tree(tree, frontier, stacks):
    """ Adds the nodes in the frontier that are pointing
        towards the tree.

        Then, clears the tree from the stacks.

        Returns True if nodes were added to the tree, otherwise False.
    """
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

def get_src_from_outward_nodes(frontier, stacks):
    outward_nodes = []
    for node in frontier:
        if stacks[node] not in frontier:
            outward_nodes.append(node)
    return random.choice(outward_nodes)
