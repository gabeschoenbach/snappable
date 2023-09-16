from tree_partition import sample_uniform_partition_of_tree
import tree_utils as tu
import networkx as nx
from tqdm import tqdm
import click
import os

def make_biclique(n):
    """
    Generates a complete bipartite graph (biclique) where one part of the graph
    has size 2n and the other has size 4n.
    """
    g = nx.Graph()
    small_side = range(1, 2 * n + 1)
    big_side = range(2 * n + 1, 6 * n + 1)
    g.add_nodes_from(small_side, bipartite = 0)
    g.add_nodes_from(big_side, bipartite = 1)
    edges = [(s, b) for s in small_side for b in big_side]
    g.add_edges_from(edges)
    
    assert nx.is_bipartite(g)
    return g
    

def generate_graph(size, kind):
    if kind == "square":
        graph = nx.grid_graph(dim=(size,size))
    elif kind == "clique":
        graph = nx.complete_graph(size)
    elif kind == "biclique":
        graph = make_biclique(size)
    else:
        raise Exception("Can only accept 'square', 'clique', or 'biclique' graphs.")
    return graph
    
def count_snaps(trials, size, kind):
    graph = generate_graph(size, kind)
    num_snaps = 0
    print(f"{kind} {size}")
    for _ in tqdm(range(trials)):
        tree = tu.uniform_random_spanning_tree(graph)
        for node in tree.nodes:
            tree.nodes[node]['pop'] = 1
        part_size = len(tree) / 2
        _, num_balance_edges = sample_uniform_partition_of_tree(tree, 2, part_size, part_size, "pop")
        if num_balance_edges:
            num_snaps += 1
    return num_snaps / trials

@click.command()
@click.argument('kind')
@click.argument('trials', type=int)
def run_experiment(kind, trials):
    os.makedirs("heuristics/", exist_ok=True)
    with open(f"heuristics/{kind}_{trials}.csv", "a") as f:
        f.write("size,proportion_snaps\n")
    
    for size in [2, 4, 8, 16]:
        proportion = count_snaps(trials, size, kind)
        with open(f"heuristics/{kind}_{trials}.csv", "a") as f:
            f.write(f"{size},{proportion}\n")
    return

if __name__=="__main__":
    run_experiment()
