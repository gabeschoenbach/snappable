from tree_partition import sample_uniform_partition_of_tree
import tree_utils as tu
import networkx as nx
from tqdm import tqdm
import click
import os

def generate_graph(size, kind):
    if kind == "square":
        graph = nx.grid_graph(dim=(size,size))
    elif kind == "complete":
        graph = nx.complete_graph(size)
    else:
        raise Exception("Can only accept 'square' or 'complete' graphs.")
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
    
    for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        proportion = count_snaps(trials, size, kind)
        with open(f"heuristics/{kind}_{trials}.csv", "a") as f:
            f.write(f"{size},{proportion}\n")
    return

if __name__=="__main__":
    run_experiment()