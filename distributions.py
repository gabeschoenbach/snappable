import numpy as np
import pandas as pd
from tqdm import tqdm
import tree_utils as tu
import matplotlib.pyplot as plt
from tree_partition import sample_uniform_partition_of_tree

def get_sp_score(g, partition):
    prod = 1
    for part in partition:
        subgraph = g.subgraph(part)
        NST = tu.find_NST(subgraph)
        prod *= NST
    return prod

def run_thing(x, y):
    # Generate grid graph
    dims = (x,y)
    g = tu.generate_grid_graph(dims, queen=False)

    print(f"Generating all trees of the {dims} grid graph...")
    all_trees = [tree for tree in tu.enumerate_all_trees(g)]

    print("Identifying snappable trees...")
    part_size = g.number_of_nodes() / 2
    all_snappables = []
    for tree in tqdm(all_trees):
        for n in tree:
            tree.nodes[n]['pop'] = 1
        partition, num_balance_edges = sample_uniform_partition_of_tree(tree, 2, part_size, part_size, "pop")
        if num_balance_edges:
            sp_score = get_sp_score(g, partition)
            for part in partition:
                if (0,0) in part:
                    all_snappables.append((tree, set(part), sp_score))

    print("Pulling out all partitions...")
    all_snappables = sorted(all_snappables, key=lambda x:x[2], reverse=True)
    all_partitions = []
    for tup in all_snappables:
        partition = tup[1]
        if partition not in all_partitions:
            all_partitions.append(partition)
    print(f"There are {len(all_partitions)} balanced 2-partitions of the {dims} grid-graph")

    def indexer(partition):
        return all_partitions.index(partition)

    partition_freqs = {indexer(partition): 0 for partition in all_partitions}
    for tup in all_snappables:
        partition_freqs[indexer(tup[1])] += 1/len(all_snappables)

    partition_scores = {}
    for tup in all_snappables:
        partition = tup[1]
        if indexer(partition) not in partition_scores:
            partition_scores[indexer(partition)] = tup[2]

    total_sp_score = sum(partition_scores.values())
    for idx in partition_scores:
        partition_scores[idx] /= total_sp_score

    freqs_df = pd.DataFrame.from_dict(partition_freqs, orient='index', columns=['freq'])
    scores_df = pd.DataFrame.from_dict(partition_scores, orient='index', columns=['score'])
    df = pd.concat([freqs_df, scores_df], axis=1)
    df.to_csv(f"{x}x{y}_distributions.csv")
    
if __name__=="__main__":
    run_thing(2,4)
    run_thing(3,4)
    run_thing(4,4)
    run_thing(5,4)