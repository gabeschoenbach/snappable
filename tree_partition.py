from gerrychain.tree import *
from gerrychain.random import random


def get_parent(graph, node):
    for neighbor in graph.neighbors(node):
        return neighbor


def sample_uniform_partition_of_tree(spanning_tree, num_parts, min_pop, max_pop, pop_col):
    # Initialize spanning tree and m, which stores partial partitions of nodes that have been removed.
    #print(spanning_tree.nodes[0][pop_col])
    spanning_tree = spanning_tree.copy()
    m = {node: {(spanning_tree.nodes[node][pop_col], 0, False): ([], [node], 1)} for node in spanning_tree}

    # Main loop, contracts leaves and updates m until there is only one node left.
    leaves = [node for node in spanning_tree if spanning_tree.degree(node) == 1]
    last_node = None
    while last_node is None:
        # Contract leaf from graph.
        leaf = leaves.pop()
        parent = get_parent(spanning_tree, leaf)
        spanning_tree.remove_node(leaf)
        deg = spanning_tree.degree(parent)
        if deg == 0:
            last_node = parent
        elif deg == 1:
            leaves.append(parent)

        # Compute all possible partitions of the parent subtree, labeled by residue and partition size.
        parent_partitions = {}
        def add_parent_partition(residue_size, partition_size, partition, residue, used, weight):
            if partition_size < num_parts or (partition_size == num_parts and residue_size == 0):
                residue_and_partition_size_and_used = (residue_size, partition_size, used)
                partition_and_residue_and_weight = (partition, residue, weight)
                if residue_and_partition_size_and_used in parent_partitions:
                    parent_partitions[residue_and_partition_size_and_used].append(partition_and_residue_and_weight)
                else:
                    parent_partitions[residue_and_partition_size_and_used] = [partition_and_residue_and_weight]
        for (p_residue_size, p_partition_size, p_used), (p_partition, p_residue, p_weight) in m[parent].items():
            for (l_residue_size, l_partition_size, l_used), (l_partition, l_residue, l_weight) in m[leaf].items():
                if p_used and not l_used:
                    continue
                residue_size = p_residue_size + l_residue_size
                if residue_size > max_pop:
                    continue
                partition_size = p_partition_size + l_partition_size
                partition = p_partition + l_partition
                residue = p_residue + l_residue
                weight = p_weight * l_weight
                add_parent_partition(residue_size, partition_size, partition, residue, False, weight)
                if residue_size >= min_pop and not l_used:
                    add_parent_partition(0, partition_size + 1, partition + [residue], [], True, weight)

        # Update m[parent] by sampling a random partition for each possible residue and partition size.
        m.pop(leaf)
        m[parent] = {}
        for residue_and_partition_size_and_used, partial_partitions in parent_partitions.items():
            partitions_and_residues = []
            weights = []
            total_weight = 0
            for partition, residue, weight in partial_partitions:
                partitions_and_residues.append((partition, residue))
                weights.append(weight)
                total_weight += weight
            m[parent][residue_and_partition_size_and_used] =\
                random.choices(partitions_and_residues, weights=weights, k=1)[0] + (total_weight,)
    assert spanning_tree.order() == 1 and last_node in spanning_tree

    # If a valid partition of the entire tree exists, it should contain zero population in its residue.
    #print(m)
    if (0, num_parts, True) in m[last_node]:
        final_partition, empty_residue, num_partitions = m[last_node][(0, num_parts, True)]
        return final_partition, num_partitions
    else:
        return None, 0


