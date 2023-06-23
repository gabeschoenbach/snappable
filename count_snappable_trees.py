import tree_utils as tu
from tqdm import tqdm
from tree_partition import sample_uniform_partition_of_tree

d = 0
trials = 1000

for N in [100]:
  print(f"Checking {N}x{N} grids: ")
  graph = tu.generate_grid_graph(N)
  snappables = 0
  for _ in tqdm(range(trials)):
      tree = tu.uniform_random_spanning_tree(graph)
      # tree = tu.random_minimum_spanning_tree(graph)
      for n in tree:
          tree.nodes[n]['pop'] = 1
          
      ideal_size = len(tree) / 2
      partition, num_balance_edges = sample_uniform_partition_of_tree(tree, 2, ideal_size, ideal_size, "pop")
    #   balance_edges = tu.find_balance_edge(tree)
      if num_balance_edges:
          snappables += 1
  print(snappables/trials)
