import os
from defense_finder_posttreat import df_genes, df_systems, best_solutions

def run():
    bs = best_solutions.get()

    dir = os.path.join('/', 'tmp', 'defense-finder', 'results')
    if not os.path.exists(dir):
        os.mkdir(dir)

    df_genes.export_defense_finder_genes(bs)
    df_systems.export_defense_finder_systems(bs)
