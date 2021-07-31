import os
from defense_finder_posttreat import df_genes, df_systems, best_solutions, df_hmmer

def run():
    bs = best_solutions.get()

    dir = os.path.join('/', 'tmp', 'defense-finder', 'output')
    if not os.path.exists(dir):
        os.mkdir(dir)

    df_genes.export_defense_finder_genes(bs)
    df_systems.export_defense_finder_systems(bs)
    df_hmmer.export_defense_finder_hmmer_hits()

run()
