import os

import shutil
from defense_finder_posttreat import df_genes, df_systems, best_solution, df_hmmer
from datetime import datetime

def run(tmp_dir, outdir):
    # Run analysis
    bs = best_solution.get(tmp_dir)

    df_genes.export_defense_finder_genes(bs, outdir)
    df_systems.export_defense_finder_systems(bs, outdir)
    df_hmmer.export_defense_finder_hmmer_hits(tmp_dir, outdir)

    print("Analysis results were written to {}".format(outdir))
