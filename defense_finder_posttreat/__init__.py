import os

import shutil
from defense_finder_posttreat import df_genes, df_systems, best_solution, df_hmmer
from datetime import datetime
import colorlog

logger = colorlog.getLogger("Defense_Finder")
def run(tmp_dir, outdir, filename):
    # Run analysis
    bs = best_solution.get(tmp_dir)

    df_genes.export_defense_finder_genes(bs, outdir, filename)
    df_systems.export_defense_finder_systems(bs, outdir, filename)
    df_hmmer.export_defense_finder_hmmer_hits(tmp_dir, outdir, filename)

    logger.info("Analysis results were written to {}".format(outdir))
