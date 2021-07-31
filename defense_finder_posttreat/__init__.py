import os
import shutil
from defense_finder_posttreat import df_genes, df_systems, best_solutions, df_hmmer
from datetime import datetime

def run(outpath):
    bs = best_solutions.get()

    dir = os.path.join('/', 'tmp', 'defense-finder', 'output')
    if not os.path.exists(dir):
        os.mkdir(dir)

    df_genes.export_defense_finder_genes(bs)
    df_systems.export_defense_finder_systems(bs)
    df_hmmer.export_defense_finder_hmmer_hits()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder = "{}-defense-finder".format(timestamp)
    default_outpath = os.getcwd()
    outpath = outpath if outpath != None else default_outpath

    outdir = os.path.join(outpath, folder)

    if not os.path.isabs(outdir):
        outdir = os.path.join(os.getcwd(), outdir)

    shutil.move(dir, outdir)
    print("Analysis results were written to {}".format(outdir))

