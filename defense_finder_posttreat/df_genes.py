import os


def export_defense_finder_genes(defense_finder_genes, outdir, filename):
    defense_finder_genes.to_csv(os.path.join(outdir, filename+'_defense_finder_genes.tsv'), sep='\t', index=False)
