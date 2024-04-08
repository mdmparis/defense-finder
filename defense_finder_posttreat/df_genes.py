import os
from defense_finder_posttreat import best_solution

def export_defense_finder_genes(defense_finder_genes, outdir, filename):
    defense_finder_genes.to_csv(outdir+'/'+filename+'_defense_finder_genes.tsv',sep='\t',index=False)


def write_defense_finder_genes(defense_finder_genes_list, outdir, filename):
    filepath = os.path.join(outdir,  f'{filename}_defense_finder_genes.tsv')
    defense_finder_genes_list.to_csv('filepath',sep='\t',index=False)
    