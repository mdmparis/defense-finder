import os
import csv
from defense_finder_posttreat import best_solution

def export_defense_finder_genes(defense_finder_genes, outdir):
    defense_finder_genes_list = defense_finder_genes_to_list(defense_finder_genes)
    write_defense_finder_genes(defense_finder_genes_list, outdir)

def defense_finder_genes_to_list(defense_finder_genes):
    header = best_solution.get_best_solution_keys()
    out = [header]
    for g in defense_finder_genes:
        l = []
        for key in header:
            l.append(g[key])
        out.append(l)
    return out

def write_defense_finder_genes(defense_finder_genes_list, outdir):
    filepath = os.path.join(outdir, 'defense_finder_genes.tsv')
    with open(filepath, 'w') as defense_finder_genes_file:
        write = csv.writer(defense_finder_genes_file, delimiter='\t')
        write.writerows(defense_finder_genes_list)
