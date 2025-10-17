import os
import pandas as pd

def export_defense_finder_genes(defense_finder_genes, outdir, filename):
    esm_file_output = os.path.join(outdir, filename+'_ESMDF.tsv')
    if os.path.exists(esm_file_output):
        esmdf = pd.read_table(esm_file_output)
        esmdf2 = esmdf.merge(defense_finder_genes[['hit_id', 'gene_name', 'sys_id', 'replicon']],
                              on="hit_id", how="outer")
        #with open(esm_file_output, "w") as esmout:
        #    esmout.write(esmdf2.__repr__())
        esmdf2.to_csv(esm_file_output, sep='\t', mode="w")

    defense_finder_genes.to_csv(os.path.join(outdir, filename+'_defense_finder_genes.tsv'), sep='\t', index=False)
