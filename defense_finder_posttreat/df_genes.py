import os
import pandas as pd

def export_defense_finder_genes(defense_finder_genes, outdir, filename):
    esm_file_output = os.path.join(outdir, filename+'_ESMDF.tsv')
    genclr_file_output = os.path.join(outdir, filename+'_GeneCLR_DF.tsv')
    ESM_done = False
    GCLR_done = False
    
    if os.path.exists(esm_file_output):
        esmdf = pd.read_table(esm_file_output)
        esmdf2 = esmdf.merge(defense_finder_genes[['hit_id', 'gene_name', 'sys_id', 'replicon']],
                              on="hit_id", how="outer")
        ESM_done = True
    
    if os.path.exists(genclr_file_output):
        genclrdf = pd.read_table(genclr_file_output)
        genclrdf2 = genclrdf.merge(defense_finder_genes[['hit_id', 'gene_name', 'sys_id', 'replicon']],
                              on="hit_id", how="outer")
        GCLR_done = True

    if ESM_done and GCLR_done:
        outdf = esmdf.merge(genclrdf2, suffixes=("_ESMDF", "_GeneCLRDF"), on="hit_id")
        with open(os.path.join(outdir, filename+'_ESM_GeneCLR_DF.tsv'), "w") as outdfout:
            outdf.to_csv(outdfout, sep="\t", index=False)
        #os.remove(esm_file_output)
        #os.remove(genclr_file_output)

    else:
        if GCLR_done:
            with open(genclr_file_output, "w") as genclrout:
                genclrdf2.to_csv(genclrout, sep="\t", index=False)
        if ESM_done:
            with open(esm_file_output, "w") as esmout:
                esmdf2.to_csv(esmout, sep="\t", index=False)

    defense_finder_genes.to_csv(os.path.join(outdir, filename+'_defense_finder_genes.tsv'), sep='\t', index=False)
