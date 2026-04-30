import os
import pandas as pd

def export_defense_finder_genes(defense_finder_genes, outdir, filename):
    esm_file_output = os.path.join(outdir, filename+'_ESMDF.tsv')
    genclr_file_output = os.path.join(outdir, filename+'_GeneCLR_DF.tsv')
    ESM_done = False
    GCLR_done = False
    breakpoint()
    if os.path.exists(esm_file_output):
        esmdf = pd.read_table(esm_file_output)
        esmdf2 = esmdf.merge(defense_finder_genes[['hit_id', 'gene_name', 'sys_id', 'replicon']],
                              on="hit_id", how="outer")
        ESM_done = True
    elif os.path.exists(genclr_file_output):
        genclrdf = pd.read_table(genclr_file_output)
        genclrdf2 = genclrdf.merge(defense_finder_genes[['hit_id', 'gene_name', 'sys_id', 'replicon']],
                              on="hit_id", how="outer")
        GCLR_done = True

    if ESM_done and GCLR_done:
        outdf = esmdf2.merge(genclrdf, suffixes=("_ESMDF", "_GeneCLRDF"))
        print(outdf)
        with open(os.path.join(outdir, filename+'_ESM_GeneCLR_DF.tsv'), "w") as esmout:
            esmout.write(esmdf2.__repr__())
        os.remove(esm_file_output)
        os.remove(genclr_file_output)

    else:
        if GCLR_done:
            with open(genclr_file_output, "w") as genclrout:
                genclrout.write(genclrdf2.__repr__())
        if ESM_done:
            with open(esm_file_output, "w") as esmout:
                esmout.write(esmdf2.__repr__())

    defense_finder_genes.to_csv(os.path.join(outdir, filename+'_defense_finder_genes.tsv'), sep='\t', index=False)
