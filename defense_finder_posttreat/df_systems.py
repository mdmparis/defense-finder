import os
import pandas as pd

def export_defense_finder_systems(defense_finder_genes, outdir, filename):
    systems = build_defense_finder_systems(defense_finder_genes)
    systems.to_csv(outdir+'/'+filename+'_defense_finder_systems.tsv',sep='\t',index=False)


def build_defense_finder_systems(defense_finder_genes):
    sys=defense_finder_genes.drop_duplicates('sys_id')[['sys_id' , 'type' , 'subtype', 'activity']]

    sys_beg=defense_finder_genes.sort_values('hit_pos').drop_duplicates('sys_id').rename({'hit_id' : 'sys_beg'},axis=1)[['sys_id','sys_beg']]
    sys_end=defense_finder_genes.sort_values('hit_pos' , ascending=False).drop_duplicates('sys_id').rename({'hit_id' : 'sys_end'},axis=1)[['sys_id','sys_end']]
    protein_in_syst=defense_finder_genes.groupby('sys_id').hit_id.apply(lambda x: ",".join(x.sort_values())).reset_index().rename({'hit_id':'protein_in_syst'},axis=1)
    name_of_profiles_in_sys=defense_finder_genes.groupby('sys_id').gene_name.apply(lambda x : ",".join(x.sort_values())).reset_index().rename({'hit_id' : 'protein_in_syst'},axis = 1)
    genes_count=defense_finder_genes.sys_id.value_counts().reset_index()
    genes_count.columns=['sys_id','genes_count']


    out=sys.merge(sys_beg,on = 'sys_id')
    out=out.merge(sys_end,on = 'sys_id')
    out=out.merge(protein_in_syst,on = 'sys_id')
    out=out.merge(genes_count,on = 'sys_id')
    out=out.merge(name_of_profiles_in_sys,on = 'sys_id')
    
    return(out)

