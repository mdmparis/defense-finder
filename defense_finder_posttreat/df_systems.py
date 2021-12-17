import os
import csv
from itertools import groupby
from functools import reduce

def export_defense_finder_systems(defense_finder_genes, outdir):
    systems = build_defense_finder_systems(defense_finder_genes)
    systems_list = systems_to_list(systems)
    write_defense_finder_systems(systems_list, outdir)

def systems_to_list(systems):
    header = get_system_keys()
    out = [header]
    for s in systems:
        l = []
        for key in header:
            l.append(s[key])
        out.append(l)
    return out

def write_defense_finder_systems(systems_list, outdir):
    filepath = os.path.join(outdir, 'defense_finder_systems.tsv')
    with open(filepath, 'w') as defense_finder_systems_file:
        write = csv.writer(defense_finder_systems_file, delimiter='\t')
        write.writerows(systems_list)

def get_system_keys():
    return [ 'sys_id', 'type', 'subtype', 'sys_beg', 'sys_end', 'protein_in_syst', 'genes_count', 'name_of_profiles_in_sys' ]

def projection(val):
    return val['sys_id']

def build_defense_finder_systems(defense_finder_genes):
    system_goups = [list(it) for k, it in groupby(defense_finder_genes, projection)]
    out = []
    for system_group in system_goups:
        item = {}
        first_item = system_group[0]
        last_item = system_group[-1]
        item['sys_id'] = first_item['sys_id']
        item['sys_beg'] = first_item['hit_id']
        item['sys_end'] = last_item['hit_id']
        item['type'] = first_item['type']
        item['subtype'] = first_item['subtype']
        item['protein_in_syst'] = reduce(lambda acc, s: acc + ',' + s['hit_id'], system_group, '')[1:]
        item['genes_count'] = len(system_group)
        item['name_of_profiles_in_sys'] = reduce(lambda acc, s: acc + ',' + s['gene_name'], system_group, '')[1:]
        out.append(item)
    return out
