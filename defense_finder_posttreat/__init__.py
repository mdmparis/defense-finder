import subprocess
import csv
import os
from itertools import groupby
from functools import reduce

def run():
    all_best_solutions = parse_all_best_solutions()
    defense_finder_genes = format_defense_finder_genes(all_best_solutions)

    dir = os.path.join('/', 'tmp', 'defense-finder', 'results')
    if not os.path.exists(dir):
        os.mkdir(dir)

    defense_finder_genes_list = defense_finder_genes_to_list(defense_finder_genes)
    write_defense_finder_genes(defense_finder_genes_list)

    systems = build_defense_finder_systems(defense_finder_genes)
    systems_list = systems_to_list(systems)
    write_defense_finder_systems(systems_list)

def systems_to_list(systems):
    header = get_system_keys()
    out = [header]
    for s in systems:
        l = []
        for key in header:
            l.append(s[key])
        out.append(l)
    return out

def write_defense_finder_systems(systems_list):
    with open('/tmp/defense-finder/results/defense_finder_systems.tsv', 'w') as defense_finder_systems_file:
        write = csv.writer(defense_finder_systems_file, delimiter='\t')
        write.writerows(systems_list)

def get_system_keys():
    return [ 'sys_id', 'type', 'subtype', 'sys_beg', 'sys_end', 'protein_in_syst', 'genes_count', 'name_of_profiles_in_sys' ]

def projection(val): return val['sys_id']

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

def defense_finder_genes_to_list(defense_finder_genes):
    header = get_defense_finder_genes_keys()
    out = [header]
    for g in defense_finder_genes:
        l = []
        for key in header:
            l.append(g[key])
        out.append(l)
    return out

def write_defense_finder_genes(defense_finder_genes):
    with open('/tmp/defense-finder/results/defense_finder_genes.tsv', 'w') as defense_finder_genes_file:
        write = csv.writer(defense_finder_genes_file, delimiter='\t')
        write.writerows(defense_finder_genes)

def cut_defense_finder_genes_line(l):
    l = l[:13]
    return l

def handle_defense_finder_genes_header(l):
    l = cut_defense_finder_genes_line(l)
    l.pop(5)
    l.append('type')
    l.append('subtype')
    return l

def get_defense_finder_genes_keys():
    return [
            'sol_id', 'replicon', 'hit_id', 'gene_name',
            'hit_pos', 'sys_id', 'sys_loci', 'locus_num',
            'sys_wholeness', 'sys_score', 'sys_occ',
            'hit_gene_ref', 'type', 'subtype'
            ]

def format_defense_finder_genes(p):
    out = []
    for l in p:
        gene_ref = l['model_fqn']
        gene_ref_elements = gene_ref.split('/')
        type = gene_ref_elements[1]
        subtype = gene_ref_elements[2]
        native_keys = list(filter(lambda i: i not in ['type', 'subtype'], get_defense_finder_genes_keys()))
        new_line = { key: l[key] for key in native_keys }
        new_line['type'] = type
        new_line['subtype'] = subtype
        out.append(new_line)
    return out

def parse_all_best_solutions():
    tsv_file = open('/tmp/defense-finder/all_best_solutions.tsv')
    tsv = csv.reader(tsv_file, delimiter='\t')
    data = []
    for row in tsv: data.append(row)
    tsv_file.close()
    data = data[3:]
    header = data.pop(0)
    out = []
    for l in data:
        if not l: continue
        line_as_dict = {}
        for idx, val in enumerate(header):
            line_as_dict[val] = l[idx]
        out.append(line_as_dict)
    return out
