import os
import csv

def get(tmp_dir):
    results = os.listdir(tmp_dir)
    acc = []

    for family_dir in results:
        family_path = os.path.join(tmp_dir, family_dir)
        acc = acc + parse_best_solution(family_path)

    return format_best_solution(acc)

def parse_best_solution(dir):
    tsv_file = open(os.path.join(dir, 'best_solution.tsv'))
    tsv = csv.reader(tsv_file, delimiter='\t')
    data = []
    for row in tsv: data.append(row)
    tsv_file.close()
    if "No Systems found" in ' '.join(data[2]):
        return []
    else:
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

def get_best_solution_keys():
    return [
            'replicon', 'hit_id', 'gene_name',
            'hit_pos', 'model_fqn', 'sys_id', 'sys_loci', 'locus_num',
            'sys_wholeness', 'sys_score', 'sys_occ','hit_gene_ref',
            'hit_status','hit_seq_len','hit_i_eval', 'hit_score',
            'hit_profile_cov', 'hit_seq_cov', 'hit_begin_match',
            'hit_end_match', 'counterpart', 'used_in'
            ]

def format_best_solution(p):
    out = []
    for l in p:
        gene_ref = l['model_fqn']
        gene_ref_elements = gene_ref.split('/')
        type = gene_ref_elements[2]
        subtype = gene_ref_elements[3]
        native_keys = list(filter(lambda i: i not in ['type', 'subtype'], get_best_solution_keys()))
        new_line = { key: l[key] for key in native_keys }
        new_line['type'] = type
        new_line['subtype'] = subtype
        out.append(new_line)
    return out
