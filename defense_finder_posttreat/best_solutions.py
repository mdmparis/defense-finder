import csv

def get():
    return format_best_solutions(parse_all())

def parse_all():
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

def get_best_solutions_keys():
    return [
            'sol_id', 'replicon', 'hit_id', 'gene_name',
            'hit_pos', 'sys_id', 'sys_loci', 'locus_num',
            'sys_wholeness', 'sys_score', 'sys_occ',
            'hit_gene_ref', 'type', 'subtype'
            ]

def format_best_solutions(p):
    out = []
    for l in p:
        gene_ref = l['model_fqn']
        gene_ref_elements = gene_ref.split('/')
        type = gene_ref_elements[1]
        subtype = gene_ref_elements[2]
        native_keys = list(filter(lambda i: i not in ['type', 'subtype'], get_best_solutions_keys()))
        new_line = { key: l[key] for key in native_keys }
        new_line['type'] = type
        new_line['subtype'] = subtype
        out.append(new_line)
    return out