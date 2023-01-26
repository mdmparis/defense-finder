import os
import csv

from macsypy.serialization import TsvSystemSerializer
from macsypy.registries import split_def_name

def get(tmp_dir):
    results = os.listdir(tmp_dir)
    acc = []

    for family_dir in results:
        family_path = os.path.join(tmp_dir, family_dir)
        acc = acc + parse_best_solution(family_path)

    return format_best_solution(acc)

def uncomment(csvfile):
    """
    generator which yield lines in macsyfinder/best_solution files but skip comments or empty lines

    :param csvfile: the csv to parse
    :type cvsfile: file object
    """
    for line in csvfile:
        uncommented = line.split('#')[0].strip()
        if uncommented:
            yield uncommented


def parse_best_solution(dir):
    """
    :param dir: the macsyfinder result directory path
    :type dir: str
    """
    delimiter = '\t'
    with open(os.path.join(dir, 'best_solution.tsv')) as tsv_file:
        tsv = csv.DictReader(uncomment(tsv_file),
                             fieldnames=get_best_solution_keys(delimiter=delimiter),
                             delimiter=delimiter)
        try:
            next(tsv)
        except StopIteration:
            return []
        data = list(tsv)
    return data


def get_best_solution_keys(delimiter='\t'):
    return TsvSystemSerializer.header.split(delimiter)


def format_best_solution(p):
    out = []
    for l in p:
        gene_ref = l['model_fqn']
        gene_ref_elements = split_def_name(gene_ref)
        print("#### gene_ref_elements", gene_ref_elements)
        type = gene_ref_elements[-2]
        subtype = gene_ref_elements[-1]
        native_keys = list(filter(lambda i: i not in ['type', 'subtype'], get_best_solution_keys()))
        new_line = { key: l[key] for key in native_keys }
        new_line['type'] = type
        new_line['subtype'] = subtype
        out.append(new_line)
    return out
