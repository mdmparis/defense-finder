import os
import pandas as pd


def remove_duplicates(hmmer_hits):
    # Remove duplicates of hit_id if they have the same gene name
    # For R-M and Retron (multi HMM) only one hit is conserved
    hmmer_hits = hmmer_hits.sort_values('hit_score', ascending=False).drop_duplicates(['hit_id', 'gene_name'])

    return hmmer_hits


def export_defense_finder_hmmer_hits(tmp_dir, outdir, filename):
    paths = get_hmmer_paths(tmp_dir)
    hmmer_hits = pd.DataFrame()

    for path in paths:
        d = parse_hmmer_results_file(path)
        if d.empty is False:
            hmmer_hits = pd.concat([hmmer_hits, remove_duplicates(d)])
    if hmmer_hits.empty is True:
        hmmer_hits = pd.DataFrame(columns=get_hmmer_keys())

    hmmer_hits = remove_duplicates(hmmer_hits)
    hmmer_hits = hmmer_hits.sort_values(['hit_pos', 'hit_score'])

    write_defense_finder_hmmer(hmmer_hits, outdir, filename)


def write_defense_finder_hmmer(hmmer_hits, outdir, filename):
    hmmer_hits.to_csv(os.path.join(outdir, filename+"_defense_finder_hmmer.tsv"), sep='\t', index=False)


def get_hmmer_keys():
    return ['hit_id', 'replicon', 'hit_pos', 'hit_sequence_length', 'gene_name', 'i_eval', 'hit_score', 'hit_profile_cov', 'hit_seq_cov', 'hit_begin_match', 'hit_end_match']


def parse_hmmer_results_file(path):
    data = pd.read_table(path, sep='\t', comment='#', names=get_hmmer_keys())
    return data


def get_hmmer_paths(results_dir):
    family_dirs = os.listdir(results_dir)
    files = []
    for family_dir in family_dirs:
        hmmer_results_dir = os.path.join(results_dir, family_dir, 'hmmer_results')
        with os.scandir(hmmer_results_dir) as it:
            for entry in it:
                if entry.name.endswith('extract') and entry.is_file():
                    files.append(entry)
    return list(map(lambda i: i.path, files))
