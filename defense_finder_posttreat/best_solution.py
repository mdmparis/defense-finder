import os
import pandas as pd

from macsypy.serialization import TsvSystemSerializer

def get(tmp_dir):
    results = os.listdir(tmp_dir)
    acc = pd.DataFrame()
    for family_dir in results:
        family_path = os.path.join(tmp_dir, family_dir)
        
        if is_file_empty(family_path+'/best_solution.tsv') == False :
            acc = pd.concat([acc , parse_best_solution(family_path)])
    if acc.empty == True :
        acc = pd.DataFrame(columns=get_best_solution_keys('\t'))

    
    return format_best_solution(acc)

def is_file_empty(path):
    prev_line = ''
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                prev_line = line
            else:
                break
    if prev_line.startswith('# No Systems found'):
        return True
    return False

def parse_best_solution(dir):
    """
    :param dir: the macsyfinder result directory path
    :type dir: str
    """
    delimiter = '\t'
    data=pd.read_table(dir+'/best_solution.tsv',sep=delimiter,comment='#')
    
    return data


def get_best_solution_keys(delimiter='\t'):
    return TsvSystemSerializer.header.split(delimiter)


def format_best_solution(p):
    p['type'] = p.model_fqn.map(lambda x : x.split('/')[-2])
    p['subtype'] = p.model_fqn.map(lambda x : x.split('/')[-1])
    
    p.loc[p.type=='CasFinder','type']='Cas'
    p.loc[p.model_fqn.str.contains("ADF"),'activity']='Antidefense'
    p.loc[~p.model_fqn.str.contains("ADF"),'activity']='Defense'

    p=p.sort_values('hit_pos')

    return(p)
